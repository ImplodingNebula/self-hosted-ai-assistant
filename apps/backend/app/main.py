"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from http import HTTPStatus

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import ProgrammingError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1 import health
from app.api.v1.router import router as v1_router
from app.core.concurrency import StreamLimiter
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.factories import build_model_provider, build_search_provider, build_storage_backend
from app.core.logging import configure_logging, get_request_id
from app.core.middleware import MaxBodySizeMiddleware, RequestIdMiddleware
from app.core.ratelimit import RateLimitMiddleware
from app.db.engine import build_engine, build_sessionmaker
from app.db.seed import seed_defaults
from app.providers.rag.qdrant_store import QdrantVectorStore
from app.providers.search.base import SearchProvider
from app.services.warmup import warm_search
from app.workers.queue import ArqJobQueue

logger = logging.getLogger(__name__)

# Slack over the per-file limit for the multipart envelope (boundaries, the
# small conversation_id/origin form fields, part headers) so the coarse body
# guard never rejects a legitimately-sized upload; the handler's exact per-file
# check does the precise enforcement.
_MULTIPART_ENVELOPE_HEADROOM_BYTES = 1_048_576


async def _warm_search_bg(provider: SearchProvider) -> None:
    """Prime the search backend in the background so the first user search is fast.

    Best-effort and fire-and-forget: a cold or briefly-unavailable SearXNG must
    never stall or crash startup, so every failure is swallowed (the first real
    search then simply pays the cold-start cost, as before). Cancelled cleanly
    on shutdown.
    """
    try:
        took_ms, results = await warm_search(provider)
        logger.info("Search backend warmed in %dms (%d results).", took_ms, results)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - warmup is best-effort; never crash startup
        logger.warning(
            "Search-backend warmup failed; the first web search may be slower.",
            exc_info=True,
        )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the process-wide singletons once; tear them down on shutdown."""
    settings = get_settings()
    app.state.model_provider = build_model_provider(settings)
    engine = build_engine(settings)
    app.state.db_engine = engine
    app.state.db_sessionmaker = build_sessionmaker(engine)
    app.state.stream_limiter = StreamLimiter(settings.max_streams_per_tenant)
    # No network I/O at construction; collections are created lazily on the
    # first ingest, so a slow Qdrant boot cannot block or crash the backend.
    app.state.vector_store = QdrantVectorStore(url=settings.qdrant_url)
    # Likewise nothing is written or created here: the storage root is made on
    # the first upload, so a missing volume surfaces when a file is stored
    # rather than taking the boot down.
    app.state.storage = build_storage_backend(settings)
    # Built eagerly (construction opens no connection) rather than on the first
    # enqueue, because the rate limiter needs a counter from request ONE - and
    # a limiter with no counter fails open, which is silent.
    app.state.job_queue = ArqJobQueue(settings.redis_url)
    # The limiter is handed a counter, not a queue: the queue happens to hold
    # the process's one Redis pool, so lending it avoids a second pool for a
    # single INCR per request. See app.core.ratelimit.RateCounter.
    app.state.rate_counter = app.state.job_queue
    app.state.search_warm_task = None
    try:
        await seed_defaults(app.state.db_sessionmaker)
    except ProgrammingError as exc:
        # The database answered but the schema is missing or stale: every data
        # request would 500 while /healthz stays green, which reads as a
        # half-broken app instead of a deploy mistake (seen live: a stale
        # backend image whose startup command predates migrate-then-serve).
        # Abort the boot with the remedy front and center.
        raise RuntimeError(
            "The database is reachable but its schema is missing or out of "
            "date - migrations have not been applied. Rebuild the backend "
            "image and refresh volumes (make rebuild), or apply migrations "
            "directly (make db-upgrade)."
        ) from exc
    except Exception:
        # Boot resilience: compose starts postgres first, but the API must
        # still come up if the DB is briefly down; requests fail until it
        # returns and the seed rows land on the next boot (or already exist).
        logger.warning("Default-tenant seeding failed; is the database up?", exc_info=True)
    # Pre-build the search provider and warm it in the background so the first
    # user web search does not pay SearXNG's cold-start (DNS, empty pool, cold
    # worker) - the "first search fails, second works" case. Non-blocking: a slow
    # SearXNG boot never delays the API coming up, and sharing this one instance
    # with the request path means the pool it warms is the one reused. Kicked off
    # AFTER seeding so a fatal seed abort above cannot orphan a pending task.
    if settings.web_search_enabled:
        search_provider = build_search_provider(settings)
        if search_provider is not None:
            app.state.search_provider = search_provider
            app.state.search_warm_task = asyncio.create_task(_warm_search_bg(search_provider))
    yield
    # Cancel the background warmup before closing the provider it uses, so it is
    # never mid-request when its HTTP client is torn down.
    warm_task = getattr(app.state, "search_warm_task", None)
    if warm_task is not None and not warm_task.done():
        warm_task.cancel()
        with suppress(asyncio.CancelledError):
            await warm_task
    try:
        for attr in (
            "model_provider",
            "vector_store",
            "reranker",
            "search_provider",
            "page_fetcher",
            # Only ever built if somebody ran a download preflight.
            "model_registry_client",
            # LocalFsStorage holds no client and so has no aclose; the loop
            # skips it. S3Storage owns a connection pool and does have one.
            "storage",
            # Only ever built if something enqueued; the pool is lazy.
            "job_queue",
        ):
            holder = getattr(app.state, attr, None)
            aclose = getattr(holder, "aclose", None)
            if callable(aclose):
                await aclose()
    finally:
        # The pool must be disposed even if a client close raises.
        await engine.dispose()


def _problem_response(
    status: int,
    title: str,
    detail: str,
    code: str,
    headers: dict[str, str] | None = None,
    **extensions: object,
) -> JSONResponse:
    """Build an RFC 7807 problem+json response stamped with the request id.

    X-Request-ID is set here as well as by the middleware because the
    catch-all Exception handler runs in Starlette's outermost error
    middleware, above RequestIdMiddleware, where its response bypasses the
    middleware's header hook.

    Args:
        status: HTTP status code.
        title: Short human-readable summary, constant per problem type.
        detail: Instance-specific explanation, safe to show to a user.
        code: Stable machine-readable code for clients.
        headers: Extra response headers (e.g. from an HTTPException).
        **extensions: Additional problem-document members (e.g. ``errors``).
    """
    request_id = get_request_id()
    body: dict[str, object] = {
        "type": "about:blank",
        "title": title,
        "status": status,
        "detail": detail,
        "code": code,
        "request_id": request_id,
    }
    body.update(extensions)
    response_headers = dict(headers or {})
    if request_id:
        response_headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=status,
        content=body,
        media_type="application/problem+json",
        headers=response_headers or None,
    )


async def _app_error_handler(request: Request, exc: AppError) -> Response:
    """Render a typed AppError as an RFC 7807 problem+json response."""
    # 4xx AppErrors are expected business outcomes; only server faults are
    # errors worth alerting on (the access line records the status either way).
    if exc.status >= 500:
        logger.error("%s: %s", exc.code, exc.detail)
    return _problem_response(exc.status, exc.title, exc.detail, exc.code)


# Contract-enumerated codes win over the phrase-derived slug: HTTPStatus(500)
# would otherwise yield "internal_server_error" where clients expect
# "internal_error".
_STATUS_TO_CODE = {
    400: "bad_request",
    404: "not_found",
    409: "conflict",
    422: "validation_error",
    429: "too_many_requests",
    500: "internal_error",
    503: "service_unavailable",
}


async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
    """Render framework HTTPExceptions (e.g. unknown-route 404s) as problem+json."""
    if exc.status_code in {204, 304}:
        # Body-less statuses must stay body-less.
        return Response(status_code=exc.status_code, headers=exc.headers)
    try:
        phrase = HTTPStatus(exc.status_code).phrase
    except ValueError:
        phrase = "HTTP Error"
    code = _STATUS_TO_CODE.get(exc.status_code, phrase.lower().replace(" ", "_"))
    detail = exc.detail if isinstance(exc.detail, str) else phrase
    return _problem_response(exc.status_code, phrase, detail, code, headers=exc.headers)


async def _validation_error_handler(request: Request, exc: RequestValidationError) -> Response:
    """Render request validation failures as a 422 problem+json.

    Only loc/msg/type are exposed per pydantic error - ``ctx`` can carry raw
    exception objects and ``input`` echoes payloads; neither belongs on the wire.
    """
    errors = [
        {"loc": list(err.get("loc", ())), "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]
    return _problem_response(
        422,
        "Validation Error",
        "Request validation failed.",
        "validation_error",
        errors=errors,
    )


async def _unhandled_exception_handler(request: Request, exc: Exception) -> Response:
    """Last-resort handler: log the real exception, return a generic 500."""
    logger.error("unhandled exception", exc_info=exc)
    return _problem_response(
        500,
        "Internal Server Error",
        "The request failed unexpectedly.",
        "internal_error",
    )


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings)
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=_lifespan)
    # An oversized body is refused before it is read, so no request can spool a
    # disk-filling upload. The ceiling is the largest legitimate body (a file
    # upload) plus headroom for the multipart envelope; the upload handler still
    # applies the exact per-file limit after parsing. Added FIRST so it sits
    # inside CORS: the 413 it writes then bubbles back out through CORS and
    # carries the same headers the handler's own per-file 413 does.
    app.add_middleware(
        MaxBodySizeMiddleware,
        max_bytes=settings.attachment_max_bytes + _MULTIPART_ENVELOPE_HEADROOM_BYTES,
    )
    # Inside CORS so its 429 carries the same CORS headers a route's own would,
    # and inside the request id so the refusal still gets an id and an access
    # line. Outside the body guard, which is free and needs no counter.
    app.add_middleware(
        RateLimitMiddleware,
        enabled=settings.rate_limit_enabled,
        per_minute=settings.rate_limit_per_minute,
        heavy_per_minute=settings.rate_limit_heavy_per_minute,
        light_per_minute=settings.rate_limit_light_per_minute,
        trusted_proxies=frozenset(settings.trusted_proxy_ips),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Added last = outermost user middleware, so the request id exists for the
    # whole request (including the 413 above and CORS preflight short-circuits).
    app.add_middleware(RequestIdMiddleware)
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
    app.include_router(health.router)
    app.include_router(v1_router)
    return app


app = create_app()
