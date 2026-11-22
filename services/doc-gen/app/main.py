"""The doc-gen sidecar: one MCP server that renders files, mounted at /docgen.

``build_app`` composes a parent Starlette app that mounts the server's
Streamable-HTTP ASGI app under a path prefix, so clients reach it at
``/docgen/mcp``. The mount is its own registration in the platform's server
registry: its own credential, its own tool toggles.

Two SDK facts shape this module, exactly as they shape the MCP chassis:

- Starlette never runs a MOUNTED app's lifespan, and a FastMCP session manager
  only accepts requests while running - so the parent app's lifespan runs the
  session manager explicitly.
- The SDK's DNS-rebinding protection ships with an empty allow list; the
  accepted Host values come from ``DOCGEN_ALLOWED_HOSTS`` and default to
  localhost plus the compose service name.
"""

from __future__ import annotations

import contextlib
import logging
import os
import secrets
from collections.abc import AsyncIterator

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import BaseRoute, Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from app.server import build_docgen_server

logger = logging.getLogger("doc_gen")

_DEFAULT_ALLOWED_HOSTS = "localhost,localhost:*,127.0.0.1,127.0.0.1:*,doc-gen,doc-gen:*"


def _allowed_hosts_from_env() -> list[str]:
    """The Host allow list, from ``DOCGEN_ALLOWED_HOSTS`` (comma-separated)."""
    raw = os.environ.get("DOCGEN_ALLOWED_HOSTS", _DEFAULT_ALLOWED_HOSTS)
    hosts = [host.strip() for host in raw.split(",") if host.strip()]
    if not hosts:
        # An empty list is the SDK's reject-everything state: every request
        # would 421 with nothing in the logs pointing at this env var.
        logger.warning(
            "DOCGEN_ALLOWED_HOSTS is set but empty; falling back to the default allow list (%s).",
            _DEFAULT_ALLOWED_HOSTS,
        )
        return _DEFAULT_ALLOWED_HOSTS.split(",")
    return hosts


class BearerAuthMiddleware:
    """Pure ASGI middleware requiring ``Authorization: Bearer <key>`` on every request.

    Pure ASGI rather than ``BaseHTTPMiddleware`` because the MCP server streams
    SSE responses and BaseHTTPMiddleware buffers bodies. ``/healthz`` stays open
    so an orchestrator can probe liveness without holding the key.
    """

    def __init__(
        self,
        app: ASGIApp,
        api_key: str,
        open_paths: frozenset[str] = frozenset({"/healthz"}),
    ) -> None:
        self._app = app
        # Exact-bytes compare, scheme case included: RFC 7235 allows "bearer",
        # but strictness keeps the whole check one constant-time comparison.
        self._expected = f"Bearer {api_key}".encode()
        self._open_paths = open_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Rejects unauthenticated requests; only lifespan events pass through freely."""
        if scope["type"] == "lifespan":
            await self._app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            # This service serves no websockets; denying outright keeps a future
            # websocket transport from landing behind zero auth.
            await receive()
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["path"] in self._open_paths and scope.get("method") == "GET":
            await self._app(scope, receive, send)
            return
        supplied = next(
            (value for name, value in scope["headers"] if name == b"authorization"),
            b"",
        )
        # compare_digest, not ==: a timing oracle on the key is cheap to avoid.
        if not secrets.compare_digest(supplied, self._expected):
            response = PlainTextResponse(
                "unauthorized",
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)


async def _healthz(_: Request) -> PlainTextResponse:
    """Liveness probe; deliberately outside the auth gate."""
    return PlainTextResponse("ok")


def build_app() -> Starlette:
    """Composes the doc-gen app from env configuration.

    Returns:
        The ASGI app uvicorn serves: the Streamable-HTTP mount at ``/docgen``
        plus ``GET /healthz``, wrapped in bearer-token auth when
        ``DOCGEN_API_KEY`` is set.
    """
    allowed_hosts = _allowed_hosts_from_env()
    server = build_docgen_server(allowed_hosts)

    routes: list[BaseRoute] = [
        Route("/healthz", _healthz, methods=["GET"]),
        Mount("/docgen", app=server.streamable_http_app()),
    ]

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        # Starlette never runs a mounted app's lifespan, so the session manager
        # (which FastMCP's own lifespan would have started) is run here, for the
        # parent app's whole lifetime.
        async with server.session_manager.run():
            yield

    api_key = os.environ.get("DOCGEN_API_KEY")
    if api_key is not None and not api_key.strip():
        # A set-but-blank key is what a failed compose variable substitution
        # produces; booting silently open would defeat the intended auth.
        logger.warning(
            "DOCGEN_API_KEY is set but blank - an empty value usually means a "
            "failed variable substitution; treating it as unset."
        )
        api_key = None
    if api_key:
        middleware = [Middleware(BearerAuthMiddleware, api_key=api_key)]
    else:
        middleware = []
        logger.warning(
            "DOCGEN_API_KEY is not set; doc-gen accepts unauthenticated requests "
            "(suitable only on a private network)."
        )
    return Starlette(routes=routes, lifespan=lifespan, middleware=middleware)


app = build_app()
