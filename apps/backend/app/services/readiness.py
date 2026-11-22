"""The readiness probe: can this deployment actually serve a request?

Liveness (``/healthz``) says the process is running. Readiness says the things
it cannot work without are reachable, and it is deliberately a NARROWER claim
than "everything is healthy".

Two classes of dependency, and the split is the whole design:

Required
    Postgres, Redis and Qdrant. Every request path touches at least one of
    them and none has a degraded mode - a conversation cannot be written, a
    parse job cannot be queued, a passage cannot be retrieved. A failure here
    answers 503, which is what takes an instance out of a load balancer or
    holds a rolling deploy.

Informational
    The model server, the search sidecar and the reranker. These are reported
    but never vote. The reasoning is deployment-shaped: the model server is
    routinely REMOTE and can spend well over a minute loading a cold model, so
    making it a hard dependency would flap an otherwise-fine deployment out of
    service on every model swap. The other two are optional by design and the
    app already degrades around them (retrieval goes dense-only, a turn stays
    local). Publishing them still matters - a missing reranker is invisible
    from the outside, and "quietly worse retrieval" is exactly the kind of
    thing an operator should be able to see.

Every probe runs concurrently under its own timeout, so the endpoint costs
about as long as its slowest check rather than the sum of them.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.providers.base import ModelProvider
from app.providers.rag.base import VectorStore
from app.schemas.health import DependencyCheck, DependencyState, ReadinessReport
from app.workers.queue import JobQueue

logger = logging.getLogger(__name__)

# Per-probe wall-clock budgets. Constants rather than settings: these bound how
# long a health check may take, and a probe endpoint that can be configured to
# hang is a probe endpoint that can wedge whatever polls it. The required ones
# are tighter because their answer gates traffic; an informational one may take
# a moment longer, since nothing waits on its verdict.
REQUIRED_TIMEOUT_SECONDS = 2.0
INFORMATIONAL_TIMEOUT_SECONDS = 3.0

# Where each optional sidecar answers a liveness question. Different paths
# because these are different products, not one convention.
_SEARXNG_HEALTH_PATH = "/healthz"
_RERANKER_HEALTH_PATH = "/health"


def _probe(call: Callable[..., Awaitable[object]], *args: object) -> Callable[[], Awaitable[None]]:
    """Bind a call into a zero-argument probe, discarding whatever it returns.

    Binding happens NOW rather than inside a lambda, so a probe built in a loop
    cannot read the loop variable after it has moved on.
    """

    async def run() -> None:
        await call(*args)

    return run


async def _timed(
    name: str, *, required: bool, probe: Callable[[], Awaitable[None]], budget_seconds: float
) -> DependencyCheck:
    """Run one probe under its budget and render the outcome as a check.

    Every exception is caught deliberately: a probe that raised has answered
    the question it was asked - "is this reachable" - and letting it escape
    would turn one dead dependency into a 500 for the whole report.

    Args:
        name: The dependency's name, as it appears in the report.
        required: Whether a failure here makes the deployment unready.
        probe: The zero-argument coroutine function to run.
        budget_seconds: Wall-clock budget for the probe.

    Returns:
        The check result, with the elapsed time in milliseconds.
    """
    started = time.perf_counter()
    state: DependencyState = "ok"
    detail: str | None = None
    try:
        await asyncio.wait_for(probe(), timeout=budget_seconds)
    except TimeoutError:
        state = "unreachable"
        detail = f"No answer within {budget_seconds:g}s."
    except Exception as exc:  # noqa: BLE001 - a failed probe is a result, not a fault
        state = "unreachable"
        # The exception TYPE only. Its message can quote a URL, and a URL can
        # carry a credential in its userinfo; this report is unauthenticated.
        detail = f"Unreachable ({type(exc).__name__})."
        logger.debug("Readiness probe for %s failed", name, exc_info=exc)
    return DependencyCheck(
        name=name,
        required=required,
        state=state,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        detail=detail,
    )


def _unconfigured(name: str) -> DependencyCheck:
    """Report a dependency this deployment deliberately does not have."""
    return DependencyCheck(
        name=name,
        required=False,
        state="unconfigured",
        latency_ms=0.0,
        detail="Not configured for this deployment.",
    )


def verdict(checks: Sequence[DependencyCheck]) -> Literal["ready", "unready"]:
    """Decide the overall status from the individual checks.

    Required checks alone decide it. This is the one line the whole endpoint
    exists to protect: an informational dependency folded into the verdict
    would pull a working deployment out of service every time a remote model
    server took its time loading a model.

    Args:
        checks: Every dependency result, required and informational alike.

    Returns:
        ``"ready"`` when every REQUIRED check passed, else ``"unready"``.
    """
    return "ready" if all(c.state == "ok" for c in checks if c.required) else "unready"


async def _probe_database(sessionmaker: async_sessionmaker[AsyncSession]) -> None:
    """Open a session and round-trip the cheapest statement there is."""
    async with sessionmaker() as session:
        await session.execute(text("SELECT 1"))


async def _probe_url(client: httpx.AsyncClient, url: str) -> None:
    """GET ``url`` and require a non-error status."""
    response = await client.get(url)
    response.raise_for_status()


async def check_readiness(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    job_queue: JobQueue,
    vector_store: VectorStore,
    model_provider: ModelProvider,
    searxng_url: str | None,
    reranker_url: str | None,
    http_client: httpx.AsyncClient | None = None,
) -> ReadinessReport:
    """Probe every backing service concurrently and report what they said.

    Collaborators are passed in rather than resolved here, so this stays free
    of the web layer and testable without a running stack.

    Args:
        sessionmaker: The process session factory (the postgres probe).
        job_queue: The job-queue seam (the redis probe).
        vector_store: The vector-store seam (the qdrant probe).
        model_provider: The model seam (the informational model-server probe).
        searxng_url: The search sidecar's base URL, or None when unconfigured.
        reranker_url: The reranker sidecar's base URL, or None when unconfigured.
        http_client: Client for the two sidecar probes. One is built and closed
            here when omitted - a probe endpoint is polled every few seconds at
            most, so a connection per call costs nothing worth pooling for.

    Returns:
        The full report, including the overall ready/unready verdict.
    """
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=INFORMATIONAL_TIMEOUT_SECONDS)
    unconfigured: list[str] = []
    try:
        probes: list[Awaitable[DependencyCheck]] = [
            _timed(
                "postgres",
                required=True,
                probe=_probe(_probe_database, sessionmaker),
                budget_seconds=REQUIRED_TIMEOUT_SECONDS,
            ),
            _timed(
                "redis",
                required=True,
                probe=_probe(job_queue.ping),
                budget_seconds=REQUIRED_TIMEOUT_SECONDS,
            ),
            _timed(
                "qdrant",
                required=True,
                probe=_probe(vector_store.ping),
                budget_seconds=REQUIRED_TIMEOUT_SECONDS,
            ),
            # Informational, and the one it matters most to keep that way: this
            # server is routinely remote, and a cold load holds the connection
            # open far longer than any readiness budget should wait.
            _timed(
                "ollama",
                required=False,
                probe=_probe(model_provider.list_models),
                budget_seconds=INFORMATIONAL_TIMEOUT_SECONDS,
            ),
        ]
        for name, base_url, path in (
            ("searxng", searxng_url, _SEARXNG_HEALTH_PATH),
            ("reranker", reranker_url, _RERANKER_HEALTH_PATH),
        ):
            if not base_url:
                unconfigured.append(name)
                continue
            probes.append(
                _timed(
                    name,
                    required=False,
                    probe=_probe(_probe_url, client, f"{base_url.rstrip('/')}{path}"),
                    budget_seconds=INFORMATIONAL_TIMEOUT_SECONDS,
                )
            )
        checks = list(await asyncio.gather(*probes))
    finally:
        if owns_client:
            await client.aclose()

    checks.extend(_unconfigured(name) for name in unconfigured)
    return ReadinessReport(status=verdict(checks), checks=checks)
