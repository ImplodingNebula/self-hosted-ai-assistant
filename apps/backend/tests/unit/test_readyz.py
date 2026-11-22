"""Readiness probe: the hard/informational split, the verdict, and the timeouts.

The one behaviour worth stating twice: an informational dependency must NEVER
change the status code. The deployment this ships into runs its model server on
a remote pod that can spend well over a minute loading a model, so a probe that
counted it would pull a perfectly healthy API out of rotation on every swap.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from redis.exceptions import ConnectionError as RedisConnectionError

import app.workers.queue as queue_module
from app.core.config import Settings, get_settings
from app.main import create_app
from app.providers.base import ProviderUnavailableError
from app.providers.fake_provider import FakeProvider
from app.providers.rag.base import VectorStoreUnavailableError
from app.schemas.health import DependencyCheck
from app.services import readiness
from app.services.readiness import check_readiness, verdict
from app.workers.queue import ArqJobQueue, FakeJobQueue, JobQueueUnavailableError
from tests.conftest import make_memory_store


class StubStore:
    """Only ``ping`` is ever called on the store by the readiness service."""

    def __init__(self, *, fail: bool = False, hang: bool = False) -> None:
        self.fail = fail
        self.hang = hang
        self.pings = 0

    async def ping(self) -> None:
        self.pings += 1
        if self.hang:
            await asyncio.sleep(30)
        if self.fail:
            raise VectorStoreUnavailableError("The vector store is unavailable.")


class StubProvider(FakeProvider):
    """A model provider whose inventory call fails the way a dead server does."""

    async def list_models(self):
        raise ProviderUnavailableError("The model server is unavailable.")


def _sidecar_client(status: int = 200) -> httpx.AsyncClient:
    """An httpx client whose every request answers with ``status``."""
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _req: httpx.Response(status)))


async def _report(db_sessionmaker, **over):
    kwargs = {
        "sessionmaker": db_sessionmaker,
        "job_queue": FakeJobQueue(),
        "vector_store": StubStore(),
        "model_provider": FakeProvider(),
        "searxng_url": None,
        "reranker_url": None,
    }
    kwargs.update(over)
    return await check_readiness(**kwargs)


def _by_name(report) -> dict:
    return {check.name: check for check in report.checks}


async def test_everything_up_is_ready(db_sessionmaker) -> None:
    report = await _report(db_sessionmaker)

    checks = _by_name(report)
    assert report.status == "ready"
    assert [checks[n].state for n in ("postgres", "redis", "qdrant")] == ["ok"] * 3
    assert checks["ollama"].state == "ok"
    assert all(c.latency_ms >= 0 for c in report.checks)


async def test_every_required_dependency_is_probed_for_real(db_sessionmaker) -> None:
    # Not a stand-in each: the sqlite sessionmaker really executes SELECT 1, the
    # fake queue really answers ping, and the store really counts its call.
    store = StubStore()
    report = await _report(db_sessionmaker, vector_store=store)

    assert store.pings == 1
    assert _by_name(report)["postgres"].state == "ok"


async def test_a_dead_required_dependency_makes_it_unready(db_sessionmaker) -> None:
    report = await _report(db_sessionmaker, job_queue=FakeJobQueue(unavailable=True))

    checks = _by_name(report)
    assert report.status == "unready"
    assert checks["redis"].state == "unreachable"
    assert checks["redis"].required is True
    # One dead dependency must not smear over the others.
    assert checks["postgres"].state == "ok"
    assert checks["qdrant"].state == "ok"


async def test_a_dead_vector_store_makes_it_unready(db_sessionmaker) -> None:
    report = await _report(db_sessionmaker, vector_store=StubStore(fail=True))

    assert report.status == "unready"
    assert _by_name(report)["qdrant"].state == "unreachable"


async def test_informational_failures_never_flip_the_verdict(db_sessionmaker) -> None:
    # THE guard: every optional dependency is down at once - the model server is
    # unreachable and both sidecars answer 500 - and the verdict is still ready.
    report = await _report(
        db_sessionmaker,
        model_provider=StubProvider(),
        searxng_url="http://searxng:8080",
        reranker_url="http://reranker:7997",
        http_client=_sidecar_client(500),
    )

    checks = _by_name(report)
    assert report.status == "ready"
    assert checks["ollama"].state == "unreachable"
    assert checks["searxng"].state == "unreachable"
    assert checks["reranker"].state == "unreachable"
    assert all(checks[n].required is False for n in ("ollama", "searxng", "reranker"))


def test_verdict_ignores_every_informational_check() -> None:
    # The verdict function on its own, so the rule survives a refactor of the
    # probes around it.
    required_ok = DependencyCheck(name="postgres", required=True, state="ok", latency_ms=1.0)
    optional_dead = DependencyCheck(
        name="ollama", required=False, state="unreachable", latency_ms=1.0
    )
    required_dead = DependencyCheck(
        name="redis", required=True, state="unreachable", latency_ms=1.0
    )

    assert verdict([required_ok, optional_dead]) == "ready"
    assert verdict([required_ok, required_dead]) == "unready"
    assert verdict([]) == "ready"


async def test_configured_sidecars_that_answer_are_ok(db_sessionmaker) -> None:
    report = await _report(
        db_sessionmaker,
        searxng_url="http://searxng:8080/",
        reranker_url="http://reranker:7997",
        http_client=_sidecar_client(200),
    )

    checks = _by_name(report)
    assert checks["searxng"].state == "ok"
    assert checks["reranker"].state == "ok"


async def test_sidecar_probes_hit_each_products_own_health_path(db_sessionmaker) -> None:
    seen: list[str] = []

    def _record(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(_record))
    # A trailing slash on the base URL must not double up in the probe path.
    await _report(
        db_sessionmaker,
        searxng_url="http://searxng:8080/",
        reranker_url="http://reranker:7997",
        http_client=client,
    )

    assert sorted(seen) == ["http://reranker:7997/health", "http://searxng:8080/healthz"]


async def test_absent_sidecars_are_unconfigured_not_unreachable(db_sessionmaker) -> None:
    # A deployment without a reranker is a NORMAL deployment (the dev tier is
    # exactly that), so it must not read as a broken one.
    report = await _report(db_sessionmaker, searxng_url=None, reranker_url="")

    checks = _by_name(report)
    assert report.status == "ready"
    assert checks["searxng"].state == "unconfigured"
    assert checks["reranker"].state == "unconfigured"


async def test_a_hanging_probe_is_cut_off_and_reported(
    db_sessionmaker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(readiness, "REQUIRED_TIMEOUT_SECONDS", 0.05)
    started = time.perf_counter()
    report = await _report(db_sessionmaker, vector_store=StubStore(hang=True))
    elapsed = time.perf_counter() - started

    checks = _by_name(report)
    assert report.status == "unready"
    assert checks["qdrant"].state == "unreachable"
    assert "0.05s" in (checks["qdrant"].detail or "")
    # The point of the timeout: the probe slept for 30s and the report did not.
    assert elapsed < 5


async def test_probes_run_concurrently(db_sessionmaker) -> None:
    # Asserted by OBSERVED OVERLAP rather than by wall clock: a timing
    # threshold on a loaded machine is a flake waiting to happen, and it only
    # ever measured concurrency indirectly. Counting how many probes are in
    # flight at once says it exactly.
    live = 0
    peak = 0

    async def _tracked() -> None:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        try:
            # One event-loop turn is enough: every probe must have been started
            # before any of them is awaited to completion.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        finally:
            live -= 1

    class _Tracked(StubStore):
        async def ping(self) -> None:
            await _tracked()

    class _TrackedQueue(FakeJobQueue):
        async def ping(self) -> None:
            await _tracked()

    class _TrackedProvider(FakeProvider):
        async def list_models(self):
            await _tracked()
            return []

    report = await _report(
        db_sessionmaker,
        vector_store=_Tracked(),
        job_queue=_TrackedQueue(),
        model_provider=_TrackedProvider(),
    )

    assert report.status == "ready"
    # Serial execution can never show more than one probe in flight.
    assert peak == 3


async def test_probe_detail_never_quotes_the_exception_text(db_sessionmaker) -> None:
    # The report is unauthenticated, and a connection error routinely quotes the
    # URL it failed to reach - which can carry a credential in its userinfo.
    class _Leaky(StubStore):
        async def ping(self) -> None:
            raise VectorStoreUnavailableError("cannot reach http://user:hunter2@qdrant:6333")

    report = await _report(db_sessionmaker, vector_store=_Leaky())

    detail = _by_name(report)["qdrant"].detail or ""
    assert "hunter2" not in detail
    assert "VectorStoreUnavailableError" in detail


async def test_the_real_qdrant_store_answers_a_ping() -> None:
    # The seam method against genuine client code, not a stub of it.
    store = make_memory_store()
    try:
        await store.ping()
    finally:
        await store.aclose()


async def test_the_arq_queue_reports_an_unreachable_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Pool:
        async def ping(self) -> None:
            raise RedisConnectionError("connection refused")

    async def _create(*_args, **_kwargs):
        return _Pool()

    monkeypatch.setattr(queue_module, "create_pool", _create)

    with pytest.raises(JobQueueUnavailableError):
        await ArqJobQueue("redis://localhost:6379/0").ping()


async def test_the_arq_queue_pings_a_reachable_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    pinged = 0

    class _Pool:
        async def ping(self) -> None:
            nonlocal pinged
            pinged += 1

    async def _create(*_args, **_kwargs):
        return _Pool()

    monkeypatch.setattr(queue_module, "create_pool", _create)
    await ArqJobQueue("redis://localhost:6379/0").ping()

    assert pinged == 1


# --- the endpoint --------------------------------------------------------


def _local_settings() -> Settings:
    """Settings with both optional sidecars off, so the probe reaches no network."""
    return get_settings().model_copy(update={"searxng_url": None, "reranker_url": None})


async def _probe_app(db_sessionmaker, *, job_queue=None):
    """An app whose every dependency is in-process, so the probe touches no network."""
    app = create_app()
    app.dependency_overrides[get_settings] = _local_settings
    app.state.db_sessionmaker = db_sessionmaker
    app.state.vector_store = make_memory_store()
    app.state.job_queue = job_queue or FakeJobQueue()
    # Without this the lazy fallback builds a real OllamaProvider and the
    # informational probe dials a model server that is not there.
    app.state.model_provider = FakeProvider()
    return app


async def test_readyz_answers_200_when_the_required_dependencies_answer(db_sessionmaker) -> None:
    app = await _probe_app(db_sessionmaker)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/readyz")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    names = {c["name"] for c in body["checks"]}
    assert {"postgres", "redis", "qdrant", "ollama", "searxng", "reranker"} == names
    assert all("latency_ms" in c for c in body["checks"])


async def test_readyz_reports_search_as_unconfigured_when_the_provider_is_none(
    db_sessionmaker,
) -> None:
    # A URL is still set, but the deployment switched search off. Reporting a
    # sidecar it deliberately does not use - and dialling it every probe -
    # would be the same inference SEARCH_PROVIDER exists to remove.
    def _off() -> Settings:
        return get_settings().model_copy(
            update={"search_provider": "none", "searxng_url": "http://searxng:8080"}
        )

    app = await _probe_app(db_sessionmaker)
    app.dependency_overrides[get_settings] = _off
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/readyz")

    assert resp.status_code == 200
    searxng = next(c for c in resp.json()["checks"] if c["name"] == "searxng")
    assert searxng["state"] == "unconfigured"


async def test_readyz_answers_503_when_a_required_dependency_is_down(db_sessionmaker) -> None:
    app = await _probe_app(db_sessionmaker, job_queue=FakeJobQueue(unavailable=True))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/readyz")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "unready"
    redis = next(c for c in body["checks"] if c["name"] == "redis")
    assert redis["state"] == "unreachable"


async def test_readyz_is_logged_at_debug(db_sessionmaker, caplog: pytest.LogCaptureFixture) -> None:
    app = await _probe_app(db_sessionmaker)
    transport = ASGITransport(app=app)
    with caplog.at_level(logging.DEBUG, logger="app.access"):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/readyz")

    assert resp.status_code == 200
    line = next(r for r in caplog.records if r.name == "app.access" and r.path == "/readyz")
    assert line.levelno == logging.DEBUG
