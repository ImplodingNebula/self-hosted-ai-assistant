"""SSE heartbeat tests: the with_heartbeat wrapper and its endpoint wiring."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_model_provider
from app.core.concurrency import StreamLimiter
from app.core.config import Settings, get_settings
from app.main import create_app
from app.providers.fake_provider import FakeProvider
from app.providers.rag.embedder import FakeEmbedder
from app.sse import KEEP_ALIVE_FRAME, with_heartbeat
from app.workers.queue import FakeJobQueue
from tests.conftest import make_memory_store

CHAT_URL = "/api/v1/chat/stream"


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, model_provider="fake", **overrides)


@asynccontextmanager
async def client_for(provider: FakeProvider, db_sessionmaker, settings: Settings):
    """Build an in-process client with overridden settings (heartbeat interval)."""
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.db_sessionmaker = db_sessionmaker
    app.state.vector_store = make_memory_store()
    app.state.embedder = FakeEmbedder(dim=32)
    app.state.stream_limiter = StreamLimiter(settings.max_streams_per_tenant)
    # Faked like storage: the lazy fallback opens a real Redis pool, and a first
    # turn now queues the background title job.
    app.state.job_queue = FakeJobQueue()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client:
        yield http_client
    app.dependency_overrides.clear()


# --- with_heartbeat unit behavior ---


async def test_passthrough_when_interval_disabled() -> None:
    async def source():
        yield "a"
        await asyncio.sleep(0.1)
        yield "b"

    frames = [frame async for frame in with_heartbeat(source(), 0)]
    assert frames == ["a", "b"]


async def test_fast_source_passes_through_unchanged() -> None:
    async def source():
        for frame in ["a", "b", "c"]:
            yield frame

    frames = [frame async for frame in with_heartbeat(source(), 5.0)]
    assert frames == ["a", "b", "c"]


async def test_keep_alive_inserted_during_quiet_gap() -> None:
    async def source():
        yield "first"
        await asyncio.sleep(0.2)
        yield "second"

    frames = [frame async for frame in with_heartbeat(source(), 0.05)]
    assert frames[0] == "first"
    assert frames[-1] == "second"
    assert KEEP_ALIVE_FRAME in frames[1:-1]
    # Nothing but keep-alives was inserted; source frames stayed in order.
    assert [f for f in frames if f != KEEP_ALIVE_FRAME] == ["first", "second"]


async def test_producer_exception_propagates_after_yielded_frames() -> None:
    async def failing():
        yield "a"
        raise RuntimeError("boom")

    collected: list[str] = []
    with pytest.raises(RuntimeError, match="boom"):
        async for frame in with_heartbeat(failing(), 0.05):
            collected.append(frame)
    assert collected == ["a"]


async def test_early_close_cancels_producer_and_closes_source() -> None:
    closed = asyncio.Event()

    async def source():
        try:
            yield "a"
            await asyncio.sleep(10)
            yield "never"
        finally:
            closed.set()

    gen = with_heartbeat(source(), 0.05)
    assert await anext(gen) == "a"
    await gen.aclose()
    assert closed.is_set()


# --- endpoint wiring ---


async def test_endpoint_emits_keep_alive_between_slow_chunks(db_sessionmaker) -> None:
    provider = FakeProvider(["a", "b"], delay=0.15)
    settings = make_settings(sse_heartbeat_seconds=0.05)
    async with client_for(provider, db_sessionmaker, settings) as client:
        resp = await client.post(CHAT_URL, json={"message": "hi"})

    assert resp.status_code == 200
    assert ": keep-alive" in resp.text
    assert "event: done" in resp.text


async def test_endpoint_emits_no_keep_alive_when_disabled(db_sessionmaker) -> None:
    provider = FakeProvider(["a", "b"], delay=0.12)
    settings = make_settings(sse_heartbeat_seconds=0)
    async with client_for(provider, db_sessionmaker, settings) as client:
        resp = await client.post(CHAT_URL, json={"message": "hi"})

    assert resp.status_code == 200
    assert ": keep-alive" not in resp.text
    assert "event: done" in resp.text
