"""Chat SSE endpoint tests: event mapping, validation, and the concurrency cap."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_tenant, get_model_provider
from app.core.concurrency import StreamLimiter
from app.core.config import Settings, get_settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
from app.providers.base import ChatChunk, ModelInfo
from app.providers.fake_provider import FakeProvider
from app.providers.rag.embedder import FakeEmbedder
from app.workers.queue import FakeJobQueue
from tests.conftest import make_memory_store

CHAT_URL = "/api/v1/chat/stream"

# A chat pair. Every fake here reports holding them AND make_settings below
# configures the default as one of them, so these stream-shape tests exercise
# a normally provisioned machine: holding the tag without configuring it (or
# the reverse) would send each turn down the missing-model rescue path instead
# of the event mapping they are about.
DEV_PAIR = [ModelInfo(name="qwen3:4b-instruct"), ModelInfo(name="qwen3:4b")]


def provisioned(*args: object, **kwargs: object) -> FakeProvider:
    """A FakeProvider that reports holding the configured chat pair."""
    kwargs.setdefault("models", list(DEV_PAIR))
    return FakeProvider(*args, **kwargs)  # type: ignore[arg-type]


# A hang guard, never an ordering device: every wait below is on an event that
# is already set or is set by this test, so a healthy run never approaches it.
GATE_TIMEOUT = 10.0


class GatedProvider(FakeProvider):
    """A FakeProvider whose stream announces its start, then waits to be let go.

    The concurrency cap is about two requests genuinely OVERLAPPING, and a sleep
    cannot establish an overlap - it only makes one likely, which is how the
    two requests came to swap roles on a loaded runner. These two events make
    the overlap a fact instead: ``started`` fires once the first request is past
    the yield dependency that takes the slot, and ``release`` keeps the first
    request inside its stream - holding that slot - until the assertions about
    the second one are done.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Create the fake with the usual script, plus its two gate events."""
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def chat_stream(self, *args: object, **kwargs: object) -> AsyncIterator[ChatChunk]:
        """Signal that the stream has begun, wait for the gate, then stream."""
        self.started.set()
        await self.release.wait()
        async for chunk in super().chat_stream(*args, **kwargs):  # type: ignore[arg-type]
            yield chunk


@pytest.fixture
def fake_provider() -> FakeProvider:
    # Override the conftest default with a deterministic two-token script.
    return provisioned(["Hello", " world"])


def make_settings(**overrides: object) -> Settings:
    # These stream-shape tests exercise the pre-tool path; the shipped default
    # AGENCY_MODE=agentic would add tool offers and a tools_not_used notice that
    # are not what they assert, so pin the deterministic mode. Overridable.
    overrides.setdefault("agency_mode", "deterministic")
    # With no tier profile left to supply a default, an unconfigured deployment
    # answers ANY turn with a "no default chat model" status frame ahead of
    # everything else - not what these tests are about, so configure the pair's
    # default explicitly (held by every fake here; see DEV_PAIR above).
    overrides.setdefault("default_chat_model", "qwen3:4b-instruct")
    return Settings(_env_file=None, model_provider="fake", **overrides)


@asynccontextmanager
async def client_for(provider: FakeProvider, db_sessionmaker, settings: Settings | None = None):
    """Build an in-process client wired to a specific provider (and settings)."""
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    settings = settings or make_settings()
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.db_sessionmaker = db_sessionmaker
    app.state.vector_store = make_memory_store()
    app.state.embedder = FakeEmbedder(dim=32)
    app.state.stream_limiter = StreamLimiter(settings.max_streams_per_tenant)
    # Faked for the same reason as storage: the lazy fallback builds a real Redis
    # pool, so a first turn (which queues the title job) would try to connect.
    app.state.job_queue = FakeJobQueue()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client:
        yield http_client
    app.dependency_overrides.clear()


async def collect_events(client: AsyncClient, payload: dict) -> list[tuple[str, dict]]:
    """POST a chat request and parse the SSE stream into (event, data) pairs."""
    events: list[tuple[str, dict]] = []
    current_event = ""
    async with client.stream("POST", CHAT_URL, json=payload) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("event: "):
                current_event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((current_event, json.loads(line.removeprefix("data: "))))
    return events


async def test_chat_stream_emits_tokens_and_done(client: AsyncClient) -> None:
    lines: list[str] = []
    async with client.stream("POST", CHAT_URL, json={"message": "hi"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for line in resp.aiter_lines():
            if line:
                lines.append(line)

    body = "\n".join(lines)
    assert "event: token" in body
    assert "Hello" in body and "world" in body
    assert "event: done" in body


async def test_chat_stream_rejects_empty_message(client: AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json={"message": ""})
    assert resp.status_code == 422


async def test_chat_stream_rejects_bad_thinking_mode(client: AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json={"message": "hi", "thinking": "maybe"})
    assert resp.status_code == 422


async def test_thinking_frames_emitted_before_tokens(db_sessionmaker) -> None:
    provider = provisioned(["answer"], thinking_script=["step one", " step two"])
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": "hi"})

    names = [name for name, _ in events]
    assert names == ["meta", "thinking", "thinking", "token", "done"]
    assert events[1][1] == {"delta": "step one"}


async def test_the_stream_opens_by_naming_its_conversation(db_sessionmaker) -> None:
    # The meta frame precedes everything - tokens, thinking, even a pre-token
    # status - so a client that never reaches done (an interrupted turn, a
    # dropped connection) can still bind its pane to the thread the backend
    # created, follow up on it, and watch for its model-written title.
    provider = provisioned(["ok"])
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": "hi"})

    assert events[0][0] == "meta"
    assert [n for n, _ in events].count("meta") == 1
    assert events[0][1]["conversation_id"] == dict(events)["done"]["conversation_id"]


async def test_scripted_error_maps_code_and_ends_stream(db_sessionmaker) -> None:
    provider = provisioned([], error="model exploded", error_code="bad_request")
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": "hi"})

    assert events[-1][0] == "error"
    assert events[-1][1] == {"message": "model exploded", "code": "bad_request"}
    assert all(name != "done" for name, _ in events)


async def test_raising_provider_yields_generic_internal_error(db_sessionmaker) -> None:
    provider = provisioned(["tok"], raise_exc=RuntimeError("secret internal detail"))
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": "hi"})

    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "internal_error"
    assert "secret internal detail" not in json.dumps(events)


async def test_done_frame_carries_usage_and_persisted_ids(db_sessionmaker) -> None:
    provider = provisioned(["a", "b", "c"])
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": "hi"})

    done = dict(events)["done"]
    assert done["usage"]["completion_tokens"] == 3
    assert done["message_id"]
    assert done["conversation_id"]
    # The context meter rides the same frame: how much of the window this
    # turn's prompt filled, both numbers the backend's own - tagged with the
    # model that actually ran, so a client can tell whose window it was.
    assert done["context_usage"]["num_ctx"] == 8192
    assert done["context_usage"]["prompt_tokens"] > 0
    assert done["context_usage"]["model"] == "qwen3:4b-instruct"


async def test_a_server_with_no_chat_model_maps_the_first_run_error(db_sessionmaker) -> None:
    # A deployment pointed at a server holding nothing it can chat with. The
    # turn cannot start, but it is not a crash: the client gets a named code
    # to act on and a message that says what to do about it.
    provider = provisioned(["never streamed"], models=[])
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": "hi"})

    assert [name for name, _ in events] == ["meta", "error"]
    assert events[-1][1]["code"] == "no_chat_model_configured"
    assert "Pull a chat model" in events[-1][1]["message"]


async def test_legacy_history_reaches_provider(db_sessionmaker) -> None:
    provider = provisioned(["ok"])
    payload = {
        "message": "and now?",
        "history": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ],
    }
    async with client_for(provider, db_sessionmaker) as client:
        await collect_events(client, payload)

    contents = [m.content for m in provider.last_messages if m.role != "system"]
    assert contents == ["first", "second", "and now?"]


async def test_chat_stream_emits_status_frame(db_sessionmaker) -> None:
    provider = provisioned(["ok"], status="Loading the model into memory.")
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": "hi"})

    # The status is the first frame after meta (the stream names its thread
    # before it narrates anything).
    assert events[1] == (
        "status",
        {"message": "Loading the model into memory.", "code": "loading_model"},
    )
    assert events[-1][0] == "done"


async def test_concurrent_streams_capped_then_slot_released(db_sessionmaker) -> None:
    # Cap of 1: while a stream holds the slot, a second request must get a
    # problem+json 429; once the first completes, the slot is free again.
    provider = GatedProvider(["a", "b", "c"], models=list(DEV_PAIR))
    settings = make_settings(max_streams_per_tenant=1, sse_heartbeat_seconds=0)
    async with client_for(provider, db_sessionmaker, settings) as client:
        first = asyncio.create_task(client.post(CHAT_URL, json={"message": "one"}))
        # The slot is held from here on: the fake only reaches its gate after
        # the yield dependency that acquires it, and it will not stream a
        # single chunk until this test releases it.
        await asyncio.wait_for(provider.started.wait(), timeout=GATE_TIMEOUT)

        second = await client.post(CHAT_URL, json={"message": "two"})
        assert second.status_code == 429
        assert second.headers["content-type"].startswith("application/problem+json")
        assert second.json()["code"] == "too_many_requests"

        provider.release.set()
        first_resp = await asyncio.wait_for(first, timeout=GATE_TIMEOUT)
        assert first_resp.status_code == 200
        assert "event: done" in first_resp.text

        third = await client.post(CHAT_URL, json={"message": "three"})
        assert third.status_code == 200
        assert "event: done" in third.text


async def test_one_members_streams_do_not_spend_a_colleagues_budget(db_sessionmaker) -> None:
    # MUTATION GUARD: key the limiter on the tenant alone and Bob's request 429s
    # here, which is the starvation this change exists to stop - one colleague's
    # long tool turns holding the whole workspace's budget.
    #
    # Alice's slot is taken directly rather than by holding a real stream open,
    # because what is under test is the KEY the slot is filed under, and a
    # limiter primed by hand states that without a second gated turn in flight.
    provider = provisioned(["ok"])
    settings = make_settings(max_streams_per_tenant=1)
    alice = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")
    bob = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="member")
    acting = alice

    limiter = StreamLimiter(settings.max_streams_per_tenant)
    assert await limiter.acquire(alice.tenant_id, alice.user_id) is True

    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_current_tenant] = lambda: acting
    app.state.db_sessionmaker = db_sessionmaker
    app.state.vector_store = make_memory_store()
    app.state.embedder = FakeEmbedder(dim=32)
    app.state.stream_limiter = limiter
    app.state.job_queue = FakeJobQueue()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Alice is at her own cap and is refused, so the cap is still a cap.
        refused = await client.post(CHAT_URL, json={"message": "hi"})
        assert refused.status_code == 429
        assert refused.json()["code"] == "too_many_requests"

        acting = bob
        served = await client.post(CHAT_URL, json={"message": "hi"})
        assert served.status_code == 200
        assert "event: done" in served.text
    app.dependency_overrides.clear()


async def test_the_single_user_deployment_sees_no_change(db_sessionmaker) -> None:
    # Every request today carries the same pair of ids, so the per-user key
    # collapses to exactly the behaviour the per-tenant one had: one slot, and
    # the seeded principal's second concurrent request is refused.
    limiter = StreamLimiter(1)
    assert await limiter.acquire(DEFAULT_TENANT_ID, DEFAULT_USER_ID) is True
    assert await limiter.acquire(DEFAULT_TENANT_ID, DEFAULT_USER_ID) is False

    provider = provisioned(["ok"])
    settings = make_settings(max_streams_per_tenant=1)
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.db_sessionmaker = db_sessionmaker
    app.state.vector_store = make_memory_store()
    app.state.embedder = FakeEmbedder(dim=32)
    app.state.stream_limiter = limiter
    app.state.job_queue = FakeJobQueue()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # The seeded principal - no dependency override - lands on the slot
        # taken above, proving the route keys on the ids it actually carries.
        refused = await client.post(CHAT_URL, json={"message": "hi"})
        assert refused.status_code == 429
    app.dependency_overrides.clear()


async def test_disconnect_before_response_start_releases_slot(db_sessionmaker) -> None:
    # A client gone before http.response.start means the response body is
    # never iterated; the slot release must not depend on the server starting
    # the SSE generator. Simulated at the raw ASGI layer: the first send()
    # raises, exactly like uvicorn on a pre-response disconnect (spec >= 2.4).
    limiter = StreamLimiter(1)
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provisioned(["never sent"])
    app.state.db_sessionmaker = db_sessionmaker
    app.state.vector_store = make_memory_store()
    app.state.embedder = FakeEmbedder(dim=32)
    app.state.stream_limiter = limiter
    app.state.job_queue = FakeJobQueue()

    body = json.dumps({"message": "hi"}).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": CHAT_URL,
        "raw_path": CHAT_URL.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    body_sent = False

    async def receive() -> dict:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        raise OSError("client disconnected before the response started")

    # The disconnect surfaces as ClientDisconnect/OSError (version-dependent);
    # what matters here is the slot, not the exception shape.
    with suppress(BaseException):
        await app(scope, receive, send)

    # Deterministic: the slot returns via dependency teardown as the request
    # unwinds - never via generator finalization at garbage collection.
    assert await limiter.acquire(DEFAULT_TENANT_ID, DEFAULT_USER_ID), (
        "stream slot leaked after a pre-response-start disconnect"
    )
    app.dependency_overrides.clear()
