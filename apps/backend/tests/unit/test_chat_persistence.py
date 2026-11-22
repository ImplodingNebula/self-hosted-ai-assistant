"""Persistence behavior of the chat stream: conversations, messages, history rebuild."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete

from app.api.deps import get_model_provider
from app.core.concurrency import StreamLimiter
from app.core.config import Settings, get_settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
from app.models import Message, utcnow
from app.providers.base import ChatChunk, ChatMessage, GenerationOptions, ModelInfo
from app.providers.fake_provider import FakeProvider
from app.providers.rag.embedder import FakeEmbedder
from app.repositories.conversations import ConversationRepository
from app.repositories.messages import MessageRepository
from app.schemas.chat import ChatRequest
from app.services.chat_service import stream_chat
from app.services.model_resolution import resolve_chat_models
from app.workers.queue import FakeJobQueue
from tests.conftest import make_memory_store

CHAT_URL = "/api/v1/chat/stream"
TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")

# The configured chat pair. Every fake here reports holding them, so these
# persistence tests run against a normally provisioned machine: a fake listing
# none of them would have the turn substitute (or adopt) some other tag, and the
# persisted row would record that rescue rather than the routing these tests are
# about.
CHAT_MODEL = "qwen3:4b-instruct"
THINKING_MODEL = "qwen3:4b"
DEV_PAIR = [ModelInfo(name=CHAT_MODEL), ModelInfo(name=THINKING_MODEL)]


def provisioned(*args: object, **kwargs: object) -> FakeProvider:
    """A FakeProvider that reports holding the configured chat pair.

    Args:
        *args: Passed through to :class:`FakeProvider`.
        **kwargs: Passed through to :class:`FakeProvider`.

    Returns:
        The fake, listing the pair unless the caller seeded its own models.
    """
    kwargs.setdefault("models", list(DEV_PAIR))
    return FakeProvider(*args, **kwargs)  # type: ignore[arg-type]


def make_settings(**overrides: object) -> Settings:
    """Settings for a deployment configured with the pair above.

    Args:
        **overrides: Fields to set on top of the pair configuration.

    Returns:
        Settings built without reading the developer's own ``.env``.
    """
    # These persistence tests assert the pre-tool turn shape (no tool offers, no
    # tools_not_used notice); the shipped default AGENCY_MODE=agentic would add
    # both, so pin the deterministic mode. Overridable.
    overrides.setdefault("agency_mode", "deterministic")
    overrides.setdefault("default_chat_model", CHAT_MODEL)
    overrides.setdefault("thinking_chat_model", THINKING_MODEL)
    return Settings(_env_file=None, model_provider="fake", **overrides)


@asynccontextmanager
async def client_for(provider: FakeProvider, db_sessionmaker, limiter: StreamLimiter | None = None):
    """Build an in-process client wired to a specific provider instance."""
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: make_settings()
    app.state.db_sessionmaker = db_sessionmaker
    app.state.vector_store = make_memory_store()
    app.state.embedder = FakeEmbedder(dim=32)
    app.state.stream_limiter = limiter or StreamLimiter(get_settings().max_streams_per_tenant)
    # Faked like storage: the lazy fallback opens a real Redis pool, and a first
    # turn now queues the background title job.
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


async def load_conversation(db_sessionmaker, conversation_id: UUID):
    """Fetch a conversation and its ordered messages in a fresh session."""
    async with db_sessionmaker() as session:
        repo = ConversationRepository(session, TENANT)
        return await repo.get_with_messages(conversation_id)


async def test_first_turn_creates_conversation_and_persists_turn(db_sessionmaker) -> None:
    provider = provisioned(["Nice ", "to ", "meet ", "you."])
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": "Hello there, who are you?"})

    done = dict(events)["done"]
    assert done["conversation_id"] and done["message_id"]

    conversation, messages = await load_conversation(db_sessionmaker, UUID(done["conversation_id"]))
    assert conversation.title == "Hello there, who are you?"
    assert conversation.user_id == DEFAULT_USER_ID
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "Hello there, who are you?"
    assert messages[1].content == "Nice to meet you."
    # The persisted model is the resolved one - the same chain the chat path
    # routes through, not the tag the request happened to reach.
    assert messages[1].model == resolve_chat_models(make_settings()).default
    # A finished reply keeps what it took to write and nothing else: the fake
    # counts its four tokens and times none of them.
    assert messages[1].meta == {"usage": {"completion_tokens": 4, "generation_ms": None}}
    assert str(messages[1].id) == done["message_id"]


async def test_first_message_titles_a_pre_created_empty_conversation(db_sessionmaker) -> None:
    # A conversation created empty (e.g. to receive an attachment before the
    # first turn) has no title; its first message must derive one, the same as
    # an implicitly created conversation gets one.
    async with db_sessionmaker() as session:
        conversation = await ConversationRepository(session, TENANT).create()
        await session.commit()
        conversation_id = str(conversation.id)
    assert conversation.title is None

    provider = provisioned(["ok"])
    async with client_for(provider, db_sessionmaker) as client:
        await collect_events(
            client, {"message": "Summarise the attached report", "conversation_id": conversation_id}
        )

    conv, messages = await load_conversation(db_sessionmaker, UUID(conversation_id))
    assert conv.title == "Summarise the attached report"
    assert [m.role for m in messages] == ["user", "assistant"]


async def test_second_message_does_not_retitle_a_pre_created_conversation(db_sessionmaker) -> None:
    # Only the FIRST message titles a title-less conversation; a later turn must
    # never overwrite the title it already earned.
    async with db_sessionmaker() as session:
        conversation = await ConversationRepository(session, TENANT).create()
        await session.commit()
        conversation_id = str(conversation.id)

    provider = provisioned(["a", "b"])
    async with client_for(provider, db_sessionmaker) as client:
        await collect_events(
            client, {"message": "first real message", "conversation_id": conversation_id}
        )
        await collect_events(client, {"message": "a follow-up", "conversation_id": conversation_id})

    conv, _ = await load_conversation(db_sessionmaker, UUID(conversation_id))
    assert conv.title == "first real message"


async def test_second_turn_replays_history_and_bumps_updated_at(db_sessionmaker) -> None:
    provider = provisioned(["assistant reply"])
    async with client_for(provider, db_sessionmaker) as client:
        events1 = await collect_events(client, {"message": "first question"})
        conversation_id = dict(events1)["done"]["conversation_id"]
        conv1, _ = await load_conversation(db_sessionmaker, UUID(conversation_id))

        events2 = await collect_events(
            client, {"message": "second question", "conversation_id": conversation_id}
        )

    # The provider saw: system prompt, then the persisted turn, then the new message.
    contents = [m.content for m in provider.last_messages if m.role != "system"]
    assert contents == ["first question", "assistant reply", "second question"]

    done2 = dict(events2)["done"]
    assert done2["conversation_id"] == conversation_id
    conv2, messages = await load_conversation(db_sessionmaker, UUID(conversation_id))
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    assert str(messages[-1].id) == done2["message_id"]
    assert conv2.updated_at > conv1.updated_at


async def test_unknown_conversation_id_returns_404_problem_json(client: AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json={"message": "hi", "conversation_id": str(uuid4())})
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["code"] == "not_found"


async def test_malformed_conversation_id_returns_400_problem_json(client: AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json={"message": "hi", "conversation_id": "not-a-uuid"})
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["code"] == "bad_request"


async def test_rejected_conversation_releases_stream_slot(db_sessionmaker) -> None:
    # A 404 raised in the prepare phase must not leak the acquired slot.
    limiter = StreamLimiter(1)
    async with client_for(provisioned(), db_sessionmaker, limiter=limiter) as client:
        resp = await client.post(CHAT_URL, json={"message": "hi", "conversation_id": str(uuid4())})
        assert resp.status_code == 404
    assert await limiter.acquire(DEFAULT_TENANT_ID, DEFAULT_USER_ID) is True


async def test_token_budget_truncation_drops_oldest_persisted_turns(db_sessionmaker) -> None:
    # Budget of 5 tokens ~ 20 chars: only the 3 newest 6-char turns survive.
    settings = make_settings(history_token_budget=5)
    async with db_sessionmaker() as session:
        conv_repo = ConversationRepository(session, TENANT)
        msg_repo = MessageRepository(session, TENANT)
        conversation = await conv_repo.create(title="seeded")
        base = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
        for i in range(5):
            await msg_repo.add(
                conversation.id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"turn-{i}",
                created_at=base.replace(minute=i),
            )
        await session.commit()

    provider = provisioned(["ok"])
    gen = await stream_chat(
        ChatRequest(message="latest", conversation_id=str(conversation.id)),
        provider,
        settings,
        db_sessionmaker,
        TENANT,
    )
    async for _ in gen:
        pass

    contents = [m.content for m in provider.last_messages if m.role != "system"]
    assert contents == ["turn-2", "turn-3", "turn-4", "latest"]


async def test_thinking_and_foreign_roles_not_replayed(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        conv_repo = ConversationRepository(session, TENANT)
        msg_repo = MessageRepository(session, TENANT)
        conversation = await conv_repo.create(title="seeded")
        base = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
        await msg_repo.add(conversation.id, role="user", content="q1", created_at=base)
        await msg_repo.add(
            conversation.id,
            role="assistant",
            content="a1",
            thinking="secret reasoning",
            created_at=base.replace(minute=1),
        )
        await msg_repo.add(
            conversation.id,
            role="system",
            content="db-system-note",
            created_at=base.replace(minute=2),
        )
        await session.commit()

    provider = provisioned(["ok"])
    gen = await stream_chat(
        ChatRequest(message="next", conversation_id=str(conversation.id)),
        provider,
        make_settings(),
        db_sessionmaker,
        TENANT,
    )
    async for _ in gen:
        pass

    non_system = [m for m in provider.last_messages if m.role != "system"]
    assert [m.content for m in non_system] == ["q1", "a1", "next"]
    assert all("secret reasoning" not in m.content for m in provider.last_messages)
    assert all(m.content != "db-system-note" for m in provider.last_messages)


async def test_legacy_history_is_forwarded_but_never_persisted(db_sessionmaker) -> None:
    provider = provisioned(["fresh reply"])
    payload = {
        "message": "new question",
        "history": [
            {"role": "user", "content": "old q"},
            {"role": "assistant", "content": "old a"},
        ],
    }
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, payload)

    contents = [m.content for m in provider.last_messages if m.role != "system"]
    assert contents == ["old q", "old a", "new question"]

    conversation_id = UUID(dict(events)["done"]["conversation_id"])
    _, messages = await load_conversation(db_sessionmaker, conversation_id)
    assert [m.content for m in messages] == ["new question", "fresh reply"]


async def test_legacy_history_ignored_when_continuing_a_conversation(db_sessionmaker) -> None:
    provider = provisioned(["reply one"])
    async with client_for(provider, db_sessionmaker) as client:
        events1 = await collect_events(client, {"message": "first question"})
        conversation_id = dict(events1)["done"]["conversation_id"]

        await collect_events(
            client,
            {
                "message": "second question",
                "conversation_id": conversation_id,
                "history": [
                    {"role": "user", "content": "smuggled user turn"},
                    {"role": "assistant", "content": "smuggled assistant turn"},
                ],
            },
        )

    # Existing conversations rebuild context ONLY from persisted turns; a
    # client-supplied history must never leak into the prompt.
    contents = [m.content for m in provider.last_messages if m.role != "system"]
    assert contents == ["first question", "reply one", "second question"]


async def test_provider_error_persists_partial_as_interrupted(db_sessionmaker) -> None:
    provider = provisioned(["par", "tial"], error="boom", error_code="provider_unavailable")
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": "hi"})

    # The error chunk crosses the seam unchanged; there is no done event.
    assert events[-1] == ("error", {"message": "boom", "code": "provider_unavailable"})
    assert all(name != "done" for name, _ in events)

    async with db_sessionmaker() as session:
        conversations, _ = await ConversationRepository(session, TENANT).list_summaries(10, 0)
        _, messages = await load_conversation(db_sessionmaker, conversations[0].id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content == "partial"
    assert messages[1].meta == {"interrupted": True}


async def test_mid_stream_exception_persists_partial_as_interrupted(db_sessionmaker) -> None:
    provider = provisioned(["tok"], raise_exc=RuntimeError("kaboom"))
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": "hi"})

    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "internal_error"

    async with db_sessionmaker() as session:
        conversations, _ = await ConversationRepository(session, TENANT).list_summaries(10, 0)
        _, messages = await load_conversation(db_sessionmaker, conversations[0].id)
    assert messages[1].content == "tok"
    assert messages[1].meta == {"interrupted": True}


async def test_client_disconnect_persists_partial_as_interrupted(db_sessionmaker) -> None:
    # A mid-stream client disconnect: the SSE layer closes the service generator
    # (aclose), whose finally must persist whatever streamed so far, marked
    # interrupted, shielded so the write survives the cancellation a real
    # disconnect raises. Drives the generator directly and aborts after two
    # tokens - exactly what app/api/v1/chat.py::_stream does on an early exit.
    provider = provisioned(["one ", "two ", "three ", "four"])
    gen = await stream_chat(
        ChatRequest(message="hi"),
        provider,
        make_settings(),
        db_sessionmaker,
        TENANT,
    )
    tokens = 0
    async for chunk in gen:
        if chunk.type == "token":
            tokens += 1
            if tokens == 2:
                break
    await gen.aclose()

    async with db_sessionmaker() as session:
        conversations, _ = await ConversationRepository(session, TENANT).list_summaries(10, 0)
    _, messages = await load_conversation(db_sessionmaker, conversations[0].id)
    assert [m.role for m in messages] == ["user", "assistant"]
    # Only the two tokens that streamed before the disconnect are saved.
    assert messages[1].content == "one two "
    assert messages[1].meta == {"interrupted": True}


async def test_partial_persist_survives_a_cancelled_scope(db_sessionmaker) -> None:
    # Reproduce the real disconnect failure: Starlette wraps the stream in an
    # anyio cancel scope, and once cancelled EVERY await inside it re-raises
    # CancelledError. The shielded disconnect-persist must still land the partial
    # (an unshielded await is torn on its first DB call and the partial is lost -
    # remove the asyncio.shield in _generate and this test fails). A small provider
    # delay gives the scope a checkpoint to cancel at a token boundary. The title
    # enqueue rides the SAME shielded coroutine, so it is pinned here too: moved
    # outside the shield, a real disconnect tears it and the interrupted first
    # exchange is never named.
    import anyio

    from tests.unit.test_conversation_titles import RecordingJobQueue

    queue = RecordingJobQueue()
    provider = provisioned(["a ", "b ", "c ", "d"], delay=0.001)
    gen = await stream_chat(
        ChatRequest(message="hi"),
        provider,
        make_settings(),
        db_sessionmaker,
        TENANT,
        queue=queue,
    )
    with anyio.CancelScope() as scope:
        tokens = 0
        async for chunk in gen:
            if chunk.type == "token":
                tokens += 1
                if tokens == 2:
                    scope.cancel()  # the client disconnects mid-stream

    # The shielded persist detaches from the cancelled scope and finishes on the
    # loop, so poll briefly for it rather than racing it.
    messages: list = []
    for _ in range(50):
        await asyncio.sleep(0.005)
        async with db_sessionmaker() as session:
            conversations, _ = await ConversationRepository(session, TENANT).list_summaries(10, 0)
        if conversations:
            _, messages = await load_conversation(db_sessionmaker, conversations[0].id)
            if len(messages) == 2:
                break
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content == "a b "
    assert messages[1].meta == {"interrupted": True}
    # The enqueue runs a beat behind the save inside the same detached
    # coroutine; poll it the same way the save was polled.
    for _ in range(50):
        if queue.titles():
            break
        await asyncio.sleep(0.005)
    assert len(queue.titles()) == 1


async def test_empty_generation_persists_no_assistant_message(db_sessionmaker) -> None:
    provider = provisioned([])
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": "hi"})

    done = dict(events)["done"]
    # Synthetic id keeps the wire contract; canonical dashed form like every
    # persisted id, so clients never see a second id format.
    assert done["message_id"]
    assert str(UUID(done["message_id"])) == done["message_id"]
    _, messages = await load_conversation(db_sessionmaker, UUID(done["conversation_id"]))
    assert [m.role for m in messages] == ["user"]


async def test_followup_turn_orders_after_newest_message_despite_clock_stepback(
    db_sessionmaker,
) -> None:
    # Seed a conversation whose newest message is in the future, simulating a
    # wall clock that stepped back between turns; the follow-up turn must still
    # sort strictly after it or the messages-ASC contract order breaks.
    async with db_sessionmaker() as session:
        conv_repo = ConversationRepository(session, TENANT)
        msg_repo = MessageRepository(session, TENANT)
        conversation = await conv_repo.create(title="seeded")
        future = utcnow() + timedelta(hours=1)
        await msg_repo.add(conversation.id, role="user", content="q1", created_at=future)
        await msg_repo.add(
            conversation.id,
            role="assistant",
            content="a1",
            created_at=future + timedelta(microseconds=1),
        )
        await session.commit()

    provider = provisioned(["reply"])
    gen = await stream_chat(
        ChatRequest(message="follow-up", conversation_id=str(conversation.id)),
        provider,
        make_settings(),
        db_sessionmaker,
        TENANT,
    )
    async for _ in gen:
        pass

    _, messages = await load_conversation(db_sessionmaker, conversation.id)
    assert [m.content for m in messages] == ["q1", "a1", "follow-up", "reply"]


async def test_thinking_deltas_persisted_on_assistant_message(db_sessionmaker) -> None:
    provider = provisioned(["ans"], thinking_script=["think1 ", "think2"])
    async with client_for(provider, db_sessionmaker) as client:
        events = await collect_events(client, {"message": "hi"})

    conversation_id = UUID(dict(events)["done"]["conversation_id"])
    _, messages = await load_conversation(db_sessionmaker, conversation_id)
    assert messages[1].content == "ans"
    assert messages[1].thinking == "think1 think2"


# --------------------------------------------------------------------------- #
# The anchor guard: a reply is only ever saved beside its question             #
# --------------------------------------------------------------------------- #


class HookedProvider(FakeProvider):
    """A fake that runs a callback once, immediately after its first token.

    The anchor guard defends against a RACE, and a race needs something to
    happen while the answer is still streaming. The hook stands in for whatever
    the user did in another tab during the wait - regenerating this very turn,
    deleting the conversation - so the test drives the real generator against a
    transcript that genuinely moves underneath it.
    """

    def __init__(self, script: list[str], *, after_first_token: Callable[[], Awaitable[None]]):
        super().__init__(script=script, models=list(DEV_PAIR))
        self._hook = after_first_token
        self._fired = False

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Stream the scripted reply, firing the hook after the first token."""
        async for chunk in super().chat_stream(messages, model=model, tools=tools, options=options):
            yield chunk
            if chunk.type == "token" and not self._fired:
                self._fired = True
                await self._hook()


async def test_no_reply_is_saved_once_its_question_is_gone(db_sessionmaker) -> None:
    # The question this turn is answering is removed mid-generation (a regenerate
    # from another tab truncating the thread, or the conversation being deleted
    # outright). A reply is only meaningful beside its question, so nothing is
    # written: no orphaned assistant row, and - deleting the question is what a
    # truncation does - no second reply left standing above the replacement.

    async def truncate() -> None:
        async with db_sessionmaker() as session:
            await session.execute(delete(Message).where(Message.role == "user"))
            await session.commit()

    provider = HookedProvider(["one ", "two"], after_first_token=truncate)
    gen = await stream_chat(
        ChatRequest(message="hi"),
        provider,
        make_settings(),
        db_sessionmaker,
        TENANT,
    )
    async for _ in gen:
        pass

    async with db_sessionmaker() as session:
        conversations, _ = await ConversationRepository(session, TENANT).list_summaries(10, 0)
    _, messages = await load_conversation(db_sessionmaker, conversations[0].id)
    assert messages == [], "a reply whose question is gone must not be persisted"


async def test_a_regenerate_takes_back_a_reply_that_landed_mid_turn(db_sessionmaker) -> None:
    # The mirror race: a stopped turn's shielded save lands AFTER the regenerate
    # deleted the reply it meant to replace. The regenerate's own save clears any
    # reply sitting after its question, in the same transaction as the write, so
    # the thread cannot come out of this holding two answers to one question.
    async with db_sessionmaker() as session:
        conversation = await ConversationRepository(session, TENANT).create(title="seeded")
        question = await MessageRepository(session, TENANT).add(
            conversation.id, role="user", content="ask me again"
        )
        await session.commit()

    async def zombie_partial() -> None:
        """The discarded turn's partial, arriving late."""
        async with db_sessionmaker() as session:
            await MessageRepository(session, TENANT).add(
                conversation.id,
                role="assistant",
                content="the reply that was being replaced",
                meta={"interrupted": True},
            )
            await session.commit()

    provider = HookedProvider(["the new answer"], after_first_token=zombie_partial)
    gen = await stream_chat(
        ChatRequest(conversation_id=str(conversation.id), regenerate=True),
        provider,
        make_settings(),
        db_sessionmaker,
        TENANT,
    )
    async for _ in gen:
        pass

    _, messages = await load_conversation(db_sessionmaker, conversation.id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].id == question.id
    assert messages[1].content == "the new answer"
