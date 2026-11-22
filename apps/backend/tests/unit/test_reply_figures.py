"""A reply's figures: what it took to write, stored so a reload still says.

The footer under a reply names its model, how many tokens were written and how
fast. A live stream used to be the only place the last two existed, so a
reloaded conversation showed the model and nothing else. The chat service now
sums each finished round's report into one :class:`TurnUsage`, stores it with
the reply and carries the same object on ``done``. These pin the sum (tool
rounds count, an untimed round withholds the speed), the two saves that must
NOT store figures (a cut-short reply, a failed one), the carry across a
confirmation pause, and the lenient read on the way back out.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import anyio
import pytest

from app.models import Message, utcnow
from app.providers.base import ChatChunk, ChatMessage, ToolCall, TurnUsage, Usage
from app.providers.fake_provider import FakeProvider, FakeTurn
from app.repositories.conversations import ConversationRepository
from app.schemas.conversations import MessageOut
from app.services.chat_service import _generate, _TurnTally
from tests.conftest import make_memory_store
from tests.unit import test_chat_confirmation as confirmation
from tests.unit import test_chat_persistence as persistence
from tests.unit import test_tool_loop as loop
from tests.unit.test_chat_mcp_tools import FakeMcpSource, mcp_chat_client
from tests.unit.test_chat_mcp_tools import make_settings as mcp_settings
from tests.unit.test_chat_mcp_tools import provisioned as mcp_provisioned

SEARCH = ToolCall(id="c", name="web_search", arguments={})


def _done(chunks: list[ChatChunk]) -> ChatChunk:
    return next(chunk for chunk in chunks if chunk.type == "done")


async def _stored_usage(db_sessionmaker, conversation_id: UUID) -> dict | None:
    _, messages = await loop._load(db_sessionmaker, conversation_id)
    assistant = next(m for m in messages if m.role == "assistant")
    return (assistant.meta or {}).get("usage")


# --------------------------------------------------------------------------- #
# The tally                                                                    #
# --------------------------------------------------------------------------- #


def test_the_tally_sums_every_round_that_reported() -> None:
    tally = _TurnTally()
    tally.add(Usage(completion_tokens=12, completion_ms=400.4))
    tally.add(Usage(completion_tokens=30, completion_ms=600.4))
    assert tally.total() == TurnUsage(completion_tokens=42, generation_ms=1001)


def test_a_round_with_no_count_adds_nothing() -> None:
    tally = _TurnTally()
    tally.add(None)
    tally.add(Usage(completion_tokens=None, completion_ms=900))
    tally.add(Usage(completion_tokens=0, completion_ms=900))
    assert tally.total() is None
    # Nor does its time leak into a later round's speed.
    tally.add(Usage(completion_tokens=10, completion_ms=500))
    assert tally.total() == TurnUsage(completion_tokens=10, generation_ms=500)


def test_one_untimed_round_withholds_the_speed_not_the_count() -> None:
    # 40 tokens over the 500ms that only 10 of them took would read as 80
    # tokens a second. No speed is the honest figure.
    tally = _TurnTally()
    tally.add(Usage(completion_tokens=10, completion_ms=500))
    tally.add(Usage(completion_tokens=30, completion_ms=None))
    assert tally.total() == TurnUsage(completion_tokens=40, generation_ms=None)


def test_a_span_that_rounds_to_nothing_is_no_span() -> None:
    tally = _TurnTally()
    tally.add(Usage(completion_tokens=1, completion_ms=0.2))
    assert tally.total() == TurnUsage(completion_tokens=1, generation_ms=None)


def test_the_tally_survives_its_checkpoint() -> None:
    tally = _TurnTally()
    tally.add(Usage(completion_tokens=12, completion_ms=400))
    restored = _TurnTally.from_checkpoint(tally.to_checkpoint())
    restored.add(Usage(completion_tokens=30, completion_ms=600))
    assert restored.total() == TurnUsage(completion_tokens=42, generation_ms=1000)
    # An untimed first half stays untimed on the far side.
    untimed = _TurnTally()
    untimed.add(Usage(completion_tokens=12))
    restored = _TurnTally.from_checkpoint(untimed.to_checkpoint())
    restored.add(Usage(completion_tokens=30, completion_ms=600))
    assert restored.total() == TurnUsage(completion_tokens=42, generation_ms=None)


def test_an_unusable_checkpoint_entry_reads_as_nothing_counted() -> None:
    for raw in (
        None,
        "12",
        [],
        {},
        {"completion_tokens": "12", "generation_ms": 400, "timed": True},
        {"completion_tokens": -1, "generation_ms": 400, "timed": True},
        {"completion_tokens": True, "generation_ms": 400, "timed": True},
        {"completion_tokens": 12, "generation_ms": -5, "timed": True},
        {"completion_tokens": 12, "generation_ms": None, "timed": True},
    ):
        restored = _TurnTally.from_checkpoint(raw)
        assert restored.total() is None, raw
        # And nothing of it survives to skew the rounds that follow.
        restored.add(Usage(completion_tokens=30, completion_ms=600))
        assert restored.total() == TurnUsage(completion_tokens=30, generation_ms=600), raw


# --------------------------------------------------------------------------- #
# The loop                                                                     #
# --------------------------------------------------------------------------- #


async def test_a_tool_turn_reports_and_stores_the_sum_of_its_rounds(db_sessionmaker) -> None:
    provider = FakeProvider(
        turns=[
            # The round that asks for the tool wrote 12 tokens no one sees.
            FakeTurn(tool_calls=[SEARCH], completion_tokens=12, completion_ms=400),
            FakeTurn(tokens=["a ", "b ", "c"], completion_ms=600),
        ]
    )
    conversation_id, chunks = await loop._run(
        db_sessionmaker, provider, offered_tools=[loop.ScriptedTool()]
    )

    done = _done(chunks)
    assert done.turn_usage == TurnUsage(completion_tokens=15, generation_ms=1000)
    # The provider's own report stays what it was: the last round alone.
    assert done.usage.completion_tokens == 3
    # And the row holds the very figures the stream closed on.
    assert await _stored_usage(db_sessionmaker, conversation_id) == {
        "completion_tokens": 15,
        "generation_ms": 1000,
    }


async def test_a_runtime_that_times_nothing_stores_the_count_alone(db_sessionmaker) -> None:
    provider = FakeProvider(["a ", "b ", "c"])
    conversation_id, chunks = await loop._run(db_sessionmaker, provider, offered_tools=None)

    assert _done(chunks).turn_usage == TurnUsage(completion_tokens=3, generation_ms=None)
    assert await _stored_usage(db_sessionmaker, conversation_id) == {
        "completion_tokens": 3,
        "generation_ms": None,
    }


async def test_a_reply_with_nothing_to_save_reports_no_figures(db_sessionmaker) -> None:
    # The runtime counted tokens but no text came of them, so no row is
    # written. Figures on the live frame would be ones no reload could show.
    provider = FakeProvider(turns=[FakeTurn(completion_tokens=5, completion_ms=600)])
    conversation_id, chunks = await loop._run(db_sessionmaker, provider, offered_tools=None)

    assert _done(chunks).turn_usage is None
    _, messages = await loop._load(db_sessionmaker, conversation_id)
    assert messages == []


async def test_a_reply_cut_short_stores_no_figures(db_sessionmaker) -> None:
    # The tool round finished and reported 12 tokens; the answer was cut after
    # its first token. Twelve is a count of the round nobody reads, so the
    # partial reply must carry no figures at all.
    provider = FakeProvider(
        turns=[
            FakeTurn(tool_calls=[SEARCH], completion_tokens=12, completion_ms=400),
            FakeTurn(tokens=["a ", "b ", "c"], completion_ms=600),
        ],
        delay=0.001,
    )
    conversation_id = await loop._seed_conversation(db_sessionmaker)
    gen = _generate(
        provider=provider,
        sessionmaker=db_sessionmaker,
        tenant=loop.TENANT,
        messages=[ChatMessage(role="user", content="q")],
        model="fake",
        think=None,
        conversation_id=conversation_id,
        user_at=utcnow(),
        settings=loop.make_settings(),
        offered_tools=[loop.ScriptedTool()],
        tool_context=loop._context(conversation_id),
    )
    with anyio.CancelScope() as scope:
        async for chunk in gen:
            if chunk.type == "token":
                scope.cancel()

    messages: list = []
    for _ in range(50):
        await asyncio.sleep(0.005)
        _, messages = await loop._load(db_sessionmaker, conversation_id)
        if messages:
            break
    assert messages[0].meta["interrupted"] is True
    assert "usage" not in messages[0].meta


async def test_a_failed_reply_stores_no_figures(db_sessionmaker) -> None:
    provider = persistence.provisioned(
        ["par", "tial"], error="boom", error_code="provider_unavailable"
    )
    async with persistence.client_for(provider, db_sessionmaker) as client:
        events = await persistence.collect_events(client, {"message": "hi"})

    assert all(name != "done" for name, _ in events)
    async with db_sessionmaker() as session:
        conversations, _ = await ConversationRepository(session, persistence.TENANT).list_summaries(
            10, 0
        )
    _, messages = await persistence.load_conversation(db_sessionmaker, conversations[0].id)
    assert messages[1].meta == {"interrupted": True}


# --------------------------------------------------------------------------- #
# The wire, live and reloaded                                                  #
# --------------------------------------------------------------------------- #


async def test_done_and_the_reloaded_reply_carry_the_same_figures(db_sessionmaker) -> None:
    provider = persistence.provisioned(["Nice ", "to ", "meet ", "you."], completion_ms=250)
    async with persistence.client_for(provider, db_sessionmaker) as client:
        events = await persistence.collect_events(client, {"message": "Hello there"})
        done = dict(events)["done"]
        detail = (await client.get(f"/api/v1/conversations/{done['conversation_id']}")).json()

    figures = {"completion_tokens": 4, "generation_ms": 250}
    assert done["turn_usage"] == figures
    question, reply = detail["messages"]
    assert reply["usage"] == figures
    # A question was not written by a model and has none.
    assert question["usage"] is None


async def test_a_provider_that_counts_nothing_sends_no_figures(db_sessionmaker) -> None:
    # A reply of thinking alone: the fake counts answer tokens, and there are
    # none - the shape of a runtime that reports no count for the turn.
    provider = persistence.provisioned([], thinking_script=["hmm"])
    async with persistence.client_for(provider, db_sessionmaker) as client:
        events = await persistence.collect_events(client, {"message": "Hello there"})
        done = dict(events)["done"]
        detail = (await client.get(f"/api/v1/conversations/{done['conversation_id']}")).json()

    assert done["turn_usage"] is None
    assert detail["messages"][-1]["usage"] is None


# --------------------------------------------------------------------------- #
# Across a confirmation pause                                                  #
# --------------------------------------------------------------------------- #


def _pausing_provider() -> FakeProvider:
    return mcp_provisioned(
        turns=[
            FakeTurn(
                tool_calls=[
                    ToolCall(id="c1", name="create_record", arguments={"query": "login bug"})
                ],
                completion_tokens=12,
                completion_ms=400,
            ),
            FakeTurn(tokens=["Filed ", "it."], completion_ms=600),
        ]
    )


@pytest.mark.parametrize("decision", ["approve", "deny"])
async def test_the_figures_cover_both_sides_of_a_confirmation(
    db_sessionmaker, decision: str
) -> None:
    # Whichever way the reader answers, the model wrote the round that asked
    # and the round that replied.
    source = FakeMcpSource([confirmation._danger()])
    async with mcp_chat_client(
        _pausing_provider(),
        db_sessionmaker,
        mcp_settings(),
        store=make_memory_store(),
        mcp_sources=[source],
    ) as client:
        conversation_id, confirmation_id, _ = await confirmation._pause(client)
        events = await confirmation._resume(client, conversation_id, confirmation_id, decision)

    figures = {"completion_tokens": 14, "generation_ms": 1000}
    assert confirmation._events("done", events)[0]["turn_usage"] == figures
    meta = await confirmation._assistant_meta(db_sessionmaker, conversation_id)
    assert meta["usage"] == figures


async def test_a_checkpoint_written_before_the_tally_still_resumes(db_sessionmaker) -> None:
    # A confirmation already in flight across a deploy has no tally key. The
    # resume must finish the turn and report the half it can account for,
    # rather than raising on the missing key or inventing the first half.
    source = FakeMcpSource([confirmation._danger()])
    async with mcp_chat_client(
        _pausing_provider(),
        db_sessionmaker,
        mcp_settings(),
        store=make_memory_store(),
        mcp_sources=[source],
    ) as client:
        conversation_id, confirmation_id, _ = await confirmation._pause(client)
        async with db_sessionmaker() as session:
            conversation = await ConversationRepository(session, confirmation.TENANT).get(
                UUID(conversation_id)
            )
            payload = dict(conversation.pending_confirmation)
            payload.pop("turn_usage")
            conversation.pending_confirmation = payload
            session.add(conversation)
            await session.commit()
        events = await confirmation._resume(client, conversation_id, confirmation_id, "approve")

    assert confirmation._events("done", events)[0]["turn_usage"] == {
        "completion_tokens": 2,
        "generation_ms": 600,
    }


# --------------------------------------------------------------------------- #
# Reading a stored row back                                                    #
# --------------------------------------------------------------------------- #


def _reply(meta: dict) -> Message:
    return Message(
        tenant_id=persistence.TENANT.tenant_id,
        user_id=persistence.TENANT.user_id,
        conversation_id=UUID(int=1),
        role="assistant",
        content="an answer",
        meta=meta,
    )


def test_a_reply_saved_before_figures_existed_reads_as_none() -> None:
    assert MessageOut.from_entity(_reply({})).usage is None


def test_malformed_stored_figures_read_as_none_not_a_failed_reload() -> None:
    for usage in (
        "12",
        [],
        {},
        {"generation_ms": 500},
        {"completion_tokens": 0},
        {"completion_tokens": -3, "generation_ms": 500},
        {"completion_tokens": "many"},
        {"completion_tokens": 12, "generation_ms": 0},
        {"completion_tokens": 12, "generation_ms": -1},
    ):
        out = MessageOut.from_entity(_reply({"usage": usage}))
        assert out.usage is None, usage
        # The rest of the reply is untouched by the bad key.
        assert out.content == "an answer"


def test_stored_figures_without_a_time_keep_their_count() -> None:
    out = MessageOut.from_entity(
        _reply({"usage": {"completion_tokens": 12, "generation_ms": None}})
    )
    assert out.usage is not None
    assert (out.usage.completion_tokens, out.usage.generation_ms) == (12, None)
