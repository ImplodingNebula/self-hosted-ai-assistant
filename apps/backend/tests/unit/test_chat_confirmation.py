"""Side-effecting tool confirmation: the pause, the resume, and the guards (slice 5).

Slice 5 stops dropping a side-effecting tool and instead ASKS before it runs. The
loop offers such a tool only to a client that can resume, pauses on the call
(checkpointing the turn onto its conversation and emitting a terminal
``tool_confirmation`` event), and the client continues the SAME turn with a second
request carrying an approve/deny. These drive the real endpoint end to end - the
first request suspends, the second resumes - and assert what the model was, or was
not, allowed to do, plus the guards that keep a side-effecting action from running
twice, running stale, or running unasked.

The harness (a FakeProvider scripting the tool call, fake MCP sources injected
where a deployment's are) is shared with test_chat_mcp_tools; only the flow under
test is new.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import delete

from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.models import Message, utcnow
from app.providers.base import ToolCall
from app.providers.fake_provider import FakeProvider, FakeTurn
from app.providers.tools.base import ToolRecord
from app.repositories.conversations import ConversationRepository
from app.repositories.messages import MessageRepository
from app.schemas.chat import ChatRequest
from app.schemas.conversations import MessageOut
from app.services.chat_service import _TOOL_MESSAGE_FRAMING_TOKENS
from app.services.context_budget import ANSWER_RESERVE_TOKENS, approx_tokens
from app.workers.main import JOB_GENERATE_TITLE, title_job_id
from tests.conftest import make_memory_store
from tests.unit.test_chat_mcp_tools import (
    CHAT_URL,
    FakeMcpSource,
    RemoteTool,
    collect_events,
    drain_record_ingests,
    make_settings,
    mcp_chat_client,
    offered_names,
    provisioned,
    seed_agentic_verdict,
    status_codes,
)
from tests.unit.test_conversation_titles import RecordingJobQueue

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID)

# No freshness cue and no wired search provider, so the only tools in play are the
# local search_knowledge and the remote create_record - keeping the assertions on
# the side-effecting one rather than on web offering.
ASK = "file a bug about the login page"


def _tokens(events: list[tuple[str, dict]]) -> str:
    return "".join(d.get("delta", "") for n, d in events if n == "token")


def _events(name: str, events: list[tuple[str, dict]]) -> list[dict]:
    return [d for n, d in events if n == name]


def _pausing_provider(answer: str = "Filed the bug for you.") -> FakeProvider:
    """A model that asks for the side-effecting tool, then (on resume) answers."""
    return provisioned(
        turns=[
            FakeTurn(
                tool_calls=[
                    ToolCall(id="c1", name="create_record", arguments={"query": "login bug"})
                ]
            ),
            FakeTurn(tokens=[answer]),
        ]
    )


def _danger() -> RemoteTool:
    """The side-effecting tool under test - unannotated, so fail-closed by default."""
    return RemoteTool(
        "create_record",
        side_effecting=True,
        result="A RECORD WAS CREATED IN THE REMOTE SYSTEM",
    )


async def _conversation(db_sessionmaker, conversation_id: str):
    async with db_sessionmaker() as session:
        return await ConversationRepository(session, TENANT).get(UUID(conversation_id))


async def _roles(db_sessionmaker, conversation_id: str) -> list[str]:
    async with db_sessionmaker() as session:
        rows = await MessageRepository(session, TENANT).list_recent(UUID(conversation_id), 50)
    return [row.role for row in rows]


async def _pause(client: AsyncClient) -> tuple[str, str, list[tuple[str, dict]]]:
    """Send the turn that suspends; return (conversation_id, confirmation_id, events)."""
    events = await collect_events(client, {"message": ASK, "supports_confirmation": True})
    confirmations = _events("tool_confirmation", events)
    assert len(confirmations) == 1, "the turn should suspend on exactly one confirmation"
    confirmation = confirmations[0]
    return confirmation["conversation_id"], confirmation["id"], events


async def _resume(client: AsyncClient, conversation_id: str, confirmation_id: str, decision: str):
    return await collect_events(
        client,
        {
            "conversation_id": conversation_id,
            "confirmation": {"id": confirmation_id, "decision": decision},
        },
    )


# --------------------------------------------------------------------------- #
# Offering                                                                     #
# --------------------------------------------------------------------------- #


async def test_a_side_effecting_tool_is_offered_only_to_a_confirm_capable_client(
    db_sessionmaker,
) -> None:
    # The whole premise: the fail-closed drop becomes conditional. A client that
    # says it can confirm is offered the side-effecting tool; the same turn from a
    # client that does not is not - the tool stays withheld exactly as before.
    source = FakeMcpSource([_danger()])
    settings = make_settings()

    async def offered(supports: bool) -> set[str]:
        provider = provisioned(script=["ok"])
        async with mcp_chat_client(
            provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
        ) as client:
            await collect_events(client, {"message": ASK, "supports_confirmation": supports})
        return set(offered_names(provider))

    assert "create_record" in await offered(True)
    assert "create_record" not in await offered(False)


# --------------------------------------------------------------------------- #
# The pause                                                                    #
# --------------------------------------------------------------------------- #


async def test_calling_a_side_effecting_tool_suspends_the_turn(db_sessionmaker) -> None:
    # The model asks for create_record. The turn must NOT run it: it emits a
    # tool_confirmation carrying the pending call and its arguments, ends without a
    # done, and checkpoints itself - with no assistant reply persisted, because
    # there is no answer yet.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider()
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, events = await _pause(client)

    # The action did not happen, and the turn did not pretend to answer.
    assert danger.calls == []
    assert _tokens(events) == ""
    assert not _events("done", events)
    # The pending call is surfaced for the user, with the arguments it would run.
    confirmation = _events("tool_confirmation", events)[0]
    assert [call["name"] for call in confirmation["calls"]] == ["create_record"]
    assert confirmation["calls"][0]["arguments"] == {"query": "login bug"}
    # The checkpoint is on the conversation, keyed by the id the event carried...
    conversation = await _conversation(db_sessionmaker, conversation_id)
    assert conversation.pending_confirmation is not None
    assert conversation.pending_confirmation["id"] == confirmation_id
    # ...and no reply was saved - only the user's message stands so far.
    assert "assistant" not in await _roles(db_sessionmaker, conversation_id)


async def test_a_pause_does_not_cancel_the_record_ingest_an_earlier_round_started(
    db_sessionmaker,
) -> None:
    # A pause is the one ending that is not an ending: the turn stops here and
    # resumes later, possibly much later. The suspend must therefore not take the
    # in-flight ingest down with it, or the resumed turn - and every turn after
    # it - would look for those records in the session scope and find nothing.
    #
    # Staged so the ingest is demonstrably still in flight when the pause lands:
    # the store's first call sleeps well past the microseconds the checkpoint
    # write takes. The pause deliberately does NOT wait for it (the approve/deny
    # dialog is what the user is waiting on), so the assertion is that the write
    # completes afterwards - which it cannot if anything on the way out cancels
    # the task.
    upserted: list[str] = []

    class _SlowStore:
        """The real memory store, with a first ingest call slow enough to catch."""

        def __init__(self, inner) -> None:
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def doc_fingerprints(self, *args, **kwargs):
            await asyncio.sleep(0.3)
            return await self._inner.doc_fingerprints(*args, **kwargs)

        async def upsert(self, *args, **kwargs):
            upserted.append("done")
            return await self._inner.upsert(*args, **kwargs)

    reader = RemoteTool(
        "search_records",
        side_effecting=False,
        records=[ToolRecord(id="rec-1", title="Login bug", text="The login page 500s on submit.")],
    )
    provider = provisioned(
        turns=[
            FakeTurn(
                tool_calls=[ToolCall(id="c1", name="search_records", arguments={"query": "login"})]
            ),
            FakeTurn(
                tool_calls=[
                    ToolCall(id="c2", name="create_record", arguments={"query": "login bug"})
                ]
            ),
            FakeTurn(tokens=["Filed the bug for you."]),
        ]
    )
    async with mcp_chat_client(
        provider,
        db_sessionmaker,
        make_settings(),
        store=_SlowStore(make_memory_store()),
        mcp_sources=[FakeMcpSource([reader, _danger()])],
    ) as client:
        await _pause(client)

    # Still running when the stream ended - which is the point: it survived.
    assert upserted == []
    await drain_record_ingests()
    assert upserted == ["done"]


# --------------------------------------------------------------------------- #
# Resume: approve and deny                                                     #
# --------------------------------------------------------------------------- #


async def test_approving_runs_the_tool_and_answers(db_sessionmaker) -> None:
    # The resume with approve is the other half of the turn: the tool runs, its
    # result feeds the model, the answer streams and persists, and the checkpoint
    # is consumed so it cannot run again.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider(answer="Filed the bug for you.")
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        events = await _resume(client, conversation_id, confirmation_id, "approve")

    # The action ran exactly once, with the arguments the user saw.
    assert [args for args, _ in danger.calls] == [{"query": "login bug"}]
    # The answer streamed and the turn completed.
    assert _tokens(events) == "Filed the bug for you."
    assert _events("done", events)
    # The approved run is surfaced to the tool card too: a tool_call and its
    # result bracket the execution on the resume stream.
    assert [d["name"] for d in _events("tool_call", events)] == ["create_record"]
    assert [d["name"] for d in _events("tool_result", events)] == ["create_record"]
    # A text-only result cannot be counted, and the resume stream says so.
    assert [d["count"] for d in _events("tool_result", events)] == [None]
    # The model was handed the tool's real result on the continuation round.
    tool_message = next(m for m in provider.calls[1] if m.role == "tool")
    assert "A RECORD WAS CREATED" in tool_message.content
    # The checkpoint is gone and the reply is persisted.
    conversation = await _conversation(db_sessionmaker, conversation_id)
    assert conversation.pending_confirmation is None
    assert "assistant" in await _roles(db_sessionmaker, conversation_id)


async def test_an_approved_call_that_returns_records_reports_its_count(db_sessionmaker) -> None:
    # The resume path builds its own tool_result chunk, so it needs its own proof
    # that the count survives the pause (the live row would otherwise lose it).
    records = [ToolRecord(id=f"rec-{n}", title=f"Record {n}", text=f"Body {n}.") for n in (1, 2, 3)]
    danger = RemoteTool("create_record", side_effecting=True, records=records)
    provider = _pausing_provider(answer="Done.")
    async with mcp_chat_client(
        provider,
        db_sessionmaker,
        make_settings(),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([danger])],
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        events = await _resume(client, conversation_id, confirmation_id, "approve")
        await drain_record_ingests()

    assert [d["count"] for d in _events("tool_result", events)] == [3]


async def test_denying_skips_the_tool_and_answers_without_it(db_sessionmaker) -> None:
    # Deny is a real branch, not an error: the tool never runs, the model is told
    # so (a synthetic result), a standing tool_declined notice records it, and the
    # turn still answers.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider(answer="I did not file it, since you declined.")
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        events = await _resume(client, conversation_id, confirmation_id, "deny")

    # The side effect never happened - and the refusal is surfaced to the trace as
    # its own (denied) card: a tool_call opens it and a denied tool_result settles
    # it with no result, so it reads as a blocked action, not a tool that returned.
    assert danger.calls == []
    assert [d["name"] for d in _events("tool_call", events)] == ["create_record"]
    results = _events("tool_result", events)
    assert [d["name"] for d in results] == ["create_record"]
    assert results[0]["denied"] is True
    assert results[0]["result"] == ""
    # The decline is recorded as a standing notice and the answer still landed.
    assert "tool_declined" in status_codes(events)
    assert _tokens(events) == "I did not file it, since you declined."
    assert _events("done", events)
    # The model was told the action was refused, not handed a fabricated success.
    tool_message = next(m for m in provider.calls[1] if m.role == "tool")
    assert "declined" in tool_message.content.lower()
    assert "A RECORD WAS CREATED" not in tool_message.content
    # And the checkpoint is consumed.
    conversation = await _conversation(db_sessionmaker, conversation_id)
    assert conversation.pending_confirmation is None


async def _assistant_meta(db_sessionmaker, conversation_id: str) -> dict:
    async with db_sessionmaker() as session:
        rows = await MessageRepository(session, TENANT).list_recent(UUID(conversation_id), 50)
    return next(row for row in reversed(rows) if row.role == "assistant").meta or {}


async def _assistant_out(db_sessionmaker, conversation_id: str) -> MessageOut:
    """The last assistant reply as the reload DTO clients actually receive."""
    async with db_sessionmaker() as session:
        rows = await MessageRepository(session, TENANT).list_recent(UUID(conversation_id), 50)
    row = next(row for row in reversed(rows) if row.role == "assistant")
    return MessageOut.from_entity(row)


async def test_an_approved_call_persists_a_tool_step(db_sessionmaker) -> None:
    # The approved run is part of the persisted trace: a reload replays the
    # executed step (its own activity card), not just the final answer. This is the
    # resume path threading the trace across the pause into the saved message.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider(answer="Filed the bug for you.")
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        await _resume(client, conversation_id, confirmation_id, "approve")

    steps = (await _assistant_meta(db_sessionmaker, conversation_id)).get("tool_steps", [])
    assert [s["name"] for s in steps] == ["create_record"]
    assert steps[0]["arguments"] == {"query": "login bug"}
    assert steps[0]["is_error"] is False
    # The authorization is part of the trace: the reloaded card is badged "approved".
    assert steps[0]["approved"] is True
    assert steps[0]["denied"] is False
    # And it survives the reload DTO, not just the raw meta - that round-trip is the
    # whole reason the marker is persisted.
    out = await _assistant_out(db_sessionmaker, conversation_id)
    assert out.tool_steps[0].approved is True
    assert out.tool_steps[0].denied is False


async def test_a_denied_call_persists_a_denied_step_and_the_notice(db_sessionmaker) -> None:
    # A denied action shows in the reloaded trace as its own (badged "denied") card,
    # symmetric with an approved run, alongside the tool_declined notice that
    # explains it in words. The step records the refusal: it ran nothing, so it has
    # no result - and the model's pre-pause narration rides it, just as it would an
    # approved step.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = provisioned(
        turns=[
            FakeTurn(
                tokens=["I'll file that for you. "],
                tool_calls=[
                    ToolCall(id="c1", name="create_record", arguments={"query": "login bug"})
                ],
            ),
            FakeTurn(tokens=["I did not file it, since you declined."]),
        ]
    )
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        await _resume(client, conversation_id, confirmation_id, "deny")

    meta = await _assistant_meta(db_sessionmaker, conversation_id)
    steps = meta.get("tool_steps", [])
    assert [s["name"] for s in steps] == ["create_record"]
    assert steps[0]["denied"] is True
    assert steps[0]["approved"] is False
    assert steps[0]["result"] is None
    assert steps[0]["narration"] == "I'll file that for you. "
    assert "tool_declined" in [n["code"] for n in meta.get("notices", [])]
    # The denied marker and the empty result survive the reload DTO, so a reloaded
    # transcript renders the "denied" card exactly as the live turn did.
    out = await _assistant_out(db_sessionmaker, conversation_id)
    assert out.tool_steps[0].denied is True
    assert out.tool_steps[0].approved is False
    assert out.tool_steps[0].result is None


async def test_an_approved_step_persists_the_pre_pause_narration(db_sessionmaker) -> None:
    # The model's narration before a side-effecting call ("I'll file that now.") is
    # streamed live above the confirmation card and folded into the checkpointed
    # tool-call message. On approve it must ride the executed step too, matching a
    # non-paused round - otherwise a confirmed turn's reloaded trace drops it.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = provisioned(
        turns=[
            FakeTurn(
                tokens=["I'll file that ticket for you. "],
                tool_calls=[
                    ToolCall(id="c1", name="create_record", arguments={"query": "login bug"})
                ],
            ),
            FakeTurn(tokens=["Filed the bug for you."]),
        ]
    )
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        await _resume(client, conversation_id, confirmation_id, "approve")

    steps = (await _assistant_meta(db_sessionmaker, conversation_id)).get("tool_steps", [])
    assert [s["name"] for s in steps] == ["create_record"]
    assert steps[0]["narration"] == "I'll file that ticket for you. "


# --------------------------------------------------------------------------- #
# The guards                                                                   #
# --------------------------------------------------------------------------- #


async def test_a_confirmation_can_be_resumed_only_once(db_sessionmaker) -> None:
    # A double-tapped approve must not run a side-effecting action twice. The first
    # resume consumes the checkpoint; the second finds nothing and is a clean 404,
    # and the tool ran a single time.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider()
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        await _resume(client, conversation_id, confirmation_id, "approve")
        again = await client.post(
            CHAT_URL,
            json={
                "conversation_id": conversation_id,
                "confirmation": {"id": confirmation_id, "decision": "approve"},
            },
        )

    assert again.status_code == 404
    assert len(danger.calls) == 1


async def test_a_resume_stream_also_opens_with_one_meta_frame(db_sessionmaker) -> None:
    # The wire contract promises meta as the FIRST frame of EVERY stream. A
    # resume yields frames of its own (tool activity, statuses) before it
    # re-enters generation, so it names the thread itself up front - and
    # generation must not repeat it: exactly one meta, first, both halves.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider()
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, first = await _pause(client)
        resumed = await _resume(client, conversation_id, confirmation_id, "approve")

    for events in (first, resumed):
        assert events[0][0] == "meta"
        assert [n for n, _ in events].count("meta") == 1
    assert resumed[0][1]["conversation_id"] == conversation_id


async def test_an_unknown_confirmation_id_is_rejected(db_sessionmaker) -> None:
    # An id that matches no pending confirmation is a 404 - and leaves the real one
    # intact (a mismatched id must not consume it).
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider()
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        wrong = await client.post(
            CHAT_URL,
            json={
                "conversation_id": conversation_id,
                "confirmation": {"id": str(uuid4()), "decision": "approve"},
            },
        )
        assert wrong.status_code == 404
        # The genuine confirmation still resolves.
        events = await _resume(client, conversation_id, confirmation_id, "approve")

    assert danger.calls
    assert _events("done", events)


async def test_an_expired_confirmation_is_rejected(db_sessionmaker) -> None:
    # A confirmation past its TTL is refused rather than run late. Rewrite the
    # stored expiry into the past (deterministic, no sleep), then resume: a 404,
    # and the action never runs.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider()
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        async with db_sessionmaker() as session:
            conversation = await ConversationRepository(session, TENANT).get(UUID(conversation_id))
            payload = dict(conversation.pending_confirmation)
            payload["expires_at"] = "2000-01-01T00:00:00+00:00"
            conversation.pending_confirmation = payload
            session.add(conversation)
            await session.commit()
        expired = await client.post(
            CHAT_URL,
            json={
                "conversation_id": conversation_id,
                "confirmation": {"id": confirmation_id, "decision": "approve"},
            },
        )

    assert expired.status_code == 404
    assert danger.calls == []


async def test_a_new_turn_clears_a_pending_confirmation(db_sessionmaker) -> None:
    # Starting a fresh turn on the thread supersedes an abandoned confirmation, so
    # a stale approve can never resume onto a conversation that has moved on.
    danger = _danger()
    source = FakeMcpSource([danger])
    # First turn suspends; the next turn (no confirmation) just answers.
    provider = provisioned(
        turns=[
            FakeTurn(
                tool_calls=[ToolCall(id="c1", name="create_record", arguments={"query": "bug"})]
            ),
            FakeTurn(tokens=["Sure, ask away."]),
        ]
    )
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        # A normal follow-up turn on the same conversation.
        await collect_events(
            client,
            {
                "message": "never mind, something else",
                "conversation_id": conversation_id,
                "supports_confirmation": True,
            },
        )
        # The old confirmation is gone.
        conversation = await _conversation(db_sessionmaker, conversation_id)
        assert conversation.pending_confirmation is None
        stale = await client.post(
            CHAT_URL,
            json={
                "conversation_id": conversation_id,
                "confirmation": {"id": confirmation_id, "decision": "approve"},
            },
        )

    assert stale.status_code == 404
    assert danger.calls == []


async def test_a_resume_without_a_conversation_id_is_a_bad_request(db_sessionmaker) -> None:
    # A resume has to name the conversation it continues; there is nowhere else to
    # find the checkpoint.
    settings = make_settings()
    provider = provisioned(script=["ok"])
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store()
    ) as client:
        resp = await client.post(
            CHAT_URL, json={"confirmation": {"id": str(uuid4()), "decision": "approve"}}
        )

    assert resp.status_code == 400


async def test_denying_still_declines_if_the_tool_is_reclassified_read_only(
    db_sessionmaker,
) -> None:
    # The decision is honored against what the USER was shown, not a live re-read.
    # If the server reclassifies its tool read-only during the confirmation window
    # (readOnlyHint is advisory and untrusted), a DENY must STILL decline it - the
    # action the user refused must never run because the server changed its mind.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider(answer="OK, I did not file it.")
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        # The server drifts: the same tool now claims to be read-only.
        danger._side_effecting = False
        events = await _resume(client, conversation_id, confirmation_id, "deny")

    # The denied action never ran, despite the tool now looking harmless.
    assert danger.calls == []
    assert "tool_declined" in status_codes(events)
    tool_message = next(m for m in provider.calls[1] if m.role == "tool")
    assert "declined" in tool_message.content.lower()


async def test_a_resumed_round_that_spends_the_budget_says_so(db_sessionmaker) -> None:
    # The paused round is resolved outside the loop, so the loop's own budget check
    # never sees it. With one iteration allowed, resolving the round spends the
    # budget - and the resume must raise tool_limit_reached itself, or a resumed
    # turn silently drops a standing notice its non-paused equivalent carries.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider()
    settings = make_settings(tool_max_iterations=1)
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        events = await _resume(client, conversation_id, confirmation_id, "approve")

    assert danger.calls  # the round did run
    assert "tool_limit_reached" in status_codes(events)


# --------------------------------------------------------------------------- #
# The request contract                                                         #
# --------------------------------------------------------------------------- #


def test_a_confirmation_resume_must_not_carry_a_new_message() -> None:
    # A resume is a resume only: letting it also carry words would smuggle a fresh
    # turn past the tool floors the first pass applied.
    with pytest.raises(ValueError, match="must not carry a new message"):
        ChatRequest(
            message="and also delete everything",
            confirmation={"id": "abc", "decision": "approve"},
        )


def test_a_confirmation_only_request_is_valid() -> None:
    # The one shape that is neither words nor files: a bare approve/deny.
    request = ChatRequest(conversation_id="c", confirmation={"id": "abc", "decision": "deny"})
    assert request.confirmation is not None
    assert request.confirmation.decision == "deny"


async def test_the_resume_honours_the_turns_budget_not_the_env_ceiling(db_sessionmaker) -> None:
    # A turn resolves a tool-result SPEND budget from what its assembled prompt
    # can afford, which on a full prompt is far below the env ceiling. The resume
    # deliberately skips the affordability floor (a resumed prompt legitimately
    # already carries tool results), so there is nothing to recompute it from -
    # the checkpoint must carry it. Without that, a paused turn comes back with
    # the ceiling's room restored and can overrun num_ctx, which Ollama truncates
    # from the HEAD, silently dropping the system prompt's tool guidance and
    # injection defense.
    #
    # Rewriting the stored budget is the same technique the expiry test uses:
    # deterministic, and it isolates the carry from prompt-size arithmetic.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider()
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        async with db_sessionmaker() as session:
            conversation = await ConversationRepository(session, TENANT).get(UUID(conversation_id))
            payload = dict(conversation.pending_confirmation)
            # Any result at all exceeds this, so the approved round must be the
            # last one. Against the 2000 ceiling it would not be.
            payload["turn_result_budget"] = 1
            conversation.pending_confirmation = payload
            session.add(conversation)
            await session.commit()
        events = await _resume(client, conversation_id, confirmation_id, "approve")

    # The approved action still ran - a budget carry must not cancel the thing
    # the user just authorized.
    assert danger.calls
    # ...but the turn is now out of room, says so, and answers.
    assert "tool_limit_reached" in status_codes(events)
    assert _events("done", events)
    # And the resumed generation offered NO tools (last_tools reflects the final
    # chat_stream call), because it was on its last round from the outset. This
    # is the half that the status assertion alone would miss.
    assert offered_names(provider) == []


async def test_a_checkpoint_written_before_the_budget_carry_still_resumes(db_sessionmaker) -> None:
    # Backward compatibility for confirmations already in flight across a deploy.
    # Such a checkpoint has no turn_result_budget key; the resume must fall back
    # to the env ceiling (what that turn was actually running under) rather than
    # raising a KeyError and stranding a turn the user already approved.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider()
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        async with db_sessionmaker() as session:
            conversation = await ConversationRepository(session, TENANT).get(UUID(conversation_id))
            payload = dict(conversation.pending_confirmation)
            payload.pop("turn_result_budget")
            conversation.pending_confirmation = payload
            session.add(conversation)
            await session.commit()
        events = await _resume(client, conversation_id, confirmation_id, "approve")

    assert danger.calls
    assert _events("done", events)
    # The small result is nowhere near the ceiling, so the turn is NOT capped -
    # which is what distinguishes the fallback from a budget of 0.
    assert "tool_limit_reached" not in status_codes(events)


async def test_the_checkpoint_carries_the_question_the_turn_is_answering(db_sessionmaker) -> None:
    # A confirmation can sit for its whole TTL, which is ample time for the
    # thread to be regenerated or deleted underneath it. The resume re-checks
    # that the question still exists before it saves, so the checkpoint has to
    # carry which question that is.
    source = FakeMcpSource([_danger()])
    provider = _pausing_provider()
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, _, _ = await _pause(client)

    conversation = await _conversation(db_sessionmaker, conversation_id)
    async with db_sessionmaker() as session:
        rows = await MessageRepository(session, TENANT).list_recent(UUID(conversation_id), 50)
    question = next(row for row in rows if row.role == "user")
    assert conversation.pending_confirmation["user_message_id"] == str(question.id)


async def test_a_resume_is_dropped_when_its_question_no_longer_exists(db_sessionmaker) -> None:
    # The guard the key above exists for. The thread is truncated while the
    # confirmation waits (a regenerate in another tab), so the reply the resume
    # streams has nothing left to answer - and is not written, rather than
    # landing as a second answer to a question that already moved on.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider()
    settings = make_settings()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        async with db_sessionmaker() as session:
            await session.execute(
                delete(Message).where(Message.conversation_id == UUID(conversation_id))
            )
            await session.commit()
        events = await _resume(client, conversation_id, confirmation_id, "approve")

    # The turn still ran and still streamed - the user authorized the action and
    # it is not silently cancelled - but nothing was persisted beside a question
    # that is gone.
    assert danger.calls
    assert _tokens(events) == "Filed the bug for you."
    assert await _roles(db_sessionmaker, conversation_id) == []


async def _seed_answered_question(db_sessionmaker, question: str, answer: str):
    """A conversation already holding one answered question, deterministically ordered."""
    async with db_sessionmaker() as session:
        conversation = await ConversationRepository(session, TENANT).create(title=question)
        msg_repo = MessageRepository(session, TENANT)
        at = utcnow()
        asked = await msg_repo.add(conversation.id, role="user", content=question, created_at=at)
        replied = await msg_repo.add(
            conversation.id,
            role="assistant",
            content=answer,
            created_at=at + timedelta(microseconds=1),
        )
        await session.commit()
    return conversation, asked, replied


async def _rows(db_sessionmaker, conversation_id: str):
    async with db_sessionmaker() as session:
        return await MessageRepository(session, TENANT).list_for_conversation(UUID(conversation_id))


async def test_a_regenerate_that_pauses_still_replaces_the_reply_on_resume(db_sessionmaker) -> None:
    # A regenerate sweeps the reply it is replacing at SAVE time, and a
    # confirmation pause puts the save on the far side of the pause - so the old
    # reply is still standing when the checkpoint is written. The checkpoint has
    # to carry that debt or the resumed save appends a second answer beside the
    # one it was sent to replace, deterministically, on the plain
    # regenerate-then-approve flow.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider(answer="Filed the bug for you.")
    settings = make_settings()
    conversation, asked, stale = await _seed_answered_question(
        db_sessionmaker, ASK, "The answer being replaced."
    )
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        events = await collect_events(
            client,
            {
                "conversation_id": str(conversation.id),
                "regenerate": True,
                "supports_confirmation": True,
            },
        )
        confirmation = _events("tool_confirmation", events)[0]
        resumed = await _resume(
            client, confirmation["conversation_id"], confirmation["id"], "approve"
        )

    assert danger.calls, "the approved action should have run"
    assert _tokens(resumed) == "Filed the bug for you."
    rows = await _rows(db_sessionmaker, str(conversation.id))
    assert [m.role for m in rows] == ["user", "assistant"], "exactly one reply, not two"
    assert rows[0].id == asked.id
    assert rows[1].id != stale.id
    assert rows[1].content == "Filed the bug for you."


async def test_a_checkpoint_written_before_the_sweep_flag_still_resumes(db_sessionmaker) -> None:
    # Backward compatibility, same treatment as the other carried keys. A
    # checkpoint written before this key existed cannot have been a regenerate -
    # the shape did not exist - so reading a missing key as "no sweep" restores
    # exactly what those turns did: append the reply, disturb nothing above it.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider(answer="Filed it.")
    settings = make_settings()
    conversation, _, earlier = await _seed_answered_question(
        db_sessionmaker, "an earlier question", "an earlier answer"
    )
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        events = await collect_events(
            client,
            {
                "message": ASK,
                "conversation_id": str(conversation.id),
                "supports_confirmation": True,
            },
        )
        confirmation = _events("tool_confirmation", events)[0]
        async with db_sessionmaker() as session:
            row = await ConversationRepository(session, TENANT).get(conversation.id)
            payload = dict(row.pending_confirmation)
            payload.pop("replace_following")
            row.pending_confirmation = payload
            session.add(row)
            await session.commit()
        resumed = await _resume(
            client, confirmation["conversation_id"], confirmation["id"], "approve"
        )

    assert _events("done", resumed), "a missing key must not strand a resume"
    rows = await _rows(db_sessionmaker, str(conversation.id))
    # The earlier exchange is untouched: no key, no sweep.
    assert [m.role for m in rows] == ["user", "assistant", "user", "assistant"]
    assert rows[1].id == earlier.id


async def _seed_two_exchanges(db_sessionmaker):
    """A conversation holding two answered questions, deterministically ordered."""
    async with db_sessionmaker() as session:
        conversation = await ConversationRepository(session, TENANT).create(title="seeded")
        msg_repo = MessageRepository(session, TENANT)
        at = utcnow()
        rows = []
        for index, (role, content) in enumerate(
            [
                ("user", ASK),
                ("assistant", "first answer"),
                ("user", "and then?"),
                ("assistant", "second answer"),
            ]
        ):
            rows.append(
                await msg_repo.add(
                    conversation.id,
                    role=role,
                    content=content,
                    created_at=at + timedelta(microseconds=index),
                )
            )
        await session.commit()
    return conversation, rows


async def test_a_targeted_regenerate_that_pauses_still_discards_its_tail(db_sessionmaker) -> None:
    # The other half of the carried debt. A regenerate of an OLDER reply captures
    # the rows it will discard at prepare time and deletes them at save time - and
    # a confirmation pause puts that save on the far side of the pause. Without
    # the list in the checkpoint the resumed turn appends its answer under a tail
    # that was supposed to be gone.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider(answer="A better first answer.")
    settings = make_settings()
    conversation, rows = await _seed_two_exchanges(db_sessionmaker)
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        events = await collect_events(
            client,
            {
                "conversation_id": str(conversation.id),
                "regenerate": True,
                "message_id": str(rows[1].id),
                "supports_confirmation": True,
            },
        )
        confirmation = _events("tool_confirmation", events)[0]
        # Nothing has gone yet: the pause is not the save.
        assert len(await _rows(db_sessionmaker, str(conversation.id))) == 4
        resumed = await _resume(
            client, confirmation["conversation_id"], confirmation["id"], "approve"
        )

    assert _tokens(resumed) == "A better first answer."
    after = await _rows(db_sessionmaker, str(conversation.id))
    assert [m.role for m in after] == ["user", "assistant"]
    assert after[0].id == rows[0].id
    assert after[1].content == "A better first answer."


async def test_a_checkpoint_without_the_captured_tail_discards_nothing(db_sessionmaker) -> None:
    # Backward compatibility, and the safe direction. A checkpoint written before
    # the key existed cannot have named a tail, so a missing key reads as "nothing
    # to discard": the resume leaves an extra reply, which is visible and fixable,
    # rather than deleting rows from a list nobody can vouch for.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider(answer="Filed it.")
    settings = make_settings()
    conversation, rows = await _seed_two_exchanges(db_sessionmaker)
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        events = await collect_events(
            client,
            {
                "conversation_id": str(conversation.id),
                "regenerate": True,
                "message_id": str(rows[1].id),
                "supports_confirmation": True,
            },
        )
        confirmation = _events("tool_confirmation", events)[0]
        async with db_sessionmaker() as session:
            row = await ConversationRepository(session, TENANT).get(conversation.id)
            payload = dict(row.pending_confirmation)
            payload.pop("replace_ids")
            row.pending_confirmation = payload
            session.add(row)
            await session.commit()
        resumed = await _resume(
            client, confirmation["conversation_id"], confirmation["id"], "approve"
        )

    assert _events("done", resumed), "a missing key must not strand a resume"
    after = await _rows(db_sessionmaker, str(conversation.id))
    assert [m.id for m in after[:4]] == [m.id for m in rows], "no key, no deletion"
    assert after[-1].content == "Filed it."


async def test_an_edit_that_pauses_still_re_names_the_thread_on_resume(db_sessionmaker) -> None:
    # The third carried debt. Rewording a conversation's OPENING question hands
    # the naming back to the worker under a job id of its own, and that enqueue
    # happens where the turn finishes - which for a paused turn is the resume. A
    # checkpoint that forgot the discriminator would enqueue under the plain
    # per-conversation id instead, which arq deduplicates against the job that
    # named the thread the first time, so the rename would silently never happen.
    danger = _danger()
    source = FakeMcpSource([danger])
    provider = _pausing_provider(answer="Reworded answer.")
    settings = make_settings()
    queue = RecordingJobQueue()
    conversation, asked, _ = await _seed_answered_question(
        db_sessionmaker, ASK, "The first answer."
    )
    async with mcp_chat_client(
        provider,
        db_sessionmaker,
        settings,
        store=make_memory_store(),
        mcp_sources=[source],
        job_queue=queue,
    ) as client:
        events = await collect_events(
            client,
            {
                "conversation_id": str(conversation.id),
                "edit": {"message_id": str(asked.id), "message": "Reworded question?"},
                "supports_confirmation": True,
            },
        )
        confirmation = _events("tool_confirmation", events)[0]
        # Nothing is queued while the turn is only paused - it has no reply yet.
        assert [c for c in queue.calls if c[0] == JOB_GENERATE_TITLE] == []
        resumed = await _resume(
            client, confirmation["conversation_id"], confirmation["id"], "approve"
        )

    assert _events("done", resumed)
    titles = [call for call in queue.calls if call[0] == JOB_GENERATE_TITLE]
    assert len(titles) == 1, "the resumed edit must ask for the thread's new name"
    assert titles[0][2]["job_id"] != title_job_id(conversation.id), (
        "and under an id of its own, or arq deduplicates it away"
    )


async def test_a_checkpoint_written_before_the_anchor_still_resumes(db_sessionmaker) -> None:
    # Backward compatibility for confirmations already in flight across a deploy,
    # the same treatment turn_result_budget gets: no key means no anchor to
    # check, which is exactly the behaviour those turns already had. It must not
    # raise, and it must still save its reply.
    danger = _danger()
    source = FakeMcpSource([danger])
    settings = make_settings()
    provider = _pausing_provider()
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        async with db_sessionmaker() as session:
            conversation = await ConversationRepository(session, TENANT).get(UUID(conversation_id))
            payload = dict(conversation.pending_confirmation)
            payload.pop("user_message_id")
            conversation.pending_confirmation = payload
            session.add(conversation)
            await session.commit()
        events = await _resume(client, conversation_id, confirmation_id, "approve")

    assert danger.calls
    assert _events("done", events)
    assert "assistant" in await _roles(db_sessionmaker, conversation_id)


async def test_a_resumed_oversized_result_still_fits_the_window(db_sessionmaker) -> None:
    # The resume is the one path that may exceed the turn's computed result
    # budget, because by design it never DROPS the call it paused on - the user
    # was asked and said yes, so answering "actually, no room" would make the
    # question a lie. What it must never do is overrun num_ctx itself: above
    # that line the runtime discards from the HEAD, taking the system prompt
    # and the tool guidance with it, which is exactly the eviction the clamp on
    # the normal path exists to prevent.
    #
    # So the assertion here is the weaker, honest one: the prompt fits the
    # window. Some of the answer's reserve may be spent; the head may not.
    # The window has to be nearly full BEFORE the result lands, or the clamp is
    # never the thing that decides anything and the test passes on an empty
    # window. A ~6.4k-token question against 8192 leaves a few hundred tokens of
    # result budget - far under the per-result env ceiling, which is what makes
    # the difference between the two observable.
    danger = RemoteTool("create_record", side_effecting=True, result="RECORD " + "x" * 40_000)
    source = FakeMcpSource([danger])
    provider = _pausing_provider(answer="Filed it.")
    settings = make_settings()
    long_ask = f"{ASK} " + "context " * 3_000
    # The subject is the RESUME clamp, so the turn has to actually be agentic:
    # at assisted the pre-fetch and the ambient knowledge base share this
    # deliberately-tight window, and the tool budget floor drops the tools
    # before anything gets far enough to pause. A question this long also reads
    # as one to reason over, and only an agentic deployment resolves that in
    # favour of the tools - which is the other half of what the seeding buys.
    await seed_agentic_verdict(db_sessionmaker)
    async with mcp_chat_client(
        provider, db_sessionmaker, settings, store=make_memory_store(), mcp_sources=[source]
    ) as client:
        events = await collect_events(client, {"message": long_ask, "supports_confirmation": True})
        confirmation = _events("tool_confirmation", events)[0]
        events = await _resume(
            client, confirmation["conversation_id"], confirmation["id"], "approve"
        )

    assert _tokens(events) == "Filed it."
    # The approved call really did run and really did return the huge payload -
    # otherwise this passes by never exercising the clamp at all.
    assert len(danger.calls) == 1
    continuation = provider.calls[-1]
    assert any(m.role == "tool" for m in continuation)
    # The bound: the resume may exceed the turn's computed budget by the framing
    # allowance of the one call it is honouring, and by nothing more. Drop the
    # clamp back to the env per-result ceiling and the result arrives hundreds of
    # tokens fatter, eating the answer's reserve rather than nibbling it.
    assert (
        sum(approx_tokens(m.content) for m in continuation)
        + ANSWER_RESERVE_TOKENS
        - _TOOL_MESSAGE_FRAMING_TOKENS
        <= 8192
    )


# --------------------------------------------------------------------------- #
# The resumed turn is a fully-equipped turn                                    #
# --------------------------------------------------------------------------- #

OLD_EMBEDDER = "embed:written-by-someone-else"


async def test_a_resumed_turn_is_a_fully_equipped_turn(db_sessionmaker) -> None:
    """A resume must not be a second-class turn that quietly lost half its wiring.

    The turn context used to be assembled BELOW the confirmation branch, so
    every resumed turn re-entered the generation loop without it - and
    everything downstream that reads it simply went quiet for the rest of that
    stream. The stranded-vectors alarm is the observable consequence used here:
    the resumed turn searches an unreachable knowledge base and must still be
    able to say so.
    """
    from app.providers.rag.base import ChunkPayload, ChunkPoint
    from tests.unit.test_chat_rag import counting_store

    settings = make_settings()

    # An agentic turn, so the knowledge base is reached through the TOOL rather
    # than by the ambient pre-fetch - the tool is what the resumed stream has
    # to still be able to run, and what the alarm rides out on.
    await seed_agentic_verdict(db_sessionmaker)

    store = counting_store()
    await store.ensure_ready(OLD_EMBEDDER, 32)
    await store.upsert(
        "kb",
        OLD_EMBEDDER,
        str(DEFAULT_TENANT_ID),
        [
            ChunkPoint(
                id=str(uuid4()),
                vector=[1.0] + [0.0] * 31,
                payload=ChunkPayload(
                    tenant_id=str(DEFAULT_TENANT_ID),
                    doc_id="doc-stranded",
                    text="everything the workspace knows",
                    created_at=1000.0,
                    embed_model=OLD_EMBEDDER,
                ),
            )
        ],
    )
    # Pause on the side-effecting call, then - once resumed - search the
    # knowledge base and find nothing, which is the anomaly branch.
    provider = provisioned(
        turns=[
            FakeTurn(
                tool_calls=[
                    ToolCall(id="c1", name="create_record", arguments={"query": "login bug"})
                ]
            ),
            FakeTurn(
                tool_calls=[
                    ToolCall(id="c2", name="search_knowledge", arguments={"query": "anything"})
                ]
            ),
            FakeTurn(tokens=["Filed it."]),
        ]
    )
    async with mcp_chat_client(
        provider,
        db_sessionmaker,
        settings,
        store=store,
        mcp_sources=[FakeMcpSource([_danger()])],
    ) as client:
        conversation_id, confirmation_id, _ = await _pause(client)
        events = await _resume(client, conversation_id, confirmation_id, "approve")

    assert "vectors_stranded" in status_codes(events)
