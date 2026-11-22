"""A file a tool produced, all the way to the wire and the saved transcript.

These drive the real endpoint with a FakeProvider scripting the tool call and a
fake MCP source offering a tool that hands a file back, and assert the three
things a client and a reader depend on:

- the ``file`` event arrives, on BOTH loop paths - the ordinary one and the
  confirmation resume, which is a separate emission site and would otherwise be
  the half nobody notices is missing;
- the assistant message records what it produced, on a clean finish AND on one
  cut short, because the row and its bytes are already committed by then;
- the bytes never appear anywhere else. Not in the prompt the model is sent, not
  in the SSE payloads, not in the persisted trace. That is the property the whole
  design exists for: a base64 payload in the token-budgeted text path evicts the
  prompt HEAD, taking the tool guidance and the injection defense with it.

The harness (settings, fake sources, event collection) is shared with
test_chat_mcp_tools and test_chat_confirmation; only the subject is new.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

from httpx import AsyncClient

from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.models import ATTACHMENT_ORIGIN_GENERATED, ATTACHMENT_READY, Attachment, Message
from app.providers.base import ChatChunk, ChatMessage, GenerationOptions, ToolCall
from app.providers.fake_provider import FakeProvider, FakeTurn
from app.providers.tools.base import Tool, ToolArtifact, ToolContext, ToolResult, ToolSpec
from app.repositories.attachments import AttachmentRepository
from app.repositories.conversations import ConversationRepository
from app.repositories.messages import MessageRepository
from app.schemas.conversations import MessageOut
from app.services.chat_service import _history_content, _referenced_attachments
from app.storage.local import FakeStorageBackend
from tests.conftest import make_memory_store
from tests.unit.test_chat_mcp_tools import (
    DEV_PAIR,
    FakeMcpSource,
    collect_events,
    make_settings,
    mcp_chat_client,
    provisioned,
)

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID)

# Deliberately recognisable, so a leak anywhere is a substring search away. Both
# spellings are checked: the raw bytes and the base64 an MCP server would send.
SENTINEL = b"SENTINEL-BYTES-MUST-NOT-REACH-THE-PROMPT"
SENTINEL_B64 = base64.b64encode(SENTINEL).decode()

ASK = "make me a report of last quarter"
SUMMARY = "Generated report.pdf (2 pages)."

_QUERY_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "What to report on."}},
    "required": ["query"],
}


class FileTool(Tool):
    """A tool that returns a one-line summary plus one or more files.

    This is the seam view of what a document-generating MCP server sends: the
    text is what the model reads, the artifacts are what the user gets.
    """

    def __init__(
        self,
        name: str = "create_report",
        *,
        side_effecting: bool = False,
        files: int = 1,
        is_error: bool = False,
    ) -> None:
        self._name = name
        self._side_effecting = side_effecting
        self._files = files
        self._is_error = is_error
        self.calls: list[dict] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description="Build a report document.",
            parameters=_QUERY_SCHEMA,
            side_effecting=self._side_effecting,
        )

    async def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(
            content=SUMMARY,
            is_error=self._is_error,
            artifacts=[
                ToolArtifact(
                    filename=f"report{index}.pdf" if self._files > 1 else "report.pdf",
                    content_type="application/pdf",
                    data=SENTINEL,
                )
                for index in range(self._files)
            ],
        )


def _calling_provider(
    name: str = "create_report", answer: str = "Here is the report."
) -> FakeProvider:
    return provisioned(
        turns=[
            FakeTurn(
                tool_calls=[ToolCall(id="c1", name=name, arguments={"query": "last quarter"})]
            ),
            FakeTurn(tokens=[answer]),
        ]
    )


class _FailsAfterReadingTheResult(FakeProvider):
    """Requests the tool, then streams a partial answer and dies mid-stream.

    The tool really ran, so its file is really stored - which is exactly the
    state an interrupted save has to record. ``turns`` alone cannot express this:
    a scripted turn always ends cleanly.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._passes = 0

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[ChatChunk]:
        self._passes += 1
        if self._passes == 1:
            self.calls.append(list(messages))
            self.last_tools = tools
            yield ChatChunk(
                type="tool_call",
                tool_call=ToolCall(id="c1", name="create_report", arguments={"query": "q"}),
            )
            yield ChatChunk(type="done", finish_reason="stop")
            return
        self.calls.append(list(messages))
        yield ChatChunk(type="token", delta="partial")
        yield ChatChunk(type="error", error="boom", error_code="provider_unavailable")


def _events(name: str, events: list[tuple[str, dict]]) -> list[dict]:
    return [payload for event, payload in events if event == name]


async def _attachments(db_sessionmaker, conversation_id: str):
    async with db_sessionmaker() as session:
        return await AttachmentRepository(session, TENANT).list_for_conversation(
            UUID(conversation_id)
        )


async def _make_conversation(db_sessionmaker) -> UUID:
    """An empty conversation to hang staged attachment rows on."""
    async with db_sessionmaker() as session:
        conversation = await ConversationRepository(session, TENANT).create()
        await session.commit()
        return conversation.id


async def _stage_attachment(
    db_sessionmaker, conversation_id: UUID, *, filename: str, origin: str
) -> Attachment:
    """Write one READY attachment row directly, bypassing the upload path.

    Direct because the point is the pair: a generated row cannot be produced by
    POST /attachments (that endpoint refuses the origin, by design), so the two
    states have to be staged side by side to be compared at all.
    """
    async with db_sessionmaker() as session:
        repo = AttachmentRepository(session, TENANT)
        row = Attachment(
            tenant_id=TENANT.tenant_id,
            user_id=TENANT.user_id,
            conversation_id=conversation_id,
            filename=filename,
            content_type="application/octet-stream",
            size_bytes=len(SENTINEL),
            storage_key=f"{TENANT.tenant_id}/{conversation_id}/{uuid4()}",
            status=ATTACHMENT_READY,
            origin=origin,
        )
        repo.add(row)
        await session.commit()
        return row


async def _messages(db_sessionmaker, conversation_id: str) -> list[MessageOut]:
    async with db_sessionmaker() as session:
        rows = await MessageRepository(session, TENANT).list_recent(UUID(conversation_id), 50)
    return [MessageOut.from_entity(row) for row in rows]


# --------------------------------------------------------------------------- #
# The ordinary loop                                                            #
# --------------------------------------------------------------------------- #


async def test_a_generated_file_is_stored_and_announced(db_sessionmaker) -> None:
    # The slice in one turn. The file exists as a real attachment on the
    # conversation, and the client is told about it by id - which is the only
    # thing that makes it reachable, since the bytes never cross the stream.
    tool = FileTool()
    storage = FakeStorageBackend()
    provider = _calling_provider()
    async with mcp_chat_client(
        provider,
        db_sessionmaker,
        make_settings(),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([tool])],
        storage=storage,
    ) as client:
        events = await collect_events(client, {"message": ASK})

    files = _events("file", events)
    assert len(files) == 1
    announced = files[0]
    assert announced["filename"] == "report.pdf"
    assert announced["content_type"] == "application/pdf"
    assert announced["size_bytes"] == len(SENTINEL)
    assert announced["origin"] == ATTACHMENT_ORIGIN_GENERATED

    conversation_id = _events("meta", events)[0]["conversation_id"]
    rows = await _attachments(db_sessionmaker, conversation_id)
    assert [str(row.id) for row in rows] == [announced["attachment_id"]]
    assert rows[0].status == ATTACHMENT_READY
    assert rows[0].origin == ATTACHMENT_ORIGIN_GENERATED
    assert rows[0].content_hash is None
    assert storage.objects[rows[0].storage_key] == SENTINEL


async def test_the_file_frame_lands_between_its_tool_result_and_done(db_sessionmaker) -> None:
    # Where the frame sits in the stream: after the tool_result that settles the
    # call (so a client can attach it to that trace card) and before done.
    #
    # Note what this does NOT prove. The stronger property - that the row is
    # committed BEFORE the frame is emitted - is structural rather than tested:
    # `store_tool_artifacts` is a plain coroutine, not a generator, so it returns
    # only after its commit and nothing can be emitted from inside it. There is
    # no ordering a test could perturb without restructuring that function, so
    # this asserts the frame ordering it can actually observe and says so.
    tool = FileTool()
    async with mcp_chat_client(
        _calling_provider(),
        db_sessionmaker,
        make_settings(),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([tool])],
    ) as client:
        events = await collect_events(client, {"message": ASK})

    names = [event for event, _ in events]
    assert names.index("tool_result") < names.index("file")
    assert names.index("file") < names.index("done")


async def test_the_bytes_never_reach_the_model_the_trace_or_the_wire(db_sessionmaker) -> None:
    # THE ONE THAT MATTERS. Three separate places a blob could leak, each with a
    # different consequence: the prompt (evicts the system prompt and its
    # injection defense), the SSE payloads (megabytes of base64 down a token
    # stream), and the persisted trace (a JSON column re-read on every later
    # turn). None of them may hold either spelling of the bytes.
    provider = _calling_provider()
    async with mcp_chat_client(
        provider,
        db_sessionmaker,
        make_settings(),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([FileTool()])],
    ) as client:
        events = await collect_events(client, {"message": ASK})

    sent = json.dumps(
        [[message.content for message in call] for call in provider.calls], ensure_ascii=False
    )
    streamed = json.dumps(events, ensure_ascii=False)
    conversation_id = _events("meta", events)[0]["conversation_id"]
    messages = await _messages(db_sessionmaker, conversation_id)
    persisted = json.dumps(
        [message.model_dump(mode="json") for message in messages], ensure_ascii=False
    )

    for where, blob in (("prompt", sent), ("stream", streamed), ("transcript", persisted)):
        assert SENTINEL.decode() not in blob, f"raw artifact bytes leaked into the {where}"
        assert SENTINEL_B64 not in blob, f"base64 artifact bytes leaked into the {where}"

    # And the model still learned what the tool did - via the tool's own summary,
    # which is the whole point of the split.
    assert SUMMARY in sent


async def test_the_tool_card_names_the_file_and_its_size(db_sessionmaker) -> None:
    # What the user sees instead of the bytes: a name and a size, built from the
    # host's OWN validated record rather than from anything the server said.
    async with mcp_chat_client(
        _calling_provider(),
        db_sessionmaker,
        make_settings(),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([FileTool()])],
    ) as client:
        events = await collect_events(client, {"message": ASK})

    result = _events("tool_result", events)[0]["result"]
    assert "report.pdf" in result
    assert "B)" in result or "KB)" in result


async def test_the_assistant_message_records_the_file_it_produced(db_sessionmaker) -> None:
    # There is no message-to-attachment foreign key, so this denormalized
    # snapshot is the ONLY thing that lets a reloaded transcript show the
    # document an answer produced. `origin` is what tells a client this file
    # belongs to the answer rather than being an unsent draft of the user's.
    async with mcp_chat_client(
        _calling_provider(),
        db_sessionmaker,
        make_settings(),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([FileTool()])],
    ) as client:
        events = await collect_events(client, {"message": ASK})

    conversation_id = _events("meta", events)[0]["conversation_id"]
    messages = await _messages(db_sessionmaker, conversation_id)
    reply = next(message for message in messages if message.role == "assistant")
    assert [(a.filename, a.origin) for a in reply.attachments] == [
        ("report.pdf", ATTACHMENT_ORIGIN_GENERATED)
    ]
    assert reply.attachments[0].id == _events("file", events)[0]["attachment_id"]


async def test_a_generated_file_is_kept_out_of_the_composer_restore_list(db_sessionmaker) -> None:
    # The two lists in a conversation detail mean different things, and only one
    # of them is about the composer.
    #
    # The TOP-LEVEL list is what a reopened pane puts back as unsent drafts. The
    # client picks those by "no message carries it", which a generated file can
    # satisfy perfectly honestly - so left in, the assistant's own document turns
    # into a chip in the user's composer and rides out with their next question.
    #
    # The MESSAGE-level refs are the transcript, and the generated file has to
    # stay there or the reply that produced it points at nothing.
    async with mcp_chat_client(
        _calling_provider(),
        db_sessionmaker,
        make_settings(),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([FileTool()])],
    ) as client:
        events = await collect_events(client, {"message": ASK})
        conversation_id = _events("meta", events)[0]["conversation_id"]
        # A real upload on the same thread, so the test proves a FILTER rather
        # than an empty list - the upload must survive it.
        upload = await client.post(
            "/api/v1/attachments",
            data={"conversation_id": conversation_id},
            files={"file": ("notes.txt", b"a genuine draft the user left", "text/plain")},
        )
        assert upload.status_code == 201
        detail = (await client.get(f"/api/v1/conversations/{conversation_id}")).json()

    generated_id = _events("file", events)[0]["attachment_id"]
    # Both rows really are on the conversation...
    rows = await _attachments(db_sessionmaker, conversation_id)
    assert {row.origin for row in rows} == {ATTACHMENT_ORIGIN_GENERATED, "upload"}
    # ...but only the upload is offered back to the composer.
    assert [a["filename"] for a in detail["attachments"]] == ["notes.txt"]
    assert generated_id not in {a["id"] for a in detail["attachments"]}
    # And the reply still names the file it made, which is the half the
    # transcript renders from.
    reply = next(m for m in detail["messages"] if m["role"] == "assistant")
    assert [a["id"] for a in reply["attachments"]] == [generated_id]


async def test_an_interrupted_turn_still_records_the_file_it_already_made(db_sessionmaker) -> None:
    # The file is committed the moment the tool returns, long before the answer
    # finishes. A turn that dies afterwards must still point at it, or the row
    # exists with nothing in the transcript naming it - storage nobody can find.
    provider = _FailsAfterReadingTheResult(models=list(DEV_PAIR))
    async with mcp_chat_client(
        provider,
        db_sessionmaker,
        make_settings(),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([FileTool()])],
    ) as client:
        events = await collect_events(client, {"message": ASK})

    assert _events("file", events)
    conversation_id = _events("meta", events)[0]["conversation_id"]
    messages = await _messages(db_sessionmaker, conversation_id)
    reply = next(message for message in messages if message.role == "assistant")
    assert reply.interrupted is True
    assert reply.content == "partial"
    assert [a.filename for a in reply.attachments] == ["report.pdf"]


async def test_the_per_turn_file_cap_drops_the_rest_and_says_so(db_sessionmaker) -> None:
    # Mutation check for this one: raise tool_artifact_max_per_turn and the
    # single-file assertion below fails, because both files are then kept.
    settings = make_settings(tool_artifact_max_per_turn=1)
    storage = FakeStorageBackend()
    async with mcp_chat_client(
        _calling_provider(),
        db_sessionmaker,
        settings,
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([FileTool(files=2)])],
        storage=storage,
    ) as client:
        events = await collect_events(client, {"message": ASK})

    assert len(_events("file", events)) == 1
    assert len(storage.objects) == 1
    # Dropped, but never silently: the user's card says so.
    assert "not kept" in _events("tool_result", events)[0]["result"]


async def test_a_failed_call_keeps_none_of_the_files_it_attached(db_sessionmaker) -> None:
    # A server that reports an error AND attaches a file is describing something
    # that did not finish. Keeping the file would hand the user a document the
    # answer then says never happened - and attribute it to a call that failed.
    #
    # Mutation check: drop the `and result.is_error` branch so materialization
    # runs regardless, and the three assertions below all fail (a file event
    # appears, a row exists, bytes land in storage).
    tool = FileTool(is_error=True)
    storage = FakeStorageBackend()
    async with mcp_chat_client(
        _calling_provider(),
        db_sessionmaker,
        make_settings(),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([tool])],
        storage=storage,
    ) as client:
        events = await collect_events(client, {"message": ASK})

    assert tool.calls, "the tool must actually have run for this to mean anything"
    assert _events("file", events) == []
    assert storage.objects == {}
    conversation_id = _events("meta", events)[0]["conversation_id"]
    assert await _attachments(db_sessionmaker, conversation_id) == []
    # The failure is still surfaced the way any tool error is.
    assert "tool_error" in [payload.get("code") for _, payload in events if _ == "status"]


async def test_the_same_tool_succeeding_does_keep_its_file(db_sessionmaker) -> None:
    # The control for the test above: identical setup minus the error flag. Without
    # this, "no file was kept" could be measuring a broken harness rather than the
    # is_error rule.
    storage = FakeStorageBackend()
    async with mcp_chat_client(
        _calling_provider(),
        db_sessionmaker,
        make_settings(),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([FileTool(is_error=False)])],
        storage=storage,
    ) as client:
        events = await collect_events(client, {"message": ASK})

    assert len(_events("file", events)) == 1
    assert len(storage.objects) == 1


async def test_a_tool_that_returns_no_file_announces_nothing(db_sessionmaker) -> None:
    # The no-regression half. Every tool that existed before this slice returns
    # no artifacts, and such a turn must be byte-identical to what it was.
    tool = FileTool()
    tool._files = 0
    async with mcp_chat_client(
        _calling_provider(),
        db_sessionmaker,
        make_settings(),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([tool])],
    ) as client:
        events = await collect_events(client, {"message": ASK})

    assert _events("file", events) == []
    conversation_id = _events("meta", events)[0]["conversation_id"]
    assert await _attachments(db_sessionmaker, conversation_id) == []
    messages = await _messages(db_sessionmaker, conversation_id)
    reply = next(message for message in messages if message.role == "assistant")
    assert reply.attachments == []


# --------------------------------------------------------------------------- #
# A generated file must not be mistaken for one the user asked about            #
# --------------------------------------------------------------------------- #


async def test_a_later_question_about_the_report_resolves_the_upload_not_the_generated_file(
    db_sessionmaker,
) -> None:
    # The composer's tray empties on send, so a question on a LATER turn about a
    # file uploaded earlier is matched by name (see file_reference). A generated
    # file lands in the same conversation, and readiness is what makes it
    # dangerous here: it is born `ready` with no parse job, so it passes the
    # status filter while holding no indexed text at all.
    #
    # Both files here answer to "report". Without the origin exclusion the
    # generated report.pdf is pulled in beside - or instead of - the uploaded
    # report.txt the question was actually about, and the turn announces
    # "Searching report.pdf for relevant context..." before finding nothing.
    #
    # Mutation check: delete `and row.origin != ATTACHMENT_ORIGIN_GENERATED` from
    # _referenced_attachments and this fails - both rows come back.
    conversation_id = await _make_conversation(db_sessionmaker)
    upload = await _stage_attachment(
        db_sessionmaker, conversation_id, filename="report.txt", origin="upload"
    )
    await _stage_attachment(
        db_sessionmaker,
        conversation_id,
        filename="report.pdf",
        origin=ATTACHMENT_ORIGIN_GENERATED,
    )

    resolved = await _referenced_attachments(
        db_sessionmaker,
        TENANT,
        conversation_id,
        "what does the report say?",
        already_named=[],
    )

    assert [row.filename for row in resolved] == ["report.txt"]
    assert [row.id for row in resolved] == [upload.id]


def test_a_wordless_reply_that_made_a_file_does_not_speak_in_attachments() -> None:
    # _history_content exists so a USER turn of files and no words replays as
    # something the model can use ("(attached notes.txt)") instead of as silence.
    #
    # An assistant reply can now carry files too, and the same treatment there is
    # a lie about who said what: a thinking-only reply that generated a document
    # would be replayed to the model as the ASSISTANT having said "(attached
    # report.pdf)", words it never produced. A wordless reply replays as
    # wordless.
    #
    # Constructed rows rather than a staged turn: the state needs a persisted
    # assistant message with empty content, non-empty thinking AND a generated
    # file reference, and driving a real turn into exactly that shape would test
    # the harness more than the guard. The guard is one role comparison.
    #
    # Mutation check: drop `or message.role != "user"` and the first assertion
    # fails.
    generated = Message(
        conversation_id=uuid4(),
        role="assistant",
        content="",
        thinking="considering the numbers",
        meta={
            "attachments": [
                {
                    "id": str(uuid4()),
                    "filename": "report.pdf",
                    "content_type": "application/pdf",
                    "size_bytes": 10,
                    "origin": ATTACHMENT_ORIGIN_GENERATED,
                }
            ]
        },
    )
    carried = Message(
        conversation_id=uuid4(),
        role="user",
        content="",
        meta={
            "attachments": [
                {
                    "id": str(uuid4()),
                    "filename": "notes.txt",
                    "content_type": "text/plain",
                    "size_bytes": 10,
                    "origin": "upload",
                }
            ]
        },
    )

    assert _history_content(generated) == ""
    # ...and the user-turn behaviour it was built for is untouched.
    assert _history_content(carried) == "(attached notes.txt)"


# --------------------------------------------------------------------------- #
# The confirmation resume - the second, separately written emission site       #
# --------------------------------------------------------------------------- #


async def _pause_and_resume(
    client: AsyncClient, decision: str = "approve"
) -> tuple[str, list[tuple[str, dict]]]:
    paused = await collect_events(client, {"message": ASK, "supports_confirmation": True})
    confirmation = _events("tool_confirmation", paused)[0]
    resumed = await collect_events(
        client,
        {
            "conversation_id": confirmation["conversation_id"],
            "confirmation": {"id": confirmation["id"], "decision": decision},
        },
    )
    return confirmation["conversation_id"], resumed


async def test_a_file_from_an_approved_side_effecting_tool_is_announced_on_resume(
    db_sessionmaker,
) -> None:
    # A tool that WRITES something is exactly the shape that produces a document,
    # so the resume is not an edge case - it is the main case for this feature.
    # Its emission is written separately from the loop's, which is why it gets
    # its own test rather than being assumed to follow.
    tool = FileTool("create_report", side_effecting=True)
    storage = FakeStorageBackend()
    provider = _calling_provider()
    async with mcp_chat_client(
        provider,
        db_sessionmaker,
        make_settings(),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([tool])],
        storage=storage,
    ) as client:
        conversation_id, resumed = await _pause_and_resume(client)

    assert tool.calls, "the approved tool should have run"
    files = _events("file", resumed)
    assert len(files) == 1
    assert files[0]["filename"] == "report.pdf"
    rows = await _attachments(db_sessionmaker, conversation_id)
    assert [str(row.id) for row in rows] == [files[0]["attachment_id"]]
    assert storage.objects[rows[0].storage_key] == SENTINEL

    messages = await _messages(db_sessionmaker, conversation_id)
    reply = next(message for message in messages if message.role == "assistant")
    assert [a.filename for a in reply.attachments] == ["report.pdf"]


async def test_a_file_made_before_the_pause_survives_the_checkpoint(db_sessionmaker) -> None:
    # The path that is easy to build and easy to get wrong. A read-only tool makes
    # a file in round one; round two asks for a side-effecting tool and the turn
    # suspends. The first file is already stored and announced by then, so the
    # checkpoint has to CARRY it: without that the resumed turn saves a message
    # naming only the second file, and the first one exists in the database with
    # nothing in the transcript pointing at it.
    #
    # It is also what makes the per-turn count cap honest across a pause - a
    # resume that started counting from zero would be a way to mint a second
    # turn's worth of files.
    reader = FileTool("draft_report", side_effecting=False)
    writer = FileTool("file_report", side_effecting=True)
    provider = provisioned(
        turns=[
            FakeTurn(tool_calls=[ToolCall(id="c1", name="draft_report", arguments={"query": "q"})]),
            FakeTurn(tool_calls=[ToolCall(id="c2", name="file_report", arguments={"query": "q"})]),
            FakeTurn(tokens=["Both done."]),
        ]
    )
    async with mcp_chat_client(
        provider,
        db_sessionmaker,
        make_settings(),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([reader, writer])],
        storage=FakeStorageBackend(),
    ) as client:
        paused = await collect_events(client, {"message": ASK, "supports_confirmation": True})
        confirmation = _events("tool_confirmation", paused)[0]
        conversation_id = confirmation["conversation_id"]
        resumed = await collect_events(
            client,
            {
                "conversation_id": conversation_id,
                "confirmation": {"id": confirmation["id"], "decision": "approve"},
            },
        )

    # One file announced on each side of the pause.
    assert [file["filename"] for file in _events("file", paused)] == ["report.pdf"]
    assert [file["filename"] for file in _events("file", resumed)] == ["report.pdf"]
    # Two real rows...
    rows = await _attachments(db_sessionmaker, conversation_id)
    assert len(rows) == 2
    # ...and the saved reply names BOTH, in the order the turn made them.
    messages = await _messages(db_sessionmaker, conversation_id)
    reply = next(message for message in messages if message.role == "assistant")
    assert [a.id for a in reply.attachments] == [str(row.id) for row in rows]


async def test_the_file_cap_is_not_reset_by_a_confirmation_pause(db_sessionmaker) -> None:
    # The same carry, read as a budget. With a cap of one, the file made before
    # the pause spends it, and the approved call after the pause gets none -
    # which is what stops a pause from being a way around the cap.
    #
    # Mutation check: seed the resume's generated_files as empty (drop the
    # checkpoint's "generated_files" key) and the second file is kept, failing
    # the single-row assertion below.
    reader = FileTool("draft_report", side_effecting=False)
    writer = FileTool("file_report", side_effecting=True)
    provider = provisioned(
        turns=[
            FakeTurn(tool_calls=[ToolCall(id="c1", name="draft_report", arguments={"query": "q"})]),
            FakeTurn(tool_calls=[ToolCall(id="c2", name="file_report", arguments={"query": "q"})]),
            FakeTurn(tokens=["Done."]),
        ]
    )
    storage = FakeStorageBackend()
    async with mcp_chat_client(
        provider,
        db_sessionmaker,
        make_settings(tool_artifact_max_per_turn=1),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([reader, writer])],
        storage=storage,
    ) as client:
        paused = await collect_events(client, {"message": ASK, "supports_confirmation": True})
        confirmation = _events("tool_confirmation", paused)[0]
        conversation_id = confirmation["conversation_id"]
        resumed = await collect_events(
            client,
            {
                "conversation_id": conversation_id,
                "confirmation": {"id": confirmation["id"], "decision": "approve"},
            },
        )

    assert len(_events("file", paused)) == 1
    assert _events("file", resumed) == []
    assert len(await _attachments(db_sessionmaker, conversation_id)) == 1
    assert len(storage.objects) == 1
    # The action still RAN - only its file was refused - and the refusal is said
    # out loud rather than leaving the model to describe a document nobody has.
    assert writer.calls
    assert "not kept" in _events("tool_result", resumed)[0]["result"]


async def test_a_denied_tool_produces_no_file_at_all(db_sessionmaker) -> None:
    # The refusal has to be total. A denied action that still left its document
    # behind would be the worst possible reading of "no".
    tool = FileTool("create_report", side_effecting=True)
    storage = FakeStorageBackend()
    async with mcp_chat_client(
        _calling_provider(),
        db_sessionmaker,
        make_settings(),
        store=make_memory_store(),
        mcp_sources=[FakeMcpSource([tool])],
        storage=storage,
    ) as client:
        conversation_id, resumed = await _pause_and_resume(client, decision="deny")

    assert tool.calls == []
    assert _events("file", resumed) == []
    assert storage.objects == {}
    assert await _attachments(db_sessionmaker, conversation_id) == []
