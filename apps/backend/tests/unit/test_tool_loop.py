"""The re-entrant tool loop in chat_service._generate.

Drives ``_generate`` directly with a scripted FakeProvider (per-call ``turns``)
and a scripted in-process tool, because ``stream_chat`` does not offer tools
until a later slice. Covers: a tool round feeds its result back and answers, an
empty offer stays a single pass, a failing or invented tool is reported not
crashed, the budget terminates a model that never stops calling, and a
disconnect mid-loop still persists the partial.
"""

from __future__ import annotations

import asyncio
import json
from uuid import UUID

import anyio

from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.models import utcnow
from app.providers.base import ChatChunk, ChatMessage, ToolCall
from app.providers.fake_provider import FakeProvider, FakeTurn
from app.providers.tools.base import (
    Tool,
    ToolContext,
    ToolError,
    ToolResult,
    ToolResultSource,
    ToolSpec,
)
from app.repositories.conversations import ConversationRepository
from app.services.chat_service import _generate
from app.services.context_budget import ANSWER_RESERVE_TOKENS, approx_tokens
from app.services.tools.offer import tools_to_wire

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, model_provider="fake", **overrides)


class ScriptedTool(Tool):
    """A tool whose result (or failure) is fixed, recording each call it gets."""

    def __init__(
        self,
        name: str = "web_search",
        *,
        result: str = "the tool result",
        side_effecting: bool = False,
        raises: Exception | None = None,
        delay: float = 0.0,
        sources: list[ToolResultSource] | None = None,
        count: int | None = None,
        is_error: bool = False,
    ) -> None:
        self._is_error = is_error
        self._count = count
        self._name = name
        self._result = result
        self._side = side_effecting
        self._raises = raises
        self._delay = delay
        self._sources = sources or []
        self.calls: list[tuple[dict, ToolContext]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description="Look something up.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            side_effecting=self._side,
        )

    async def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        self.calls.append((arguments, context))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raises is not None:
            raise self._raises
        return ToolResult(
            content=self._result,
            sources=self._sources,
            count=self._count,
            is_error=self._is_error,
        )


async def _seed_conversation(db_sessionmaker) -> UUID:
    async with db_sessionmaker() as session:
        conversation = await ConversationRepository(session, TENANT).create(title="seeded")
        await session.commit()
        return conversation.id


async def _load(db_sessionmaker, conversation_id: UUID):
    async with db_sessionmaker() as session:
        return await ConversationRepository(session, TENANT).get_with_messages(conversation_id)


def _context(conversation_id: UUID) -> ToolContext:
    return ToolContext(
        tenant_id=str(TENANT.tenant_id),
        user_id=str(TENANT.user_id),
        conversation_id=str(conversation_id),
        session_id=str(conversation_id),
    )


async def _run(
    db_sessionmaker,
    provider: FakeProvider,
    *,
    offered_tools: list[Tool] | None,
    settings: Settings | None = None,
) -> tuple[UUID, list[ChatChunk]]:
    conversation_id = await _seed_conversation(db_sessionmaker)
    chunks: list[ChatChunk] = []
    async for chunk in _generate(
        provider=provider,
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        messages=[ChatMessage(role="user", content="a question")],
        model="fake",
        think=None,
        conversation_id=conversation_id,
        user_at=utcnow(),
        settings=settings or make_settings(),
        offered_tools=offered_tools,
        tool_context=_context(conversation_id) if offered_tools else None,
    ):
        chunks.append(chunk)
    return conversation_id, chunks


def _status_codes(chunks: list[ChatChunk]) -> list[str]:
    return [c.status_code for c in chunks if c.type == "status"]


def _tokens(chunks: list[ChatChunk]) -> str:
    return "".join(c.delta for c in chunks if c.type == "token")


async def test_a_tool_round_runs_the_tool_feeds_it_back_and_answers(db_sessionmaker) -> None:
    tool = ScriptedTool(result="It is sunny.")
    provider = FakeProvider(
        turns=[
            FakeTurn(
                tool_calls=[ToolCall(id="c1", name="web_search", arguments={"query": "weather"})]
            ),
            FakeTurn(tokens=["Sunny", " today."]),
        ]
    )
    conversation_id, chunks = await _run(db_sessionmaker, provider, offered_tools=[tool])

    # The tool ran once with the model's arguments, under the CONTEXT's identity.
    assert len(tool.calls) == 1
    assert tool.calls[0][0] == {"query": "weather"}
    assert tool.calls[0][1].tenant_id == str(TENANT.tenant_id)

    # The tools were offered on the wire, in the OpenAI-function shape.
    assert provider.last_tools is not None
    assert provider.last_tools[0]["function"]["name"] == "web_search"

    # A transient tool_running status was surfaced - and after the result, the
    # narration hands off from "Running ..." to reading the result, so a stale
    # "Running" never stands beside a finished step card.
    assert "tool_running" in _status_codes(chunks)
    statuses = [c.delta for c in chunks if c.type == "status"]
    result_index = next(i for i, c in enumerate(chunks) if c.type == "tool_result")
    after_result = [c.delta for c in chunks[result_index + 1 :] if c.type == "status"]
    assert any(s.startswith("Reading ") for s in after_result), statuses

    # The SECOND provider call carried the assistant tool-call turn and the tool
    # result, fenced as data, back to the model.
    assert len(provider.calls) == 2
    second = provider.calls[1]
    assert any(m.role == "assistant" and m.tool_calls for m in second)
    tool_msgs = [m for m in second if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].name == "web_search"
    assert "It is sunny." in tool_msgs[0].content
    assert "reference DATA" in tool_msgs[0].content
    assert "<<<" in tool_msgs[0].content

    # The final answer streamed and persisted; the tool round did not leak into it.
    assert _tokens(chunks) == "Sunny today."
    _, messages = await _load(db_sessionmaker, conversation_id)
    assert [m.role for m in messages] == ["assistant"]
    assert messages[0].content == "Sunny today."


async def test_no_tools_offered_is_a_single_pass(db_sessionmaker) -> None:
    provider = FakeProvider(["hello"])
    conversation_id, chunks = await _run(db_sessionmaker, provider, offered_tools=None)

    assert len(provider.calls) == 1
    assert provider.last_tools is None
    assert _tokens(chunks) == "hello"
    _, messages = await _load(db_sessionmaker, conversation_id)
    assert messages[0].content == "hello"


async def test_a_failing_tool_is_reported_and_the_turn_still_answers(db_sessionmaker) -> None:
    tool = ScriptedTool(raises=ToolError("upstream 500"))
    provider = FakeProvider(
        turns=[
            FakeTurn(tool_calls=[ToolCall(id="c1", name="web_search", arguments={})]),
            FakeTurn(tokens=["Answer."]),
        ]
    )
    conversation_id, chunks = await _run(db_sessionmaker, provider, offered_tools=[tool])

    assert "tool_error" in _status_codes(chunks)
    # The failure is fed back so the model knows the result is missing.
    second = provider.calls[1]
    assert any(m.role == "tool" and "failed" in m.content for m in second)
    # The turn still answered, and the standing notice persisted with it.
    assert _tokens(chunks) == "Answer."
    _, messages = await _load(db_sessionmaker, conversation_id)
    assert messages[0].content == "Answer."
    assert any(n["code"] == "tool_error" for n in messages[0].meta["notices"])


async def test_a_tool_the_model_invented_is_reported_not_crashed(db_sessionmaker) -> None:
    tool = ScriptedTool(name="web_search")
    provider = FakeProvider(
        turns=[
            FakeTurn(tool_calls=[ToolCall(id="c1", name="does_not_exist", arguments={})]),
            FakeTurn(tokens=["ok"]),
        ]
    )
    conversation_id, chunks = await _run(db_sessionmaker, provider, offered_tools=[tool])

    assert len(tool.calls) == 0  # the real tool was never touched
    assert "tool_error" in _status_codes(chunks)
    second = provider.calls[1]
    assert any(m.role == "tool" and "No tool named" in m.content for m in second)
    assert _tokens(chunks) == "ok"


async def test_the_iteration_cap_terminates_a_model_that_always_calls(db_sessionmaker) -> None:
    # A single turn that always requests a tool, replayed on every call: without
    # the cap the loop would never answer. tool_max_iterations forces a final,
    # tool-free round after two tool rounds.
    tool = ScriptedTool()
    provider = FakeProvider(
        turns=[
            FakeTurn(
                tool_calls=[ToolCall(id="c", name="web_search", arguments={})], tokens=["stop"]
            )
        ]
    )
    conversation_id, chunks = await _run(
        db_sessionmaker,
        provider,
        offered_tools=[tool],
        settings=make_settings(tool_max_iterations=2),
    )

    assert len(tool.calls) == 2
    assert "tool_limit_reached" in _status_codes(chunks)
    assert any(c.type == "done" for c in chunks)
    _, messages = await _load(db_sessionmaker, conversation_id)
    assert messages[0].content == "stop"


async def test_the_call_cap_bounds_the_tools_run_in_one_round(db_sessionmaker) -> None:
    tool = ScriptedTool()
    three_calls = [ToolCall(id=str(i), name="web_search", arguments={}) for i in range(3)]
    provider = FakeProvider(turns=[FakeTurn(tool_calls=three_calls), FakeTurn(tokens=["done"])])
    conversation_id, chunks = await _run(
        db_sessionmaker,
        provider,
        offered_tools=[tool],
        settings=make_settings(tool_max_calls_per_turn=2),
    )

    # Only two of the three requested calls run; the cap trims the rest.
    assert len(tool.calls) == 2
    assert "tool_limit_reached" in _status_codes(chunks)
    assert _tokens(chunks) == "done"


async def test_a_disconnect_during_the_loop_persists_the_partial(db_sessionmaker) -> None:
    # A tool runs, then the answer streams; the client disconnects after the
    # first answer token. The loop's finally must still persist the partial,
    # shielded, exactly as on the non-tool path.
    tool = ScriptedTool()
    provider = FakeProvider(
        turns=[
            FakeTurn(tool_calls=[ToolCall(id="c", name="web_search", arguments={})]),
            FakeTurn(tokens=["a ", "b ", "c"]),
        ],
        delay=0.001,
    )
    conversation_id = await _seed_conversation(db_sessionmaker)
    gen = _generate(
        provider=provider,
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        messages=[ChatMessage(role="user", content="q")],
        model="fake",
        think=None,
        conversation_id=conversation_id,
        user_at=utcnow(),
        settings=make_settings(),
        offered_tools=[tool],
        tool_context=_context(conversation_id),
    )
    with anyio.CancelScope() as scope:
        seen = 0
        async for chunk in gen:
            if chunk.type == "token":
                seen += 1
                if seen == 1:
                    scope.cancel()  # the client disconnects mid-answer

    # The shielded persist finishes on the loop after the scope unwinds.
    messages: list = []
    for _ in range(50):
        await asyncio.sleep(0.005)
        _, messages = await _load(db_sessionmaker, conversation_id)
        if messages:
            break
    assert len(tool.calls) == 1
    assert [m.role for m in messages] == ["assistant"]
    assert messages[0].content == "a "
    assert messages[0].meta["interrupted"] is True
    # The tool really ran before the disconnect, so its step is part of the
    # partial trace - a reloaded, cut-short turn still shows what it did.
    assert [s["name"] for s in messages[0].meta["tool_steps"]] == ["web_search"]


async def test_preamble_before_a_tool_call_is_dropped_from_the_saved_answer(
    db_sessionmaker,
) -> None:
    # A model that narrates before calling a tool: the preamble streams live but
    # is not part of the final reply, so it must not be persisted, and the done
    # chunk must carry the final text for the client to swap its glued copy.
    tool = ScriptedTool(result="42")
    provider = FakeProvider(
        turns=[
            FakeTurn(
                tokens=["Let me look. "],
                tool_calls=[ToolCall(id="c", name="web_search", arguments={})],
            ),
            FakeTurn(tokens=["The answer is 42."]),
        ]
    )
    conversation_id, chunks = await _run(db_sessionmaker, provider, offered_tools=[tool])

    # The persisted answer is only the final reply, not the preamble.
    _, messages = await _load(db_sessionmaker, conversation_id)
    assert messages[0].content == "The answer is 42."

    # The done chunk carries that final text so the client can replace what it
    # streamed (which glued the preamble onto the answer).
    done = next(c for c in chunks if c.type == "done")
    assert done.content == "The answer is 42."


def _citations(chunks: list[ChatChunk]) -> list:
    return [c.citation for c in chunks if c.type == "citation"]


async def test_tool_sources_become_numbered_citable_blocks(db_sessionmaker) -> None:
    # Slice 3c: a tool that read real material returns sources; the LOOP renders
    # them as numbered blocks (it owns the turn's numbering) and the answer's
    # [n] references emit citation events exactly like grounded retrieval.
    source = ToolResultSource(
        id="p1",
        doc_id="d1",
        text="Refunds run 30 days from delivery.",
        title="Refund policy",
        url="https://example.test/refunds",
        source_type="web",
        score=0.9,
    )
    tool = ScriptedTool(sources=[source])
    provider = FakeProvider(
        turns=[
            FakeTurn(
                tool_calls=[ToolCall(id="c1", name="web_search", arguments={"query": "refunds"})]
            ),
            FakeTurn(tokens=["30 days", " [1]."]),
        ]
    )
    _, chunks = await _run(db_sessionmaker, provider, offered_tools=[tool])

    # The model read a numbered, citable block with its label, URL and text.
    tool_msg = next(m for m in provider.calls[1] if m.role == "tool")
    assert "[1] Refund policy" in tool_msg.content
    assert "https://example.test/refunds" in tool_msg.content
    assert "Refunds run 30 days" in tool_msg.content

    # The answer referenced [1], so exactly that source came out as a citation.
    cited = _citations(chunks)
    assert [c.index for c in cited] == [1]
    assert cited[0].doc_id == "d1"
    assert cited[0].source_type == "web"
    assert cited[0].url == "https://example.test/refunds"


def _tool_results(chunks: list[ChatChunk]) -> list:
    return [c.tool_result for c in chunks if c.type == "tool_result"]


def _source(n: int) -> ToolResultSource:
    return ToolResultSource(id=f"p{n}", doc_id=f"d{n}", text=f"Passage {n}.", title=f"Doc {n}")


async def _count_of(db_sessionmaker, tool: ScriptedTool) -> int | None:
    provider = FakeProvider(
        turns=[
            FakeTurn(tool_calls=[ToolCall(id="c1", name="web_search", arguments={})]),
            FakeTurn(tokens=["Answer."]),
        ]
    )
    _, chunks = await _run(db_sessionmaker, provider, offered_tools=[tool])
    (result,) = _tool_results(chunks)
    return result.count


async def test_a_tool_that_produced_three_sources_reports_count_three(db_sessionmaker) -> None:
    assert (
        await _count_of(db_sessionmaker, ScriptedTool(sources=[_source(n) for n in (1, 2, 3)])) == 3
    )


async def test_a_tool_may_state_its_own_count(db_sessionmaker) -> None:
    # A tool whose text result is a countable list carries no sources; its typed
    # count must win over the (empty) source fallback.
    assert await _count_of(db_sessionmaker, ScriptedTool(count=4)) == 4


async def test_a_tool_that_cannot_count_reports_no_count(db_sessionmaker) -> None:
    assert await _count_of(db_sessionmaker, ScriptedTool()) is None


async def test_a_failed_call_reports_no_count(db_sessionmaker) -> None:
    # A tool may return an error result that still carries a count or sources
    # (an MCP isError reply with records); a failed call produced nothing.
    tool = ScriptedTool(is_error=True, count=3, sources=[_source(1)])
    assert await _count_of(db_sessionmaker, tool) is None


async def test_a_repeated_source_keeps_its_number_across_calls(db_sessionmaker) -> None:
    # A second lookup that surfaces the SAME document must reuse its number,
    # not mint a duplicate citation - repeated searches are the cache's normal
    # case, and unstable numbering would break the model's earlier references.
    source = ToolResultSource(id="p1", doc_id="d1", text="Same page.", title="Page")
    tool = ScriptedTool(sources=[source])
    provider = FakeProvider(
        turns=[
            FakeTurn(tool_calls=[ToolCall(id="c1", name="web_search", arguments={"query": "a"})]),
            FakeTurn(tool_calls=[ToolCall(id="c2", name="web_search", arguments={"query": "b"})]),
            FakeTurn(tokens=["Answer", " [1]."]),
        ]
    )
    _, chunks = await _run(db_sessionmaker, provider, offered_tools=[tool])

    # Both tool messages number the same document [1].
    tool_msgs = [m for m in provider.calls[2] if m.role == "tool"]
    assert len(tool_msgs) == 2
    assert all("[1] Page" in m.content for m in tool_msgs)
    # One citation, not two.
    assert [c.index for c in _citations(chunks)] == [1]


async def _run_with_window(
    db_sessionmaker,
    provider: FakeProvider,
    *,
    offered_tools: list[Tool],
    user_content: str,
    num_ctx: int,
    settings: Settings | None = None,
) -> tuple[UUID, list[ChatChunk]]:
    """Drive _generate with a real context window, which _run leaves unset.

    The tool-result affordability floor only engages when num_ctx is known, so a
    test about the spend cap cannot use the default driver.
    """
    conversation_id = await _seed_conversation(db_sessionmaker)
    chunks: list[ChatChunk] = []
    async for chunk in _generate(
        provider=provider,
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        messages=[ChatMessage(role="user", content=user_content)],
        model="fake",
        think=None,
        conversation_id=conversation_id,
        user_at=utcnow(),
        settings=settings or make_settings(),
        num_ctx=num_ctx,
        offered_tools=offered_tools,
        tool_context=_context(conversation_id),
    ):
        chunks.append(chunk)
    return conversation_id, chunks


async def test_the_loop_spends_the_turns_computed_budget_not_the_env_ceiling(
    db_sessionmaker,
) -> None:
    # The regression this pins: tool_result_total_max_tokens is a POLICY CEILING,
    # while what a turn can actually afford is num_ctx minus the answer reserve
    # minus the assembled prompt. When a full prompt makes the affordable budget
    # much smaller than the ceiling, spending the ceiling overruns num_ctx - and
    # Ollama truncates from the HEAD, silently dropping the system prompt with
    # the tool guidance and the injection defense, on precisely the turns that
    # ingest attacker-controlled web content. There is no error and no downstream
    # guard, so only this test stands between that and a release.
    #
    # Arithmetic: room = 8192 - 1024 reserve = 7168. The ~27k-char message costs
    # 6750 tokens and the tool schemas 44 more, so the computed budget is 374 -
    # above the 256 offer floor, so tools ARE still offered, and far below the
    # 2000 ceiling. One tool result (capped at tool_result_max_tokens=1000)
    # therefore exhausts the real budget but not the ceiling.
    #
    # That leaves only 118 tokens of headroom above the offer floor, so the band
    # is asserted below rather than left implicit: a change to prompt composition
    # should fail with "the fixture drifted" rather than as an opaque assertion
    # about chunks. It cannot false-PASS either way - too little room drops the
    # tools entirely (zero calls), too much skips tool_limit_reached.
    tool = ScriptedTool(result="x" * 8000)
    user_content = "q " * 13_500
    # Pin the fixture inside its intended band before driving anything.
    expected_budget = (
        8192
        - ANSWER_RESERVE_TOKENS
        - (approx_tokens(user_content) + approx_tokens(json.dumps(tools_to_wire([tool]))))
    )
    assert 256 < expected_budget < 1000, (
        f"fixture drifted out of band: budget={expected_budget}; "
        "adjust user_content so the turn can still afford tools but not the ceiling"
    )
    # Two turns: one tool round, then the forced final answer. A third scripted
    # turn would prove nothing - the loop is already on its last round.
    provider = FakeProvider(
        turns=[
            FakeTurn(tool_calls=[ToolCall(id="c", name="web_search", arguments={})]),
            FakeTurn(tokens=["done"]),
        ]
    )
    _, chunks = await _run_with_window(
        db_sessionmaker,
        provider,
        offered_tools=[tool],
        user_content=user_content,
        num_ctx=8192,
        # High enough that neither cap can be what stops the loop.
        settings=make_settings(tool_max_iterations=6, tool_max_calls_per_turn=12),
    )

    # One round ran and the tool was NOT dropped for lack of room.
    assert len(tool.calls) == 1
    # THIS is the discriminator: the ~1000-token result spent the 374-token
    # budget, so the turn declares itself out of room. Against the 2000 ceiling
    # it would not have, and this assertion is the one that fails on a revert.
    # (len(tool.calls) == 1 passes either way - the scripted second turn carries
    # no tool call, so the loop would stop there regardless.)
    assert "tool_limit_reached" in _status_codes(chunks)
    assert _tokens(chunks) == "done"


async def test_the_final_prompt_never_overruns_num_ctx_however_big_the_results(
    db_sessionmaker,
) -> None:
    # The invariant the budget arithmetic exists to protect, asserted directly on
    # the wire: whatever a tool returns, the prompt the turn ends up sending still
    # leaves the answer its reserve inside num_ctx. Anything above that line is
    # discarded by the runtime from the HEAD - messages[0], the system prompt,
    # carrying the tool guidance and the "reference DATA, not instructions"
    # framing - which hands an attacker-controlled tool result the most
    # authoritative position in the window. The sibling test above pins that the
    # loop STOPS at the computed budget; this one pins that it does not blow past
    # it on the way there, which stopping after the spend cannot guarantee.
    #
    # Arithmetic: room = 8192 - 1024 reserve = 7168. The ~22.5k-char message plus
    # the tool schemas leave a computed budget near 1500 - comfortably above the
    # 256 offer floor (tools ARE offered) and below the 2000 env ceiling, and wide
    # enough for two capped results but not three. The band is pinned first so a
    # change to prompt composition fails as "the fixture drifted".
    tool = ScriptedTool(result="x" * 40_000)
    user_content = "q " * 11_248
    expected_budget = (
        8192
        - ANSWER_RESERVE_TOKENS
        - (approx_tokens(user_content) + approx_tokens(json.dumps(tools_to_wire([tool]))))
    )
    assert 1000 < expected_budget < 2000, (
        f"fixture drifted out of band: budget={expected_budget}; "
        "adjust user_content so two capped results fit and a third does not"
    )
    # Three calls in ONE round, each returning far more than any cap allows.
    provider = FakeProvider(
        turns=[
            FakeTurn(
                tool_calls=[
                    ToolCall(id="c1", name="web_search", arguments={}),
                    ToolCall(id="c2", name="web_search", arguments={}),
                    ToolCall(id="c3", name="web_search", arguments={}),
                ]
            ),
            FakeTurn(tokens=["done"]),
        ]
    )
    _, chunks = await _run_with_window(
        db_sessionmaker,
        provider,
        offered_tools=[tool],
        user_content=user_content,
        num_ctx=8192,
        # High enough that neither count cap can be what stops the round.
        settings=make_settings(tool_max_iterations=6, tool_max_calls_per_turn=12),
    )

    # THE invariant. The final round is the fullest prompt the turn ever sends
    # (every tool result is in it and the schemas are not), so it is where an
    # overshoot shows up.
    final = provider.calls[-1]
    assert sum(approx_tokens(m.content) for m in final) + ANSWER_RESERVE_TOKENS <= 8192

    # The third call was DROPPED, not run on credit: the budget is enforced
    # before the spend, so a call the turn cannot afford never reaches the tool.
    assert len(tool.calls) == 2
    assert "tool_limit_reached" in _status_codes(chunks)

    # And the assistant turn claims only the calls that produced a result, so the
    # exchange fed back to the model has no orphaned tool_calls entry.
    assistant = next(m for m in final if m.role == "assistant" and m.tool_calls)
    tool_msgs = [m for m in final if m.role == "tool"]
    assert len(assistant.tool_calls) == len(tool_msgs) == 2
    assert _tokens(chunks) == "done"


async def test_a_roomy_turn_still_spends_the_full_env_ceiling(db_sessionmaker) -> None:
    # The other side of the same rule: narrowing the budget must not make every
    # turn stingy. With a short prompt the affordable budget exceeds the ceiling,
    # min() keeps the ceiling, and the loop behaves exactly as it did before.
    tool = ScriptedTool(result="x" * 8000)
    provider = FakeProvider(
        turns=[
            FakeTurn(tool_calls=[ToolCall(id="c", name="web_search", arguments={})]),
            FakeTurn(tool_calls=[ToolCall(id="c", name="web_search", arguments={})]),
            FakeTurn(tokens=["done"]),
        ]
    )
    _, chunks = await _run_with_window(
        db_sessionmaker,
        provider,
        offered_tools=[tool],
        user_content="a short question",
        num_ctx=8192,
        settings=make_settings(tool_max_iterations=6, tool_max_calls_per_turn=12),
    )

    # Two rounds of ~1000 tokens each to reach the 2000 ceiling.
    assert len(tool.calls) == 2
    assert "tool_limit_reached" in _status_codes(chunks)


class FatSchemaTool(ScriptedTool):
    """A tool whose SCHEMA is the expensive part, not its results.

    A remote MCP server describes itself in its own words, and those words ride
    the wire on every non-final round. This models the realistic upper end.
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description="Look something up. " + ("detail " * 400),
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            side_effecting=self._side,
        )


async def test_the_affordability_floor_counts_the_tool_schemas(db_sessionmaker) -> None:
    # The schemas are part of the prompt the turn actually sends, so a floor that
    # omits them measures a prompt that does not exist and can conclude a turn
    # can afford tools it cannot. This fixture is sized so the schemas alone are
    # decisive: counting them, the budget falls under the 256 offer floor and the
    # tools are correctly dropped; ignoring them, the turn would offer a tool
    # whose result cannot fit - which is the lie the floor exists to prevent.
    tool = FatSchemaTool(result="x" * 8000)
    user_content = "q " * 13_100
    message_tokens = approx_tokens(user_content)
    schema_tokens = approx_tokens(json.dumps(tools_to_wire([tool])))
    room = 8192 - ANSWER_RESERVE_TOKENS
    assert room - message_tokens >= 256, "without schemas the turn must still look affordable"
    assert room - message_tokens - schema_tokens < 256, "with schemas it must not"

    provider = FakeProvider(turns=[FakeTurn(tokens=["answered without tools"])])
    _, chunks = await _run_with_window(
        db_sessionmaker,
        provider,
        offered_tools=[tool],
        user_content=user_content,
        num_ctx=8192,
        settings=make_settings(tool_max_iterations=6, tool_max_calls_per_turn=12),
    )

    # No tool was offered on the wire, and none ran.
    assert provider.last_tools is None
    assert tool.calls == []
    assert _tokens(chunks) == "answered without tools"
