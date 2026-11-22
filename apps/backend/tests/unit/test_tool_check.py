"""Pure-service tests for the tool-check harness (app/services/tool_check.py).

Every test drives the harness with a scripted provider - no network, no real
tools - and pins the behaviors the verdicts hang on: each case's pass/fail
rule, the canned-result feedback loop, the depth caps, the pair routing at
serve-time options, and the harness-version fingerprint.

"Serve-time options" now means the runtime's own answer folded with whatever a
deployment has declared, on BOTH halves: the window comes from the weights'
native limit bounded by what the deployment affords, and the think flag from
the same chain a turn resolves. Neither is read off a committed per-model file
any more, which is why the pair under test here is described by scripted
CAPABILITY REPORTS rather than by a catalog fixture.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.providers.base import (
    ChatChunk,
    GenerationOptions,
    ModelCapabilities,
    ToolCall,
    Usage,
)
from app.providers.fake_provider import FakeProvider, FakeTurn
from app.services import prompts, tool_check
from app.services.model_policy import PolicyView
from app.services.model_resolution import ThinkMode
from app.services.tool_check import (
    FAST_CASE_NAMES,
    THINK_CASE_NAMES,
    case_named,
    clamping_cases,
    classify_think_mode,
    harness_version,
    run_case,
    run_tool_check,
)

# What a fast run covers now: the two load-bearing tool cases plus the thinking
# group, which joins EVERY depth (it is two short generations with no tools).
FAST_ALL = (*THINK_CASE_NAMES, *FAST_CASE_NAMES)

# The deployment's DEFAULT_CONTEXT_WINDOW as these runs see it. The harness
# reads no settings of its own, so every call names the number.
DEPLOYMENT_WINDOW = 8192

# The standing pair these tests check. Nothing declares either tag, so what
# they ARE is entirely the scripted report below.
CHAT_TAG = "chat-tag"
THINK_TAG = "think-tag"

# The runtime's report for that pair, and the one difference between the two
# rows is the whole reason it is a pair: the chat tag's template carries no
# think plumbing (a turn on it resolves ``never`` and OMITS the flag) while the
# sibling's does (a reasoning turn resolves ``hybrid`` and sends it). That
# difference is what makes a per-target resolution visible in a call log - one
# answer reused across a run would send the same flag to both.
PAIR_REPORTS: dict[str, ModelCapabilities] = {
    CHAT_TAG: ModelCapabilities(completion=True, tools=True, template_thinking=False),
    THINK_TAG: ModelCapabilities(completion=True, tools=True, template_thinking=True),
}

WEB_CALL = ToolCall(id="c1", name="web_search", arguments={"query": "latest python version"})
KB_CALL = ToolCall(id="c2", name="search_knowledge", arguments={"query": "refund window"})


async def sample(case_name: str, provider: FakeProvider) -> tool_check.CaseSample:
    return await run_case(
        provider, case_named(case_name), model=CHAT_TAG, options=GenerationOptions()
    )


async def check(
    provider: FakeProvider,
    *,
    chat_model: str = CHAT_TAG,
    thinking_model: str | None = THINK_TAG,
    default_window: int = DEPLOYMENT_WINDOW,
    **kwargs,
) -> tool_check.ToolCheckSummary:
    """Run the harness over the standing pair, with the shared arguments named.

    Args:
        provider: The scripted provider to drive.
        chat_model: The tag under check.
        thinking_model: The pair sibling, or "" for a single-model deployment.
        default_window: What this deployment affords one turn.
        **kwargs: Passed straight through (``deep``, ``context_ceiling``, the
            two injected think-mode reads, ``on_sample``, ``samples_per_case``).

    Returns:
        The run's :class:`ToolCheckSummary`.
    """
    return await run_tool_check(
        provider,
        provider_name="fake",
        default_window=default_window,
        chat_model=chat_model,
        thinking_model=thinking_model,
        **kwargs,
    )


class GoodModelProvider(FakeProvider):
    """Emulates a model that calls exactly when it should - every case passes.

    Decides from the offered tool list and the user message, the way a
    well-behaved model would: answer math directly, answer once a tool result
    arrives, route records-shaped questions to search_knowledge, and otherwise
    call the first offered tool. Records (model, options) per call so tests
    can assert the harness ran at serve-time options.

    It reports :data:`PAIR_REPORTS` for the standing pair, and a caller's own
    ``capabilities_by_model`` is merged OVER that - so a test that describes
    one extra tag does not silently drop the pair's reports and change what
    every other sample in the same run was driven with.
    """

    def __init__(self, **kwargs) -> None:
        kwargs["capabilities_by_model"] = {
            **PAIR_REPORTS,
            **(kwargs.get("capabilities_by_model") or {}),
        }
        super().__init__(**kwargs)
        self.seen: list[tuple[str | None, GenerationOptions | None]] = []

    async def chat_stream(self, messages, *, model=None, tools=None, options=None):
        self.seen.append((model, options))
        offered = [t["function"]["name"] for t in tools or []]
        user = next(m.content for m in reversed(messages) if m.role == "user")
        if options is not None and options.think:
            # A model asked to reason, reasons. Without this the fake answers
            # identically whatever flag it is sent, and case (h) - "reasoned
            # AND called AND answered" - would pass on an exchange in which
            # nothing reasoned, which is the verdict the harness must never
            # hand out.
            yield ChatChunk(type="thinking", delta="thinking it through")
        if messages[-1].role == "tool":
            yield ChatChunk(type="token", delta="Answered from the tool result.")
        elif "2 + 2" in user or not offered:
            yield ChatChunk(type="token", delta="4")
        else:
            wants_kb = "refund" in user or "returns" in user
            name = "search_knowledge" if wants_kb and "search_knowledge" in offered else offered[0]
            yield ChatChunk(
                type="tool_call",
                tool_call=ToolCall(id="g1", name=name, arguments={"query": user}),
            )
        yield ChatChunk(type="done", finish_reason="stop", usage=Usage())


# --- what a run refuses to start without ----------------------------------------


@pytest.mark.parametrize("missing", ["", None], ids=["blank", "unset"])
async def test_a_run_with_no_chat_model_refuses_rather_than_guessing(missing) -> None:
    # A deployment that has chosen no chat model resolves to the empty tag, and
    # there is nothing here that could fill it in: the harness reads no settings
    # and no inventory, so guessing would mean probing whichever tag a caller
    # forgot to name. The caller knows which model it meant to check.
    with pytest.raises(ValueError, match="chat_model is required"):
        await run_tool_check(
            GoodModelProvider(),
            provider_name="fake",
            default_window=DEPLOYMENT_WINDOW,
            chat_model=missing,
        )


# --- per-case pass/fail rules -------------------------------------------------


async def test_must_call_passes_when_the_right_tool_is_called():
    provider = FakeProvider(
        turns=[FakeTurn(tool_calls=[WEB_CALL]), FakeTurn(tokens=["3.13.5", " is latest."])]
    )
    result = await sample("must-call-freshness", provider)
    assert result.passed
    assert result.tokens == 2
    # The tool-call round streamed no tokens, so only the answer round's two
    # tokens contribute a generation interval to the tok/s aggregate.
    assert result.gen_intervals == 1


async def test_must_call_fails_when_no_tool_is_called():
    provider = FakeProvider(turns=[FakeTurn(tokens=["I cannot browse the web."])])
    result = await sample("must-call-freshness", provider)
    assert not result.passed
    assert "no tool call" in result.detail


async def test_must_not_call_passes_on_a_direct_answer():
    provider = FakeProvider(turns=[FakeTurn(tokens=["4"])])
    result = await sample("must-not-call", provider)
    assert result.passed


async def test_must_not_call_fails_on_any_call():
    provider = FakeProvider(turns=[FakeTurn(tool_calls=[WEB_CALL]), FakeTurn(tokens=["4"])])
    result = await sample("must-not-call", provider)
    assert not result.passed
    assert "web_search" in result.detail


async def test_selection_requires_search_knowledge_and_only_it():
    picker = FakeProvider(turns=[FakeTurn(tool_calls=[KB_CALL]), FakeTurn(tokens=["30 days."])])
    assert (await sample("selection", picker)).passed
    grabber = FakeProvider(turns=[FakeTurn(tool_calls=[WEB_CALL]), FakeTurn(tokens=["30 days."])])
    result = await sample("selection", grabber)
    assert not result.passed


async def test_hallucination_guard_fails_on_a_tool_never_offered():
    invented = ToolCall(id="x1", name="fetch_page", arguments={"url": "https://x"})
    provider = FakeProvider(turns=[FakeTurn(tool_calls=[invented]), FakeTurn(tokens=["done"])])
    result = await sample("hallucination-guard", provider)
    assert not result.passed
    assert "fetch_page" in result.detail


async def test_hallucination_guard_passes_on_declining_and_on_the_offered_tool():
    decliner = FakeProvider(
        turns=[FakeTurn(tokens=["I cannot check the web, but stable general advice is..."])]
    )
    assert (await sample("hallucination-guard", decliner)).passed
    caller = FakeProvider(
        turns=[FakeTurn(tool_calls=[KB_CALL]), FakeTurn(tokens=["Nothing stored on that."])]
    )
    assert (await sample("hallucination-guard", caller)).passed


async def test_loop_guard_fails_a_model_that_calls_every_round():
    # A single turn repeats forever, so the model requests web_search on every
    # pass; the harness itself must stop at the round cap and fail the sample.
    provider = FakeProvider(turns=[FakeTurn(tool_calls=[WEB_CALL])])
    result = await sample("loop-guard", provider)
    assert not result.passed
    assert "still requesting" in result.detail
    assert len(provider.calls) == 3  # the cap held: exactly _MAX_ROUNDS passes


async def test_continuation_feeds_the_canned_result_back_and_needs_an_answer():
    provider = FakeProvider(
        turns=[FakeTurn(tool_calls=[WEB_CALL]), FakeTurn(tokens=["Python 3.13.5."])]
    )
    result = await sample("continuation", provider)
    assert result.passed
    # The second provider call must carry the exchange so far: the assistant
    # message with the call, then a tool-role message with the CANNED result
    # (fenced as quoted data), correlated by the call id.
    second_call = provider.calls[1]
    assert second_call[-2].role == "assistant"
    assert second_call[-2].tool_calls == [WEB_CALL]
    tool_message = second_call[-1]
    assert tool_message.role == "tool"
    assert tool_message.name == "web_search"
    assert tool_message.tool_call_id == "c1"
    assert "Python 3.13.5" in tool_message.content  # the canned result...
    assert "<<<" in tool_message.content  # ...inside the injection fence


async def test_thinking_with_tools_fails_when_output_is_empty_despite_a_call():
    # The ollama/ollama#10976 signature: reasoning streams, a tool call is
    # made, and the ANSWER comes back empty - no error anywhere.
    provider = FakeProvider(
        turns=[
            FakeTurn(tool_calls=[WEB_CALL], thinking=["let me look that up"]),
            FakeTurn(tokens=[], thinking=["reading the result"]),
        ]
    )
    result = await run_case(
        provider,
        case_named("thinking-with-tools"),
        model=THINK_TAG,
        options=GenerationOptions(think=True),
    )
    assert not result.passed
    assert "10976" in result.detail


async def test_thinking_with_tools_passes_with_both_a_call_and_an_answer():
    provider = FakeProvider(
        turns=[
            FakeTurn(tool_calls=[WEB_CALL], thinking=["let me look that up"]),
            FakeTurn(tokens=["Python 3.13.5."], thinking=["got it"]),
        ]
    )
    result = await run_case(
        provider,
        case_named("thinking-with-tools"),
        model=THINK_TAG,
        options=GenerationOptions(think=True),
    )
    assert result.passed


# --- the thinking group: emission health, and the measured mode -----------------


class ThinkingModelProvider(FakeProvider):
    """A model whose reasoning behaviour is scripted per requested think flag.

    Emulates the three real shapes at once: what comes back on the reasoning
    channel, and what comes back in the visible answer, for ``think=True`` and
    for ``think=False`` separately. That pairing is the whole subject of the
    thinking group, and no fixed script can express it.
    """

    def __init__(
        self,
        *,
        on: tuple[str, str],
        off: tuple[str, str],
        error: str | None = None,
        capped: bool = False,
    ) -> None:
        super().__init__()
        self._on = on
        self._off = off
        self._error = error
        self._capped = capped

    async def chat_stream(self, messages, *, model=None, tools=None, options=None):
        if self._error is not None:
            yield ChatChunk(type="error", error=self._error, error_code="provider_unavailable")
            return
        reasoning, answer = self._on if (options and options.think) else self._off
        if reasoning:
            yield ChatChunk(type="thinking", delta=reasoning)
        if answer:
            yield ChatChunk(type="token", delta=answer)
        # "length" is what a runtime reports when the generation cap stopped it
        # rather than the model finishing. Scripted from the ANSWER being empty
        # while reasoning happened, which is the shape a real truncation takes.
        capped = self._capped and reasoning and not answer
        yield ChatChunk(type="done", finish_reason="length" if capped else "stop", usage=Usage())


async def think_samples(
    provider: FakeProvider, *, capped: bool = False
) -> dict[str, tool_check.CaseSample]:
    """Run both thinking cases against ``provider``, keyed by case name.

    ``capped`` drives them with the same generation cap the harness sends, so a
    test about truncation exercises the real request rather than a stand-in.
    """
    out: dict[str, tool_check.CaseSample] = {}
    for name in THINK_CASE_NAMES:
        case = case_named(name)
        options = GenerationOptions(
            think=case.think,
            max_tokens=tool_check._THINK_MAX_TOKENS if capped else None,
        )
        out[name] = await run_case(provider, case, model=CHAT_TAG, options=options)
    return out


async def test_a_clean_hybrid_passes_both_directions_and_measures_hybrid():
    provider = ThinkingModelProvider(on=("weighing it up", "Paris"), off=("", "Paris"))
    samples = await think_samples(provider)
    assert all(sample.passed for sample in samples.values())
    assert classify_think_mode(list(samples.values())) == "hybrid"


async def test_a_never_thinking_model_passes_both_and_measures_never():
    # The retry-without-think path, as the harness sees it: asked to reason, it
    # simply does not, and answers cleanly. That is a MEASUREMENT of "never",
    # not a failure - a permanent red mark on an instruct checkpoint would say
    # something untrue about it.
    provider = ThinkingModelProvider(on=("", "Paris"), off=("", "Paris"))
    samples = await think_samples(provider)
    assert all(sample.passed for sample in samples.values())
    assert classify_think_mode(list(samples.values())) == "never"


async def test_an_always_thinking_model_measures_always_from_the_off_sample():
    provider = ThinkingModelProvider(on=("still reasoning", "Paris"), off=("reasoning", "Paris"))
    samples = await think_samples(provider)
    # Both samples are HEALTHY - the reasoning stayed on its own channel - and
    # the model is still classified always, which is the split this group is
    # built on: pass/fail is emission health, the mode is the classification.
    assert all(sample.passed for sample in samples.values())
    assert classify_think_mode(list(samples.values())) == "always"


async def test_leaked_think_markup_fails_the_sample():
    # The mutation guard for the leak assertion: this is the exact shape a
    # think:false on always-reasoning weights produces, and it must FAIL. Drop
    # the leak check in _evaluate_think and this test goes green wrongly.
    provider = ThinkingModelProvider(
        on=("reasoning", "Paris"), off=("", "<think>the capital is Paris</think>Paris")
    )
    samples = await think_samples(provider)
    assert samples["think-on"].passed
    off = samples["think-off"]
    assert not off.passed
    assert "leaked" in off.detail
    # And the leak still counts as REASONING for the classification, because
    # the model plainly reasoned - it just came out of the wrong channel.
    assert classify_think_mode(list(samples.values())) == "always"


async def test_a_closing_marker_alone_still_counts_as_a_leak():
    # A stream cut short can carry only the closing tag; matching just the
    # opening one would call that answer clean.
    provider = ThinkingModelProvider(
        on=("", "the capital is Paris</think>Paris"), off=("", "Paris")
    )
    samples = await think_samples(provider)
    assert not samples["think-on"].passed


async def test_reasoning_with_no_visible_answer_fails():
    # The silent-empty-output shape, on a plain turn rather than a tool one.
    provider = ThinkingModelProvider(on=("thinking hard", ""), off=("", "Paris"))
    samples = await think_samples(provider)
    on = samples["think-on"]
    assert not on.passed
    assert "no visible answer" in on.detail
    assert "it reasoned, then said nothing" in on.detail


async def test_the_thinking_samples_are_capped():
    # The cap exists because this group runs in the FAST check, which fires
    # unattended after every download: an always-thinking tag with no cap can
    # reason for as long as it likes. The TOOL cases stay uncapped - their
    # exchange has to be free to run to the end.
    provider = GoodModelProvider()
    await check(provider)
    caps = {options.max_tokens for _model, options in provider.seen if options is not None}
    assert None in caps  # the tool cases
    assert tool_check._THINK_MAX_TOKENS in caps  # and the thinking pair


async def test_a_capped_generation_is_unmeasured_rather_than_failed():
    # The rule the cap is only acceptable under. Hitting it looks EXACTLY like
    # the empty-answer failure - no visible content either way - so a model
    # that merely reasons at length would be red-marked for being slow. The
    # sample reports a measurement that did not HAPPEN instead, and the
    # classifier drops the whole run's mode on it.
    provider = ThinkingModelProvider(on=("reasoning at length", ""), off=("", "Paris"), capped=True)
    samples = await think_samples(provider, capped=True)
    on = samples["think-on"]
    assert not on.passed
    assert on.error_code == "generation_capped"
    assert "not measured" in on.detail
    assert classify_think_mode(list(samples.values())) is None


async def test_a_cap_that_did_not_truncate_says_nothing():
    # The control. Reading "capped" off the REQUEST rather than off the finish
    # reason would mark every healthy thinking sample unmeasured.
    provider = ThinkingModelProvider(on=("brief", "Paris"), off=("", "Paris"), capped=True)
    samples = await think_samples(provider, capped=True)
    assert all(sample.error_code is None for sample in samples.values())
    assert classify_think_mode(list(samples.values())) == "hybrid"


async def test_a_provider_error_leaves_the_mode_unmeasured():
    # A sample that never reached the model measured the network. Reading its
    # empty reasoning channel as "never" would fabricate a routing input.
    provider = ThinkingModelProvider(
        on=("", ""), off=("", ""), error="cannot reach the model server"
    )
    samples = await think_samples(provider)
    assert classify_think_mode(list(samples.values())) is None


def test_a_run_missing_half_the_pair_is_unmeasured():
    lone = tool_check.CaseSample(
        case="think-on", model="m", passed=True, detail="", latency_s=0.0, tokens=1, thought=True
    )
    assert classify_think_mode([lone]) is None
    assert classify_think_mode([]) is None


async def test_the_thinking_group_never_blames_the_agency_verdict():
    # The mutation guard for the group split: a model that leaks on BOTH
    # thinking samples still earns its depth's cap, because which channel
    # reasoning comes out on says nothing about driving a tool loop. Add the
    # think cases to either scored set and this goes red.
    class Leaky(GoodModelProvider):
        async def chat_stream(self, messages, *, model=None, tools=None, options=None):
            if tools:
                async for chunk in super().chat_stream(
                    messages, model=model, tools=tools, options=options
                ):
                    yield chunk
                return
            self.seen.append((model, options))
            yield ChatChunk(type="token", delta="<think>hmm</think>Paris")
            yield ChatChunk(type="done", finish_reason="stop", usage=Usage())

    summary = await check(Leaky())
    assert all(summary.tallies[name] == (0, 1) for name in THINK_CASE_NAMES)
    assert summary.earned == "assisted"  # the fast cap, unaffected
    assert clamping_cases("fast", summary.tallies) == ()
    assert summary.measured_think_mode == "always"


# --- depths, verdicts, and routing ---------------------------------------------


async def test_fast_caps_at_assisted_even_when_everything_passes():
    provider = GoodModelProvider()
    summary = await check(provider)
    assert summary.depth == "fast"
    assert summary.earned == "assisted"
    assert summary.thinking_with_tools is False  # not earned at this depth
    assert {s.case for s in summary.samples} == set(FAST_ALL)
    # Fast never touches the pair's thinking tag - and neither does the
    # thinking group at any depth, which is the point of running it here.
    assert all(s.model == CHAT_TAG for s in summary.samples)
    assert summary.digest is None  # digest capture lands with model_probes
    assert summary.harness_version == harness_version()


async def test_the_progress_hook_may_be_async_and_runs_between_samples():
    # The verify job takes a live residency reading in this hook, and the
    # reading is only worth anything if it lands while THIS sample's model is
    # still the resident one - so an awaitable return value is awaited before
    # the next sample starts, rather than left to run whenever.
    provider = GoodModelProvider()
    seen: list[int] = []

    async def hook(sample: tool_check.CaseSample) -> None:
        # How many generations the provider had served when this hook ran. A
        # hook fired and forgotten would run DURING the next sample and see a
        # higher count, so the exact numbers are the property under test:
        # The thinking pair runs first at one toolless generation each, then
        # must-call takes two calls (request, then answer from the result) and
        # must-not-call answers in one.
        seen.append(len(provider.seen))

    summary = await check(provider, on_sample=hook)
    assert seen == [1, 2, 4, 5]
    assert len(summary.samples) == len(seen)


async def test_a_plain_sync_progress_hook_still_works():
    # The command-line diagnostic passes one; nothing about it changed.
    seen: list[str] = []
    await check(GoodModelProvider(), on_sample=lambda sample: seen.append(sample.case))
    assert seen == list(FAST_ALL)


async def test_fast_failure_earns_deterministic():
    # Never calls: must-call fails, so the coarse screen fails closed.
    provider = FakeProvider(turns=[FakeTurn(tokens=["I cannot browse the web."])])
    summary = await check(provider)
    assert summary.earned == "deterministic"


async def test_deep_earns_agentic_and_thinking_with_tools_on_the_pair():
    provider = GoodModelProvider()
    summary = await check(provider, deep=True)
    assert summary.depth == "deep"
    assert summary.earned == "agentic"
    assert summary.thinking_with_tools is True
    # Three samples per TOOL case, every case run; the thinking group keeps its
    # one sample per direction at either depth.
    tool_tallies = {
        name: counts for name, counts in summary.tallies.items() if name not in THINK_CASE_NAMES
    }
    assert all(run == 3 and passed == 3 for passed, run in tool_tallies.values())
    assert all(summary.tallies[name] == (1, 1) for name in THINK_CASE_NAMES)
    # Case (h) routes to the thinking tag; everything else stays on chat.
    thinking_samples = [s for s in summary.samples if s.case == "thinking-with-tools"]
    assert len(thinking_samples) == 3
    assert all(s.model == THINK_TAG for s in thinking_samples)
    assert all(s.model == CHAT_TAG for s in summary.samples if s.case != "thinking-with-tools")
    # Serve-time options: nothing here has a native window smaller than the
    # deployment's affordance, so every sample runs at the affordance...
    assert all(options.num_ctx == DEPLOYMENT_WINDOW for _, options in provider.seen)
    # ...and the think flag is normalized per TAG. Exactly four (model, think)
    # shapes reach the provider, and each says something different: the tool
    # cases normalize (the chat tag's template carries no think plumbing, so
    # the flag is omitted; the sibling's does, so a reasoning case sends it),
    # while the thinking group sends the flag LITERALLY on the chat tag both
    # ways. Asserted as a set because the group deliberately re-uses the chat
    # tag with two different flags, which a per-model dict would collapse.
    assert {(model, options.think) for model, options in provider.seen} == {
        (CHAT_TAG, None),
        (THINK_TAG, True),
        (CHAT_TAG, True),
        (CHAT_TAG, False),
    }


async def test_deep_guard_failure_clamps_to_deterministic():
    # Calls web_search on every round of every case: the loop guard fails (it
    # never stops) and the hallucination guard fails (web_search is not
    # offered there), so the verdict is the kill switch.
    provider = FakeProvider(turns=[FakeTurn(tool_calls=[WEB_CALL])])
    summary = await check(provider, deep=True, samples_per_case=1)
    assert summary.earned == "deterministic"
    assert summary.thinking_with_tools is False


async def test_provider_error_chunks_fail_the_sample_with_the_typed_code():
    provider = FakeProvider(
        error="cannot reach the model server", error_code="provider_unavailable"
    )
    result = await sample("must-call-freshness", provider)
    assert not result.passed
    assert result.error_code == "provider_unavailable"


def make_rate_sample(intervals: int, window: float) -> tool_check.CaseSample:
    return tool_check.CaseSample(
        case="x",
        model="m",
        passed=True,
        detail="",
        latency_s=0.0,
        tokens=intervals + 1,
        gen_window_s=window,
        gen_intervals=intervals,
    )


def test_aggregate_rate_is_window_weighted_and_none_when_nothing_generated():
    fast = make_rate_sample(10, 1.0)  # 10 tok/s over 1s
    slow = make_rate_sample(20, 4.0)  # 5 tok/s over 4s
    assert tool_check._aggregate_rate([fast, slow]) == 6.0  # 30 intervals / 5s
    assert tool_check._aggregate_rate([make_rate_sample(0, 0.0)]) is None
    assert tool_check._aggregate_rate([]) is None


# --- the window every sample runs at ---------------------------------------------


async def test_every_sample_is_sized_the_way_a_turn_would_size_it():
    # A probe used to run UNSIZED wherever no committed file named the tag -
    # the runtime picking its own VRAM-scaled window while every real turn on
    # that tag asked for a resolved one. So the verdict described a
    # configuration nothing serves, and the disagreement cost a model reload
    # between the check and the re-warm that follows it. The resolution here is
    # the turn's own: the deployment's affordance, since these weights report
    # no native limit.
    #
    # Deliberately a number no other test in this module uses, so a harness
    # that fell back to some other source would show up rather than pass by
    # coincidence.
    provider = GoodModelProvider()
    await check(provider, default_window=32768)

    windows = {options.num_ctx for _model, options in provider.seen if options is not None}
    assert windows == {32768}


async def test_the_workspace_ceiling_clamps_every_probe_generation():
    # Deep, so the pair's thinking sibling is driven too: both tags are sized
    # per target, and a ceiling that reached only the chat tag would still
    # certify case (h) at a window the workspace forbids.
    provider = GoodModelProvider()
    await check(provider, deep=True, context_ceiling=4096)

    windows = {options.num_ctx for _model, options in provider.seen if options is not None}
    assert windows == {4096}


async def test_every_sample_records_the_window_it_ran_at():
    # What makes a verdict self-describing: the window is the one serve-time
    # option that moves without the weights moving, so a digest says nothing
    # about it and the sample has to.
    provider = GoodModelProvider()
    summary = await check(provider, deep=True, context_ceiling=4096)

    assert {sample.num_ctx for sample in summary.samples} == {4096}


async def test_each_target_is_sized_for_itself_rather_than_from_one_answer():
    # The property _target_for exists to protect: a run touches two tags, and
    # resolving the window ONCE for the run would size case (h) for the wrong
    # model. Visible only when the two tags differ, so the chat tag's weights
    # report a native window BELOW the deployment's affordance while the
    # sibling's report one above it - the shape a mismatched pair really takes.
    provider = GoodModelProvider(
        capabilities_by_model={
            CHAT_TAG: ModelCapabilities(completion=True, tools=True, context_length=4096),
            THINK_TAG: ModelCapabilities(
                completion=True, tools=True, template_thinking=True, context_length=131072
            ),
        }
    )
    await check(provider, deep=True)

    windows = {model: options.num_ctx for model, options in provider.seen if options is not None}
    assert windows == {
        # Its own native limit, which is smaller than what the deployment would
        # otherwise afford it.
        CHAT_TAG: 4096,
        # Native room to spare, so the affordance decides - a different answer
        # from its partner's, resolved for itself.
        THINK_TAG: DEPLOYMENT_WINDOW,
    }


async def test_an_unsized_sample_records_no_window_rather_than_a_guess():
    # run_case with no options is the direct-drive path a test or a future
    # caller can take. Nothing was requested, so nothing is claimed - the
    # honest record of "the runtime chose", never a number invented here.
    result = await sample("must-not-call", FakeProvider(turns=[FakeTurn(tokens=["4"])]))
    assert result.num_ctx is None


# --- the think flag every sample runs at -----------------------------------------

# A tag whose weights the code-shipped quirk seed knows always reason (the
# qwen3 2507 refresh). It is the shape a free-text pull takes, and the exact
# case a committed per-model file used to answer nothing for.
ALWAYS_THINKING_TAG = "qwen3:4b"

# A tag nothing describes at all, so the runtime's own template report is the
# only source left - the bottom rung of the resolution, and the one a caller
# with no database session still reaches.
UNKNOWN_TAG = "mystery:1b"

# What that tag's runtime says: a readable template with no think plumbing,
# which resolves to ``never`` and omits the flag.
NO_THINK_TEMPLATE = ModelCapabilities(tools=True, completion=True, template_thinking=False)


def tool_flags(summary: tool_check.ToolCheckSummary) -> set[bool | None]:
    """The think flags the TOOL samples ran with (the thinking group excluded)."""
    return {sample.think for sample in summary.samples if sample.case not in THINK_CASE_NAMES}


def think_flags(summary: tool_check.ToolCheckSummary) -> dict[str, bool | None]:
    """The think flag each THINKING-group sample ran with, by case name."""
    return {
        sample.case: sample.think for sample in summary.samples if sample.case in THINK_CASE_NAMES
    }


async def test_the_tool_cases_carry_the_flag_a_turn_would_send():
    # The defect this repairs: the tool cases normalized against a committed
    # per-model file alone, which answered "unknown" for a tag it did not name -
    # so an always-thinking model was driven with think:false, the one
    # configuration that puts a raw <think> block in the visible answer, and the
    # harness's own emission check could then score the leak as a failure of the
    # model. Cut the seed and the report out of the chain and every flag below
    # comes back False.
    provider = GoodModelProvider()
    summary = await check(provider, chat_model=ALWAYS_THINKING_TAG, thinking_model="")

    assert tool_flags(summary) == {True}
    # Read off the recorded samples above, and off the PROVIDER here: the two
    # together say the flag was both requested and written down, rather than
    # one of the two.
    assert {options.think for _model, options in provider.seen} == {True, False}


async def test_the_thinking_group_still_sends_its_flag_literally():
    # The group's exception, on the tag most likely to hide a regression: for a
    # model the seed already calls always-thinking, resolving BOTH halves would
    # send think:true twice, and the run would then classify "always" from the
    # seed rather than from the weights - measuring the claim under test.
    summary = await check(GoodModelProvider(), chat_model=ALWAYS_THINKING_TAG, thinking_model="")

    assert think_flags(summary) == {"think-on": True, "think-off": False}


async def test_a_stored_declaration_reaches_the_tool_cases():
    # The rung the harness cannot read for itself. An operator who confirmed
    # that these weights never reason has said the flag must be OMITTED, not
    # sent as false - and the quirk seed says the opposite about this tag, so
    # only the injected read can produce this answer.
    async def declared(tag: str) -> PolicyView:
        return PolicyView(think_mode="never")

    summary = await check(
        GoodModelProvider(),
        chat_model=ALWAYS_THINKING_TAG,
        thinking_model="",
        fetch_policy=declared,
    )

    assert tool_flags(summary) == {None}
    # Untouched: the group is measuring what the declaration claims.
    assert think_flags(summary) == {"think-on": True, "think-off": False}


async def test_a_previous_measurement_reaches_the_tool_cases():
    # The other injected rung, and the only one that outranks a declaration:
    # weights watched reasoning with the flag cleared get it forced back on, so
    # the reasoning lands on the collapsed channel instead of in the answer. The
    # chat tag's own template reports no think plumbing, which is what makes the
    # flag it would otherwise carry (omitted) different from this one.
    async def measured(tag: str) -> ThinkMode:
        return "always"

    summary = await check(GoodModelProvider(), fetch_measured_think=measured)

    assert tool_flags(summary) == {True}
    assert think_flags(summary) == {"think-on": True, "think-off": False}


async def test_a_run_with_no_fetchers_resolves_the_sessionless_half():
    # The command-line diagnostic's shape, and the degradation is a NAMED half
    # rather than an accident: with no session there is no stored declaration
    # and no previous measurement, while the quirk seed and the runtime's
    # template report both still answer. Nothing describes this tag but its
    # template, and a template with no reasoning plumbing means the flag is
    # OMITTED - where a file that did not name the tag said "unknown" and sent
    # false.
    provider = GoodModelProvider(capabilities_by_model={UNKNOWN_TAG: NO_THINK_TEMPLATE})
    summary = await check(provider, chat_model=UNKNOWN_TAG, thinking_model="")

    assert tool_flags(summary) == {None}


async def test_a_tag_that_never_reasons_cannot_earn_thinking_with_tools():
    # The hazard the serve-time resolution opens on case (h), and the reason
    # its pass rule needs all THREE halves. Driving the tool cases at the flag a
    # turn would send means a tag resolved "never" is sent NO flag - so the one
    # case whose whole question is "reason AND call tools in the same turn" runs
    # with nothing reasoning. A well-behaved model still calls and answers, so
    # every other condition holds, and the run would record that these weights
    # can reason with tools on evidence containing no reasoning. This is the axis
    # that fails as silent EMPTY OUTPUT, so the optimistic answer is the
    # dangerous one - drop the reasoning check in _evaluate and this goes red.
    provider = GoodModelProvider(capabilities_by_model={UNKNOWN_TAG: NO_THINK_TEMPLATE})
    summary = await check(
        provider,
        deep=True,
        chat_model=UNKNOWN_TAG,
        thinking_model="",  # single-model deployment: case (h) runs on the tag itself
    )

    assert {sample.think for sample in summary.samples if sample.case == "thinking-with-tools"} == {
        None
    }
    assert summary.thinking_with_tools is False
    assert all(
        "no reasoning came back" in sample.detail
        for sample in summary.samples
        if sample.case == "thinking-with-tools"
    )
    # And it stays out of the AGENCY verdict, exactly as before: case (h) is in
    # neither scored set, so a model that cannot reason is not thereby held down.
    assert summary.earned == "agentic"


async def test_each_target_resolves_its_own_think_flag():
    # The pair shape, and the property one shared resolution would break: an
    # env override can put an undescribed chat tag beside a sibling the seed
    # knows well, and one answer reused across the run would drive case (h)
    # with the other tag's mode.
    #
    # The two targets must resolve DIFFERENTLY or this proves nothing - which is
    # exactly how an earlier version of this test passed with the resolution
    # keyed on the chat tag. So the chat tag is one whose template reports no
    # think plumbing (resolves "never", flag omitted) while the sibling is the
    # seeded always-thinking tag (resolves "always", flag forced on). Key the
    # resolution on the chat tag and case (h) runs with no flag.
    #
    # Read off the samples rather than the provider's call log, because only the
    # samples say which CASE a flag belonged to - and the group's literal flags
    # land on the same tag as the tool cases'.
    provider = GoodModelProvider(capabilities_by_model={UNKNOWN_TAG: NO_THINK_TEMPLATE})
    summary = await check(
        provider, deep=True, chat_model=UNKNOWN_TAG, thinking_model=ALWAYS_THINKING_TAG
    )

    assert {(sample.case, sample.model, sample.think) for sample in summary.samples} >= {
        # The chat tag: nothing but its template describes it, and that template
        # has no think plumbing, so every tool case omits the flag.
        ("must-not-call", UNKNOWN_TAG, None),
        ("continuation", UNKNOWN_TAG, None),
        # The sibling, resolved for ITSELF: the seed calls these weights
        # always-thinking, so case (h) forces the flag on.
        ("thinking-with-tools", ALWAYS_THINKING_TAG, True),
        # And the group keeps its literal pair on the tag under check.
        ("think-on", UNKNOWN_TAG, True),
        ("think-off", UNKNOWN_TAG, False),
    }


# --- which cases held a verdict down ---------------------------------------------


# A run where every case passed outright, to vary one case at a time from.
CLEAN_DEEP = {
    "must-call-freshness": (3, 3),
    "must-call-workspace": (3, 3),
    "must-not-call": (3, 3),
    "selection": (3, 3),
    "hallucination-guard": (3, 3),
    "continuation": (3, 3),
    "loop-guard": (3, 3),
    "thinking-with-tools": (3, 3),
}


def deep(**overrides: tuple[int, int]) -> dict[str, tuple[int, int]]:
    """The clean deep tallies with named cases replaced."""
    return {**CLEAN_DEEP, **{name.replace("_", "-"): counts for name, counts in overrides.items()}}


def test_a_clean_run_blames_nothing() -> None:
    assert clamping_cases("deep", CLEAN_DEEP) == ()
    assert clamping_cases("fast", {"must-call-freshness": (1, 1), "must-not-call": (1, 1)}) == ()


def test_a_deciding_case_blames_only_once_it_loses_the_majority() -> None:
    # 2/3 is a PASS: the deep depth scores the deciding cases by per-case
    # majority, which is exactly why "imperfect" cannot be read as "failed".
    assert clamping_cases("deep", deep(selection=(2, 3))) == ()
    assert clamping_cases("deep", deep(selection=(1, 3))) == ("selection",)


def test_a_guard_case_blames_on_a_single_lost_sample() -> None:
    # Inventing a tool and chasing tools forever are the two actively
    # dangerous behaviors, so one observed instance is enough.
    assert clamping_cases("deep", deep(hallucination_guard=(2, 3))) == ("hallucination-guard",)
    assert clamping_cases("deep", deep(loop_guard=(2, 3))) == ("loop-guard",)


def test_a_fast_run_blames_any_lost_sample() -> None:
    # One sample per case, so there is no majority to take.
    assert clamping_cases("fast", {"must-call-freshness": (0, 1), "must-not-call": (1, 1)}) == (
        "must-call-freshness",
    )


def test_a_case_that_never_ran_blames_too() -> None:
    # A truncated run has demonstrated nothing, so it fails closed.
    assert "selection" in clamping_cases("deep", {})
    assert "must-not-call" in clamping_cases("fast", {})


def test_the_reasoning_case_is_in_neither_scored_set() -> None:
    # Case (h) decides thinking_with_tools alone; losing it all three times
    # must not appear as a reason the AGENCY verdict was held down.
    assert clamping_cases("deep", deep(thinking_with_tools=(0, 3))) == ()


@pytest.mark.parametrize("depth", ["fast", "deep"])
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"selection": (2, 3)},
        {"selection": (1, 3)},
        {"must_not_call": (0, 3)},
        {"hallucination_guard": (2, 3)},
        {"loop_guard": (0, 3)},
        {"thinking_with_tools": (0, 3)},
        {"selection": (1, 3), "loop_guard": (2, 3)},
    ],
)
def test_the_verdict_and_the_blame_never_disagree(depth: str, overrides) -> None:
    # The single-source pin: _earned_verdict is written on top of
    # clamping_cases, so a run that blames nothing must have earned its
    # depth's cap and a run that blames something must not have. A second copy
    # of these thresholds would drift, and the drift would surface as a card
    # naming a case that passed.
    tallies = deep(**overrides)
    blamed = clamping_cases(depth, tallies)
    earned = tool_check._earned_verdict(depth, tallies)
    cap = "assisted" if depth == "fast" else "agentic"
    assert (earned == cap) == (blamed == ())
    # And a blamed GUARD is the only thing that reaches all the way down.
    guards = {"hallucination-guard", "loop-guard"}
    assert (earned == "deterministic") == (
        bool(guards & set(blamed)) or (depth == "fast" and blamed != ())
    )


# --- the version fingerprint ----------------------------------------------------


def test_harness_version_is_stable_and_tracks_case_edits(monkeypatch):
    before = harness_version()
    assert harness_version() == before  # deterministic across calls
    assert before == tool_check.HARNESS_VERSION  # the import-time snapshot agrees
    cases = list(tool_check.CASES)
    cases[0] = dataclasses.replace(cases[0], message=cases[0].message + " (edited)")
    monkeypatch.setattr(tool_check, "CASES", tuple(cases))
    assert harness_version() != before


def test_harness_version_tracks_the_rules_that_live_in_code(monkeypatch):
    # The case specs are DATA, so editing one moves the fingerprint by itself.
    # The normalization rules are CODE, where nothing about rewriting them
    # would move a hash - and a fingerprint that does not move is a promise that
    # two rows measured the same thing. Drop _CODE_RULES out of the payload and
    # this goes red, which is the whole reason it is in there.
    before = harness_version()
    monkeypatch.setitem(tool_check._CODE_RULES, "tool_case_think", "sent as the case declares it")
    assert harness_version() != before


# Fingerprints this harness has RETIRED, newest first. Each names a run whose
# tool cases were driven at a think configuration this harness no longer
# produces, so a verdict earned under one of them describes a deployment that
# no longer exists and must never certify again (app.services.earned requires
# the RUNNING version before a row may certify). Pinned so the step that
# retired each cannot be undone silently.
RETIRED_HARNESS_VERSIONS = (
    # The think chain still had a committed per-model file between the stored
    # declaration and the quirk seed; retiring the file removed that rung, and
    # with it the answer an undescribed tag used to get.
    "15fe26b877fa",
    # Before that, the tool cases normalized their flag against that file ALONE
    # - no stored declaration, no previous measurement, no template inference.
    "8093591126f0",
)


@pytest.mark.parametrize("retired", RETIRED_HARNESS_VERSIONS)
def test_the_think_chain_repairs_stepped_the_harness_version(retired: str):
    assert harness_version() != retired


def test_harness_version_tracks_prompt_stack_edits(monkeypatch):
    # The harness drives the SHIPPED prompt stack, and prompt wording is
    # measured to move verdict-deciding cases (one guidance sentence took
    # continuation 3/6 -> 6/6 on the same weights) - so editing either layer
    # must invalidate old verdicts exactly like a case edit does.
    before = harness_version()
    monkeypatch.setattr(prompts, "_WEB_SEARCH_GUIDANCE", prompts._WEB_SEARCH_GUIDANCE + " (edited)")
    after_guidance = harness_version()
    assert after_guidance != before
    monkeypatch.setattr(
        tool_check, "DEFAULT_SYSTEM_PROMPT", tool_check.DEFAULT_SYSTEM_PROMPT + " (edited)"
    )
    assert harness_version() not in (before, after_guidance)
