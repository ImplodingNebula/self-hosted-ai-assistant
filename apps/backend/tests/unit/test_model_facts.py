"""The runtime-facts cache, the never-None window, and the effective resolvers.

Every load path in the app - the chat turn, the title job, warmup - resolves
its window through :func:`app.services.model_facts.resolve_num_ctx`, and the
model listing leans on :func:`runtime_capabilities`' caching contract, so the
promises those docstrings make are pinned here: every window resolves
runtime-first, a probe failure is remembered only briefly, and the answer is
never None.

The window rule takes the smaller of what the WEIGHTS natively take and what
this DEPLOYMENT affords one turn. The workspace ceiling is the second half of
that affordance rather than a clamp bolted on top, so it moves the window in
BOTH directions - which is the thing most worth pinning, because composing it
as a ``min`` instead would silently pin an operator who raised it at the
deployment default with nothing to point at.

The ``effective_*`` resolvers make one more promise, and it is the whole shape
of this slice: the runtime's report answers, folded with a stored declaration
and the code-shipped quirk seed. Nothing committed to a file describes a model
any more, so a server holding an inventory nobody has ever written down is
described honestly instead of blacked out - and the one POLICY resolver among
them, :func:`effective_agency_ceiling`, still stops a report one rung short of
letting a model drive a turn.
"""

from __future__ import annotations

import pytest

from app.providers.base import ModelCapabilities, ProviderUnavailableError
from app.providers.fake_provider import FakeProvider
from app.services import model_facts
from app.services.model_facts import (
    clamp_num_ctx,
    effective_agency_ceiling,
    effective_num_ctx,
    effective_think_mode,
    effective_tools_mode,
    effective_vision,
    resolve_num_ctx,
    runtime_capabilities,
    runtime_capabilities_many,
)
from app.services.model_policy import PolicyView


class CountingProvider(FakeProvider):
    """A fake whose capability probes are counted and can be made to fail.

    ``fail_first`` makes only the FIRST probe raise, which is the exact shape
    the failure-TTL rule is about: a transient blip is remembered for half a
    minute, never for the five the positive answers get.
    """

    def __init__(self, *, fail_first: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.probes = 0
        self._fail_first = fail_first

    async def capabilities(self, model: str) -> ModelCapabilities:
        self.probes += 1
        if self._fail_first and self.probes == 1:
            raise ProviderUnavailableError("scripted transient failure")
        return await super().capabilities(model)


# --- the window: native, bounded by what the deployment affords ---


async def test_a_native_window_is_clamped_down_to_the_deployment_default() -> None:
    provider = CountingProvider(
        capabilities_by_model={"wild:8b": ModelCapabilities(completion=True, context_length=131072)}
    )
    window = await resolve_num_ctx(provider, "fake", "wild:8b", default_window=4096)
    assert window == 4096


async def test_a_native_window_smaller_than_the_affordance_stands() -> None:
    provider = CountingProvider(
        capabilities_by_model={"tiny:1b": ModelCapabilities(completion=True, context_length=2048)}
    )
    window = await resolve_num_ctx(provider, "fake", "tiny:1b", default_window=8192)
    assert window == 2048


async def test_an_unknown_native_window_falls_to_the_affordance() -> None:
    # FakeProvider's fallback report carries no context_length at all.
    provider = CountingProvider()
    window = await resolve_num_ctx(provider, "fake", "wild:8b", default_window=4096)
    assert window == 4096


async def test_every_tag_is_sized_from_the_runtime_rather_than_from_a_file() -> None:
    # The re-homing this slice is about. Nothing written down can size a tag
    # ahead of the runtime any more, so the number a turn requests is always
    # one somebody checked against the weights in front of it - and the probe
    # is what produces it, on every tag, not only on the undescribed ones.
    provider = CountingProvider(
        capabilities_by_model={"sized:1b": ModelCapabilities(completion=True, context_length=4096)}
    )
    window = await resolve_num_ctx(provider, "fake", "sized:1b", default_window=8192)
    assert window == 4096
    assert provider.probes == 1


async def test_raising_the_deployment_default_raises_a_tags_window() -> None:
    # The reason the number is a setting rather than a constant: an operator
    # with real memory raises it and windows follow, up to what the weights
    # natively take.
    provider = CountingProvider(
        capabilities_by_model={"wild:8b": ModelCapabilities(completion=True, context_length=131072)}
    )
    window = await resolve_num_ctx(provider, "fake", "wild:8b", default_window=32768)
    assert window == 32768


async def test_a_workspace_ceiling_raises_a_tags_window() -> None:
    # The CRITICAL asymmetry. The ceiling REPLACES the deployment default
    # rather than clamping it, so a console ceiling of 16384 yields 16384 even
    # though the deployment affords 8192 by default.
    #
    # This is also the proof that the two ceiling reads compose to a no-op
    # rather than a contradiction: the resolved window still passes through
    # `clamp_num_ctx` with the same ceiling, and a window resolved AGAINST the
    # ceiling can never exceed it. Composing the ceiling as a clamp only -
    # min(native, default, ceiling) - would pin this at 8192 instead.
    provider = CountingProvider(
        capabilities_by_model={"wild:8b": ModelCapabilities(completion=True, context_length=262144)}
    )
    window = await resolve_num_ctx(provider, "fake", "wild:8b", default_window=8192, ceiling=16384)
    assert window == 16384


async def test_a_workspace_ceiling_still_lowers_a_tags_window() -> None:
    # Replacement cuts both ways: an operator asking for less than the
    # deployment default gets less.
    provider = CountingProvider(
        capabilities_by_model={"wild:8b": ModelCapabilities(completion=True, context_length=131072)}
    )
    window = await resolve_num_ctx(provider, "fake", "wild:8b", default_window=32768, ceiling=4096)
    assert window == 4096


async def test_a_ceiling_cannot_ask_for_more_window_than_the_weights_have() -> None:
    # The bound that survives the upward replacement. The ceiling replaces the
    # DEPLOYMENT's affordance, never the model's native limit - asking a
    # runtime for a window the weights do not have is how a load fails.
    provider = CountingProvider(
        capabilities_by_model={"tiny:1b": ModelCapabilities(completion=True, context_length=8192)}
    )
    window = await resolve_num_ctx(
        provider, "fake", "tiny:1b", default_window=4096, ceiling=1_048_576
    )
    assert window == 8192


def test_the_clamp_only_ever_narrows() -> None:
    # Stated on its own because the resolution above makes it a no-op by
    # construction, and it is kept anyway so the rule reads the same wherever a
    # window is produced. A ceiling that could RAISE would hand a turn more KV
    # cache than the weights admit, and the failure names nothing.
    assert clamp_num_ctx(8192, 4096) == 4096
    assert clamp_num_ctx(4096, 8192) == 4096
    assert clamp_num_ctx(4096, None) == 4096


async def test_the_two_window_entry_points_agree_about_a_raised_ceiling() -> None:
    # `effective_num_ctx` is what a management card shows and `resolve_num_ctx`
    # is what a turn requests; they share the rule so the number on the card is
    # the number that runs. A ceiling that raises one must raise both.
    #
    # The two are fed the SAME ceiling here, so this pins the shared rule and
    # not the wiring: the manager row derives its ceiling through
    # `model_manager._row_ceiling` first, and that a card still names the
    # turn's window is pinned where that wiring lives
    # (test_model_manager.test_a_card_names_the_window_a_turn_on_it_would_request).
    report = ModelCapabilities(completion=True, context_length=262144)
    provider = CountingProvider(capabilities_by_model={"wild:8b": report})
    shown = effective_num_ctx(report, default_window=8192, ceiling=16384)
    requested = await resolve_num_ctx(
        provider, "fake", "wild:8b", default_window=8192, ceiling=16384
    )
    assert shown == requested == 16384


async def test_the_window_is_never_none_even_when_the_probe_fails() -> None:
    provider = CountingProvider(fail_first=True)
    window = await resolve_num_ctx(provider, "fake", "wild:8b", default_window=4096)
    assert window == 4096


async def test_a_zero_window_report_is_ignored() -> None:
    provider = CountingProvider(
        capabilities_by_model={"weird:1b": ModelCapabilities(completion=True, context_length=0)}
    )
    window = await resolve_num_ctx(provider, "fake", "weird:1b", default_window=4096)
    assert window == 4096


# --- the cache ---


async def test_a_report_is_cached_inside_the_ttl() -> None:
    provider = CountingProvider()
    first = await runtime_capabilities(provider, "fake", "wild:8b")
    second = await runtime_capabilities(provider, "fake", "wild:8b")
    assert first is not None and second is not None
    assert provider.probes == 1


async def test_a_failed_probe_is_remembered_briefly_then_retried(clock: FrozenClock) -> None:
    provider = CountingProvider(fail_first=True)
    assert await runtime_capabilities(provider, "fake", "wild:8b") is None
    # Held, not re-probed. "The runtime would not say" is an answer a TURN now
    # depends on, so a tag on a half-answering server would otherwise pay a
    # round trip before every single generation.
    assert await runtime_capabilities(provider, "fake", "wild:8b") is None
    assert provider.probes == 1
    # ...and held only briefly, so a runtime that came back is seen within half
    # a minute rather than at the end of the five-minute success TTL.
    assert model_facts._FAILURE_TTL_SECONDS < model_facts._TTL_SECONDS
    clock.now = model_facts._FAILURE_TTL_SECONDS
    report = await runtime_capabilities(provider, "fake", "wild:8b")
    assert report is not None and report.completion
    assert provider.probes == 2


async def test_a_recovered_probe_replaces_the_remembered_failure(clock: FrozenClock) -> None:
    # One entry, not two tables: the success has to overwrite the negative, or
    # the next reader could still be served the older of the two.
    provider = CountingProvider(fail_first=True)
    assert await runtime_capabilities(provider, "fake", "wild:8b") is None
    clock.now = model_facts._FAILURE_TTL_SECONDS
    assert await runtime_capabilities(provider, "fake", "wild:8b") is not None
    clock.now += 1.0
    assert await runtime_capabilities(provider, "fake", "wild:8b") is not None
    assert provider.probes == 2  # the success is now cached under the success TTL


async def test_the_shared_fetch_rule_skips_a_blank_tag_only() -> None:
    # A deployment that has chosen no chat model resolves to the empty tag, and
    # there is nothing to ask a runtime about it. Everything else goes to the
    # seam - the old "skip anything a committed file described" shortcut is
    # gone with the file, and it is exactly what left the harness driving an
    # undescribed tag at a window and a think flag no turn uses.
    provider = CountingProvider()
    assert await model_facts.report_for(provider, "fake", "") is None
    assert provider.probes == 0
    assert await model_facts.report_for(provider, "fake", "stray:7b") is not None
    assert provider.probes == 1


class FrozenClock:
    """A monotonic clock the test moves by hand.

    Held rather than exhausted (an iterator of readings raises during teardown,
    where the cache fixtures read the clock again), and time only ever moves
    when a test says so - which is the point: these are cache-AGE tests, and a
    real clock would make them race the suite's own runtime.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> FrozenClock:
    frozen = FrozenClock()
    monkeypatch.setattr(model_facts.time, "monotonic", frozen)
    return frozen


async def test_a_stricter_caller_reprobes_past_its_own_limit(clock: FrozenClock) -> None:
    # One cache, per-caller freshness. The vision gate accepts a much fresher
    # answer than a card does, because the failure directions are opposite: a
    # stale capability read is harmless on a card and keeps sending pixels here.
    provider = CountingProvider()
    assert await runtime_capabilities(provider, "fake", "wild:8b") is not None
    clock.now = 120.0
    # Two minutes on, the shared TTL is still happy...
    assert await runtime_capabilities(provider, "fake", "wild:8b") is not None
    assert provider.probes == 1
    # ...but a caller that will only accept a minute is not, and re-probes.
    assert await runtime_capabilities(provider, "fake", "wild:8b", max_age_seconds=60.0) is not None
    assert provider.probes == 2


async def test_a_stricter_caller_leaves_the_fresher_answer_behind(clock: FrozenClock) -> None:
    # The re-probe refreshes the ONE entry rather than filling a second cache,
    # so the laxer reader that follows gets the newer answer for free.
    provider = CountingProvider()
    await runtime_capabilities(provider, "fake", "wild:8b")
    clock.now = 120.0
    await runtime_capabilities(provider, "fake", "wild:8b", max_age_seconds=60.0)
    assert provider.probes == 2
    clock.now = 121.0
    await runtime_capabilities(provider, "fake", "wild:8b")
    assert provider.probes == 2


async def test_a_caller_cannot_ask_for_a_staler_answer_than_the_ttl(clock: FrozenClock) -> None:
    # max_age only ever tightens: the shared TTL is the ceiling, so no caller
    # can extend the window a re-pull has to survive.
    provider = CountingProvider()
    await runtime_capabilities(provider, "fake", "wild:8b")
    clock.now = 400.0
    await runtime_capabilities(provider, "fake", "wild:8b", max_age_seconds=9_999.0)
    assert provider.probes == 2


async def test_a_bounded_fan_out_answers_one_report_per_tag_in_order() -> None:
    provider = CountingProvider(fail_first=True)
    reports = await runtime_capabilities_many(provider, "fake", ["a:1b", "b:1b", "c:1b"])
    # One slot per input, positionally, so a caller can zip and see WHICH tag
    # went unanswered - a shorter list would silently misalign them.
    assert len(reports) == 3
    assert reports[0] is None
    assert all(report is not None for report in reports[1:])


# --- the effective resolvers: the report, folded with a declaration and the seed ---

# A runtime that says yes to everything, with a template wired for reasoning.
GENEROUS = ModelCapabilities(
    completion=True, tools=True, vision=True, thinking=True, template_thinking=True
)
# One that says no to everything, with a template that has no think plumbing.
STINGY = ModelCapabilities(completion=True, template_thinking=False)

# A tag the quirk seed says nothing about, so the report answers alone.
STRAY = "stray:7b"


def test_the_mechanical_resolvers_read_the_report_and_nothing_else() -> None:
    # Whether a template takes a tool list and whether weights accept pixels
    # are facts only the serving stack can see - it is the party that renders
    # the template and loads the weights. There is nothing here for a
    # written-down claim to add, and a blackout over every tag nobody had
    # described was the wrong answer for the deployments this is pointed at.
    assert effective_tools_mode(GENEROUS) == "native"
    assert effective_vision(STRAY, GENEROUS) is True
    assert effective_tools_mode(STINGY) == "none"
    assert effective_vision(STRAY, STINGY) is False


def test_a_failed_probe_keeps_every_answer_at_the_safe_floor() -> None:
    # "The runtime would not say" is not evidence of capability. Every one of
    # these is the fail-closed answer, and none of them is a measurement.
    assert effective_tools_mode(None) == "none"
    assert effective_vision(STRAY, None) is False
    assert effective_think_mode(STRAY, None) == "unknown"
    assert effective_agency_ceiling(None) == "deterministic"


def test_a_blank_tag_is_never_shown_images() -> None:
    # The first-run state: nothing has chosen a chat model, so there is no
    # model to send pixels to however generous a report happens to be in hand.
    assert effective_vision("", GENEROUS) is False


def test_a_template_that_was_not_reported_leaves_the_mode_unknown() -> None:
    # Distinct from "never": one is a runtime that exposes no template, the
    # other is a model whose template cannot reason.
    quiet = ModelCapabilities(completion=True)
    assert quiet.template_thinking is None
    assert effective_think_mode(STRAY, quiet) == "unknown"
    assert effective_think_mode(STRAY, STINGY) == "never"
    assert effective_think_mode(STRAY, GENEROUS) == "hybrid"


# --- the agency ceiling: the one POLICY question among them ---


def test_a_vouched_template_reaches_assisted_and_no_further() -> None:
    # The whole point: a server holding an inventory nobody described stops
    # handing out a blanket deterministic. And the ceiling stops at assisted -
    # "this template accepts a tool list" is a mechanical fact about a
    # template, and letting a model DRIVE a turn is a decision about weights
    # that only a human makes.
    assert effective_agency_ceiling(GENEROUS) == "assisted"


def test_an_unvouched_template_keeps_the_safe_floor() -> None:
    # No tools reported, and a probe that could not be taken, are the same
    # answer for the same reason: nothing said this model can take a tool list.
    assert effective_agency_ceiling(STINGY) == "deterministic"
    assert effective_agency_ceiling(None) == "deterministic"


def test_a_stored_ceiling_overrides_the_report_upward() -> None:
    # An operator arming a model raises the CEILING only. What a turn actually
    # gets is still min(CEILING, EARNED, ...) - this function is one input to
    # that, never the decision. It is also the ONE input that can grant
    # agentic, which is exactly what the report cap was reserving that rung for.
    armed = PolicyView(ceiling="agentic")
    assert effective_agency_ceiling(GENEROUS, policy=armed) == "agentic"
    # ...and it does not need the report's blessing to do it: a declaration is
    # a decision about weights, so it answers even where nothing was reported.
    assert effective_agency_ceiling(None, policy=armed) == "agentic"


def test_a_stored_ceiling_overrides_the_report_downward() -> None:
    # The other direction, and the one that has to actuate immediately: an
    # operator clamping a model down is not asking for it to take effect at
    # some later moment.
    clamped = PolicyView(ceiling="deterministic")
    assert effective_agency_ceiling(GENEROUS, policy=clamped) == "deterministic"


def test_a_policy_that_declares_no_ceiling_changes_nothing() -> None:
    # Each axis is independent: a stored window says nothing about agency.
    windowed = PolicyView(num_ctx=16384)
    assert effective_agency_ceiling(GENEROUS, policy=windowed) == "assisted"


def test_a_pair_ceiling_can_only_clamp_a_sibling_down() -> None:
    # Declarations are stored per tag and a pair is two tags, so an operator
    # who clamps a model from the console has otherwise clamped only the half
    # that answers directly - every thinking turn routes to the sibling and
    # keeps whatever the sibling's own report allows, which reads as the clamp
    # being ignored.
    assert effective_agency_ceiling(GENEROUS, pair_ceiling="deterministic") == "deterministic"
    # One-directional: a permissive default must not lift a sibling that is
    # clamped in its own right.
    clamped = PolicyView(ceiling="deterministic")
    assert (
        effective_agency_ceiling(GENEROUS, policy=clamped, pair_ceiling="agentic")
        == "deterministic"
    )


# --- the think mode: measured always, then a declaration, then the seed ---


def test_a_stored_think_mode_outranks_the_seed_and_the_template() -> None:
    # A stored confirmation is a check that drove THESE weights and a human who
    # then agreed to it; the seed is a family fact and the template is a prior.
    confirmed = PolicyView(think_mode="hybrid")
    assert effective_think_mode("qwen3:4b", STINGY) == "always"
    assert effective_think_mode("qwen3:4b", STINGY, policy=confirmed) == "hybrid"


def test_a_measured_always_outranks_every_declaration() -> None:
    # The pre-confirm routing clamp: a check WATCHED these weights reason with
    # the flag cleared, so the flag goes back on and the reasoning lands on the
    # collapsed channel instead of arriving as raw markup in the answer. It
    # outranks the stored confirmation too, which is the only source it could
    # meet - both describe the same installed weights, and only one of them
    # measured what actually came back.
    assert effective_think_mode(STRAY, STINGY) == "never"
    assert effective_think_mode(STRAY, STINGY, measured="always") == "always"
    confirmed = PolicyView(think_mode="hybrid")
    assert effective_think_mode(STRAY, STINGY, policy=confirmed, measured="always") == "always"


@pytest.mark.parametrize("measured", ["hybrid", "never"])
def test_no_other_measured_direction_moves_the_routing(measured: str) -> None:
    # The safe-direction asymmetry, and the mutation guard for it. Measured
    # ALWAYS repairs a leak and can only ask a model to reason that was going
    # to anyway; every other direction REMOVES reasoning, or offers a flag to
    # weights that reject it, on a measurement nobody has confirmed. Those wait
    # for the confirm endpoint. Widen the clamp to `measured is not None` and
    # this goes red.
    assert effective_think_mode("qwen3:4b", None, measured=measured) == "always"
    confirmed = PolicyView(think_mode="never")
    assert effective_think_mode("qwen3:4b", None, policy=confirmed, measured=measured) == "never"


def test_the_seed_answers_above_the_template() -> None:
    # The template report says "wired for reasoning", which for the 2507
    # instruct checkpoint is exactly the wrong prior - it has no IsThinkSet
    # wiring at all and never reasons. The seed knows, and it outranks.
    assert effective_think_mode("qwen3:4b-instruct", GENEROUS) == "never"
    assert effective_think_mode("qwen3:4b", STINGY) == "always"


def test_a_tag_the_seed_omits_falls_through_to_the_template() -> None:
    # The protection behind leaving an unmeasured family OUT: with no entry,
    # the runtime's own report answers. A wrong seeded "never" here would omit
    # the think flag on a family whose template does carry think plumbing, and
    # an omitted flag is what a serving stack promotes to think:true on a model
    # advertising the capability - the dangerous triple, reached from a guess.
    assert effective_think_mode("gemma3:12b", GENEROUS) == "hybrid"
    assert effective_think_mode("deepseek-r1:7b", STINGY) == "never"
