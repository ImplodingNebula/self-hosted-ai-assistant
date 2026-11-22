"""Unit tests for plan_turn: the consolidated pre-generation decision.

The chat-level tests (test_chat_agency.py) already pin the end-to-end
behaviour through the app; these drive the planner directly with fakes for
its injected I/O boundaries, pinning the properties the consolidation
must not lose:

- a turn that cannot carry tools never pays the earned-verdict I/O;
- an explicit model pick is never routed;
- substitution re-resolves the ROUTED model's claims and can drop the tools;
- a reasoning turn on a model that cannot combine thinking and tools loses
  the tools, not the reasoning;
- the final drive flags derive from the tool set that actually survived.

Plus what the runtime report decides: a tag nothing describes is resolved from
what the serving stack says about it - capped at ``assisted``, never
``agentic``, which only a stored operator declaration can grant.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.providers.base import ModelCapabilities
from app.schemas.chat import ChatRequest
from app.schemas.settings import TurnDefaults
from app.services.earned import EffectiveVerdicts
from app.services.model_policy import PolicyView
from app.services.model_resolution import AgencyMode, resolve_chat_models
from app.services.turn_plan import plan_turn

# A runtime that vouches for a tool-taking template wired for reasoning...
TOOLED = ModelCapabilities(completion=True, tools=True, template_thinking=True)
# ...one that vouches for the tools but not the reasoning...
TOOLED_NO_THINK = ModelCapabilities(completion=True, tools=True, template_thinking=False)
# ...and one that says the template takes no tool list at all.
TOOLLESS = ModelCapabilities(completion=True, template_thinking=True)

# The deployment these tests plan turns for. Nothing writes model facts down
# any more, so each tag's meaning comes from exactly two places, and the names
# are chosen so both are legible:
#
# * how it treats the think flag comes from the code-shipped quirk seed
#   (app.services.known_models), which is why these are real qwen3 tags -
#   qwen3:8b is a hybrid, qwen3:4b and qwen3-vl:2b always reason;
# * everything else (tools, ceiling, reason-and-call) comes from the injected
#   report / policy / verdict fakes below, per test.
DEFAULT_MODEL = "qwen3:8b"
THINKING_MODEL = "qwen3:4b"
# A second always-thinking tag a router can substitute to.
OTHER_REASONER = "qwen3-vl:2b"
# A tag whose template takes no tool list at all (see BASE_REPORTS).
TOOLLESS_MODEL = "toolless:1b"
# Tags nothing describes: no seed entry, and no report or policy unless a test
# hands one over.
STRAY = "stray:7b"
GHOST = "ghost:1b"

# What the runtime says about the configured inventory. Every tag a turn can
# land on needs an entry, because a report that could not be taken floors the
# tag to `deterministic` - which used to be what an absent catalog entry meant
# and is now what an absent REPORT means.
BASE_REPORTS: dict[str, ModelCapabilities] = {
    DEFAULT_MODEL: TOOLED,
    THINKING_MODEL: TOOLED,
    OTHER_REASONER: TOOLED,
    TOOLLESS_MODEL: TOOLLESS,
}

# What an operator declared about them. A report lifts a tag as far as
# `assisted` and no further, so `agentic` - the mode most of these tests are
# about - exists only because these declarations do.
BASE_POLICIES: dict[str, PolicyView] = {
    DEFAULT_MODEL: PolicyView(ceiling="agentic"),
    THINKING_MODEL: PolicyView(ceiling="agentic"),
    OTHER_REASONER: PolicyView(ceiling="agentic"),
    TOOLLESS_MODEL: PolicyView(ceiling="agentic"),
}


def make_settings(**overrides) -> Settings:
    """Settings for a deployment configured with the pair above.

    Args:
        **overrides: Fields to set on top of the pair configuration.

    Returns:
        Settings built without reading the developer's own ``.env``.
    """
    return Settings(
        _env_file=None,
        model_provider="fake",
        default_chat_model=DEFAULT_MODEL,
        thinking_chat_model=THINKING_MODEL,
        **overrides,
    )


class VerdictFake:
    """Records every lookup; answers from a per-tag map with a full-trust default.

    The default mirrors the real lookup's grace regime: with no probe evidence
    the earned axis simply IS the ceiling the caller resolved, so a test whose
    turn comes out ``assisted`` is showing the ceiling doing the work rather
    than a fake that happened to say so.
    """

    def __init__(self, by_tag: dict[str, EffectiveVerdicts] | None = None):
        self.calls: list[str] = []
        self.ceilings: list[AgencyMode] = []
        self.by_tag = by_tag or {}

    async def __call__(self, tag: str, ceiling: AgencyMode) -> EffectiveVerdicts:
        self.calls.append(tag)
        self.ceilings.append(ceiling)
        return self.by_tag.get(tag, EffectiveVerdicts(earned=ceiling, thinking_with_tools=True))


class ReportFake:
    """Records every capability probe; answers from a per-tag map.

    The default is None - "the runtime would not say" - so a test that wants a
    report has to hand one over, and the safe floor is what an unconfigured tag
    lands on.
    """

    def __init__(self, by_tag: dict[str, ModelCapabilities] | None = None):
        self.calls: list[str] = []
        self.by_tag = by_tag or {}

    async def __call__(self, tag: str) -> ModelCapabilities | None:
        self.calls.append(tag)
        return self.by_tag.get(tag)


def reports_for(extra: dict[str, ModelCapabilities] | None = None) -> ReportFake:
    """The configured inventory's reports, with a test's own entries on top.

    Args:
        extra: Per-tag reports this test wants, overriding the base map.

    Returns:
        A fresh :class:`ReportFake` over the merged map.
    """
    return ReportFake({**BASE_REPORTS, **(extra or {})})


class RouteFake:
    """Records every routing call; substitutes when told to."""

    def __init__(self, substitute: str | None = None, note: str | None = None):
        self.calls: list[tuple[str, bool]] = []
        self.substitute = substitute
        self.note = note

    async def __call__(self, model: str, think: bool) -> tuple[str, str | None]:
        self.calls.append((model, think))
        if self.substitute is not None:
            return self.substitute, self.note
        return model, None


class CtxFake:
    """Records which tag the window was resolved for; answers a fixed size.

    Stands in for the real resolution (the smaller of the runtime's native
    window and what the deployment affords), which the planner deliberately
    does not do itself.
    """

    def __init__(self, num_ctx: int = 8192):
        self.calls: list[str] = []
        self.num_ctx = num_ctx

    async def __call__(self, model: str) -> int:
        self.calls.append(model)
        return self.num_ctx


class PolicyFake:
    """Records every declaration lookup; answers from a per-tag map.

    The default is None - nothing declared about that tag - which is what an
    undeclared model looks like, and leaves its ceiling to the report alone.
    """

    def __init__(self, by_tag: dict[str, PolicyView] | None = None):
        self.calls: list[str] = []
        self.by_tag = by_tag or {}

    async def __call__(self, tag: str) -> PolicyView | None:
        self.calls.append(tag)
        return self.by_tag.get(tag)


def policies_for(extra: dict[str, PolicyView] | None = None) -> PolicyFake:
    """The configured inventory's declarations, with a test's own on top.

    Args:
        extra: Per-tag declarations this test wants, replacing the base entry
            for that tag outright (a declaration is one row, not a merge).

    Returns:
        A fresh :class:`PolicyFake` over the merged map.
    """
    return PolicyFake({**BASE_POLICIES, **(extra or {})})


class MeasuredThinkFake:
    """Records every measured-think lookup; answers from a per-tag map.

    The default is None - nothing measured - so every test that is not about
    the routing clamp resolves the think flag exactly as it did before the
    measurement existed.
    """

    def __init__(self, by_tag: dict[str, str] | None = None):
        self.calls: list[str] = []
        self.by_tag = by_tag or {}

    async def __call__(self, tag: str):
        self.calls.append(tag)
        return self.by_tag.get(tag)


async def run_plan(
    req: ChatRequest,
    *,
    settings: Settings | None = None,
    verdicts: VerdictFake | None = None,
    route: RouteFake | None = None,
    resolve_ctx: CtxFake | None = None,
    reports: ReportFake | None = None,
    policies: PolicyFake | None = None,
    measured: MeasuredThinkFake | None = None,
    search_wired: bool = True,
    rag_wired: bool = True,
    defaults: TurnDefaults | None = None,
):
    """Plan one turn against the fakes, defaulting to the configured inventory.

    Args:
        req: The chat request to plan.
        settings: The deployment's settings; defaults to the pair above.
        verdicts: The earned-verdict fetcher.
        route: The routing fake.
        resolve_ctx: The window resolver.
        reports: The runtime-report fetcher; defaults to the base inventory,
            so a test only has to state the tags it cares about.
        policies: The declaration fetcher; defaults to the base declarations.
        measured: The unconfirmed-measurement fetcher.
        search_wired: Whether a search provider is wired this turn.
        rag_wired: Whether the RAG collaborators are wired this turn.
        defaults: The merged stored turn defaults.

    Returns:
        The resolved :class:`~app.services.turn_plan.TurnPlan`.
    """
    settings = settings or make_settings()
    return await plan_turn(
        req=req,
        history=[],
        settings=settings,
        chat_pair=resolve_chat_models(settings),
        search_wired=search_wired,
        rag_wired=rag_wired,
        defaults=defaults,
        fetch_verdicts=verdicts or VerdictFake(),
        route=route or RouteFake(),
        resolve_ctx=resolve_ctx or CtxFake(),
        fetch_report=reports if reports is not None else reports_for(),
        fetch_policy=policies if policies is not None else policies_for(),
        fetch_measured_think=measured or MeasuredThinkFake(),
    )


@pytest.mark.anyio
async def test_deterministic_clamp_stays_io_free():
    """An env-clamped deterministic turn never pays the earned-verdict lookup."""
    verdicts = VerdictFake()
    plan = await run_plan(
        ChatRequest(message="hello"),
        settings=make_settings(agency_mode="deterministic"),
        verdicts=verdicts,
    )
    assert verdicts.calls == []
    assert plan.offer.mode == "deterministic"
    assert plan.offer_tools is False


@pytest.mark.anyio
async def test_no_offerable_tool_stays_io_free():
    """With every tool gated off there is nothing to earn, so no lookup runs."""
    verdicts = VerdictFake()
    plan = await run_plan(
        ChatRequest(message="hello", web_search="off"),
        verdicts=verdicts,
        search_wired=False,
        rag_wired=False,
    )
    assert verdicts.calls == []
    assert plan.offer.mode == "deterministic"


@pytest.mark.anyio
async def test_offering_turn_fetches_the_candidate_once():
    """A tool-carrying turn resolves the earned axis exactly once, for the candidate."""
    verdicts = VerdictFake()
    plan = await run_plan(ChatRequest(message="hello"), verdicts=verdicts)
    assert verdicts.calls == [DEFAULT_MODEL]
    assert plan.offer.mode == "agentic"
    assert plan.offer_tools is True
    assert plan.model == DEFAULT_MODEL
    assert plan.num_ctx == 8192


@pytest.mark.anyio
async def test_the_window_is_resolved_for_the_routed_model():
    """The window follows the model that RUNS, not the one the config named.

    A substitution can land the turn on an uncatalogued tag whose window is a
    runtime fact, so resolving before routing would size the wrong weights -
    and every budget on the turn is measured against this number.
    """
    ctx = CtxFake(num_ctx=4096)
    plan = await run_plan(
        ChatRequest(message="hello"),
        route=RouteFake(substitute="llama3.1:8b", note="substituted"),
        resolve_ctx=ctx,
    )
    assert ctx.calls == ["llama3.1:8b"]
    assert plan.num_ctx == 4096


@pytest.mark.anyio
async def test_explicit_model_is_never_routed():
    """An explicit req.model is honored verbatim - the router is not consulted."""
    route = RouteFake(substitute=THINKING_MODEL)
    plan = await run_plan(ChatRequest(message="hello", model=DEFAULT_MODEL), route=route)
    assert route.calls == []
    assert plan.model == DEFAULT_MODEL
    assert plan.status_note is None


@pytest.mark.anyio
async def test_substitution_reresolves_the_routed_model():
    """Routing to different weights re-resolves THEIR claims; an unvouched tag drops tools."""
    verdicts = VerdictFake()
    route = RouteFake(substitute=GHOST, note="substituted")
    plan = await run_plan(ChatRequest(message="hello"), verdicts=verdicts, route=route)
    # Both lookups happened: the candidate's for the offer, the routed model's
    # for the floor - and the uncatalogued tag, which no report vouches for
    # either, keeps the deterministic floor.
    assert verdicts.calls == [DEFAULT_MODEL, GHOST]
    assert plan.model == GHOST
    assert plan.status_note == "substituted"
    assert plan.offer_tools is False


@pytest.mark.anyio
async def test_substitution_to_a_toolless_template_drops_tools():
    """The routed model's own template floor drops tools even at full trust.

    chat-d is catalogued and agentic with fully trusting (faked) verdicts, so
    nothing but the recomputed tools-none runtime floor can force the drop -
    this is the case that dies if the routed-floors recomputation is lost.
    """
    verdicts = VerdictFake()
    route = RouteFake(substitute=TOOLLESS_MODEL)
    plan = await run_plan(ChatRequest(message="hello"), verdicts=verdicts, route=route)
    assert verdicts.calls == [DEFAULT_MODEL, TOOLLESS_MODEL]
    assert plan.offer_tools is False
    # The offer is REWRITTEN, not merely stripped of its tools: leaving
    # mode="agentic" standing on a turn that carries none would make the plan -
    # the one value the trace and the logs read - describe a turn nobody ran.
    assert (plan.offer.mode, plan.offer.think, plan.offer.agentic) == (
        "deterministic",
        False,
        False,
    )


@pytest.mark.anyio
async def test_reasoner_that_cannot_combine_loses_tools_not_reasoning():
    """A thinking turn landing on a no-combine model keeps reasoning, drops tools."""
    verdicts = VerdictFake(
        by_tag={OTHER_REASONER: EffectiveVerdicts(earned="agentic", thinking_with_tools=False)}
    )
    route = RouteFake(substitute=OTHER_REASONER)
    plan = await run_plan(
        ChatRequest(message="hello", thinking="on"), verdicts=verdicts, route=route
    )
    # The offer was resolved for chat-b (the pair's thinking member), the turn
    # ran on chat-c: catalogued and agentic, so only the combine verdict drops it.
    assert verdicts.calls == [THINKING_MODEL, OTHER_REASONER]
    assert plan.think_flag is True
    assert plan.offer_tools is False


@pytest.mark.anyio
async def test_outcomes_derive_from_the_surviving_tool_set():
    """Agentic needs a surviving tool; the pre-fetch and the notice follow from it."""
    plan = await run_plan(ChatRequest(message="hello", web_search="on"))
    assert plan.resolved_web is True

    with_tools = plan.outcomes(offered_any=True)
    assert with_tools.agentic is True
    assert with_tools.do_search is False
    assert with_tools.announce_tools_not_used is True

    without_tools = plan.outcomes(offered_any=False)
    assert without_tools.agentic is False
    assert without_tools.do_search is True
    assert without_tools.announce_tools_not_used is False


@pytest.mark.anyio
async def test_explicit_request_beats_a_conversation_default():
    """A request switch set to on/off wins over any stored default."""
    plan = await run_plan(
        ChatRequest(message="hello", web_search="on", thinking="off"),
        defaults=TurnDefaults(thinking="on", web_search="off"),
    )
    assert plan.effective_web == "on"
    assert plan.effective_thinking == "off"
    assert plan.resolved_web is True


@pytest.mark.anyio
async def test_stored_default_applies_when_the_request_defers():
    """An auto request adopts the merged stored default."""
    plan = await run_plan(
        ChatRequest(message="hello"),
        defaults=TurnDefaults(thinking="on", web_search="on"),
    )
    assert plan.effective_thinking == "on"
    assert plan.effective_web == "on"
    assert plan.resolved_web is True


@pytest.mark.anyio
async def test_conversation_web_off_withholds_the_tool_too():
    """The locked floor holds for the EFFECTIVE mode, not just the request.

    A conversation-level off with a bare auto request must suppress the
    deterministic pre-fetch AND the web tool - this is the test that dies if
    the tool-offer gate keeps reading the raw request switch.
    """
    plan = await run_plan(
        ChatRequest(message="hello"),
        defaults=TurnDefaults(web_search="off"),
    )
    assert plan.effective_web == "off"
    assert plan.resolved_web is False
    assert plan.offer.offer_web is False


@pytest.mark.anyio
async def test_no_defaults_is_byte_identical():
    """The regression pin: empty stored tiers produce the pre-chain plan."""
    bare = await run_plan(ChatRequest(message="hello"))
    with_empty = await run_plan(ChatRequest(message="hello"), defaults=TurnDefaults())
    assert bare == with_empty


# --- the runtime report: tags nothing describes ----------------------------------


@pytest.mark.anyio
async def test_a_vouched_undeclared_tag_reaches_assisted_and_stops():
    """The runtime clears an unknown tag for tools - and agentic stays out of reach.

    The bug this closes: against a server holding an inventory nobody had
    described, every tag clamped to deterministic and no turn was ever handed a
    tool. The ceiling it reaches is the no-regression mode, not the full one,
    and AGENCY_MODE=agentic must not be able to lift it: driving a turn is a
    human declaration, and a report about a chat template is not one.
    """
    verdicts = VerdictFake()
    reports = reports_for({STRAY: TOOLED})
    plan = await run_plan(
        ChatRequest(message="hello", web_search="on", model=STRAY),
        settings=make_settings(agency_mode="agentic"),
        verdicts=verdicts,
        reports=reports,
    )
    assert reports.calls == [STRAY]
    assert verdicts.ceilings == ["assisted"]
    assert plan.offer.mode == "assisted"
    assert plan.offer_tools is True
    assert plan.offer.agentic is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("report", "why"),
    [(None, "the probe failed"), (TOOLLESS, "the runtime says the template takes no tools")],
    ids=["no-report", "toolless-report"],
)
async def test_an_unvouched_undeclared_tag_keeps_the_safe_floor(report, why):
    """No declaration and no vouching report leaves the blackout floor in place."""
    reports = reports_for({STRAY: report} if report is not None else {})
    plan = await run_plan(
        ChatRequest(message="hello", model=STRAY),
        settings=make_settings(agency_mode="agentic"),
        reports=reports,
    )
    assert plan.offer.mode == "deterministic", why
    assert plan.offer_tools is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("report", "expected"),
    [
        # A template with no think plumbing: omit the flag rather than send one
        # the serving stack rejects with a 400 (and a retry) on every turn.
        (TOOLED_NO_THINK, None),
        # Wired for reasoning: the decision passes through explicitly.
        (TOOLED, True),
    ],
    ids=["never-thinking-template", "hybrid-template"],
)
async def test_the_think_flag_follows_an_undescribed_tags_template(report, expected):
    plan = await run_plan(
        ChatRequest(message="hello", thinking="on", model=STRAY),
        reports=reports_for({STRAY: report}),
    )
    assert plan.offer.think is True
    assert plan.think_flag is expected


@pytest.mark.anyio
async def test_an_unreported_template_passes_the_decision_through():
    """No report means unknown, not never: send the flag and let the retry handle it."""
    plan = await run_plan(
        ChatRequest(message="hello", thinking="on", model=STRAY),
        reports=ReportFake(),
    )
    assert plan.think_flag is plan.offer.think


@pytest.mark.anyio
async def test_the_think_flag_is_resolved_even_when_the_turn_carries_no_tools():
    """Think correctness must not depend on the agency clamp.

    A deterministic deployment still generates, so it still has to send (or
    omit) the right flag. If the report were fetched only on the tool path, a
    clamped deployment would normalize off a report it declined to read.
    """
    reports = reports_for({STRAY: TOOLED_NO_THINK})
    plan = await run_plan(
        ChatRequest(message="hello", thinking="on", model=STRAY),
        settings=make_settings(agency_mode="deterministic"),
        reports=reports,
    )
    assert reports.calls == [STRAY]
    assert plan.offer.mode == "deterministic"
    assert plan.think_flag is None


@pytest.mark.anyio
async def test_an_undeclared_reasoner_loses_the_tools_not_the_reasoning():
    """No declaration means no thinking-with-tools claim, so a thinking turn drops tools.

    The optimistic default is the dangerous one here: reasoning plus tools
    fails as EMPTY OUTPUT rather than as an error (ollama/ollama#10976), so an
    undeclared tag has to lose the tools until a deep verdict says otherwise.
    The false verdict below is what the real lookup returns for a tag with no
    declaration to revoke (see app.services.earned.effective_verdicts).
    """
    plan = await run_plan(
        ChatRequest(message="hello", thinking="on", model=STRAY),
        verdicts=VerdictFake(
            by_tag={STRAY: EffectiveVerdicts(earned="assisted", thinking_with_tools=False)}
        ),
        reports=reports_for({STRAY: TOOLED}),
    )
    assert plan.think_flag is True
    assert plan.offer_tools is False


@pytest.mark.anyio
async def test_routing_onto_an_undeclared_tag_re_resolves_its_own_report():
    """The routed model's floors read the ROUTED model's report, not the candidate's.

    Both tags are probed - every window and every capability is runtime-first
    now - but the substitution lands on weights nothing has DECLARED anything
    about, and that difference is what decides whether the offer survives.

    And SURVIVING is not the same as surviving intact. The default is declared
    ``agentic``, so the offer resolved before routing is agentic; the routed
    tag's ceiling is only assisted, and the difference between the two is the
    GROUNDING, not the tools. Left unclamped the turn would withdraw the
    deterministic pre-fetch and the ambient knowledge base on weights that
    never earned the right to - an ungrounded answer with the plan still
    claiming otherwise.
    """
    kept = reports_for({STRAY: TOOLED})
    plan = await run_plan(
        ChatRequest(message="hello", web_search="on"),
        route=RouteFake(substitute=STRAY, note="substituted"),
        reports=kept,
    )
    assert kept.calls == [DEFAULT_MODEL, STRAY]
    assert plan.offer_tools is True
    assert (plan.offer.mode, plan.offer.think, plan.offer.agentic) == ("assisted", False, False)
    # ...so the pre-fetch the agentic mode would have withdrawn still runs.
    outcomes = plan.outcomes(offered_any=True)
    assert (outcomes.agentic, outcomes.do_search) == (False, True)


@pytest.mark.anyio
async def test_the_routed_clamp_re_runs_the_arbitration_it_does_not_edit_the_offer():
    """A demoted turn gets the ROUTED mode's own tie-break, not the candidate's.

    The two modes break thinking-vs-tools in OPPOSITE directions on a model
    that cannot do both: agentic turns an auto-thinking decision OFF to keep
    its tools, assisted keeps the reasoning and gives the tools up. So an
    offer demoted agentic -> assisted by editing its mode alone would carry the
    agentic tie-break under an assisted label - the user's auto-thinking turn
    silently losing its reasoning AND gaining tools, a combination the resolver
    cannot produce.
    """
    # An auto-thinking turn: the heuristic reads a fenced code block as one.
    asked = "why is this wrong\n```py\nx = 1\n```"
    verdicts = VerdictFake(
        by_tag={
            # The candidate reasons but cannot call tools while doing it, so
            # the agentic arbitration drops the reasoning to keep the tools...
            THINKING_MODEL: EffectiveVerdicts(earned="agentic", thinking_with_tools=False),
            # ...and the routed stray, with no declaration to grant it, cannot
            # either - which under assisted means the reasoning wins instead.
            STRAY: EffectiveVerdicts(earned="assisted", thinking_with_tools=False),
        }
    )
    plan = await run_plan(
        ChatRequest(message=asked),
        verdicts=verdicts,
        route=RouteFake(substitute=STRAY, note="substituted"),
        reports=reports_for({STRAY: TOOLED}),
    )
    assert plan.heuristic_think is True
    assert plan.model == STRAY
    # The coherent assisted resolution: reasoning kept, no tools this turn.
    assert (plan.offer.mode, plan.offer.think, plan.offer.agentic) == ("deterministic", True, False)
    assert plan.think_flag is True
    assert plan.offer_tools is False


@pytest.mark.anyio
async def test_a_substitution_can_never_raise_the_turn():
    """Routing onto stronger weights does not lift a clamp the candidate earned."""
    verdicts = VerdictFake(
        by_tag={DEFAULT_MODEL: EffectiveVerdicts(earned="assisted", thinking_with_tools=True)}
    )
    plan = await run_plan(
        ChatRequest(message="hello"),
        verdicts=verdicts,
        route=RouteFake(substitute=OTHER_REASONER, note="substituted"),
    )
    # The substitute is declared agentic with fully trusting verdicts, so its
    # own arbitration would come out agentic - and it must not be adopted.
    assert verdicts.calls == [DEFAULT_MODEL, OTHER_REASONER]
    assert plan.offer.mode == "assisted"
    assert plan.offer.agentic is False

    dropped = reports_for({STRAY: TOOLLESS})
    plan = await run_plan(
        ChatRequest(message="hello"),
        route=RouteFake(substitute=STRAY, note="substituted"),
        reports=dropped,
    )
    assert plan.offer_tools is False


@pytest.mark.anyio
async def test_the_candidate_report_is_reused_when_routing_keeps_the_model():
    """One tag, one probe: routing that changes nothing must not double the I/O."""
    reports = reports_for({STRAY: TOOLED})
    await run_plan(ChatRequest(message="hello", model=STRAY), reports=reports)
    assert reports.calls == [STRAY]


# --- the stored declarations -----------------------------------------------------


@pytest.mark.anyio
async def test_a_stored_clamp_holds_the_turn_down():
    """The default is declared agentic; the operator clamped it. The turn obeys."""
    verdicts = VerdictFake()
    plan = await run_plan(
        ChatRequest(message="hello"),
        verdicts=verdicts,
        policies=policies_for({DEFAULT_MODEL: PolicyView(ceiling="deterministic")}),
    )
    assert plan.offer.mode == "deterministic"
    assert plan.offer_tools is False
    # Floored by the ceiling itself, so there was nothing for a verdict to add.
    assert verdicts.ceilings == ["deterministic"]


@pytest.mark.anyio
async def test_a_stored_ceiling_reaches_the_verdict_lookup():
    """The ceiling handed to the earned lookup is the POLICY one, not the report's.

    A lookup resolved against a different ceiling than the offer would let the
    page, the turn and the log disagree about one model - the exact drift the
    ceiling is passed in to prevent.
    """
    verdicts = VerdictFake()
    await run_plan(
        ChatRequest(message="hello"),
        verdicts=verdicts,
        policies=policies_for({DEFAULT_MODEL: PolicyView(ceiling="assisted")}),
    )
    assert verdicts.ceilings == ["assisted"]


@pytest.mark.anyio
async def test_the_routed_model_gets_its_own_policy():
    """Substitution re-resolves against the ROUTED tag's declarations, not the candidate's."""
    policies = policies_for({OTHER_REASONER: PolicyView(ceiling="deterministic")})
    plan = await run_plan(
        ChatRequest(message="hello"),
        route=RouteFake(substitute=OTHER_REASONER, note="substituted"),
        policies=policies,
    )
    assert policies.calls == [DEFAULT_MODEL, OTHER_REASONER]
    assert plan.offer.mode == "deterministic"


@pytest.mark.anyio
async def test_the_candidate_policy_is_reused_when_routing_keeps_the_model():
    """One tag, one query - the same rule the report fetch follows."""
    policies = PolicyFake()
    await run_plan(ChatRequest(message="hello"), policies=policies)
    assert policies.calls == [DEFAULT_MODEL]


@pytest.mark.anyio
async def test_a_stored_grant_reaches_the_io_free_default():
    """A turn that never queries the log still honours a thinking-with-tools grant.

    This tag takes no tool list, so the planner takes its IO-free path and never
    calls the verdict fetcher - and the declared value it carries has to be the
    stored one, or a granted model would silently lose its grant on exactly the
    turns that skip the lookup.
    """
    verdicts = VerdictFake()
    plan = await run_plan(
        ChatRequest(message="hello", model=TOOLLESS_MODEL),
        verdicts=verdicts,
        policies=policies_for({TOOLLESS_MODEL: PolicyView(thinking_with_tools=True)}),
    )
    assert verdicts.calls == []
    assert plan.verdicts.thinking_with_tools is True


@pytest.mark.anyio
async def test_a_measured_always_forces_the_think_flag_on():
    """The routing clamp: measured always beats a declared hybrid, immediately.

    The default tag is a seeded HYBRID and this turn's heuristic wants no
    reasoning, so the flag would normally go out as False - which on weights
    that reason regardless means the serving stack parses no reasoning channel
    and the raw ``<think>`` block lands in the answer. The measurement forces it
    back on without waiting for anyone to confirm.
    """
    plain = await run_plan(ChatRequest(message="hello", model=DEFAULT_MODEL))
    assert plain.think_flag is False
    clamped = await run_plan(
        ChatRequest(message="hello", model=DEFAULT_MODEL),
        measured=MeasuredThinkFake({DEFAULT_MODEL: "always"}),
    )
    assert clamped.think_flag is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("model", "mode", "thinking", "flag"),
    [
        # Measured never over a seeded hybrid: applying it would OMIT the
        # flag on a turn that explicitly asked to reason.
        (DEFAULT_MODEL, "never", "on", True),
        # Measured hybrid over a seeded always: applying it would let a
        # non-thinking turn send `think: false` to weights that reason anyway -
        # which is the very leak the one allowed direction exists to repair.
        (OTHER_REASONER, "hybrid", "auto", True),
    ],
)
async def test_no_other_measured_direction_touches_the_turn(
    model: str, mode: str, thinking: str, flag: bool
):
    """The safe-direction asymmetry, at the turn level.

    Both of these are decisions about how to run a model, taken on a
    measurement nobody has confirmed, so they wait for the confirm endpoint -
    and the turn resolves exactly as it did before the measurement existed.
    Each case is chosen so that APPLYING the measurement would visibly change
    the flag: a parametrization where both answers agree would pass either way.
    """
    plan = await run_plan(
        ChatRequest(message="hello", model=model, thinking=thinking),
        measured=MeasuredThinkFake({model: mode}),
    )
    assert plan.think_flag is flag


@pytest.mark.anyio
async def test_the_clamp_also_strips_tools_without_a_reason_and_call_grant():
    """The composition the clamp lands in, pinned so the copy cannot drift from it.

    Forcing the flag on is one decision; what happens next is the #10976 gate,
    which drops the tools of any turn that will reason on a model without a
    reason-and-call-tools grant. Nothing grants that by default - only an
    operator declaration does - so on an ordinary deployment the clamp costs
    the model its tools too, and the card says so, which is only honest while
    this stays true.

    The turn is NOT left ungrounded by it: with no tool surviving, the plan's
    own outcomes run the deterministic pre-fetch exactly as they did before
    tools existed.
    """
    plan = await run_plan(
        ChatRequest(message="what is the latest python version", model=OTHER_REASONER),
        verdicts=VerdictFake(
            {OTHER_REASONER: EffectiveVerdicts(earned="agentic", thinking_with_tools=False)}
        ),
        measured=MeasuredThinkFake({OTHER_REASONER: "always"}),
    )
    assert plan.think_flag is True  # the clamp landed
    assert plan.offer_tools is False  # and took the tools with it
    # The grounding the tools would have done still happens.
    assert plan.outcomes(offered_any=False).do_search is True


@pytest.mark.anyio
async def test_the_clamp_keeps_the_tools_where_the_model_may_reason_and_call():
    """The other direction, and the proof the stripping is the GATE and not the clamp.

    Same forced flag, on a model whose verdict grants reason-and-call-tools: the
    tools stay. A clamp that dropped them on its own would fail here, and the
    card's sentence about it would be a lie on every properly-granted model.
    """
    plan = await run_plan(
        ChatRequest(message="what is the latest python version", model=THINKING_MODEL),
        verdicts=VerdictFake(
            {THINKING_MODEL: EffectiveVerdicts(earned="agentic", thinking_with_tools=True)}
        ),
        measured=MeasuredThinkFake({THINKING_MODEL: "always"}),
    )
    assert plan.think_flag is True
    assert plan.offer_tools is True


@pytest.mark.anyio
async def test_the_measured_lookup_asks_only_about_the_routed_model():
    """One query, for the tag whose flag this turn actually sets.

    The candidate's mode changes nothing here - the flag is normalized against
    the ROUTED model - so asking about it would be a query for an answer the
    plan throws away.
    """
    measured = MeasuredThinkFake()
    await run_plan(
        ChatRequest(message="hello"),
        route=RouteFake(substitute=OTHER_REASONER, note="substituted"),
        measured=measured,
    )
    assert measured.calls == [OTHER_REASONER]
