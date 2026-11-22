"""One inspectable plan for how a chat turn will drive.

:func:`plan_turn` gathers every pre-generation routing decision - the thinking
and web-search heuristics, the agency clamps, the tool offer, model routing,
and think-flag normalization - into a single :class:`TurnPlan` value, in
exactly the order the chat service used to make them inline. The decisions are
unchanged; what changed is that they are now one value that can be logged,
asserted on, and later shown in the tool-activity trace.

Two things are deliberately NOT here:

- **The token-budget floors.** Both are evaluated against the actually
  assembled prompt (once before generation, once inside it after grounding and
  attachments land) and are pinned by tests as separate evaluation points; a
  plan computed before assembly cannot know either answer.
- **Tool instantiation and selection.** Which tool objects exist is wiring;
  the plan only decides whether the turn may carry tools at all.

The five I/O boundaries - the earned-verdict lookup (database), the
installed-model routing (provider), the context-window resolution (the
runtime's native window, clamped by the deployment's affordance), the runtime
capability report (provider) and the stored policy declarations (database) -
are injected as callables, so the planner calls no provider or database
machinery itself (only value types and pure resolvers are imported) and unit
tests drive it with plain fakes.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from app.core.config import Settings
from app.providers.base import ChatMessage, ModelCapabilities
from app.schemas.chat import ChatRequest
from app.schemas.settings import SettingMode, TurnDefaults
from app.services import web_search
from app.services.agency import (
    ToolOffer,
    agency_rank,
    parse_agency_mode,
    resolve_tool_offer,
)
from app.services.context_budget import ANSWER_RESERVE_TOKENS, THINKING_ANSWER_RESERVE_TOKENS
from app.services.earned import EffectiveVerdicts, declared_thinking_with_tools
from app.services.model_facts import (
    effective_agency_ceiling,
    effective_think_mode,
    effective_tools_mode,
)
from app.services.model_policy import PolicyView
from app.services.model_resolution import (
    AgencyMode,
    ChatModelPair,
    ThinkMode,
)
from app.services.thinking import normalize_think, resolve_thinking

logger = logging.getLogger(__name__)

# The earned-verdict lookup for one tag AND the effective ceiling already
# resolved for it; wired to app.services.earned by the chat service, replaced by
# a fake in unit tests. The ceiling travels with the tag rather than being
# re-derived inside, because it depends on a runtime report the planner has
# already paid for - re-deriving it there would either
# duplicate the probe or let the two disagree about the same model.
VerdictFetcher = Callable[[str, AgencyMode], Awaitable[EffectiveVerdicts]]
# Installed-model routing for a config-routed tag: (model, think) -> the model
# to use plus an optional substitution status note.
ModelRouter = Callable[[str, bool], Awaitable[tuple[str, str | None]]]
# The context window one tag runs with. Injected like the two boundaries above,
# because resolving it can reach the provider: a tag's native window is a
# runtime fact, so the planner would otherwise have to do I/O.
ContextResolver = Callable[[str], Awaitable[int]]
# The runtime's own capability report for one tag, or None when the probe
# failed. Wired to the shared short-TTL facts cache, so a turn that needs one
# pays a round trip at most once per tag per TTL - and a turn whose tags are all
# already described never calls this at all.
ReportFetcher = Callable[[str], Awaitable[ModelCapabilities | None]]
# The stored operator declarations for one tag, folded and already digest-gated
# (None when nothing is declared). Unlike the report above this is fetched for
# EVERY tag: a declaration outranks the runtime's own report, so skipping it
# would be skipping the one source that can overrule the
# answer. Behind its own short-TTL cache, so the cost is one query per model per
# window rather than per turn.
PolicyFetcher = Callable[[str], Awaitable[PolicyView | None]]
# What a tool check MEASURED about one tag's think flag, already bound to the
# installed weights and the running harness (None when nothing measured them).
# Separate from the policy fetcher above because the two are different kinds of
# statement: that one is what an operator DECLARED, this is what a probe
# WATCHED and nobody has confirmed yet. Only one direction of it moves a turn -
# see :func:`app.services.model_facts.effective_think_mode`.
MeasuredThinkFetcher = Callable[[str], Awaitable[ThinkMode | None]]


@dataclass(frozen=True)
class TurnOutcomes:
    """The plan's final derivations, once the offered tool set is known.

    Attributes:
        agentic: The turn drives itself (agentic offer AND at least one tool
            actually survived instantiation, selection, and the budget floors).
        do_search: Run the deterministic web pre-fetch this turn.
        announce_tools_not_used: Stand the freshness-gated notice if the model
            then answers without calling a tool.
    """

    agentic: bool
    do_search: bool
    announce_tools_not_used: bool


@dataclass(frozen=True)
class TurnPlan:
    """Every pre-generation decision for one chat turn, in one value.

    Attributes:
        effective_thinking: The thinking mode after the settings chain - the
            request when explicit, else the stored default, else auto.
        effective_web: The web-search mode after the same chain.
        heuristic_think: The raw think decision from :func:`resolve_thinking`,
            before the thinking-vs-tools arbitration.
        resolved_web: The web-search decision (request switch + heuristic).
        candidate: The model the capability lookups were resolved against.
        verdicts: The candidate's effective (earned, thinking_with_tools)
            verdicts - declared values when the turn never needed the lookup.
        offer: The arbitrated :class:`~app.services.agency.ToolOffer`.
        model: The model the turn actually runs on, after routing.
        status_note: Human-safe substitution note for the SSE status channel,
            or None when nothing was substituted.
        model_verdicts: The ROUTED model's verdicts (same object as
            ``verdicts`` unless substitution forced a re-resolution).
        think_flag: The normalized think flag for :class:`GenerationOptions`.
        num_ctx: The context window this turn requests. Always a number: the
            budgets downstream read an unknown window as unlimited, which is
            how a prompt they thought fitted got truncated from the head (see
            :func:`app.services.model_facts.resolve_num_ctx`).
        reserve_tokens: The answer reserve matching ``think_flag``.
        offer_tools: Whether the turn may carry tools at all, after the
            routing floors (the budget floors downstream can still drop them).
    """

    effective_thinking: SettingMode
    effective_web: SettingMode
    heuristic_think: bool
    resolved_web: bool
    candidate: str
    verdicts: EffectiveVerdicts
    offer: ToolOffer
    model: str
    status_note: str | None
    model_verdicts: EffectiveVerdicts
    think_flag: bool | None
    num_ctx: int
    reserve_tokens: int
    offer_tools: bool

    def outcomes(self, *, offered_any: bool) -> TurnOutcomes:
        """Derive the turn's final drive flags from the surviving tool set.

        The ``tools_not_used`` notice is FRESHNESS-GATED, not blanket. Modern
        assistants stay silent when the model answers from its own knowledge
        and speak up only by attribution when a tool DID ground the answer
        (citations). Announce a non-lookup only when the turn looked like it
        wanted current/external info - the same heuristic the deterministic
        pre-fetch keys on - so a poem or an "explain recursion" turn is silent
        while a "latest ... today" turn the model skipped is flagged. On a
        small model that declines tools often, this is the difference between
        an honest signal and a banner nobody reads.

        Args:
            offered_any: Whether at least one tool survived instantiation,
                selection, and the pre-generation budget floor.

        Returns:
            The :class:`TurnOutcomes` the generation call acts on.
        """
        agentic = self.offer.agentic and offered_any
        return TurnOutcomes(
            agentic=agentic,
            do_search=self.resolved_web and not agentic,
            announce_tools_not_used=agentic and self.resolved_web,
        )


async def plan_turn(
    *,
    req: ChatRequest,
    history: Sequence[ChatMessage],
    settings: Settings,
    chat_pair: ChatModelPair,
    search_wired: bool,
    rag_wired: bool,
    fetch_verdicts: VerdictFetcher,
    route: ModelRouter,
    resolve_ctx: ContextResolver,
    fetch_report: ReportFetcher,
    fetch_policy: PolicyFetcher | None = None,
    fetch_measured_think: MeasuredThinkFetcher | None = None,
    defaults: TurnDefaults | None = None,
    message: str | None = None,
) -> TurnPlan:
    """Resolve every pre-generation decision for one turn.

    The order is load-bearing and mirrors the decisions' data dependencies:
    heuristics first (they need only the request), then the offer against the
    CANDIDATE model (the offer must exist before routing, because routing keys
    on the arbitrated think decision), then routing, then the floors that
    protect the ROUTED model from an offer resolved for different weights.

    Args:
        req: The validated chat request (thinking/web switches, optional
            explicit model, the developer agency clamp, and the message the
            heuristics read - ``req.message``, unless ``message`` overrides it,
            and never a synthesized prompt).
        history: The capped conversation history (heuristic context only).
        settings: Active settings (web-search enablement, the ``AGENCY_MODE``
            clamp, and the earned-verification switch the fetcher honours).
        chat_pair: The tier's resolved (default, thinking) chat tags.
        search_wired: Whether a web-search provider is wired.
        rag_wired: Whether the RAG collaborators (embedder + store) are wired.
        fetch_verdicts: Earned-verdict lookup, called at most twice and only
            when a tool could actually reach a model.
        route: Installed-model routing for config-routed tags; never called
            for an explicit ``req.model``.
        resolve_ctx: The window lookup for the ROUTED tag, called once per
            turn. Every caller that loads or generates on a model resolves it
            through the same helper, so a chat turn and a background job never
            ask the runtime for two different windows on one model - which
            would make it reload the weights between them.
        fetch_report: The runtime's capability report for one tag. Called at
            most twice (the candidate, and the routed model when routing landed
            somewhere else), behind the shared short-TTL cache.
        fetch_policy: The stored declarations for one tag, called for the
            candidate and (when routing moved) for the routed model. None
            leaves every answer to the report, which is what a caller with no
            database - and every test that is not about the store - wants.
        fetch_measured_think: What a check measured about one tag's think
            flag. Called ONCE, for the ROUTED model, because that is the only
            tag whose flag this turn actually sets - the candidate's mode
            affects nothing here, and asking about it would be a query for an
            answer the plan throws away. None (a caller with no database)
            routes on the declaration alone, exactly as before the measurement
            existed.
        defaults: The merged stored tiers (conversation over workspace), or
            None when neither is set - in which case every effective mode
            equals the request and the plan is byte-identical to the
            pre-chain behaviour.
        message: The words the turn was asked in, when they are not the
            request's own. Only a regenerate passes this: it carries no message
            because the question it re-answers is already stored, and reading
            intent out of the empty string would resolve a repeat of "what is
            the weather now" as an ordinary offline turn. Still the USER's
            words either way - never a synthesized prompt. None (every other
            caller) reads ``req.message``.

    Returns:
        The complete :class:`TurnPlan`.
    """
    # The settings chain, per axis: an explicit request value wins; "auto"
    # defers to the stored default; an unset default leaves "auto" standing
    # so the heuristics decide, exactly as before the chain existed.
    stored_thinking = defaults.thinking if defaults is not None else None
    stored_web = defaults.web_search if defaults is not None else None
    effective_thinking: SettingMode = (
        req.thinking if req.thinking != "auto" else (stored_thinking or "auto")
    )
    effective_web: SettingMode = (
        req.web_search if req.web_search != "auto" else (stored_web or "auto")
    )

    # On the user's own words, never a synthesized prompt: the synthesized ask is
    # words the user did not write, and reading intent out of them is reading our
    # own tail. An attachment-only turn resolves both to their defaults.
    asked = req.message if message is None else message
    heuristic_think = resolve_thinking(effective_thinking, asked)
    # History-aware: a context-dependent follow-up in a fresh-info thread
    # searches even without a freshness cue of its own.
    resolved_web = web_search.resolve_web_search(effective_web, asked, history)

    # The candidate model for capability lookup: whatever routing WOULD pick
    # from the raw thinking decision. On the dev pair both members share one
    # set of declarations, so the ceiling / tools / thinking_with_tools it yields
    # hold for either member - which is what lets the offer be decided before
    # routing.
    if req.model:
        candidate = req.model
    elif heuristic_think and chat_pair.thinking:
        candidate = chat_pair.thinking
    else:
        candidate = chat_pair.default

    # Whether each tool MAY be offered, folding in the user's switches, the
    # workspace settings, and whether its collaborator is wired. web_search is
    # gated on the web switch so the locked "web search off = fully local"
    # stays literally true; search_knowledge needs the RAG collaborators and
    # honours RAG_ENABLED itself.
    # Keyed on the EFFECTIVE mode, not the raw request: a conversation-level
    # "off" with a bare request must withhold the web tool too, or the locked
    # "web search off = fully local" floor would hold for the pre-fetch but
    # not for the tool the model could still call.
    web_offered = effective_web != "off" and settings.web_search_enabled and search_wired
    knowledge_offered = rag_wired
    # What the RUNTIME says about the candidate: the mechanical questions only
    # the serving stack can answer about its own templates and weights. Against
    # a server holding an inventory nobody has described, this is the
    # difference between every tag clamping to deterministic - no tools, on
    # weights that support them - and the deployment working.
    #
    # The blank tag is excluded because it names nothing to probe: a deployment
    # that has not chosen a chat model reaches here before routing raises, and
    # a metadata round trip for "" would be a guaranteed failure on the way to
    # a clearer one.
    candidate_report = None if not candidate else await fetch_report(candidate)
    # What an OPERATOR declared about the candidate, which outranks the report:
    # a declaration exists precisely to overrule what the runtime says.
    candidate_policy = (
        await fetch_policy(candidate) if fetch_policy is not None and candidate else None
    )
    ceiling = effective_agency_ceiling(candidate_report, policy=candidate_policy)
    env_clamp = parse_agency_mode(settings.agency_mode)
    request_clamp = parse_agency_mode(req.agency)
    tools_native = effective_tools_mode(candidate_report) == "native"

    # The EARNED half of the resolution: what the tool-check log proves about
    # the installed weights (digest-bound, cached). Looked up only when a tool
    # could actually reach the model - a deterministic clamp, a template that
    # takes no tools, or an empty tool set already floors the turn, so those
    # turns stay IO-free and the declared values stand in as the (irrelevant)
    # earned axis.
    verdicts = EffectiveVerdicts(
        earned=ceiling,
        thinking_with_tools=declared_thinking_with_tools(candidate_policy),
    )
    if (
        tools_native
        and "deterministic" not in (env_clamp, request_clamp)
        and (web_offered or knowledge_offered)
    ):
        verdicts = await fetch_verdicts(candidate, ceiling)
    offer = resolve_tool_offer(
        ceiling=ceiling,
        earned=verdicts.earned,
        env_mode=env_clamp,
        request_mode=request_clamp,
        tools_native=tools_native,
        thinking_with_tools=verdicts.thinking_with_tools,
        requested_thinking=effective_thinking,
        heuristic_think=heuristic_think,
        web_offered=web_offered,
        knowledge_offered=knowledge_offered,
    )

    # Route the model on the ARBITRATED think decision (offer.think), never the
    # raw heuristic: an agentic auto turn that yielded its reasoning to keep
    # the tools must route to the non-thinking model, not the always-thinking
    # one. An explicit req.model is honored verbatim and fails loudly (typed
    # model_not_found) rather than being second-guessed.
    status_note: str | None = None
    if req.model:
        model = req.model
    else:
        model = chat_pair.thinking if offer.think and chat_pair.thinking else chat_pair.default
        model, status_note = await route(model, offer.think)
    # The ROUTED model's report - reused
    # outright when routing kept the candidate, so the common case costs one
    # fetch and not two. Resolved BEFORE the agency floors below and
    # unconditionally, because the very next line needs it: how a model treats
    # the think flag has nothing to do with whether this turn carries tools, and
    # making the fetch conditional on the tool path would leave an agency-clamped
    # turn normalizing the flag off a report it declined to read.
    if model == candidate:
        routed_report = candidate_report
        routed_policy = candidate_policy
    else:
        routed_report = None if not model else await fetch_report(model)
        routed_policy = await fetch_policy(model) if fetch_policy is not None and model else None
    num_ctx = await resolve_ctx(model)

    # Final safety floors, keyed on the ROUTED model. Substitution (or an
    # explicit pick) can land the turn on different weights than the candidate
    # the offer was resolved for; the offer must never outrank the model that
    # actually runs.
    model_verdicts = verdicts
    if offer.mode != "deterministic" and model != candidate:
        # Re-resolve the ROUTED model's own claims - ceiling, earned, template -
        # by running the SAME arbitration again over them. On a properly paired
        # deployment both members share their declarations, so this resolves to
        # answer (and the cache makes it free); the guard exists for the
        # misconfigured case where they do not.
        #
        # Re-running rather than editing the offer in place, because the modes
        # do not differ only in rank: they break the thinking-vs-tools tie in
        # OPPOSITE directions. An agentic auto-thinking turn turns reasoning OFF
        # to keep its tools, while assisted keeps the reasoning and takes the
        # tools away. Rewriting mode alone would leave the agentic tie-break
        # standing under an assisted label - a turn that silently lost its
        # reasoning AND gained the tools, which is a combination the resolver
        # cannot produce and no rule in the platform describes.
        routed_ceiling = effective_agency_ceiling(
            routed_report,
            policy=routed_policy,
            # The clamp an operator put on the tag this turn STARTED from
            # reaches the sibling it routed to. Declarations are per tag and a
            # pair is two tags, so without this a console clamp on the
            # direct-answer half leaves every thinking turn running at whatever
            # the sibling's own entry allows - which reads, correctly, as the
            # clamp having been ignored. Clamp-down only; a permissive default
            # cannot lift a sibling held back in its own right.
            pair_ceiling=ceiling,
        )
        model_verdicts = await fetch_verdicts(model, routed_ceiling)
        routed_offer = resolve_tool_offer(
            ceiling=routed_ceiling,
            earned=model_verdicts.earned,
            env_mode=env_clamp,
            request_mode=request_clamp,
            tools_native=effective_tools_mode(routed_report) == "native",
            thinking_with_tools=model_verdicts.thinking_with_tools,
            requested_thinking=effective_thinking,
            heuristic_think=heuristic_think,
            web_offered=web_offered,
            knowledge_offered=knowledge_offered,
        )
        # Adopted only when it is no more autonomous than what the candidate
        # earned: the routed model is authoritative about ITSELF, but the turn
        # was already clamped for the weights the offer was resolved against,
        # and letting a substitution raise the mode would be an escalation
        # through the back door.
        #
        # The model is deliberately NOT re-routed on the new offer. Routing ran
        # once, on the candidate-resolved think decision, and the routed tag
        # stays - `normalize_think` against ITS mode below absorbs any flip in
        # the think value. Re-routing would invite a fixpoint chase (route,
        # re-resolve, route again) for no behavioural gain: the second hop
        # could only land on a model whose claims would have to be re-resolved
        # in turn.
        if agency_rank(routed_offer.mode) <= agency_rank(offer.mode):
            if routed_offer != offer:
                logger.info(
                    "Routed model %r resolves the turn as %s (was %s); adopting its arbitration",
                    model,
                    routed_offer.mode,
                    offer.mode,
                )
            offer = routed_offer

    # Keyed on the NORMALIZED flag, and computed AFTER the floors so it reads
    # the offer that actually stands: a turn the policy marked non-thinking can
    # still land on an always-thinking model (explicit pick, pair
    # substitution). Absent a declaration the mode is inferred from the
    # runtime's template report, which is what keeps a never-thinking tag from
    # being sent a flag its serving stack rejects with a 400 on every turn.
    #
    # The measured layer joins here and only here. A check that WATCHED these
    # weights reason with the flag cleared forces it back on, so the reasoning
    # goes to the collapsed channel instead of arriving as raw <think> markup
    # in the answer - without waiting for anyone to confirm the measurement.
    # Nothing else it could measure changes this line; see effective_think_mode.
    routed_measured = (
        await fetch_measured_think(model) if fetch_measured_think is not None and model else None
    )
    think_flag = normalize_think(
        offer.think,
        effective_think_mode(model, routed_report, policy=routed_policy, measured=routed_measured),
    )
    reserve_tokens = THINKING_ANSWER_RESERVE_TOKENS if think_flag else ANSWER_RESERVE_TOKENS
    offer_tools = offer.mode != "deterministic"

    # Offering tools to a model that WILL reason but cannot reason and call
    # tools at once walks straight into the empty-output bug
    # (ollama/ollama#10976) - so drop the tools, keep the reasoning, and let
    # the turn fall back to the deterministic pipeline.
    if offer_tools and think_flag and not model_verdicts.thinking_with_tools:
        logger.info(
            "Model %r will reason but cannot combine thinking and tools; "
            "dropping tools for this turn",
            model,
        )
        offer_tools = False

    return TurnPlan(
        effective_thinking=effective_thinking,
        effective_web=effective_web,
        heuristic_think=heuristic_think,
        resolved_web=resolved_web,
        candidate=candidate,
        verdicts=verdicts,
        offer=offer,
        model=model,
        status_note=status_note,
        model_verdicts=model_verdicts,
        think_flag=think_flag,
        num_ctx=num_ctx,
        reserve_tokens=reserve_tokens,
        offer_tools=offer_tools,
    )
