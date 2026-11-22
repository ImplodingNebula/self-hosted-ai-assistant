"""Cached runtime facts about installed models, read through the provider seam.

This module answers what the runtime itself REPORTS about a tag - the
capability flags and the native context length a ``/api/show``-style endpoint
exposes - and then folds that report together with the other things known
about the same weights into one answer per question.

Three sources, and the order every ``effective_*`` resolver here reads is:
the stored policy declaration, then the code-shipped quirk seed, then the
runtime report or a template inference, then the safe floor. The store
(:mod:`app.services.model_policy`) sits on top because an operator declared it
about THIS deployment and it is digest-gated, so a re-pull retires it; the seed
(:mod:`app.services.known_models`) sits under it because it describes a model
FAMILY rather than this machine, and above the report because a written-down
measurement beats reading a template's shape. Policy travels in as an argument
- a resolver here stays pure and does no I/O - while the seed needs none at
all, which is why consulting it costs the background jobs and the warm path
nothing.

One of them - :func:`effective_agency_ceiling` - answers a POLICY question
rather than a mechanical one, and it is deliberately stingier on the report
half: a report lifts a tag as far as ``assisted`` and no further, because "this
template accepts a tool list" is a fact about a template and letting a model
DRIVE a turn is a decision about weights that only a human makes.

Results are cached in-process per (provider name, tag) for a short TTL: the
facts change only when weights change, and the callers sit on request paths
that must not pay a metadata round trip per hit. A FAILED probe is cached too,
for a much shorter window, because "the runtime would not say" is an answer a
turn now depends on: a tag on a half-answering server would
otherwise re-probe before every single generation. The cache is dropped
outright when a pull or a delete changes what a tag points at, so the TTL is a
ceiling on staleness rather than the only thing standing between a re-pull and
a stale answer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence

from app.providers.base import ModelCapabilities, ModelProvider
from app.services.agency import agency_rank
from app.services.known_models import quirks_for
from app.services.model_policy import PolicyView
from app.services.model_resolution import AgencyMode, ThinkMode, ToolsMode

logger = logging.getLogger(__name__)

# Runtime reports change when weights are re-pulled, which is rare; a short
# TTL keeps a stale report from outliving a pull by long while making every
# request-path hit free. Callers that need a fresher answer than this ask for
# one per call (``max_age_seconds``) rather than the cache being split in two.
_TTL_SECONDS = 300.0

# How long a FAILED probe is remembered. Deliberately far shorter than the
# positive TTL, and the trade is between two costs that pull opposite ways.
# Not caching failures at all puts a provider round trip on the turn path -
# BEFORE generation - for every turn against a runtime that will not report, so
# a dead or half-answering server taxes each turn twice. Caching them for the
# full 300s would leave a runtime that came back a minute ago still described
# as silent. Thirty seconds bounds the per-turn I/O while keeping a recovering
# runtime visible within half a minute.
#
# The vision gate shares this cache, and a stale negative is acceptable there
# for the reason its fresh read already is: an unanswered probe resolves to "no
# images", which is the same safe answer a failure gives, so the negative can
# only withhold a capability briefly - never grant one.
_FAILURE_TTL_SECONDS = 30.0

# What the VISION gate accepts, and the one place a caller wants less than the
# TTL above. Carried here beside it so the two are read together, with the
# reason the old separate vision cache recorded for choosing it, unchanged:
# the failure directions are opposite. A stale earned verdict clamps agency
# DOWN (fails safe), while a stale vision=True keeps sending pixels to weights
# that may no longer accept them (fails unsafe). A re-pull is visible within a
# minute. That matters most where in-app invalidation cannot fire at all -
# a shared remote model server someone re-pulls a tag on from another machine.
VISION_MAX_AGE_SECONDS = 60.0

# How many probes a fan-out runs at once. The probes are metadata reads, so the
# ceiling is about not opening one connection per tag against a server holding
# a large inventory - not about their cost.
_MAX_CONCURRENT_PROBES = 8

# (provider_name, model) -> (fetched_at, capabilities). A None payload is a
# remembered FAILURE, held for the shorter :data:`_FAILURE_TTL_SECONDS`; the two
# share one dict so a successful probe replaces a negative outright rather than
# racing a second table.
_cache: dict[tuple[str, str], tuple[float, ModelCapabilities | None]] = {}


def clear_facts_cache() -> None:
    """Drop cached runtime reports (test isolation, forced refresh)."""
    _cache.clear()


async def runtime_capabilities(
    provider: ModelProvider,
    provider_name: str,
    model: str,
    *,
    max_age_seconds: float | None = None,
) -> ModelCapabilities | None:
    """Return the runtime's own report about ``model``, or None when it fails.

    None means "the runtime would not say" - an unreachable server, a tag it
    does not hold, an unexpected payload - and callers must treat it as
    unknown, never as a report of no capabilities: skipping a model because a
    probe failed and skipping it because the runtime said it cannot chat are
    different decisions with different fixes.

    Args:
        provider: The model provider to ask.
        provider_name: Its configured name (``settings.model_provider``), for
            the cache key.
        model: The exact tag to ask about.
        max_age_seconds: How stale an answer this caller will accept. None
            takes the shared TTL. A caller passing less does not get a second
            cache - it re-probes past its own limit and refreshes the one
            entry, so a stricter reader always leaves a fresher answer behind
            for a laxer one. This is per-CALLER freshness precisely because
            the stakes differ: a management card reading a minute-old fact is
            harmless, while the same fact deciding whether to send pixels is
            not (see :data:`VISION_MAX_AGE_SECONDS`). It shortens the SUCCESS
            window only - a remembered failure already expires faster than any
            caller would ask for (:data:`_FAILURE_TTL_SECONDS`).

    Returns:
        The runtime's report, or None when the probe failed.
    """
    key = (provider_name, model)
    now = time.monotonic()
    hit = _cache.get(key)
    if hit is not None:
        cached_at, payload = hit
        limit = (
            _FAILURE_TTL_SECONDS
            if payload is None
            else (_TTL_SECONDS if max_age_seconds is None else min(max_age_seconds, _TTL_SECONDS))
        )
        if now - cached_at < limit:
            return payload
    try:
        caps = await provider.capabilities(model)
    except Exception:  # noqa: BLE001 - any probe failure reads as "unknown"
        logger.warning("Capability probe failed for %s on %s", model, provider_name, exc_info=True)
        _cache[key] = (now, None)
        return None
    _cache[key] = (now, caps)
    return caps


async def runtime_capabilities_many(
    provider: ModelProvider, provider_name: str, models: Sequence[str]
) -> list[ModelCapabilities | None]:
    """Probe several tags at once, bounded, one answer per tag in order.

    The fan-out every surface that describes a whole inventory needs, in one
    place so the bound is one number rather than a decision each caller makes
    again. Unbounded, a remote server holding dozens of tags would get one
    connection per tag at once, from a client whose read timeout is measured
    in minutes.

    Args:
        provider: The model provider to ask.
        provider_name: Its configured name, for the cache key.
        models: The tags to ask about.

    Returns:
        One report per input tag, positionally, with None where the probe
        failed - never a shorter list, so callers can zip and see which.
    """
    limit = asyncio.Semaphore(_MAX_CONCURRENT_PROBES)

    async def probe(model: str) -> ModelCapabilities | None:
        async with limit:
            return await runtime_capabilities(provider, provider_name, model)

    return list(await asyncio.gather(*(probe(model) for model in models)))


async def resolve_num_ctx(
    provider: ModelProvider,
    provider_name: str,
    model: str,
    *,
    default_window: int,
    ceiling: int | None = None,
) -> int:
    """Return the context window a turn on ``model`` must request. Never None.

    Every tag resolves the same way now: the smaller of the runtime's native
    window and what this DEPLOYMENT affords - see :func:`_unsized_window` for
    what "affords" resolves to, and why the workspace ceiling is part of that
    answer rather than a clamp on top of it. A committed file used to be able
    to size a tag ahead of the runtime, which meant the number a turn requested
    could be one nobody had checked against the weights in front of it.

    The workspace ceiling then clamps that, and only downward - see
    :func:`clamp_num_ctx`. It is a no-op by construction (the affordance was
    already the ceiling) and applied anyway, so the rule reads as "resolve,
    then clamp" wherever it appears.

    Always an int, deliberately: ``None`` used to mean "let the runtime pick",
    and the runtime's pick is VRAM-scaled (4096 on most cards since Ollama
    0.15.5) while every context budget in the platform - the tool-result
    eviction floor, the RAG window clamp - treats an unknown window as
    unlimited. That pair of defaults let a turn assemble a prompt the runtime
    then silently truncated from the HEAD, system prompt first. A known window
    every layer agrees on is the fix, and it is also what makes the
    context-usage meter's denominator honest.

    Every caller that loads or generates on a model resolves the window HERE:
    warmup, the title job, and the chat turn requesting different windows for
    the same tag would make the runtime reload the model between them.

    The ceiling narrows that agreement to one TENANT, and knowingly. Two
    workspaces with different ceilings sharing a runtime would reload the model
    on every alternation between them - which is a real cost, and still the
    right trade: ``DEFAULT_CONTEXT_WINDOW`` is the deployment-wide number, and
    the ceiling exists precisely so a workspace can ask for something else.
    Single-tenant today, so nothing alternates yet.

    Args:
        provider: The model provider serving this deployment.
        provider_name: Its configured name, for the facts cache key.
        model: The exact tag the caller will run.
        default_window: What this deployment affords one turn
            (``settings.default_context_window``). Passed in rather than read
            here so this module keeps doing no I/O and every caller is visibly
            answering for the same deployment.
        ceiling: The workspace's context ceiling, when the caller has the
            stored defaults in hand. None leaves the window unclamped, which
            is what every caller without a tenant to read for wants.

    Returns:
        The window in tokens.
    """
    report = await runtime_capabilities(provider, provider_name, model)
    unsized = _unsized_window(report, ceiling=ceiling, default_window=default_window)
    return clamp_num_ctx(unsized, ceiling)


def effective_num_ctx(
    report: ModelCapabilities | None,
    *,
    default_window: int,
    ceiling: int | None = None,
) -> int:
    """The window rule itself, over a report already in hand. Never None.

    Split out of :func:`resolve_num_ctx` so the surface that DISPLAYS a
    model's window and the turn that REQUESTS one cannot disagree. A card
    showing "unknown" beside a native window of 262144 reads as "unlimited",
    when the turn would in fact ask for the deployment's affordance; one rule
    in one place is what keeps the number on the card the number that runs.
    The workspace ceiling belongs to that same rule for the same reason: a
    card reporting the model's full window under a ceiling that halves it
    would be off by exactly the amount the operator asked for.

    See :func:`resolve_num_ctx` for why the answer is always an int.

    Args:
        report: The runtime's report for this tag, or None.
        default_window: What this deployment affords one turn, as on
            :func:`resolve_num_ctx`.
        ceiling: The workspace's context ceiling, or None for unclamped.

    Returns:
        The window in tokens.
    """
    window = _unsized_window(report, ceiling=ceiling, default_window=default_window)
    return clamp_num_ctx(window, ceiling)


def clamp_num_ctx(window: int, ceiling: int | None) -> int:
    """Apply the workspace context ceiling to a resolved window.

    A ``min``, deliberately and permanently. A ceiling is an operator saying
    "spend at most this much of the machine on one turn", so it may narrow a
    window and never widen one. Letting a number typed into a settings page
    RAISE a window past what the resolution granted would hand a turn more KV
    cache than the weights admit, and the failure would be an out-of-memory
    load or a silently truncated prompt, neither of which names the setting
    that caused it.

    Callers pass every window through here even though the resolution above
    already accounted for the ceiling (:func:`_unsized_window` treats it as the
    deployment's affordance, so the result cannot exceed it). It is a no-op
    there, kept so the rule reads the same wherever a window is produced.

    Args:
        window: The window the resolution granted.
        ceiling: The workspace ceiling, or None when none is set.

    Returns:
        The smaller of the two, or ``window`` when no ceiling is set.
    """
    if ceiling is None:
        return window
    return min(window, ceiling)


def _unsized_window(
    report: ModelCapabilities | None, *, ceiling: int | None, default_window: int
) -> int:
    """The window for one turn: the smaller of native and afforded.

    The half of the rule that does not need the tag, so the async and the pure
    entry points above can share it.

    Two inputs, and only one of them is about the model. The NATIVE window is
    what the weights can take, and nothing may exceed it. The AFFORDANCE is how
    much of this machine a deployment is willing to spend on one turn, which no
    platform constant can know - so it is the workspace's context ceiling when
    an operator has named one, and the deployment's configured default
    otherwise.

    The ceiling REPLACES the default rather than clamping it, and the upward
    direction is the point: on a 256K-native tag with an 8192 default, an
    operator who raises their ceiling to 16384 is asking for 16384, and
    composing the two as a ``min`` would silently pin them at the default with
    nothing to point at. That is safe precisely because it is not the declared
    path - there is no model fact being widened here, and the native window
    still bounds the result.

    Args:
        report: The runtime's report for this tag, or None when the probe
            failed or was never taken.
        ceiling: The workspace's context ceiling, or None when none is set.
        default_window: What this deployment affords when no ceiling is set.

    Returns:
        The window in tokens; the affordance alone when the native window is
        unknowable.
    """
    affordance = ceiling if ceiling is not None else default_window
    native = report.context_length if report is not None else None
    if native is not None and native > 0:
        return min(native, affordance)
    return affordance


def effective_tools_mode(report: ModelCapabilities | None) -> ToolsMode:
    """Resolve whether a model's chat template can take a tool list.

    The runtime's own report is the whole answer, and a report that could not
    be taken (``None``) keeps the safe floor. Whether a template accepts a tool
    list is mechanical - the serving stack renders that template and is the
    only party that can see it - so there is nothing here for a written-down
    declaration to add, and a blackout for every tag nobody had described was
    the wrong answer for exactly the deployments this platform is pointed at.

    This is the MECHANICAL question only - what a chat template can be handed.
    How much a model may DRIVE with it is policy, and resolves separately in
    :func:`effective_agency_ceiling`, which reads the same report far more
    stingily: a vouched template lifts a tag as far as ``assisted`` and no
    further. So a report can widen what a card says about a tag, and what a
    turn OFFERS it, without ever making it agentic.

    Args:
        report: The runtime's report for this tag, or None when it failed or
            was never taken.

    Returns:
        The resolved :data:`~app.services.model_resolution.ToolsMode`.
    """
    return "native" if report is not None and report.tools else "none"


async def report_for(
    provider: ModelProvider, provider_name: str, model: str
) -> ModelCapabilities | None:
    """Read the runtime's report for a tag a caller is about to run.

    A blank tag answers None without a round trip - a deployment that has not
    chosen a chat model yet resolves one, and there is nothing to ask about.
    Everything else goes to the seam, behind the shared short-TTL cache, so the
    warm path and the background jobs pay at most one metadata read per tag
    every few minutes.

    The chat turn does not call this; it takes the same read as an injected
    callable so the planner stays free of provider machinery. Same rule, and
    the same cache underneath.

    Args:
        provider: The model provider to ask.
        provider_name: Its configured name, for the cache key.
        model: The exact tag about to be loaded or generated on.

    Returns:
        The runtime's report, or None for a blank tag or a probe that failed.
    """
    if not model:
        return None
    return await runtime_capabilities(provider, provider_name, model)


def effective_agency_ceiling(
    report: ModelCapabilities | None,
    *,
    policy: PolicyView | None = None,
    pair_ceiling: AgencyMode | None = None,
) -> AgencyMode:
    """Resolve the strongest mode a model may ATTEMPT before any clamp.

    A stored operator declaration answers first when there is one. It arrives
    already digest-gated (:mod:`app.services.model_policy` retires a ceiling
    declared against weights the runtime no longer holds), so nothing here
    compares digests, and it overrides in BOTH directions: an operator who
    clamps a model DOWN wants that to take effect on the next turn, and one
    who arms a model UP moves only this CEILING - the EARNED axis in
    :mod:`app.services.earned` still decides what a turn actually gets, and
    ``min(CEILING, EARNED)`` is untouched by anything here.

    With nothing declared, the runtime's report answers, and it stops one rung
    short of the top:

    - ``assisted`` for a tag whose template takes a tool list. This is the
      NO-REGRESSION mode. Tools are offered, the deterministic pre-fetch and
      the ambient knowledge base still run, so a model that never calls a tool
      answers exactly as it does today. Nothing about this turn gets worse
      because a report was believed.
    - ``agentic`` withdraws the pre-fetch and hands the turn to the model. That
      is a decision about how much a deployment trusts specific weights, and a
      runtime reporting "my template accepts a tool list" is not that decision -
      it is a mechanical fact about a chat template. Agentic stays a HUMAN
      declaration, made from the console and stored digest-bound; the ``policy``
      argument above is the only input here that can grant it.
    - ``deterministic`` otherwise, including for a report that could not be
      taken - "the runtime would not say" is not evidence.

    Every surface that needs this answer - the chat turn's offer, the earned
    lookup, the management card - reads it HERE, so what a page says a model may
    do is what a turn will actually grant it.

    Args:
        report: The runtime's report for this tag, or None.
        policy: The stored declarations for this tag, already folded and
            digest-gated, or None when nothing is declared. Taken as an input
            rather than fetched here so this stays pure: the caller that needs
            it is the caller that has a database session.
        pair_ceiling: The ceiling already resolved for the tag this one is the
            THINKING SIBLING of, when that is what is being resolved. It can
            only clamp DOWN, never raise.

            Declarations are stored per tag, and a pair is two tags - so an
            operator who clamps a model to ``deterministic`` from the console
            has, without this, clamped only the half that answers directly.
            Every thinking turn on that model routes to the sibling and keeps
            whatever the sibling's own report allows, which reads as the clamp
            being ignored. The clamp is applied at RESOLUTION rather than by
            writing a second row: a row against the sibling would be a
            declaration nobody made, and it would then outlive the one that
            caused it.

            One-directional on purpose. A permissive default must not lift a
            sibling that is clamped in its own right, so this is a ``min`` and
            not a replacement.

    Returns:
        The resolved :data:`~app.services.model_resolution.AgencyMode` ceiling.
    """
    if policy is not None and policy.ceiling is not None:
        resolved: AgencyMode = policy.ceiling
    else:
        resolved = "assisted" if report is not None and report.tools else "deterministic"
    if pair_ceiling is None:
        return resolved
    return min((resolved, pair_ceiling), key=agency_rank)


def effective_vision(model: str, report: ModelCapabilities | None) -> bool:
    """Resolve whether ``model`` may actually be shown images.

    The report is the whole answer, and the clamp direction is what makes that
    safe: nothing here can say yes over a runtime that says no, and a missing
    report is always no. A tag is mutable, so a written-down opt-in was never
    enough on its own to send pixels - the measurement always had to agree -
    and with the report deciding alone, a multimodal model on a server nobody
    described is usable for images instead of silently blind.

    Args:
        model: The exact model tag the turn will generate with.
        report: The runtime's report for this tag, or None.

    Returns:
        True only when the resolution above admits the pixels.
    """
    return bool(model) and report is not None and report.vision


def effective_think_mode(
    model: str,
    report: ModelCapabilities | None,
    *,
    policy: PolicyView | None = None,
    measured: ThinkMode | None = None,
) -> ThinkMode:
    """Resolve how ``model`` treats the think flag.

    Four sources, strongest first, and the ordering is about how each one KNOWS
    what it claims:

    0. An UNCONFIRMED measurement of ``always``, and nothing else it could say.
       See the asymmetry below - this rung exists only to stop a leak, and it
       is deliberately not a general "measured beats declared" layer.
    1. A stored **policy** confirmation - a check that drove these weights and
       watched what came back, and that a human then agreed to. It is
       digest-gated, so a re-pull retires it.
    2. The **seed** (:mod:`app.services.known_models`) - a family fact somebody
       measured once and wrote down beside its provenance. It sits above the
       template inference because it describes the weights rather than their
       packaging, and below the declaration above because that one was written
       about THIS deployment.
    3. The **template inference**. A PRIOR, deliberately weak: a template wired
       to the think flag reads as ``hybrid`` (it can be asked either way) and a
       readable template without that wiring reads as ``never``. Neither is a
       measurement of what the weights do - a template says what the serving
       stack CAN ask for, not what comes back - so nothing safety-critical is
       keyed on it.

    **The measured layer is deliberately one-directional.** An unconfirmed
    measurement outranks every declaration when it says ``always``, and is
    ignored otherwise. That is not timidity about the measurement; it is that
    the two directions have opposite consequences:

    - Measured ``always`` over a ``hybrid``/``never`` resolution means the flag
      is being cleared on weights that reason anyway, so the serving stack
      parses no reasoning channel and the raw ``<think>`` block lands in the
      visible answer. Forcing the flag ON puts it back on the collapsed
      channel. The worst case if the measurement were somehow wrong is a model
      asked to reason that would have anyway.
    - Every other direction would REMOVE reasoning, or offer a flag to weights
      that reject it, on the strength of a measurement nobody has confirmed.
      Those wait for the confirm endpoint, which is a human agreeing to store
      the same fact - and which then answers from ``policy`` above.

    Args:
        model: The exact model tag.
        report: The runtime's report for this tag, or None.
        policy: The stored declarations for this tag, already folded and
            digest-gated, or None. Pure input, as on
            :func:`effective_agency_ceiling`.
        measured: What a check watched these weights do with the flag, from
            :func:`app.services.earned.measured_think_mode` - already bound to
            the installed digest and the running harness by the time it gets
            here, so nothing about weights or versions is compared in this
            function. Pure input for the same reason ``policy`` is.

    Returns:
        The resolved :data:`~app.services.model_resolution.ThinkMode`.
    """
    if measured == "always":
        return "always"
    if policy is not None and policy.think_mode is not None:
        return policy.think_mode
    quirks = quirks_for(model)
    if quirks is not None and quirks.think_mode is not None:
        return quirks.think_mode
    if report is None or report.template_thinking is None:
        return "unknown"
    return "hybrid" if report.template_thinking else "never"
