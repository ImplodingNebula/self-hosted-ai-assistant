"""The verify job: running the tool-check harness against a live deployment.

The harness itself (:mod:`app.services.tool_check`) is pure - it drives a model
and scores the exchange, and knows nothing about databases, GPUs, or the fact
that something else was using the card. This module is everything AROUND that:
the order a probe has to happen in on a real machine, and it is the only place
that order lives, so a terminal run and a UI click cannot disagree about what
was measured.

The order, and why each step is where it is:

1. **One probe at a time, process-wide.** A probe evicts models and generates
   tokens for minutes. Two at once do not merely cost double - they measure
   each other, because the second one's model load is what the first one's
   residency reading sees. The guard is not per model for the same reason.
2. **Unload deliberately, before measuring.** On a card that fits one model,
   loading the target evicts something no matter what; the choice is between
   choosing what goes and letting the runtime choose mid-measurement. Unloading
   first also guarantees the run measures a FRESH load at serve-time options,
   rather than reusing a runner somebody warmed with a different context size.
3. **Measure**, at the serve-time options this deployment's turns actually
   request - the window, and the think flag. The workspace ceiling is read
   BEFORE the probe for that reason and not only for the re-warm at the end: a
   certificate earned at a window no turn here runs describes somebody else's
   deployment, and the two requests would make the runtime reload the model
   between the check and the warm that follows it. The think flag is the same
   argument on a different axis, and this module supplies the two halves of it
   the harness cannot reach on its own - a stored declaration and a previous
   measurement, both of which live in the database this job has a session for.
   The PAIR is that argument a third time: the reasoning case drives the
   sibling a TURN would route to
   (:func:`~app.services.model_resolution.thinking_sibling_for`, stored operator
   pairing included), because a certificate for that case binds to the
   sibling's weights and one earned against a tag nothing routes to is a
   measurement of somebody else's configuration. Each finished sample streams
   out as progress.
4. **Assert where the model actually sat.** A model that spilled entirely to
   the CPU still answers every case - slowly - so a passing run says nothing
   about how the deployment serves. Such a run is MARKED, never silently
   accepted; see :func:`assess_residency` for exactly what that costs it.
5. **Record it**, bound to the weights digest, the runtime version and the
   serving host, through the same row builder the CLI uses. All three are
   axes a verdict must not migrate across: different weights, a different
   runtime, or a different machine each make it a measurement of something
   else.
6. **Re-warm chat and the embedder, unconditionally.** The probe evicted them.
   A diagnostic that leaves the deployment cold has made the app slower for the
   next person to use it, and that must not depend on the diagnostic having
   succeeded - the failure paths are exactly the ones that leave the most
   damage behind.

A verification NEVER writes a verdict it could not honestly measure: a run
where the model server was unreachable for every sample raises rather than
recording "deterministic", because a row like that clamps a healthy model on
the strength of a network outage.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing, suppress
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.errors import BadRequestError, ConflictError, NotFoundError
from app.core.tenancy import TenantContext
from app.providers.base import (
    ModelNotFoundError,
    ModelProvider,
    ModelResidency,
    ProviderUnavailableError,
    installed_tag_matches,
    tag_aliases,
)
from app.repositories.model_probes import ModelProbeRepository, row_from_summary
from app.schemas.models import (
    PullDoneEvent,
    VerifyDoneEvent,
    VerifyErrorEvent,
    VerifyProgressEvent,
    VerifyResidencyReport,
)
from app.schemas.settings import WorkspaceDefaults
from app.services import model_pull
from app.services.earned import clear_earned_cache, measured_think_mode
from app.services.embedding_resolution import resolve_embedding
from app.services.model_facts import clear_facts_cache, report_for
from app.services.model_manager import article_for
from app.services.model_names import require_model_name
from app.services.model_policy import PolicyView, clear_policy_cache, resolve_policy
from app.services.model_resolution import (
    ThinkMode,
    resolve_chat_models,
    thinking_sibling_for,
)
from app.services.tool_check import (
    THINK_CASE_NAMES,
    CaseSample,
    MeasuredThinkFetcher,
    PolicyFetcher,
    ToolCheckSummary,
    expected_sample_count,
    run_tool_check,
)
from app.services.warmup import warm_embedder, warm_model
from app.services.workspace_defaults import load_defaults

logger = logging.getLogger(__name__)

VerifyEvent = VerifyProgressEvent | VerifyDoneEvent | VerifyErrorEvent

# The model currently being probed, or None. One slot, not one per model: the
# guard exists because probes CONTEND (for the card, and for each other's
# residency readings), and two different models contend exactly as much as the
# same one twice. In-process is the honest scope, for the same reason it is on
# downloads: the case this exists for is a second click, which lands on the
# same worker.
_running: str | None = None

# Below this share of the model in GPU memory, a run is treated as having
# landed on the CPU. Deliberately a floor near zero rather than "fully
# resident": a PARTIAL split is ordinary and healthy on a small card (the dev
# tier's own chat model routinely sits around 60% GPU), so demanding full
# residency would mark every honest dev probe unrepresentative and teach a
# reader to ignore the flag. What is genuinely unrepresentative is the model
# that got no GPU at all.
_MIN_GPU_FRACTION = 0.05

_CPU_WARNING = (
    "The model ran on the CPU rather than the GPU, so this run does not "
    "describe how the deployment serves. It is recorded and its failures still "
    "count, but it certifies nothing - free the card and check again."
)

# Logged, not shown: it says something about the DEPLOYMENT rather than about
# the model being measured, and the verdict's own warning field is reserved for
# the one caveat that changes how a reading should be read.
_NO_UNLOAD_VERB = (
    "The model backend cannot unload models, so this probe measures whatever "
    "the runtime chose to evict rather than a deliberately freed card."
)


@dataclass
class VerifyOutcome:
    """What one verification run measured, and what became of it.

    Attributes:
        summary: The harness summary, with its throughput suppressed when the
            run was not representative (see :func:`assess_residency`).
        digest: The weights the chat tag pointed at, when the runtime exposes
            one - what binds this verdict to weights rather than to a mutable
            tag.
        thinking_digest: The same for the pair's thinking tag, when there is
            one.
        runtime_version: The serving runtime this run measured under.
        residency: Where the model sat while it ran, or None when the backend
            does not report residency.
        representative: Whether the run describes how the deployment serves.
        warning: Why it does not, when it does not.
        probe_id: The measurement row it was recorded as, or None when the
            write failed (the measurement still happened).
    """

    summary: ToolCheckSummary
    digest: str | None
    thinking_digest: str | None
    runtime_version: str | None
    residency: ModelResidency | None
    representative: bool
    warning: str | None
    probe_id: str | None


def serve_time_think_fetchers(
    *,
    provider: ModelProvider,
    provider_name: str,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> tuple[PolicyFetcher, MeasuredThinkFetcher]:
    """Build the two database reads the harness's think resolution needs.

    The harness resolves a tool case's think flag through the chain a turn uses
    (:func:`app.services.model_facts.effective_think_mode`), but it holds no
    session and cannot import the verdict log - which reads
    :func:`~app.services.tool_check.harness_version` back out of it - so the two
    database rungs arrive as callables. This is where the pair is built, for
    both writers, and that is the whole point of it being a function.

    A probe row carries the running harness version, which is a promise that any
    two rows under it measured the same thing. Two callers wiring these rungs
    DIFFERENTLY breaks that promise in the worst available way: the console
    would drive a declared always-thinking tag with the flag on and certify it,
    while a terminal run drove the same tag with the flag cleared, leaked a
    ``<think>`` block into its answers, failed on it, and wrote a
    ``deterministic`` row - which clamps regardless of harness version or host,
    demoting a model the console had just proven healthy. One factory makes that
    a shape rather than a convention.

    Both reads degrade internally to None on failure (see
    :func:`app.services.model_policy.policy_for` and
    :func:`app.services.earned.measured_think_for`), so a caller wires them
    whenever it has a session at all and never has to decide what a broken
    database means.

    Args:
        provider: The model seam, for the installed-digest reads both make.
        provider_name: The ``MODEL_PROVIDER`` selector value rows key on.
        sessionmaker: Session factory for the two reads (both cached).

    Returns:
        The ``(policy, measured think mode)`` fetcher pair, ready to hand to
        :func:`~app.services.tool_check.run_tool_check`.
    """

    async def policy_of(tag: str) -> PolicyView | None:
        """Read the stored declarations for one tag (cached; None on failure)."""
        return await resolve_policy(
            provider=provider,
            provider_name=provider_name,
            model=tag,
            sessionmaker=sessionmaker,
        )

    async def measured_think_of(tag: str) -> ThinkMode | None:
        """Read what a previous check measured about one tag's think flag."""
        found = await measured_think_mode(
            provider=provider,
            provider_name=provider_name,
            model=tag,
            sessionmaker=sessionmaker,
        )
        return None if found is None else found.mode

    return policy_of, measured_think_of


def verification_in_flight() -> str | None:
    """Return the model currently being probed, or None."""
    return _running


def clear_verifications() -> None:
    """Drop the in-flight claim (test isolation only)."""
    global _running
    _running = None


def claim_verification(tag: str) -> None:
    """Claim the single probe slot, or refuse.

    The check and the claim happen with no ``await`` between them, so on the
    event loop this is atomic - there is no window for a second request to
    slip through between reading the slot and taking it.

    Args:
        tag: The model about to be probed.

    Raises:
        ConflictError: A probe is already running. The message names the model
            being probed, because "a check is already running" without saying
            which one reads like a bug in the page.
    """
    global _running
    if _running is not None:
        # "Model check", not "tool check": this slot is SHARED - the embedder
        # check takes it too (embedder_verify.begin_embedder_check), so naming
        # the tool harness here told an operator whose embedder check was
        # holding the slot to go looking for a tool check that was not running.
        # The refusal names the MODEL, which is the fact that locates it either
        # way, and stays silent about which harness has it.
        raise ConflictError(
            f"A model check is already running on '{_running}'. It evicts models "
            "and takes minutes, so only one runs at a time - wait for it to "
            "finish before starting another."
        )
    _running = tag


def release_verification(tag: str) -> None:
    """Release the probe slot held for ``tag``; safe to call twice.

    Keyed on the holder, so a caller can only ever release its OWN claim. The
    unconditional version is a silent hazard in exactly the case that matters:
    a path that releases a claim it never took cancels somebody else's probe
    and hands the slot to a third caller mid-run, and nothing anywhere would
    say so. A mismatch is logged rather than raised - the release is always
    running in a ``finally``, where raising would replace whatever outcome the
    caller was actually reporting.

    Args:
        tag: The model the caller claimed.
    """
    global _running
    if _running is not None and _running != tag:
        logger.warning(
            "Refusing to release the tool-check slot for %r: it is held by %r.", tag, _running
        )
        return
    _running = None


async def begin_verification(provider: ModelProvider, *, provider_name: str, name: str) -> str:
    """Decide everything a verification can be refused for, then claim the slot.

    Split out of the stream for the same reason the download's is: all of it
    happens BEFORE the response begins, so a refusal arrives as problem+json
    rather than as an event on a stream something is already rendering.

    Args:
        provider: The model seam that holds the weights.
        provider_name: Its configured name, for the facts cache key.
        name: The requested model name, as received.

    Returns:
        The normalized tag, now claimed - the caller MUST pair this with
        :func:`release_verification` in a scope that unwinds on every outcome.

    Raises:
        UnprocessableEntityError: The name failed validation.
        BadRequestError: The runtime reports this tag cannot chat. The
            harness measures tool-calling, so a run against the embedder would
            evict the card for minutes and then append a "deterministic"
            verdict for a model that was never going to drive a turn.
        NotFoundError: The backend does not hold these weights. Probing a model
            that is not here would report a failure of the model rather than
            the plain fact that nothing was downloaded.
        ConflictError: A probe is already running.
        ProviderUnavailableError: The backend could not be reached.
    """
    tag = require_model_name(name).text
    wrong_role = await _wrong_role_reason(provider, provider_name, tag)
    if wrong_role is not None:
        raise BadRequestError(wrong_role)
    installed = {info.name for info in await provider.list_models()}
    if not any(installed_tag_matches(alias, installed) for alias in sorted(tag_aliases(tag))):
        raise NotFoundError(f"'{tag}' is not installed, so there is nothing to check.")
    claim_verification(tag)
    return tag


def assess_residency(view: ModelResidency | None) -> tuple[bool, str | None]:
    """Decide whether a run's placement makes it representative, and say why not.

    The rule, and the reasoning behind each branch:

    - **No reading at all** (the backend does not report residency, or no
      longer held the model when asked): representative. Unknown is not
      evidence, and marking every run on such a backend would make the flag
      meaningless everywhere it is not needed.
    - **A partial split**: representative. Spilling part of a model to the CPU
      is how small cards run models at all.
    - **No GPU share worth the name**: NOT representative.

    What that costs the run is deliberately narrow, and it follows the standing
    rule that a probe may only ever clamp, never raise. The pass/fail verdicts
    are kept exactly as measured and still clamp a model that failed a guard -
    a spill does not make a model's tool calls better. What is withheld is the
    CERTIFICATE: a run that never touched the GPU cannot be the proof that
    lifts a model to ``agentic`` under ``AGENCY_REQUIRE_VERIFICATION``, and its
    measured throughput is dropped rather than published, because a CPU-bound
    decode rate describes the spill and not the deployment.

    Args:
        view: The residency reading taken around the run.

    Returns:
        ``(representative, warning)`` - the warning is None when it is.
    """
    if view is None or not view.loaded:
        return True, None
    fraction = view.gpu_fraction
    if fraction is None:
        return True, None
    if fraction < _MIN_GPU_FRACTION:
        return False, _CPU_WARNING
    return True, None


def _unload_targets(
    *,
    model: str,
    thinking: str | None,
    settings: Settings,
    embedding: str | None,
) -> list[str]:
    """List what to drop from memory before probing, in the order to drop it.

    The target first, because a stale runner for it is the one thing that would
    silently change what the run measures, then everything the probe is about
    to evict anyway: the deployment's own chat pair and the embedder. Naming
    them means the eviction happens at a moment of our choosing rather than
    part-way through a measured sample.

    Args:
        model: The tag about to be measured.
        thinking: Its thinking sibling, when it has one.
        settings: The active settings, for the chat pair's resolution.
        embedding: The embedder actually in use, resolved by the caller. Naming
            the deployment's configured tag instead would unload one that is not
            resident and leave the one that IS competing for the card through
            every sample - which is the reading, not a detail of it.
    """
    pair = resolve_chat_models(settings)
    candidates = [model, thinking, pair.default, pair.thinking, embedding]
    ordered: list[str] = []
    for tag in candidates:
        if tag and tag not in ordered:
            ordered.append(tag)
    return ordered


async def _free_the_card(
    provider: ModelProvider,
    *,
    model: str,
    thinking: str | None,
    settings: Settings,
    defaults: WorkspaceDefaults,
) -> None:
    """Unload the target and everything it would have evicted.

    The embedder is resolved through the usual chain, minus its evidence rung:
    no vector store is passed, because a Qdrant round trip buys nothing here -
    the question is which weights are on the CARD, and the store cannot answer
    it. The INVENTORY rung is passed, and has to be: on a deployment that
    stored no choice - now the common case, since "the server holds exactly
    one embedder" is what usually resolves it - a stored-choice-only lookup
    answers None, and the embedder this is here to evict stays resident
    through every sample of a run that exists to measure a fresh load.

    Resolved BEFORE the first unload rather than in the gap after it, so its
    round trips are not seconds in which something else can load onto a card
    just emptied.
    """
    if not provider.management_capabilities().can_unload:
        logger.info(_NO_UNLOAD_VERB)
        return
    embedding = await resolve_embedding(
        settings=settings,
        stored=defaults,
        provider=provider,
        provider_name=settings.model_provider,
    )
    for tag in _unload_targets(
        model=model,
        thinking=thinking,
        settings=settings,
        embedding=embedding.model,
    ):
        await provider.unload_model(tag)


async def _residency(provider: ModelProvider, model: str) -> ModelResidency | None:
    """Read where ``model`` sits, degrading ANY failure to unmeasured.

    The seam documents these hooks as best-effort and the shipped provider
    swallows its own failures, but the CONTRACT does not oblige the next one
    to. Broadly caught rather than narrowed to the seam's error type, and for a
    sharper reason than tidiness: this one runs MID-PROBE, between samples, so
    an implementation that raised anything at all would abandon a run whose GPU
    minutes are already spent. Losing a reading is a far smaller outcome than
    losing the run, and the caller records None as "unmeasured", never a guess.
    """
    try:
        return await provider.residency(model)
    except Exception:  # noqa: BLE001 - a measurement must never break the measuring
        logger.info("Could not read residency for %r while probing it.", model, exc_info=True)
        return None


async def _runtime_version(provider: ModelProvider) -> str | None:
    """Read the serving runtime's version, degrading the same way."""
    try:
        return await provider.runtime_version()
    except Exception:  # noqa: BLE001 - see :func:`_residency`
        logger.info("Could not read the runtime version after a probe.", exc_info=True)
        return None


def _serving_endpoint(provider: ModelProvider) -> str | None:
    """Read the host this deployment serves from, degrading the same way.

    The seam declares this one sync and IO-free, so there is nothing here to
    time out - but it is still read AFTER the GPU minutes are spent, and this
    module's rule at that point is that nothing may turn a completed
    measurement into a failed verification. A third-party provider is free to
    raise where the shipped ones only format a string, and None is an answer
    the log already understands ("did not say", certifying nothing).
    """
    try:
        return provider.serving_endpoint()
    except Exception:  # noqa: BLE001 - see :func:`_residency`
        logger.info("Could not read the serving endpoint after a probe.", exc_info=True)
        return None


async def _workspace_defaults(
    sessionmaker: async_sessionmaker[AsyncSession], tenant: TenantContext
) -> WorkspaceDefaults:
    """Read the tenant's stored defaults, degrading a failed read to the unset view.

    Shared by the two things in this module that need them - the window the
    probe runs at, and the model the re-warm puts back - so the two read the
    settings the same way and degrade the same way.

    They read at different MOMENTS, though, and deliberately: the re-warm runs
    on the failure path too, where the probe's read may never have happened at
    all. An operator who saves the settings page mid-run therefore gets a
    re-warm against the newer value - one extra model load, at the moment they
    asked for a different one.

    Never raises. On the probe side the fallback probes unclamped, which is
    what the deployment did before a ceiling was stored; on the re-warm side it
    heats the deployment-wide resolution. Both are worse than the stored answer
    and both are far better than failing a check, or leaving a card empty,
    because a settings read did not come back.

    Args:
        sessionmaker: Session factory for the settings read.
        tenant: The acting principal whose workspace is read.

    Returns:
        The typed defaults, every field unset when nothing is stored or the
        read failed.
    """
    try:
        async with sessionmaker() as session:
            return await load_defaults(session, tenant)
    except Exception:  # noqa: BLE001 - fall back to the env/profile resolution
        logger.warning("Could not read the workspace defaults for a model check.", exc_info=True)
        return WorkspaceDefaults()


async def _capture_digests(
    provider: ModelProvider, *, model: str, thinking: str | None
) -> tuple[str | None, str | None]:
    """Read the weights digests the two tags currently point at (best-effort).

    A tag is mutable - this repo has already been bitten by a whole tag family
    being re-pointed - so a verdict that names only a tag says nothing a re-pull
    cannot quietly invalidate. None is recorded as None ("the runtime exposes no
    digest"), never guessed.
    """
    digest: str | None = None
    thinking_digest: str | None = None
    try:
        for info in await provider.list_models():
            if installed_tag_matches(model, {info.name}):
                digest = info.digest
            if thinking and installed_tag_matches(thinking, {info.name}):
                thinking_digest = info.digest
    except ProviderUnavailableError:
        # The measurement already happened; losing the binding is a weaker
        # outcome than losing the run, so it degrades to "unmeasured".
        logger.info("Could not read the installed digests after probing %r.", model)
    return digest, thinking_digest


async def _persist(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    tenant: TenantContext,
    settings: Settings,
    summary: ToolCheckSummary,
    digest: str | None,
    thinking_digest: str | None,
    runtime_version: str | None,
    serving_endpoint: str | None,
    residency: ModelResidency | None,
    representative: bool,
) -> str | None:
    """Append the measurement row; a failed write is loud but not the run's error.

    Same rule the CLI applies: the expensive part - minutes of GPU time - has
    already happened, and reporting it as a failed verification would send
    someone to re-run the probe when the thing to fix is the database.
    """
    try:
        async with sessionmaker() as session:
            row = row_from_summary(
                summary,
                provider_name=settings.model_provider,
                digest=digest,
                thinking_digest=thinking_digest,
                runtime_version=runtime_version,
                serving_endpoint=serving_endpoint,
                vram_bytes=residency.vram_bytes if residency is not None else None,
                residency=residency,
                representative=representative,
            )
            # The audit columns, populated for the first time here: a UI click
            # has an actor where a terminal run does not. Constant today, and
            # recording it anyway is the point - the trail becomes meaningful
            # the moment real identity populates it.
            row.triggered_by_tenant_id = tenant.tenant_id
            row.triggered_by_user_id = tenant.user_id
            await ModelProbeRepository(session).add(row)
            await session.commit()
            probe_id = str(row.id)
    except Exception:
        logger.exception("Tool-check verdict for %r was NOT recorded", summary.model)
        return None
    # A verdict nobody can see yet is not much of a verdict: the chat path
    # caches earned lookups for minutes, so drop them and let the next turn
    # read what this run just measured.
    clear_earned_cache()
    return probe_id


async def _rewarm(
    provider: ModelProvider,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    tenant: TenantContext,
) -> None:
    """Reload the default chat model and the embedder the probe evicted.

    The chat warm resolves through the same stored chain a turn uses, so what
    gets put back is what a default turn will actually run: the workspace
    default participates (read here exactly as POST /models/warm reads it).
    Resolving without it heated the deployment's own tag instead - a model no
    turn on that deployment runs, cold-loaded onto the card the probe just
    fought for.

    Only the deployment's steady state is restored. Any OTHER model that
    happened to be resident before the probe (an explicit warm, a
    conversation's pinned pick) reloads on its next use instead. That is a real
    cost and not a self-correcting one: the shipped keep_alive is -1, so a
    bystander this probe evicted would have stayed resident indefinitely, and
    nothing puts it back until somebody uses it. It is still the right trade,
    on the three legs that do not depend on expiry - probes are rare and
    admin-triggered, a reload is seconds, and on a small card restoring a
    bystander would evict the very default this exists to put back.

    Never raises: this runs on the failure path too, and a warm request that
    fails must not replace the outcome the caller is waiting for. A cold
    deployment recovers on its own with the next turn - slowly, which is the
    whole reason this is here.

    KNOWN GAP - this restores the weights and the persona, but not the TOOL
    BLOCK. A tools-offered turn's prompt opens with the offered tool schemas
    rendered into the template's system region, and priming that needs the
    turn's collaborators (retrieval, web search, the tenant's registered MCP
    servers), which this service does not hold. So the first turn after a
    check still pays the tool-block prefill, exactly as every turn did before
    the warm endpoint learned to prime it. Closing it means threading the
    collaborators from the two endpoints that reach here down to
    ``warm_model``'s ``tool_prefix``; it is a real cost, not a design.
    """
    defaults = await _workspace_defaults(sessionmaker, tenant)

    async def _warm_chat() -> None:
        # The sessionmaker travels so the re-warm resolves the same think mode
        # a turn will - including the measurement THIS run just recorded, which
        # can move the flag on its own where it says the model reasons
        # regardless. Priming a different flag than the turn sends would leave
        # the first real question paying the cold load anyway. The window
        # ceiling travels for the same reason, one layer down: a differently
        # sized request restarts the runner outright.
        await warm_model(
            provider,
            settings,
            workspace_model=defaults.model,
            thinking=defaults.thinking,
            # The persona travels for the same reason: it is the head of the
            # prefix, so re-warming without it puts the weights back and leaves
            # the cache pointing at a prompt no turn here sends.
            workspace_prompt=defaults.system_prompt,
            context_ceiling=defaults.context_ceiling,
            sessionmaker=sessionmaker,
        )

    async def _warm_embed() -> None:
        await warm_embedder(provider, settings)

    for label, warm in (("chat model", _warm_chat), ("embedder", _warm_embed)):
        try:
            await warm()
        except Exception:  # noqa: BLE001 - a diagnostic must not fail on cleanup
            logger.warning("Could not re-warm the %s after a tool check.", label, exc_info=True)


def _residency_report(view: ModelResidency | None) -> VerifyResidencyReport:
    """Project a residency reading onto the wire shape (None stays unknown)."""
    if view is None or not view.loaded:
        return VerifyResidencyReport()
    fraction = view.gpu_fraction
    return VerifyResidencyReport(
        gpu_resident=None if fraction is None else fraction >= _MIN_GPU_FRACTION,
        gpu_fraction=None if fraction is None else round(fraction, 3),
        vram_bytes=view.vram_bytes,
        total_bytes=view.total_bytes,
    )


def done_event(outcome: VerifyOutcome) -> VerifyDoneEvent:
    """Build the terminal frame from a finished run."""
    summary = outcome.summary
    return VerifyDoneEvent(
        model=summary.model,
        thinking_model=summary.thinking_model,
        depth=summary.depth,
        earned=summary.earned,
        thinking_with_tools=summary.thinking_with_tools,
        measured_think_mode=summary.measured_think_mode,
        tallies={name: list(counts) for name, counts in summary.tallies.items()},
        tokens_per_second=summary.tokens_per_second,
        harness_version=summary.harness_version,
        digest=outcome.digest,
        runtime_version=outcome.runtime_version,
        residency=_residency_report(outcome.residency),
        representative=outcome.representative,
        warning=outcome.warning,
        probe_id=outcome.probe_id,
    )


async def run_verification(
    *,
    provider: ModelProvider,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    tenant: TenantContext,
    model: str,
    deep: bool,
    on_sample: Callable[[CaseSample], None] | None = None,
) -> VerifyOutcome:
    """Unload, probe, assert residency, and record - steps 2 to 5 of the job.

    Assumes the probe slot is already claimed (see :func:`begin_verification`)
    and does NOT re-warm: the claim belongs to the request scope that took it,
    and the re-warm belongs to a scope that also runs when this raises. Both
    live in :func:`stream_verification`, which is the whole order.

    Args:
        provider: The model seam to drive.
        settings: The active settings (the provider name rows key on, and the
            deployment's own chat pair).
        sessionmaker: Session factory for the measurement-log write, for the
            stored defaults the probe's window resolves against, and for the
            two declaration reads the probe's think flag resolves against.
        tenant: The acting principal. Recorded as audit-only columns - and,
            until identity is real, also the workspace whose context ceiling
            the probe runs under, which is the deployment's own.
        model: The validated tag to measure.
        deep: Run every case three times (the only depth that can earn
            ``agentic`` or ``thinking_with_tools``) rather than the fast subset.
        on_sample: Called after each sample completes, for progress reporting.

    Returns:
        The :class:`VerifyOutcome`.

    Raises:
        ProviderUnavailableError: The model server was unreachable for every
            sample. Deliberately raised rather than recorded: a run like that
            measured the network, and writing its "deterministic" verdict to
            the log would clamp a perfectly healthy model.
    """
    policy_of, measured_think_of = serve_time_think_fetchers(
        provider=provider,
        provider_name=settings.model_provider,
        sessionmaker=sessionmaker,
    )
    # Read BEFORE the card is freed, so the DATABASE round trips are not sitting
    # between the unload and the first sample: the run measures a fresh load,
    # and every second between the two is a second something else could load
    # onto the card this just emptied. The harness still reads the runtime's
    # metadata inside that gap when it sizes a tag, which is a
    # far cheaper call against a server this deployment is already talking to -
    # and one that loads nothing.
    defaults = await _workspace_defaults(sessionmaker, tenant)
    # The sibling a TURN would route to, which is the one this run has to drive:
    # the reasoning case's certificate binds to the SIBLING's weights, so a run
    # that drove some other pair while turns reason on a declared partner
    # would certify weights nothing serves - and the binding would never match,
    # withholding the capability with nothing saying why. The stored pairing is
    # the rung only this job can supply (the harness holds no session), read
    # through the same cached fetcher the think resolution uses below.
    pair_policy = await policy_of(model)
    thinking = thinking_sibling_for(
        settings,
        model,
        declared=None if pair_policy is None else pair_policy.thinking_model,
    )
    await _free_the_card(
        provider,
        model=model,
        thinking=thinking,
        settings=settings,
        defaults=defaults,
    )
    during: ModelResidency | None = None

    async def relay(sample: CaseSample) -> None:
        """Take the residency reading DURING the run, then report the sample.

        The first sample the target itself ran is the moment it has
        demonstrably loaded, and reading only AFTERWARDS is wrong on exactly
        the case that matters most: a deep run's last case drives the pair's
        THINKING tag, which on a card that fits one model evicts the very tag
        this row is about - so an after-only reading would report "not loaded"
        for a model that was on the GPU the whole time it was measured.

        Awaited by the harness between samples (it accepts an async hook), so
        the reading lands at a known point rather than whenever a background
        task happened to be scheduled.
        """
        nonlocal during
        if (during is None or not during.loaded) and sample.model == model:
            # Retried until one comes back LOADED rather than kept from the
            # first sample: a reading can legitimately miss (the runtime has not
            # registered the load yet), and keeping that miss would let a run
            # that spent every sample on the CPU report an unknown placement -
            # which is precisely the run this reading exists to mark.
            during = await _residency(provider, model)
        if on_sample is not None:
            on_sample(sample)

    summary = await run_tool_check(
        provider,
        provider_name=settings.model_provider,
        deep=deep,
        chat_model=model,
        thinking_model=thinking or "",
        # The window a turn on this workspace would ask for, so the verdict
        # certifies the configuration that actually runs here - and so the
        # re-warm that follows requests the same window rather than restarting
        # the runner the probe just measured.
        context_ceiling=defaults.context_ceiling,
        # The other half of that same window, for a tag no entry sizes: what
        # this deployment affords when the workspace has named no ceiling.
        default_window=settings.default_context_window,
        # The two DATABASE rungs of the think-flag resolution, which the harness
        # cannot take itself (it holds no session, and the log it would read
        # reads the harness version back out of it). Built by the shared factory
        # above so this run and the terminal one wire the SAME pair - see
        # serve_time_think_fetchers for what a disagreement here would cost.
        #
        # These fire inside the window between the unload above and the first
        # sample, alongside the runtime metadata read the window resolution
        # already makes there. Two indexed queries and one listing, none of
        # which loads a model - which is the standard that gap holds things to.
        # Usually they are cache hits, and the declarations for the target tag
        # always are: the pair resolution above already read them, above the
        # unload, where a read that DECIDES what gets driven has to happen.
        # Still cheap enough to leave the rest here rather than hoist them,
        # alongside the settings read, which sits above because it is neither
        # indexed by digest nor shared with anything else.
        fetch_policy=policy_of,
        fetch_measured_think=measured_think_of,
        on_sample=relay,
    )
    failed = [sample for sample in summary.samples if sample.error_code == "provider_unavailable"]
    # Scoped to the samples whose failure could poison a VERDICT. The thinking
    # group decides nothing about agency - it cannot blame a case, and its own
    # classifier already refuses to read an errored sample - so a blip confined
    # to it costs the think measurement and nothing else. Discarding the whole
    # run there would throw away a perfectly good agency verdict, and on the
    # automatic post-download check that means throwing away the only one a
    # model gets. A blip anywhere in the agency group keeps the full discard.
    for sample in failed:
        logger.warning(
            "Tool-check sample could not run: case=%s model=%s detail=%s",
            sample.case,
            sample.model,
            sample.detail,
        )
    if failed and all(sample.case in THINK_CASE_NAMES for sample in failed):
        logger.warning(
            "The think-mode measurement for %r could not be taken; recording the "
            "agency verdict without it",
            model,
        )
        summary.measured_think_mode = None
        failed = []
    if failed:
        # ANY such sample in the agency group, not just a whole run of them. A
        # sample that could not reach the model server measured the NETWORK,
        # and the harness scores it as a case failure - so one blip in a
        # two-sample fast check earns "deterministic", which the log then reads
        # as failure evidence and clamps the running model with. This check now
        # fires unattended after every download, which is exactly when the box
        # is busiest; discarding a partly-measured run is much cheaper than
        # quietly demoting a healthy model.
        #
        # The code is BROADER than its name, which is why the detail lines are
        # logged before anything is discarded: the provider raises this same
        # code for a connect failure, a generation timeout, a runner crash, and
        # any error line in the backend's own response body - and that last one
        # is where a model too large for the card reports that it needs more
        # memory than the machine has. Without this log, the one sentence that
        # names the real problem would be thrown away with the run.
        #
        # TODO: the proper fix is a distinct seam error code for an in-band
        # MODEL error (or a generation timeout) so that only true
        # unreachability discards a run, and an over-large model records the
        # honest "it cannot run here" verdict instead. Deliberately not taken
        # here: those codes are part of the chat wire contract and changing
        # them reaches the UI.
        raise ProviderUnavailableError(
            "The model server was unreachable during the check, or this model "
            "could not run on this machine - a model too large for the card "
            "reports the same way. Nothing was recorded, because either answer "
            "would describe the machine rather than the model. The server logs "
            "carry what the model server actually said."
        )
    # The mid-run reading answers the question actually being asked - where the
    # model sat while it was measured - and a reading taken now covers the runs
    # that never got one (nothing ran on the target, or every reading missed).
    residency = (
        during if during is not None and during.loaded else await _residency(provider, model)
    )
    representative, warning = assess_residency(residency)
    if not representative:
        # Dropped rather than published: a CPU-bound decode rate reported as
        # this model's throughput is a false fact, and the grid renders a
        # missing number honestly as "unmeasured".
        summary.tokens_per_second = None
    digest, thinking_digest = await _capture_digests(provider, model=model, thinking=thinking)
    runtime_version = await _runtime_version(provider)
    summary.digest = digest
    probe_id = await _persist(
        sessionmaker=sessionmaker,
        tenant=tenant,
        settings=settings,
        summary=summary,
        digest=digest,
        thinking_digest=thinking_digest,
        runtime_version=runtime_version,
        serving_endpoint=_serving_endpoint(provider),
        residency=residency,
        representative=representative,
    )
    logger.info(
        "tool check ran for tenant=%s user=%s: model=%r depth=%s earned=%s "
        "representative=%s probe=%s",
        tenant.tenant_id,
        tenant.user_id,
        model,
        summary.depth,
        summary.earned,
        representative,
        probe_id,
    )
    return VerifyOutcome(
        summary=summary,
        digest=digest,
        thinking_digest=thinking_digest,
        runtime_version=runtime_version,
        residency=residency,
        representative=representative,
        warning=warning,
        probe_id=probe_id,
    )


async def stream_verification(
    *,
    provider: ModelProvider,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    tenant: TenantContext,
    model: str,
    deep: bool,
) -> AsyncIterator[VerifyEvent]:
    """Run one verification, yielding a frame per sample and one terminal frame.

    Never raises past its own stream: by the time this runs the response has
    begun, and a stream that simply stops leaves a progress display frozen with
    no way to tell a finished check from a dead connection.

    The harness reports progress through a callback (it is a pure service and
    knows nothing about streams), so the run happens in a task and each
    finished sample crosses over on a queue. The task is cancelled if the
    consumer goes away, and the re-warm below still runs. A probe cancelled
    part-way leaves no trace; one cancelled at the very last frame has already
    committed its row, which is the right way round - the measurement was made,
    and only the report of it was lost.

    Releases the probe slot when the PROBE ends - after the re-warm - rather
    than leaving it to the caller alone. Both releases are needed and neither
    is redundant, because they cover opposite failures:

    - The request scope's release (see :func:`begin_verification`) covers a
      client that hangs up before the body is ever iterated. This generator is
      then never started, an unstarted generator's ``finally`` never runs, and
      without that release the slot would wedge until a restart.
    - This one covers the reverse. On a mid-stream disconnect the server
      unwinds the request stack immediately, while the heartbeat wrapper's
      producer task keeps draining this generator and the probe keeps
      generating. A request-scoped release alone would open the slot in that
      window, and the next click would start a second probe underneath the
      first - the exact contention the guard exists to prevent.

    The release is keyed on the holder and idempotent, so whichever of the two
    arrives second is a no-op rather than a way to cancel somebody else's probe.

    Yields:
        Progress events, then exactly one terminal done or error event.
    """
    # None is the end-of-run sentinel; samples are never None.
    queue: asyncio.Queue[CaseSample | None] = asyncio.Queue()

    async def _work() -> VerifyOutcome:
        try:
            return await run_verification(
                provider=provider,
                settings=settings,
                sessionmaker=sessionmaker,
                tenant=tenant,
                model=model,
                deep=deep,
                on_sample=queue.put_nowait,
            )
        finally:
            # Reached on success, failure, and cancellation alike: the consumer
            # must always stop waiting for another sample.
            queue.put_nowait(None)

    task = asyncio.create_task(_work())
    total = expected_sample_count(deep=deep)
    completed = 0
    try:
        while True:
            sample = await queue.get()
            if sample is None:
                break
            completed += 1
            yield VerifyProgressEvent(
                case=sample.case,
                model=sample.model,
                passed=sample.passed,
                detail=sample.detail,
                latency_s=sample.latency_s,
                completed=completed,
                total=max(total, completed),
            )
        yield done_event(await task)
    except ModelNotFoundError as exc:
        yield VerifyErrorEvent(message=str(exc), code="model_not_found")
    except ProviderUnavailableError as exc:
        yield VerifyErrorEvent(message=str(exc), code="provider_unavailable")
    except Exception:
        # Log the real failure server-side; never leak exception text.
        logger.exception("Tool check failed for %s", model)
        yield VerifyErrorEvent(message="The tool check failed unexpectedly.", code="internal_error")
    finally:
        if not task.done():
            task.cancel()
            with suppress(BaseException):
                await task
        # Unconditional, and outside every branch above: the probe evicted the
        # chat model and the embedder whatever became of its verdict.
        await _rewarm(provider, settings, sessionmaker, tenant)
        # LAST, and after the re-warm: the slot has to stay held for as long as
        # this process is still touching the card. The request scope releases
        # too, but it unwinds the moment a client disconnects - while this
        # generator is still being drained by the heartbeat wrapper's producer
        # task, still probing.
        release_verification(model)


async def stream_pull_with_check(
    *,
    provider: ModelProvider,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    tenant: TenantContext,
    tag: str,
    verify: bool,
) -> AsyncIterator[model_pull.PullEvent | VerifyEvent]:
    """Download a model and, when it lands, measure it - on one stream.

    One user action stays one connection: the roadmap's "a successful pull
    auto-runs the fast tool check" is only true from a user's point of view if
    they do not have to go and start it. The download's own terminal frame is
    forwarded FIRST and unchanged in its meaning, carrying only the flags that
    say whether anything follows it - so a check that fails afterwards can
    never be mistaken for a download that failed.

    A check is skipped, with a reason on that frame, when the caller asked for
    none, when another probe is already running (the guard is the same one the
    endpoint uses - an auto-check must not be the thing that evicts a model out
    from under a deliberate one), or when the runtime says this tag is not a
    chat model. That last one matters: the harness measures tool-calling, and
    running it against an embedder would spend GPU minutes proving that an
    embedder cannot call tools.

    Yields:
        The download's events, then the check's, when one runs.
    """
    verified = False
    # ONE guard around everything from the claim onwards. The claim is taken
    # while this generator is suspended at the ``done`` frame, and the very next
    # thing that happens is the download generator's own cleanup - a real await
    # point, and a real place for a client to hang up. A release scoped to the
    # verification loop alone would be jumped straight past there, wedging the
    # process-wide slot until a restart.
    try:
        # ``aclosing`` on both halves: a consumer that walks away mid-stream has
        # to reach the inner generators' cleanup NOW - the download's connection
        # to the model server, and the check's re-warm - not at garbage
        # collection.
        async with aclosing(model_pull.stream_pull(provider, tag=tag)) as download:
            async for event in download:
                if not isinstance(event, PullDoneEvent):
                    yield event
                    continue
                # The weights under this tag just changed, so every cached
                # answer about them is now about the OLD ones. Dropped here
                # rather than only after a probe writes a row, because the
                # paths that skip or fail the check are exactly where it
                # matters: a re-pull that changed the digest must revert to
                # unverified NOW, not when a five-minute TTL happens to lapse,
                # or the old row keeps certifying the new weights in between.
                # The capability report is dropped for the same reason and is
                # the more urgent of the two: it now decides whether an
                # tag is shown pixels, and a cached "yes" over
                # re-pulled weights fails in the unsafe direction.
                #
                # The stored declarations go with them. What actually retires
                # an operator's ceiling here is the digest listing the line
                # above drops (a fold re-reading it against the new weights
                # finds the declaring row bound to the old ones), so this third
                # call is symmetry rather than the mechanism: nothing on this
                # path WRITES a declaration, so the rows themselves cannot be
                # stale. It is here so the three caches a digest-bound answer
                # depends on are dropped together, instead of a later reader
                # having to work out which of them mattered.
                clear_earned_cache()
                clear_facts_cache()
                clear_policy_cache()
                reason: str | None = None
                if verify:
                    # Nobody asked when ``verify`` is false, so there is nothing
                    # to explain; every other path either checks or says why it
                    # did not.
                    reason = await _wrong_role_reason(provider, settings.model_provider, tag)
                    if reason is None:
                        try:
                            claim_verification(tag)
                        except ConflictError as exc:
                            reason = str(exc)
                        else:
                            verified = True
                yield PullDoneEvent(
                    model=event.model,
                    status=event.status,
                    verifying=verified,
                    verify_skipped_reason=reason,
                )
        if not verified:
            return
        async with aclosing(
            stream_verification(
                provider=provider,
                settings=settings,
                sessionmaker=sessionmaker,
                tenant=tenant,
                model=tag,
                deep=False,
            )
        ) as check:
            async for event in check:
                yield event
    finally:
        # This claim was taken inside the stream rather than by the request
        # scope, so it is released here - and the pairing holds: an unstarted
        # generator never reached the claim either. Releasing by TAG so a stream
        # that never claimed cannot cancel somebody else's probe on its way out.
        if verified:
            release_verification(tag)


async def _wrong_role_reason(provider: ModelProvider, provider_name: str, tag: str) -> str | None:
    """Say why the runtime's report rules out a tool check, or None.

    Only a report that positively says these weights cannot hold a conversation
    skips. A tag whose probe FAILED is checked anyway - "the runtime would not
    say" is not evidence, and the free-text pull is exactly the case somebody
    most needs an answer about.
    """
    report = await report_for(provider, provider_name, tag)
    if report is None or report.completion:
        return None
    kind = "embedding" if report.embedding else "non-chat"
    return (
        f"'{tag}' is reported as {article_for(kind)} {kind} model, and the tool check "
        "only measures chat models."
    )
