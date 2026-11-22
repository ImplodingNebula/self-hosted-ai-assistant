"""Running the embedder-check harness against a live deployment.

The harness (:mod:`app.services.embedder_check`) is pure: it embeds a corpus
and returns bands. This module is everything around that - which tag to
measure, where the embedding work is allowed to land, what binds the reading to
weights, and what has to be put back afterwards - and it is the only place that
order lives, so a terminal run and an API call cannot disagree about what was
measured.

The order, and why each step is where it is:

1. **Decide everything a run can be refused for, first.** An unusable name,
   weights the backend does not hold, a tag the model server reports cannot
   embed: all of them are answered before any embedding happens, so a refusal
   is a refusal rather than a report with nothing in it.
2. **One probe at a time on this machine, in BOTH directions.** The embedder
   check is seconds rather than minutes, but it LOADS a model, and a tool check
   is busy deliberately unloading them to measure a fresh load. Either one
   started underneath the other puts a model back on the card mid-measurement,
   and the damage is asymmetric: a disturbed tool-check sample scores as a case
   FAILURE, and failure evidence clamps a model's earned agency whether or not
   the run was representative. So a check takes the tool check's process-wide
   slot as well as its own - see :func:`claim_probe_slots`.
3. **Embed where retrieval embeds.** ``EMBEDDING_DEVICE`` is honored, so a
   deployment that keeps its embedder off the GPU (a tight-VRAM box does,
   precisely so a retrieval cannot evict the chat model) measures it there
   rather than measuring a placement it never serves from.
4. **Record it**, bound to the weights digest and the runtime version.
5. **Put the active embedder back**, when the run measured a DIFFERENT tag. A
   diagnostic that leaves the deployment cold has made the app slower for the
   next person to use it.

What this module deliberately does NOT do is apply anything. A floors proposal
is a number for a human to put in configuration (``RAG_MIN_SCORE`` /
``RAG_REL_MARGIN``) after reading how it was derived. Retrieval's floors decide
what every answer in the deployment is allowed to cite, and a measurement that
rewrote them on its own would change every answer on the strength of one probe
corpus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.errors import BadRequestError, ConflictError, NotFoundError
from app.core.tenancy import TenantContext
from app.providers.base import (
    EmbedOptions,
    ModelProvider,
    ProviderUnavailableError,
    installed_tag_matches,
    tag_aliases,
)
from app.repositories.model_probes import (
    ModelProbeRepository,
    gate_payload,
    row_from_embedder_summary,
)
from app.schemas.models import (
    EmbedderArmReport,
    EmbedderCheckResponse,
    EmbedderComposedGate,
    EmbedderConfiguredFloors,
    EmbedderFloorCheck,
    EmbedderFloorProposal,
    EmbedderFloorsReport,
    EmbedderPrefixReport,
    EmbedderQueryGate,
    EmbedderScoreBand,
    EmbedderTripletReport,
)
from app.services import model_verify
from app.services.embedder_check import (
    ArmMeasurement,
    ComposedGate,
    EmbedderCheckSummary,
    FloorCheck,
    ScoreBand,
    compose_gate,
    resolve_query_prefix,
    run_embedder_check,
    validate_floors,
)
from app.services.embedding_resolution import EmbeddingChoice, resolve_embedding
from app.services.model_facts import report_for
from app.services.model_names import require_model_name
from app.services.model_resolution import (
    EmbeddingConfig,
)
from app.services.warmup import warm_embedder
from app.services.workspace_defaults import load_defaults

logger = logging.getLogger(__name__)

# The embedder currently being checked, or None. One slot for the process, for
# the same reason the tool check has one: two runs contend for the same
# hardware, and the in-process scope is the honest one because the case this
# exists for is a second click landing on the same worker.
_running: str | None = None


@dataclass(frozen=True)
class ConfiguredFloors:
    """The floors THIS deployment retrieves with, measured against the bands.

    The harness judges what the model-quirk seed DECLARES for an embedder,
    which is a statement about the model. Retrieval reads
    ``settings.rag_min_score`` and ``settings.rag_rel_margin``, which is a
    statement about this deployment - and the two coincide only until somebody
    edits an env file. Without this block a run could answer "the floors hold"
    while the numbers actually in force sat above the related band, dropping
    true answers on every turn, which is the precise failure the check exists
    to catch.

    Attributes:
        min_score: ``RAG_MIN_SCORE`` as this process sees it.
        rel_margin: ``RAG_REL_MARGIN`` as this process sees it.
        composed: What these two, applied together, would actually retrieve -
            the verdict, simulated exactly as ``rag_service.retrieve`` composes
            them.
        checks: Band diagnostics for the same floor. Informational; see
            :func:`~app.services.embedder_check.validate_floors`.
        hold: The composed gate's answer.
        matches_declared: Whether the configured pair is the declared pair. When
            it is, the two verdicts are one measurement reported twice; when it
            is not, this is the one that describes what the deployment does.
    """

    min_score: float
    rel_margin: float
    composed: ComposedGate
    checks: tuple[FloorCheck, ...]
    hold: bool
    matches_declared: bool


@dataclass
class EmbedderCheckOutcome:
    """What one embedder-check run measured, and what became of it.

    Attributes:
        summary: The harness reading - bands, prefix comparison, floors verdict.
        configured: How the floors actually in force fare against those bands.
        digest: The weights the tag pointed at, when the runtime exposes one.
        runtime_version: The serving runtime this run measured under.
        vram_bytes: The embedder's resident VRAM after the run, when reported.
        probe_id: The measurement row it was recorded as. None when the caller
            asked for no persistence, and also when the write failed - the
            measurement happened either way, which is why it is reported
            beside the summary rather than instead of it.
        active: Whether the tag measured is the one this deployment actually
            embeds with. Carried on the outcome rather than recomputed by the
            caller because the run already resolved it, and a second
            resolution is a chance for the report to disagree with the run
            that produced it.
    """

    summary: EmbedderCheckSummary
    configured: ConfiguredFloors
    digest: str | None
    runtime_version: str | None
    vram_bytes: int | None
    probe_id: str | None
    active: bool = False


def check_in_flight() -> str | None:
    """Return the embedder currently being checked, or None."""
    return _running


def clear_embedder_checks() -> None:
    """Drop this module's in-flight claim (test isolation only)."""
    global _running
    _running = None


def claim_check(tag: str) -> None:
    """Claim the single embedder-check slot, or refuse.

    The check and the claim happen with no ``await`` between them, so on the
    event loop this is atomic - there is no window for a second request to slip
    between reading the slot and taking it.

    Args:
        tag: The embedder about to be checked.

    Raises:
        ConflictError: A check is already running. The message names the model,
            because "a check is already running" without saying which one reads
            like a bug in the page.
    """
    global _running
    if _running is not None:
        raise ConflictError(
            f"An embedder check is already running on '{_running}'. It loads a "
            "model, so only one runs at a time - wait for it to finish."
        )
    _running = tag


def claim_probe_slots(tag: str) -> None:
    """Take BOTH probe slots for ``tag``: this module's, and the tool check's.

    Two slots rather than one, and neither is redundant.

    This module's own slot answers a second embedder check precisely ("a check
    is already running on X"). The TOOL CHECK's process-wide slot
    (:func:`app.services.model_verify.claim_verification`) is what makes the
    exclusion MUTUAL, and that is the half that matters: a tool check
    deliberately unloads the embedder to measure a fresh load, so one started
    underneath a running embedder check would have a model loaded back onto the
    card mid-measurement. What that costs is not a wasted diagnostic - a
    disturbed sample scores as a case FAILURE, and failure evidence clamps a
    model's earned agency whether or not the run was representative
    (:func:`app.services.earned._resolve_earned`). A persisted clamp caused by
    an unrelated diagnostic is the outcome this guard exists to prevent.

    Holding the tool check's slot is the only way to close that direction from
    here, since the claim it would have to consult lives in its module and this
    slice does not change it. The cost is one imprecise sentence: a tool check
    refused while an embedder check holds the slot is told a "tool check" is
    running on an embedder tag. A confusing message is a much smaller price
    than a wrong verdict in the log.

    The order is what keeps the refusals readable. This module's slot is taken
    FIRST, so a second embedder check is refused by the message that names an
    embedder check; only then is the tool check's slot consulted, so an
    embedder check refused during a probe is told a TOOL check is in the way.
    Taking the tool check's slot first would answer both cases in its wording.
    No ``await`` runs between the two, so the pair is atomic on the event loop.

    Args:
        tag: The embedder about to be checked.

    Raises:
        ConflictError: Either slot is already held.
    """
    claim_check(tag)
    try:
        probing = model_verify.verification_in_flight()
        if probing is not None:
            raise ConflictError(
                f"A tool check is running on '{probing}'. It unloads models to "
                "measure a fresh load, so an embedder check has to wait for it."
            )
        model_verify.claim_verification(tag)
    except ConflictError:
        release_check(tag)
        raise


def release_probe_slots(tag: str) -> None:
    """Release both slots held for ``tag``; safe to call twice, and holder-keyed."""
    model_verify.release_verification(tag)
    release_check(tag)


def release_check(tag: str) -> None:
    """Release this module's slot for ``tag``; safe to call twice.

    Keyed on the holder, so a caller can only ever release its OWN claim: a
    path that released a claim it never took would hand the slot away
    mid-run with nothing anywhere saying so. A mismatch is logged rather than
    raised, because the release always runs in a ``finally`` where raising
    would replace whatever outcome the caller was reporting.

    Args:
        tag: The model the caller claimed.
    """
    global _running
    if _running is not None and _running != tag:
        logger.warning(
            "Refusing to release the embedder-check slot for %r: it is held by %r.", tag, _running
        )
        return
    _running = None


def is_active_embedder(tag: str, active: str | None) -> bool:
    """True when ``tag`` names the embedder this deployment actually queries with.

    Compared through :func:`~app.providers.base.tag_aliases` on BOTH sides,
    because both sides are written-down configuration: a request may spell a
    tag ``qwen3-embedding:0.6b`` where a stored choice spells it
    ``library/qwen3-embedding:0.6b``, and treating those as different models
    would measure the active embedder while reporting it as a bystander.

    Args:
        tag: The tag being asked about.
        active: What the resolution chain settled on, or None when it settled
            on nothing (see
            :func:`app.services.embedding_resolution.resolve_embedding`).
    """
    if not active:
        return False
    return bool(tag_aliases(tag) & tag_aliases(active))


async def _wrong_role_reason(provider: ModelProvider, provider_name: str, tag: str) -> str | None:
    """Say why the runtime's report rules out an embedder check, or None.

    Only a report that positively says these weights cannot embed refuses. A
    tag whose probe FAILED is checked anyway: "the runtime would not say" is
    not evidence, and a hand-pulled embedder on a half-answering server is
    exactly the case somebody most needs bands for.
    """
    report = await report_for(provider, provider_name, tag)
    if report is None or report.embedding:
        return None
    kind = "chat" if report.completion else "non-embedding"
    return (
        f"The model server reports '{tag}' as a {kind} model, and the embedder "
        "check only measures embedding models."
    )


def resolve_target(active: str | None, name: str | None) -> str:
    """Decide which tag a check will measure.

    Args:
        active: The embedder the resolution chain settled on, or None.
        name: An explicit tag, or None/blank for the active embedder.

    Returns:
        The requested tag verbatim, or the active embedder's.

    Raises:
        BadRequestError: No tag was named and nothing resolves an embedder, so
            there is nothing to measure. Deliberately not a 404: the
            configuration is the thing missing, not a model.
    """
    requested = (name or "").strip()
    if requested:
        return requested
    if not active:
        raise BadRequestError(
            "No embedding model is configured, so there is nothing to check. "
            "Name a model explicitly to measure one anyway."
        )
    return active


async def begin_embedder_check(
    provider: ModelProvider, *, provider_name: str, active: str | None, name: str | None
) -> str:
    """Decide everything a check can be refused for, then claim the slot.

    Split out of the run for the same reason the tool check's is: all of it
    happens BEFORE any work, so a refusal arrives as problem+json rather than
    as a report with empty bands in it.

    Args:
        provider: The model seam that holds the weights.
        provider_name: Its configured name, for the facts cache key.
        active: The embedder the resolution chain settled on, or None. Resolved
            by the caller, which is the layer that holds a workspace and a
            vector store.
        name: The requested tag, or None/blank for the active embedder.

    Returns:
        The normalized tag, now claimed - the caller MUST pair this with
        :func:`release_probe_slots` in a scope that unwinds on every outcome.

    Raises:
        UnprocessableEntityError: The name failed validation.
        BadRequestError: The runtime reports this tag cannot embed, or nothing
            was named and no embedder is configured.
        NotFoundError: The backend does not hold these weights. Checking a
            model that is not here would report a failure of the model rather
            than the plain fact that nothing was downloaded.
        ConflictError: A check is already running, or a tool check is.
        ProviderUnavailableError: The backend could not be reached.
    """
    tag = require_model_name(resolve_target(active, name)).text
    wrong_role = await _wrong_role_reason(provider, provider_name, tag)
    if wrong_role is not None:
        raise BadRequestError(wrong_role)
    installed = {info.name for info in await provider.list_models()}
    if not any(installed_tag_matches(alias, installed) for alias in sorted(tag_aliases(tag))):
        raise NotFoundError(f"'{tag}' is not installed, so there is nothing to check.")
    claim_probe_slots(tag)
    return tag


async def _capture_digest(provider: ModelProvider, model: str) -> str | None:
    """Read the weights digest ``model`` currently points at (best-effort).

    A tag is mutable - this repo has already been bitten by a whole tag family
    being re-pointed - so bands recorded against a tag alone say nothing a
    re-pull cannot quietly invalidate. Read fresh rather than through the
    cached identity map: the reading is worth binding correctly, and this runs
    once per check rather than once per turn.
    """
    try:
        for info in await provider.list_models():
            if installed_tag_matches(model, {info.name}):
                return info.digest
    except ProviderUnavailableError:
        # The measurement already happened; losing the binding is a weaker
        # outcome than losing the run, so it degrades to "unmeasured".
        logger.info("Could not read the installed digest after checking %r.", model)
    return None


async def _runtime_version(provider: ModelProvider) -> str | None:
    """Read the serving runtime's version, degrading any failure to unmeasured."""
    try:
        return await provider.runtime_version()
    except Exception:  # noqa: BLE001 - a measurement must never break the measuring
        logger.info("Could not read the runtime version after an embedder check.", exc_info=True)
        return None


async def _resident_vram(provider: ModelProvider, model: str) -> int | None:
    """Read the embedder's resident VRAM, degrading the same way.

    Often zero or absent by design rather than by failure: the tight-VRAM tiers
    pin the embedder to the CPU precisely so it holds no VRAM at all.
    """
    try:
        return await provider.resident_vram_bytes(model)
    except Exception:  # noqa: BLE001 - see :func:`_runtime_version`
        logger.info("Could not read resident VRAM for %r.", model, exc_info=True)
        return None


async def _persist(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    tenant: TenantContext | None,
    settings: Settings,
    summary: EmbedderCheckSummary,
    configured: ConfiguredFloors,
    digest: str | None,
    runtime_version: str | None,
    vram_bytes: int | None,
) -> str | None:
    """Append the measurement row; a failed write is loud but not the run's error.

    Same rule the tool check applies: the measurement already happened, and
    reporting it as a failed check would send someone to re-run the probe when
    the thing to fix is the database.
    """
    try:
        async with sessionmaker() as session:
            row = row_from_embedder_summary(
                summary,
                provider_name=settings.model_provider,
                digest=digest,
                runtime_version=runtime_version,
                vram_bytes=vram_bytes,
                configured={
                    "min_score": configured.min_score,
                    "rel_margin": configured.rel_margin,
                    "hold": configured.hold,
                    "matches_declared": configured.matches_declared,
                    # Flattened by the repository's own helper, so the two
                    # composed blocks in one row cannot drift into different
                    # shapes.
                    "composed": gate_payload(configured.composed),
                    "checks": [
                        {"name": check.name, "passed": check.passed, "detail": check.detail}
                        for check in configured.checks
                    ],
                },
            )
            if tenant is not None:
                # A request has an actor where a terminal run does not.
                # Constant today, and recording it anyway is the point - the
                # trail becomes meaningful the moment real identity populates it.
                row.triggered_by_tenant_id = tenant.tenant_id
                row.triggered_by_user_id = tenant.user_id
            await ModelProbeRepository(session).add(row)
            await session.commit()
            return str(row.id)
    except Exception:
        logger.exception("Embedder-check reading for %r was NOT recorded", summary.model)
        return None


async def _resolve_active(
    provider: ModelProvider,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession] | None,
    tenant: TenantContext | None,
) -> EmbeddingChoice:
    """Resolve the embedder a check has to work around, through the usual chain.

    The workspace rung is reachable only when this run has both a session
    factory and a principal - a terminal run has neither, and resolves the
    deployment's answer instead. The INVENTORY rung is always reachable, which
    is what stops this answer diverging from the one the refusal was decided
    against: a deployment whose embedder comes from "the server holds exactly
    one" would otherwise measure the right tag, report it as a bystander, and
    skip a re-warm it did not need.

    No vector store is passed, and that IS a divergence, knowingly: the
    evidence rung would put a store round trip on a path that measures the
    model server. It costs a deployment whose embedder is known only from its
    stored passages the same misreport - narrower than before, and the honest
    fix is a store the caller already holds, which this signature admits when
    one is worth threading.
    """
    stored = None
    if sessionmaker is not None and tenant is not None:
        async with sessionmaker() as session:
            stored = await load_defaults(session, tenant)
    return await resolve_embedding(
        settings=settings,
        stored=stored,
        provider=provider,
        provider_name=settings.model_provider,
    )


async def _restore_active_embedder(
    provider: ModelProvider, settings: Settings, config: EmbeddingConfig | None
) -> None:
    """Reload the deployment's own embedder after a check measured another one.

    Never raises: a warm request that fails must not replace the outcome the
    caller is waiting for, and a cold embedder recovers on its own with the
    next retrieval - slowly, which is the whole reason this is here.
    """
    if config is None:
        return
    try:
        await warm_embedder(provider, settings, embedding=config)
    except Exception:  # noqa: BLE001 - a diagnostic must not fail on cleanup
        logger.warning("Could not re-warm the active embedder after a check.", exc_info=True)


def judge_configured_floors(summary: EmbedderCheckSummary, settings: Settings) -> ConfiguredFloors:
    """Measure the floors this deployment retrieves with against the bands.

    Reuses the harness's own :func:`~app.services.embedder_check.compose_gate`
    and :func:`~app.services.embedder_check.validate_floors` rather than
    restating the rules, so the configured pair and a declared pair are held to
    exactly the same standard - a second copy of the thresholds here would
    drift, and the drift would show up as a report that contradicts itself.

    Judged on the WRAPPED arm, because that is the arm live retrieval produces.

    Args:
        summary: The completed harness run.
        settings: The active settings, which is where the live floors are.

    Returns:
        The configured pair, what it retrieves, and whether it is the declared
        one.
    """
    declared = summary.floors.declared
    composed = compose_gate(
        summary.wrapped,
        min_score=settings.rag_min_score,
        rel_margin=settings.rag_rel_margin,
    )
    return ConfiguredFloors(
        min_score=settings.rag_min_score,
        rel_margin=settings.rag_rel_margin,
        composed=composed,
        checks=validate_floors(summary.wrapped, min_score=settings.rag_min_score),
        hold=composed.holds,
        matches_declared=(
            declared.min_score == settings.rag_min_score
            and declared.rel_margin == settings.rag_rel_margin
        ),
    )


def _gate_report(gate: ComposedGate) -> EmbedderComposedGate:
    """Project a composed-gate verdict onto the wire shape."""
    return EmbedderComposedGate(
        min_score=gate.min_score,
        rel_margin=gate.rel_margin,
        queries=[
            EmbedderQueryGate(
                name=query.name,
                top_score=query.top_score,
                top_passage=query.top_passage,
                effective_floor=query.effective_floor,
                related_retrieved=query.related_retrieved,
                supporting_retrieved=query.supporting_retrieved,
                noise_retrieved=list(query.noise_retrieved),
                noise_clearance=query.noise_clearance,
            )
            for query in gate.queries
        ],
        leaks=list(gate.leaks),
        related_lost=list(gate.related_lost),
        supporting_pruned=list(gate.supporting_pruned),
        holds=gate.holds,
        detail=gate.detail,
    )


def _band_report(band: ScoreBand) -> EmbedderScoreBand:
    """Project one score band onto the wire shape."""
    return EmbedderScoreBand(
        count=band.count, minimum=band.minimum, maximum=band.maximum, mean=band.mean
    )


def _arm_report(arm: ArmMeasurement) -> EmbedderArmReport:
    """Project one measured arm onto the wire shape."""
    return EmbedderArmReport(
        related=_band_report(arm.related),
        supporting=_band_report(arm.supporting),
        unrelated=_band_report(arm.unrelated),
        separation=arm.separation,
        triplets=[
            EmbedderTripletReport(
                name=triplet.name,
                related=triplet.related,
                supporting=triplet.supporting,
                best_unrelated=triplet.best_unrelated,
                best_unrelated_passage=triplet.best_unrelated_passage,
            )
            for triplet in arm.triplets
        ],
    )


def build_report(outcome: EmbedderCheckOutcome, *, active: bool) -> EmbedderCheckResponse:
    """Build the response body from a finished run.

    Lives here rather than in the router for the same reason the verify job's
    terminal frame does: the run's shape and the shape a client reads are one
    decision, and splitting them is how the two drift.

    Args:
        outcome: The completed run.
        active: Whether the measured tag is the deployment's own embedder.

    Returns:
        The :class:`~app.schemas.models.EmbedderCheckResponse` body.
    """
    summary = outcome.summary
    floors = summary.floors
    return EmbedderCheckResponse(
        model=summary.model,
        active=active,
        digest=outcome.digest,
        runtime_version=outcome.runtime_version,
        harness_version=summary.harness_version,
        prefix=EmbedderPrefixReport(
            source=summary.prefix.source,
            prefix=summary.prefix.prefix,
            wrapped_separation=summary.prefix.wrapped_separation,
            raw_separation=summary.prefix.raw_separation,
            delta=summary.prefix.delta,
            improves_separation=summary.prefix.improves_separation,
            degenerate=summary.prefix.degenerate,
        ),
        wrapped=_arm_report(summary.wrapped),
        raw=_arm_report(summary.raw),
        floors=EmbedderFloorsReport(
            declared_min_score=floors.declared.min_score,
            declared_rel_margin=floors.declared.rel_margin,
            source=floors.declared.source,
            composed=None if floors.composed is None else _gate_report(floors.composed),
            checks=[
                EmbedderFloorCheck(name=check.name, passed=check.passed, detail=check.detail)
                for check in floors.checks
            ],
            hold=floors.hold,
            proposal=None
            if floors.proposal is None
            else EmbedderFloorProposal(
                min_score=floors.proposal.min_score,
                rel_margin=floors.proposal.rel_margin,
                derivation=floors.proposal.derivation,
                validates=floors.proposal.validates,
                caveat=floors.proposal.caveat,
            ),
        ),
        configured=EmbedderConfiguredFloors(
            min_score=outcome.configured.min_score,
            rel_margin=outcome.configured.rel_margin,
            composed=_gate_report(outcome.configured.composed),
            checks=[
                EmbedderFloorCheck(name=check.name, passed=check.passed, detail=check.detail)
                for check in outcome.configured.checks
            ],
            hold=outcome.configured.hold,
            matches_declared=outcome.configured.matches_declared,
        ),
        probe_id=outcome.probe_id,
    )


async def run_embedder_check_job(
    *,
    provider: ModelProvider,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession] | None,
    tenant: TenantContext | None,
    model: str,
) -> EmbedderCheckOutcome:
    """Measure ``model``, record the reading, and put the deployment back.

    Assumes the slot is already claimed (see :func:`begin_embedder_check`); the
    release belongs to the scope that took it.

    Args:
        provider: The model seam to drive.
        settings: The active settings (the provider name rows key on, and the
            tier the re-warm reads).
        sessionmaker: Session factory for the measurement-log write, or None to
            skip persistence entirely (a hermetic run, or a stack whose
            database is down but whose embedder is worth measuring).
        tenant: The acting principal, recorded as audit-only columns. None for
            a terminal run, which has no actor.
        model: The validated tag to measure.

    Returns:
        The :class:`EmbedderCheckOutcome`.

    Raises:
        ProviderUnavailableError: The embedding backend was unreachable, or the
            model would not load. Propagated rather than recorded: bands from a
            run that could not embed would describe the outage, and a floors
            verdict taken on them could invalidate correct configuration.
    """
    config = (await _resolve_active(provider, settings, sessionmaker, tenant)).config
    active = is_active_embedder(model, config.model if config is not None else None)
    # The deployment's placement is honored whichever tag is measured: it is
    # this machine's policy for where embedding work runs, and probing a
    # candidate embedder onto a card the policy keeps it off would evict the
    # chat model to measure something the deployment would never do.
    options = EmbedOptions(device=settings.embedding_device)
    prefix = resolve_query_prefix(model)
    summary = await run_embedder_check(
        provider,
        model=model,
        query_prefix=prefix.prefix,
        prefix_source=prefix.source,
        options=options,
    )
    configured = judge_configured_floors(summary, settings)
    digest = await _capture_digest(provider, model)
    runtime_version = await _runtime_version(provider)
    vram_bytes = await _resident_vram(provider, model)
    summary.digest = digest
    probe_id: str | None = None
    if sessionmaker is not None:
        probe_id = await _persist(
            sessionmaker=sessionmaker,
            tenant=tenant,
            settings=settings,
            summary=summary,
            configured=configured,
            digest=digest,
            runtime_version=runtime_version,
            vram_bytes=vram_bytes,
        )
    if not active:
        # The check loaded weights the deployment does not serve from, which on
        # a backend with a small residency budget is exactly how the real
        # embedder gets evicted. Put back the one that was resolved above, not
        # whatever a fresh resolution would find: they are the same answer, and
        # asking again here would let a cleanup warm a different model from the
        # one the check displaced.
        await _restore_active_embedder(provider, settings, config)
    logger.info(
        "embedder check ran for tenant=%s user=%s: model=%r prefix=%s floors_hold=%s probe=%s",
        None if tenant is None else tenant.tenant_id,
        None if tenant is None else tenant.user_id,
        model,
        summary.prefix.source,
        summary.floors.hold,
        probe_id,
    )
    return EmbedderCheckOutcome(
        summary=summary,
        configured=configured,
        digest=digest,
        runtime_version=runtime_version,
        vram_bytes=vram_bytes,
        probe_id=probe_id,
        active=active,
    )
