"""The model_probes repository - deliberately MACHINE-scoped, not tenant-scoped.

The one sanctioned exception to the tenant-injecting base repository
(owner-approved 2026-07-15): a probe row measures the shared model server, so
scoping it to a tenant would fragment one machine's measurement history into
per-tenant shards that each answer the capability question wrongly. The
triggering tenant/user ride along as audit-only columns; no method filters by
them, and none accepts an identity argument at all - there is nothing here an
identity could scope.

Append-only by design: verdicts are history ("passed on the previous runtime,
fails on this one" IS the data), so there is no update method to misuse.

The log holds runs from TWO harnesses now, told apart by ``depth``. The
tool-check writes ``fast`` and ``deep``, which the agency resolver reads; the
embedder-check writes ``embedder``, which it does not - an embedder never
drives a turn, so its row has no business in a decision about tool-calling.
That separation is structural rather than conventional: every read on the
agency path filters to a depth by name, so a depth nobody asks for is invisible
to it, and the constant naming this one lives with the harness that measures it
(:data:`app.services.embedder_check.PROBE_DEPTH`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import ModelProbe

if TYPE_CHECKING:  # the harnesses stay import-free of persistence; only types cross
    from app.providers.base import ModelResidency
    from app.services.embedder_check import (
        ArmMeasurement,
        ComposedGate,
        EmbedderCheckSummary,
        FloorsVerdict,
        ScoreBand,
    )
    from app.services.tool_check import ToolCheckSummary


def row_from_summary(
    summary: ToolCheckSummary,
    *,
    provider_name: str,
    digest: str | None,
    thinking_digest: str | None = None,
    runtime_version: str | None,
    serving_endpoint: str | None,
    vram_bytes: int | None,
    residency: ModelResidency | None = None,
    representative: bool = True,
) -> ModelProbe:
    """Map one harness summary onto a measurement row.

    Lives beside the repository, not inside the harness: the harness is a pure
    service that must not know persistence exists, while this module's whole
    job is the shape verdicts take at rest. Samples are stored compact (case,
    tag, verdict, detail, and the serve-time options it ran at) - the
    latency/token numbers that matter roll up into ``tokens_per_second``, and a
    probe row is a verdict record, not a trace.

    Args:
        summary: The completed harness run.
        provider_name: The MODEL_PROVIDER selector value ("ollama", ...).
        digest: The weights digest the CHAT tag resolved to, when exposed.
        thinking_digest: The thinking tag's digest on a pair-routed tier -
            case (h)'s verdict is measured on those weights (stored in
            ``cases``; the column key stays the chat tag's digest).
        runtime_version: The serving runtime's version, when exposed.
        serving_endpoint: WHERE the run happened - the backend's host identity
            from :meth:`~app.providers.base.ModelProvider.serving_endpoint`,
            credentials already stripped. Required with no default, so a new
            writer has to answer the question rather than silently record an
            unbound verdict; None is the honest answer for a backend that
            names no host, and a row carrying it certifies nothing.
        vram_bytes: The model's observed resident VRAM after the run, if known.
        residency: Where the model sat while it ran, when the runtime says.
        representative: Whether the run describes how the deployment actually
            serves. Recorded, not decided, here: what counts as a spill worth
            marking is a judgement about hardware, and it belongs with the job
            that took the reading (``app.services.model_verify``). A default
            of True keeps a run with no residency reading - the CLI's, and any
            backend that reports none - exactly as trusted as it was before
            this column existed.

    Returns:
        An unsaved :class:`ModelProbe` row for :meth:`ModelProbeRepository.add`.
    """
    return ModelProbe(
        provider=provider_name,
        model=summary.model,
        digest=digest,
        runtime_version=runtime_version,
        serving_endpoint=serving_endpoint,
        harness_version=summary.harness_version,
        depth=summary.depth,
        earned_agency=summary.earned,
        thinking_with_tools=summary.thinking_with_tools,
        tokens_per_second=summary.tokens_per_second,
        vram_bytes=vram_bytes,
        cases={
            "tallies": {name: list(counts) for name, counts in summary.tallies.items()},
            "samples": [
                {
                    "case": sample.case,
                    "model": sample.model,
                    "passed": sample.passed,
                    "detail": sample.detail,
                    # The two serve-time options this sample ran at, so the row
                    # says which CONFIGURATION it measured and not only which
                    # weights. A deployment can move either without touching
                    # the weights (a workspace ceiling moves the window; a
                    # confirmed declaration or a re-pull moves the think flag),
                    # and a verdict that named neither could not be read against
                    # the turns it certifies. ``think`` carries None for the
                    # flag OMITTED, which is a third request rather than a
                    # missing field. Descriptive only: nothing reads either back
                    # into a decision, and a later change to either neither
                    # stales nor clamps this row.
                    "num_ctx": sample.num_ctx,
                    "think": sample.think,
                }
                for sample in summary.samples
            ],
            "thinking_model": summary.thinking_model,
            # What the run WATCHED the checked tag do with the think flag, as
            # opposed to what any declaration claims. Digest-bound like every
            # other fact on this row, which is the point: a tag re-pulled onto
            # weights that reason differently retires this measurement instead
            # of carrying it forward. None when the thinking group could not be
            # classified (a provider error on either half), and a reader must
            # keep that apart from the three modes.
            "measured_think_mode": summary.measured_think_mode,
            # The pair's OTHER weights: case (h)'s thinking_with_tools verdict
            # is measured on the thinking tag, so a re-pull of THAT tag must
            # also read as "these weights were never verified". The row's main
            # digest column stays the chat tag's (the primary lookup key); a
            # digest-bound consumer of thinking_with_tools checks this one too.
            "thinking_digest": thinking_digest,
            # Whether this run describes how the deployment serves. A verdict
            # measured with the model spilled off the GPU is kept and its
            # failures still count, but it must not stand as the CERTIFICATE
            # that lifts a model under AGENCY_REQUIRE_VERIFICATION - so the
            # fact travels with the row rather than being lost with the run.
            "representative": representative,
            "residency": None
            if residency is None
            else {
                "loaded": residency.loaded,
                "total_bytes": residency.total_bytes,
                "vram_bytes": residency.vram_bytes,
                "gpu_fraction": (
                    None if residency.gpu_fraction is None else round(residency.gpu_fraction, 3)
                ),
            },
        },
    )


def _band_payload(band: ScoreBand) -> dict:
    """Flatten one score band for storage."""
    return {
        "count": band.count,
        "minimum": band.minimum,
        "maximum": band.maximum,
        "mean": band.mean,
    }


def _arm_payload(arm: ArmMeasurement) -> dict:
    """Flatten one prefix arm - its bands, its corridor, its per-triplet detail."""
    return {
        "related": _band_payload(arm.related),
        "supporting": _band_payload(arm.supporting),
        "unrelated": _band_payload(arm.unrelated),
        "separation": arm.separation,
        "triplets": [
            {
                "name": triplet.name,
                "related": triplet.related,
                "supporting": triplet.supporting,
                "best_unrelated": triplet.best_unrelated,
                "best_unrelated_passage": triplet.best_unrelated_passage,
            }
            for triplet in arm.triplets
        ],
    }


def gate_payload(gate: ComposedGate) -> dict:
    """Flatten the composed-gate verdict - the answer that decides.

    Public because the row carries TWO composed gates - the declared pair's and
    the one the deployment actually retrieves with - and the second is assembled
    by the caller that knows what is configured. One flattener for both, so the
    two blocks cannot drift into different shapes inside one row.

    The per-query simulation is kept, not just the summary: when a gate starts
    leaking, the row has to say which query and which passage, or the next
    reader is left re-running the probe to find out.
    """
    return {
        "min_score": gate.min_score,
        "rel_margin": gate.rel_margin,
        "holds": gate.holds,
        "detail": gate.detail,
        "leaks": list(gate.leaks),
        "related_lost": list(gate.related_lost),
        "supporting_pruned": list(gate.supporting_pruned),
        "queries": [
            {
                "name": query.name,
                "top_score": query.top_score,
                "top_passage": query.top_passage,
                "effective_floor": query.effective_floor,
                "related_retrieved": query.related_retrieved,
                "supporting_retrieved": query.supporting_retrieved,
                "noise_retrieved": list(query.noise_retrieved),
                "noise_clearance": query.noise_clearance,
            }
            for query in gate.queries
        ],
    }


def _floors_payload(floors: FloorsVerdict) -> dict:
    """Flatten the floors verdict: what was declared, what held, what is proposed."""
    return {
        "declared": {
            "min_score": floors.declared.min_score,
            "rel_margin": floors.declared.rel_margin,
            "source": floors.declared.source,
        },
        # The verdict. ``checks`` beneath it are band DIAGNOSTICS and decide
        # nothing - a row where they disagree with ``hold`` is the ordinary
        # case, not a contradiction.
        "composed": None if floors.composed is None else gate_payload(floors.composed),
        "checks": [
            {"name": check.name, "passed": check.passed, "detail": check.detail}
            for check in floors.checks
        ],
        "hold": floors.hold,
        "proposal": None
        if floors.proposal is None
        else {
            "min_score": floors.proposal.min_score,
            "rel_margin": floors.proposal.rel_margin,
            "derivation": floors.proposal.derivation,
            "validates": floors.proposal.validates,
            # Stored with the numbers, never beside them: a proposal read back
            # out of this row without the reason it cannot be pasted into
            # configuration is exactly how a corpus-derived floor reaches
            # production.
            "caveat": floors.proposal.caveat,
            # Said out loud in the row itself, because a stored number that
            # looks like configuration invites being treated as configuration:
            # floors are applied by a human editing RAG_MIN_SCORE and
            # RAG_REL_MARGIN, and nothing reads this field back.
            "applied": False,
        },
    }


def row_from_embedder_summary(
    summary: EmbedderCheckSummary,
    *,
    provider_name: str,
    digest: str | None,
    runtime_version: str | None,
    vram_bytes: int | None = None,
    configured: dict | None = None,
) -> ModelProbe:
    """Map one embedder-check run onto a measurement row.

    The same log, a different depth. Three columns are constants here and each
    is a statement rather than a placeholder:

    - ``depth`` is :data:`~app.services.embedder_check.PROBE_DEPTH`, a value
      the agency reads never ask for, which is what keeps this row out of a
      decision about tool-calling.
    - ``earned_agency`` is ``deterministic`` because an embedder never drives a
      turn: it has no tool loop to earn anything in, and the honest verdict for
      "did these weights demonstrate they can drive" is the floor. It is stored
      rather than left null so the column keeps one meaning across both
      harnesses, and it can only ever clamp - which for a model nothing routes
      a turn to is a clamp on nothing.
    - ``thinking_with_tools`` is False for the same reason.

    ``tokens_per_second`` stays null: an embedder generates no tokens, and a
    rate invented from embedding latency would sit in the same column the grid
    reads as decode throughput.

    Args:
        summary: The completed harness run.
        provider_name: The MODEL_PROVIDER selector value ("ollama", ...).
        digest: The weights digest the tag resolved to, when exposed - what
            binds these bands to weights rather than to a mutable tag. A
            re-pull changes it, and the old bands correctly stop describing
            what is installed.
        runtime_version: The serving runtime's version, when exposed.
        vram_bytes: The embedder's observed resident VRAM, if the runtime says.
        configured: How the floors THIS deployment retrieves with fared against
            the same bands, when the caller measured them. Recorded beside the
            declared verdict rather than instead of it: the declared pair says
            something about the model, the configured pair says what this
            deployment is actually doing, and a row that kept only the first
            could read as "the floors hold" while the numbers in force were
            dropping every true answer.

    Returns:
        An unsaved :class:`ModelProbe` row for :meth:`ModelProbeRepository.add`.
    """
    return ModelProbe(
        provider=provider_name,
        model=summary.model,
        digest=digest,
        runtime_version=runtime_version,
        harness_version=summary.harness_version,
        depth=summary.depth,
        earned_agency="deterministic",
        thinking_with_tools=False,
        tokens_per_second=None,
        vram_bytes=vram_bytes,
        cases={
            # Named explicitly as well as living in the harness_version column:
            # a reader holding only this JSON should not have to know which
            # harness's fingerprint the column happens to carry.
            "embedder_harness_version": summary.harness_version,
            "prefix": {
                "source": summary.prefix.source,
                "prefix": summary.prefix.prefix,
                "wrapped_separation": summary.prefix.wrapped_separation,
                "raw_separation": summary.prefix.raw_separation,
                "delta": summary.prefix.delta,
                "improves_separation": summary.prefix.improves_separation,
                "degenerate": summary.prefix.degenerate,
            },
            "bands": {
                "wrapped": _arm_payload(summary.wrapped),
                "raw": _arm_payload(summary.raw),
            },
            "floors": _floors_payload(summary.floors) | {"configured": configured},
        },
    )


class ModelProbeRepository:
    """Append and read tool-check measurements for the shared model server."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind to a session; no tenant - see the module docstring."""
        self.session = session

    async def add(self, probe: ModelProbe) -> ModelProbe:
        """Append one measurement row (flushed, not committed - caller commits)."""
        self.session.add(probe)
        await self.session.flush()
        return probe

    async def latest(
        self, provider: str, model: str, *, digest: str | None = None, depth: str | None = None
    ) -> ModelProbe | None:
        """Return the newest probe for a (provider, model), optionally digest-bound.

        With ``digest`` given, only verdicts measured against exactly those
        weights count - a re-pull that changed the digest makes older rows
        invisible here, which is the "re-pull auto-reverts to unverified"
        rule falling out of the key rather than being enforced by code.

        WITHOUT it the read spans every digest, answering "what was measured
        last, on whatever weights were installed then". That is a REPORTING
        question and never a policy one: the row it returns may describe
        weights this machine no longer holds, so a caller has to compare
        :attr:`~app.models.entities.ModelProbe.digest` itself before treating
        the verdict as saying anything about what is installed now.

        With ``depth`` given, only verdicts measured at that depth count. The
        depths answer different questions (FAST never exercises the deciding
        agentic cases or the reasoning-with-tools case), so consumers read the
        latest verdict PER depth rather than the latest row: a newer FAST pass
        must not erase what an older DEEP run measured.
        """
        query = (
            select(ModelProbe)
            .where(ModelProbe.provider == provider, ModelProbe.model == model)
            .order_by(ModelProbe.created_at.desc())  # type: ignore[attr-defined]
            .limit(1)
        )
        if digest is not None:
            query = query.where(ModelProbe.digest == digest)
        if depth is not None:
            query = query.where(ModelProbe.depth == depth)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def history(self, provider: str, model: str, *, limit: int = 50) -> list[ModelProbe]:
        """Return the measurement history for a model, newest first."""
        result = await self.session.execute(
            select(ModelProbe)
            .where(ModelProbe.provider == provider, ModelProbe.model == model)
            .order_by(ModelProbe.created_at.desc())  # type: ignore[attr-defined]
            .limit(limit)
        )
        return list(result.scalars().all())
