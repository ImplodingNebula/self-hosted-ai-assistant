"""Check what score bands the served embedder actually produces.

Retrieval's two relevance floors - `RAG_MIN_SCORE` and `RAG_REL_MARGIN` - are
calibrated numbers on a similarity scale that belongs to one embedder and one
query wrapper. Swap either and the floors say nothing, and nothing in the
deployment reports it: a wrong floor is not an error, it is a citation that
should not be there or an answer that quietly lost its grounding.

This script is the measurement. Run it with `make embedder-check`, which pipes
this file into the backend container where the app package and the
model provider's network are all available. It embeds a fixed corpus of
query/answer/distractor triplets - twice, once with the query prefix under test
and once without - scores every pair, and reports the bands.

The headline PASS/FAIL is the COMPOSED GATE: what the two floors do when
applied together, simulated per query exactly as retrieval composes them, and
answering the two questions a user would notice - was the real answer
retrieved, and did any noise come with it. The per-floor band lines beneath it
are diagnostics that decide nothing; they say which floor to look at when the
gate fails. A margin that prunes a weak supporting passage is reported as a
TRADE, not a failure - pruning weak context is what a relative margin is for.

Where a gate fails, or a floor is undeclared, a pair is PROPOSED - reported,
never applied, and always with the caveat that this corpus scores higher than
production text, so its numbers are an upper bound to compare a real
calibration against rather than values to paste into configuration.

By default it measures the tier's active embedder; EMBEDDER_CHECK_MODEL=<tag>
measures any installed one instead (useful before switching). EMBEDDER_CHECK_JSON=1
prints the reading as JSON instead of the table; EMBEDDER_CHECK_NO_DB=1 skips
the model_probes write.

The script reports; it does not gate. A completed run exits 0 whatever the
bands say - the numbers are the point, and floors that do not fit are a finding,
not an error. Exit 1 means the run itself could not happen: no embedder
configured, the tag is not installed, or the model server is unreachable.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys

from app.core.config import Settings
from app.core.errors import AppError
from app.core.factories import build_model_provider
from app.db.engine import build_engine, build_sessionmaker
from app.providers.base import ProviderUnavailableError
from app.services import embedder_check, embedder_verify
from app.services.embedding_resolution import resolve_embedding

_TRUTHY = {"1", "true", "yes"}
AS_JSON = os.environ.get("EMBEDDER_CHECK_JSON", "").strip().lower() in _TRUTHY
# The reading persists to the model_probes measurement log by default, under
# the "embedder" depth the agency resolver never reads. NO_DB skips the write
# (hermetic runs, a stack whose database is down but whose embedder is worth
# measuring).
NO_DB = os.environ.get("EMBEDDER_CHECK_NO_DB", "").strip().lower() in _TRUTHY
# Blank measures the tier's active embedder, which is the reading the
# deployment's own floors are supposed to describe.
MODEL = os.environ.get("EMBEDDER_CHECK_MODEL", "").strip()


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _print_bands(label: str, arm: embedder_check.ArmMeasurement) -> None:
    """Render one prefix arm's three bands and its corridor."""
    print(f"\n{label} queries")
    header = f"  {'band':<12} {'count':>6} {'min':>9} {'max':>9} {'mean':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, band in (
        ("related", arm.related),
        ("supporting", arm.supporting),
        ("unrelated", arm.unrelated),
    ):
        print(
            f"  {name:<12} {band.count:>6} {_fmt(band.minimum):>9} "
            f"{_fmt(band.maximum):>9} {_fmt(band.mean):>9}"
        )
    print(f"  separation (weakest related - strongest unrelated): {_fmt(arm.separation)}")


def _print_table(outcome: embedder_verify.EmbedderCheckOutcome, *, active: bool) -> None:
    """Render the whole reading: bands, per-triplet detail, and the floors verdict."""
    summary = outcome.summary
    prefix = summary.prefix
    where = "active embedder" if active else "not the active embedder"
    print(f"model: {summary.model} ({where})")
    print(
        f"prefix: {prefix.source} ({len(prefix.prefix)} chars)  harness: {summary.harness_version}"
    )
    print(f"runtime: {outcome.runtime_version or 'unknown'}  digest: {outcome.digest or 'none'}")
    _print_bands("wrapped", summary.wrapped)
    _print_bands("raw", summary.raw)
    if prefix.degenerate:
        print("\nprefix: none applied, so the two arms are one measurement.")
    else:
        verdict = "widens" if prefix.improves_separation else "does NOT widen"
        print(
            f"\nprefix: wrapping {verdict} the corridor "
            f"({_fmt(prefix.wrapped_separation)} wrapped vs {_fmt(prefix.raw_separation)} raw, "
            f"delta {_fmt(prefix.delta)})"
        )
    print("\nper triplet (wrapped)")
    header = f"  {'triplet':<20} {'related':>9} {'support':>9} {'noise':>9}  strongest non-answer"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for triplet in summary.wrapped.triplets:
        print(
            f"  {triplet.name:<20} {_fmt(triplet.related):>9} "
            f"{_fmt(triplet.supporting):>9} {_fmt(triplet.best_unrelated):>9}  "
            f"{triplet.best_unrelated_passage}"
        )
    configured = outcome.configured
    # The headline, and the only PASS/FAIL: what the two floors do TOGETHER, on
    # the numbers this deployment is actually retrieving with.
    print(
        f"\nCOMPOSED GATE (RAG_MIN_SCORE {configured.min_score} + "
        f"RAG_REL_MARGIN {configured.rel_margin}): "
        f"{'PASS' if configured.hold else 'FAIL'}"
    )
    print(f"  {configured.composed.detail}")
    _print_gate(configured.composed)
    floors = summary.floors
    print(
        f"\ndeclared for this embedder: min_score {floors.declared.min_score}, "
        f"rel_margin {floors.declared.rel_margin} (source: {floors.declared.source})"
    )
    if floors.composed is None:
        print("  (not both floors declared, so there is nothing to compose against)")
    elif configured.matches_declared:
        print("  (the same numbers, so the composed gate above is this verdict too)")
    else:
        print(f"  composed: {'PASS' if floors.hold else 'FAIL'} - {floors.composed.detail}")
    print("\nband diagnostics (informational - the composed gate decides)")
    for check in configured.checks:
        print(f"  {'ok  ' if check.passed else 'note'} {check.name}: {check.detail}")
    if floors.proposal is not None:
        proposal = floors.proposal
        print(
            f"\nproposed: min_score {proposal.min_score}, rel_margin {proposal.rel_margin} "
            f"({'holds when composed' if proposal.validates else 'DOES NOT hold when composed'})"
        )
        print(f"  {proposal.derivation}")
        print(f"  CAVEAT: {proposal.caveat}")
        print(
            "  Reported only. Apply by setting RAG_MIN_SCORE / RAG_REL_MARGIN by hand,"
            " after reading the caveat above."
        )


def _print_gate(gate: embedder_check.ComposedGate) -> None:
    """Render the per-query simulation behind the composed verdict."""
    header = (
        f"  {'query':<20} {'top':>8} {'floor':>8} {'answer':>7} {'support':>8} {'clear':>8}  leaked"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for query in gate.queries:
        support = (
            "-"
            if query.supporting_retrieved is None
            else ("kept" if query.supporting_retrieved else "pruned")
        )
        print(
            f"  {query.name:<20} {_fmt(query.top_score):>8} "
            f"{_fmt(query.effective_floor):>8} "
            f"{'yes' if query.related_retrieved else 'LOST':>7} {support:>8} "
            f"{_fmt(query.noise_clearance):>8}  {_leaked(query.noise_retrieved)}"
        )


def _leaked(passages: list[str]) -> str:
    """Name the leaked passages, capped - a badly-mismatched embedder leaks dozens."""
    if not passages:
        return "-"
    shown = ", ".join(passages[:3])
    return shown if len(passages) <= 3 else f"{shown} (+{len(passages) - 3} more)"


def main() -> int:
    """Measure the configured embedder and report its bands and floors verdict."""
    settings = Settings()
    provider = build_model_provider(settings)

    async def run() -> tuple[embedder_verify.EmbedderCheckOutcome, str, bool]:
        # The embedder this deployment would use, resolved without a workspace
        # or a store: a terminal run has no principal, and the evidence rung
        # would put a vector-store round trip on a diagnostic that measures the
        # model server.
        active = (
            await resolve_embedding(
                settings=settings, provider=provider, provider_name=settings.model_provider
            )
        ).model
        # The provider owns a lazily-created HTTP client; close it inside the
        # loop that created it rather than letting asyncio.run tear it down.
        try:
            tag = await embedder_verify.begin_embedder_check(
                provider,
                provider_name=settings.model_provider,
                active=active,
                name=MODEL or None,
            )
            engine = None if NO_DB else build_engine(settings)
            try:
                outcome = await embedder_verify.run_embedder_check_job(
                    provider=provider,
                    settings=settings,
                    sessionmaker=None if engine is None else build_sessionmaker(engine),
                    # A terminal run has no actor, so the audit columns stay
                    # null - exactly as the tool-check CLI leaves them.
                    tenant=None,
                    model=tag,
                )
            finally:
                embedder_verify.release_probe_slots(tag)
                if engine is not None:
                    await engine.dispose()
            return outcome, tag, embedder_verify.is_active_embedder(tag, active)
        finally:
            aclose = getattr(provider, "aclose", None)
            if aclose is not None:
                await aclose()

    try:
        outcome, _tag, active = asyncio.run(run())
    except ProviderUnavailableError as exc:
        print(f"error: the model provider is unreachable: {exc}", file=sys.stderr)
        return 1
    except AppError as exc:
        # The refusals the runner decides before any embedding happens: no
        # embedder configured, an unusable name, a tag that is not installed.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if AS_JSON:
        payload = dataclasses.asdict(outcome.summary)
        payload["configured_floors"] = dataclasses.asdict(outcome.configured)
        payload["runtime_version"] = outcome.runtime_version
        payload["vram_bytes"] = outcome.vram_bytes
        payload["active"] = active
        payload["probe_id"] = outcome.probe_id
        print(json.dumps(payload, indent=2))
    else:
        _print_table(outcome, active=active)

    if NO_DB:
        print("not recorded (EMBEDDER_CHECK_NO_DB=1)", file=sys.stderr)
    elif outcome.probe_id is None:
        # Loud, but not the exit code: the measurement itself succeeded, and
        # exit 1 is reserved for "the run could not happen".
        print("warning: reading NOT recorded in model_probes", file=sys.stderr)
    else:
        short = (outcome.digest or "no-digest")[:19]
        print(
            f"recorded: model_probes {outcome.probe_id} "
            f"({short}, runtime {outcome.runtime_version or 'unknown'})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
