"""Earned-verdict lookup: fold measured tool-check probes into the agency inputs.

The two-key agency resolution reads ``effective = min(CEILING, EARNED, ...)``
(:func:`app.services.agency.resolve_agency`). CEILING is human-declared - a
stored operator declaration, else the report-derived ``assisted`` that
:func:`app.services.model_facts.effective_agency_ceiling` resolves and the
caller hands in. EARNED is measured: the tool-check harness's
verdicts, kept in the ``model_probes`` log and digest-bound so each verdict
attaches to WEIGHTS, never to a mutable tag. This module answers "what has
this model actually earned, on the weights installed right now?" for the chat
path, under two regimes:

- **Grace (the default).** A reviewed declaration counts as an asynchronous
  measurement AS FAR AS ``assisted`` - the mode where tools are merely offered
  alongside the pre-fetch that already ran, so a deployment which has never
  run a check is exactly where it was before tools existed. A fresh clone
  therefore works on day one. It does NOT reach ``agentic``: withdrawing the
  pre-fetch and handing the turn to the model is a claim about how these
  weights behave in a loop, and only a DEEP certificate bound to the installed
  digest has ever measured that. Above the floor the grace and the hardened
  regime want the same evidence.

  A probe can still only CLAMP below that, and only when it produced failure
  EVIDENCE: a verdict below what its depth could certify. FAST caps at
  ``assisted`` by design, so a FAST run that earned ``assisted`` is a pass at
  cap - evidence FOR assisted, never evidence AGAINST agentic - while a FAST
  run that earned ``deterministic`` failed a guard and clamps.

- **``AGENCY_REQUIRE_VERIFICATION=true``.** The floor goes too: earned is
  exactly what the log proves about the installed digest. Never probed on
  these weights (including a re-pull that changed the digest), or weights the
  runtime cannot digest-identify at all, resolve to ``deterministic``.

Lookups are cached in-process with a short TTL, so a turn costs no provider
round-trip and no probe query in steady state; a re-pull or a fresh
``make tool-check`` row becomes visible within the TTL.

:func:`effective_verdicts` answers the chat path's question ("what may this
turn assume?"); :func:`explain_earned` answers a reader's ("and why is it
clamped?"), reading the log fresh because a management view is a request-time
join rather than a hot path.

One thing this module deliberately does NOT consider: the runtime version a
verdict was measured under. A runtime upgrade makes a verdict STALE - a banner
fact for a human to act on - not a clamp; auto-demoting on every upgrade would
either strand a deployment on the old runtime or trigger a re-probe storm. The
staleness comparison therefore lives with the reader, against the provider's
live :meth:`~app.providers.base.ModelProvider.runtime_version`, and never here.

The SERVING HOST is the one axis of that family that IS enforced, and the
difference is worth naming because the two look alike. A runtime upgrade is
the same machine, a little newer; a different host is a different machine
entirely - different card, different memory, its own copy of the runtime - so
the measurement was never about this deployment at all. Once a model server
can be somewhere else (a rented pod, a colleague's box), a verdict that
travelled with the log would let a pod's certificate arm a laptop that never
ran a check. So a certificate must name the host serving right now
(:func:`endpoint_certifies`), while a FAILURE clamps whatever host produced it -
the same safe-direction asymmetry the harness and residency axes already use.

The same rule now covers a SECOND kind of staleness. A re-pull changes the
digest, and every verdict measured on the old weights simply falls out of the
digest-bound reads above - silently, by design, because that is what "a
re-pull auto-reverts to unverified" means. Silence is right for the RESOLUTION
and wrong for a reader, who would otherwise watch a model's measured facts
vanish with no explanation. So :func:`explain_earned` also reports the latest
verdict per depth measured on ANY weights (:attr:`EarnedExplanation
.latest_any_weights`), carrying the digest it was measured against. Comparing
that digest against what is installed is the reader's job, exactly as with the
runtime version: these are facts here, and a notice somewhere else.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.models.entities import ModelProbe
from app.providers.base import ModelProvider
from app.repositories.model_probes import ModelProbeRepository
from app.services.model_identity import (
    clear_identity_cache,
    digest_of,
    installed_digests,
    installed_name_of,
)
from app.services.model_policy import PolicyView, ReadHealth
from app.services.model_resolution import (
    KNOWN_THINK_MODES,
    AgencyMode,
    ThinkMode,
    thinking_sibling_for,
)
from app.services.tool_check import harness_version

logger = logging.getLogger(__name__)

# Lookups are re-read at most this often. Within the window a re-pull (new
# digest) or a fresh tool-check row is not yet visible - acceptable staleness
# for facts that only change by explicit operator action, and what keeps the
# chat path free of per-turn provider round-trips and probe queries.
_TTL_SECONDS = 300.0

# What a verdict at each depth can certify (the harness's own rule). A verdict
# BELOW its depth's cap is failure evidence; a verdict AT the cap is a pass
# that says nothing about the modes above it.
_DEPTH_CAP: dict[str, AgencyMode] = {"fast": "assisted", "deep": "agentic"}

# Keep in step with AgencyMode; module-local so untyped row values can be
# membership-checked before use.
_RANK: dict[str, int] = {"deterministic": 0, "assisted": 1, "agentic": 2}

# The depths every read walks, weakest first. Ordering is part of the reader
# contract: the last orphaned entry is the fullest measurement of the set.
_DEPTHS: tuple[str, ...] = ("fast", "deep")

# (provider_name, model, digest) -> (fetched_at, latest facts per depth)
_probe_cache: dict[tuple[str, str, str], tuple[float, _ProbePair]] = {}


def clear_earned_cache() -> None:
    """Drop cached digests and probe reads (test isolation, forced refresh).

    The installed listing lives in :mod:`app.services.model_identity` now, and
    is dropped from here too: every caller that clears this one does so because
    the weights or the verdicts moved, and a stale listing would keep the fresh
    verdicts bound to the wrong digest.
    """
    clear_identity_cache()
    _probe_cache.clear()


@dataclass(frozen=True)
class _ProbeFacts:
    """The verdict-relevant slice of one ``model_probes`` row.

    Five of the trailing fields are carried for READERS only - the resolution
    below never reads ``tokens_per_second``, ``vram_bytes``,
    ``runtime_version``, ``digest`` or ``imperfect_cases`` (see the module
    docstring on why neither a runtime version nor a weights comparison is a
    clamp axis here). They ride along so a reporting caller needs no second
    query for the same row.

    ``representative`` is not one of those: the verify job marks a run the
    model spilled off the GPU for, and such a run may still CLAMP (a failed
    guard is a failed guard) but may never CERTIFY. Old rows, and the CLI's,
    carry no marker and default to True - exactly as trusted as before it
    existed.

    ``serving_endpoint`` is the same kind of field and the opposite default: it
    is the HOST the run measured, and a row that names none (a legacy row, or a
    backend that reports no host) certifies nothing. Unlike the residency
    marker, silence here cannot be read generously - "measured somewhere
    unrecorded" is precisely the claim this axis exists to refuse.
    """

    earned: AgencyMode
    thinking_with_tools: bool
    thinking_digest: str | None
    harness_version: str
    at: datetime
    tokens_per_second: float | None = None
    vram_bytes: int | None = None
    runtime_version: str | None = None
    serving_endpoint: str | None = None
    representative: bool = True
    gpu_fraction: float | None = None
    # The weights this row measured. Almost always redundant on a digest-bound
    # read (it is the digest that was asked for); load-bearing on the unbound
    # one, where it is the only thing that says which weights answered.
    digest: str | None = None
    # How the run watched these weights treat the think flag, or None when it
    # could not tell. Carried for readers AND for the pre-confirm routing clamp
    # (:func:`measured_think_mode`), which is the one thing on this row that
    # changes a turn without an operator having confirmed anything.
    measured_think_mode: ThinkMode | None = None
    # The row's own id, so a writer can record WHICH measurement it acted on.
    # Never a resolution input - a probe's identity says nothing about what it
    # measured - and carried here only so the confirm path does not have to
    # re-query the log for a row this module already read.
    probe_id: str | None = None
    # Every case the run did not pass every sample of, as (name, passed, run).
    # Deliberately UNFILTERED: which of them a given report should name is a
    # display judgement (a clamp line skips the reasoning case, which decides
    # nothing about agency), and answering it here would bury that rule where
    # no reader could see it. Empty for a row written before tallies existed,
    # which is the same shape as a run that passed everything - the difference
    # is not one this layer can tell, and neither is worth naming.
    imperfect_cases: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True)
class _ProbePair:
    """The latest probe facts per depth for one (provider, model, digest).

    Read PER DEPTH, not "latest row": the depths answer different questions
    (FAST never exercises the deciding agentic cases or the reasoning-with-
    tools case), so a newer FAST pass must not erase what an older DEEP run
    measured - and a newer FAST guard FAILURE must still clamp a DEEP pass.
    """

    fast: _ProbeFacts | None
    deep: _ProbeFacts | None


# Why the EARNED axis came out below the declared ceiling. Not a clamp input -
# purely an explanation a reader can render.
EarnedReasonCode = Literal["failed_case", "unverified_weights", "endpoint_mismatch", "no_digest"]


@dataclass(frozen=True)
class EarnedReason:
    """Why a model's EARNED axis sits below its resolved ceiling.

    Attributes:
        code: ``failed_case`` when a probe produced failure evidence and
            clamped; ``endpoint_mismatch`` when a passing verdict exists for
            the installed weights and the ONLY thing keeping it from
            certifying is that it names a different serving host, or names
            none at all; ``unverified_weights`` when nothing reaching the
            ceiling has been measured on these weights; and ``no_digest`` when
            the runtime cannot identify the weights at all, so nothing could
            ever bind a verdict to them.

            ``endpoint_mismatch`` is split out of ``unverified_weights``
            because the two send a reader to opposite actions. "Nothing has
            measured this" is answered by running a check; "something measured
            this, somewhere else" is answered by running a check HERE - and
            until that distinction was on the wire, a deployment that had just
            moved to a rented pod read as though its whole inventory had never
            been probed, with the perfectly good verdicts sitting one screen
            away and nothing saying why they did not count.
        depth: The depth of the probe that clamped, when one did - or, for
            ``endpoint_mismatch``, of the discounted certificate.
        probe_at: When that probe ran, so a reader can say how old the
            evidence is.
    """

    code: EarnedReasonCode
    depth: str | None = None
    probe_at: datetime | None = None


@dataclass(frozen=True)
class ProbeView:
    """One probe row's facts as a reader sees them (no verdict logic attached).

    A public projection of :class:`_ProbeFacts`, so a management view can
    render measured numbers without importing a private type or re-querying
    the log. ``runtime_version`` is here for the staleness COMPARISON the
    reader makes against the live runtime, and ``digest`` for the same kind of
    comparison against the weights installed right now; this module never acts
    on either.

    ``serving_endpoint`` is carried for a comparison of the same SHAPE and a
    different standing: this module DOES act on it (a certificate must name
    the host serving right now), so the reader's copy exists to explain a
    verdict the resolution has already discounted, never to re-decide it.

    ``representative`` and ``gpu_fraction`` ride along for the same reason: a
    reader has to be able to say WHY a row's numbers look the way they do. A
    fully-spilled probe records a VRAM figure of literally zero, and "0 B VRAM,
    throughput unmeasured" with nothing explaining it reads as a broken page
    rather than as a model that ran on the CPU.

    ``imperfect_cases`` is the same idea applied to the verdict itself: a
    clamp that says only "the check did not pass" leaves a reader with nowhere
    to look, and the cases it missed are the one fact that turns the sentence
    into something actionable. Unfiltered here, exactly as on
    :class:`_ProbeFacts`.

    ``measured_think_mode`` rides along so a management card can compare what
    was MEASURED about the think flag against what is declared, without a
    second query for the row it is already holding. None is "the run could not
    tell", never one of the modes.
    """

    depth: str
    earned: AgencyMode
    thinking_with_tools: bool
    tokens_per_second: float | None
    vram_bytes: int | None
    runtime_version: str | None
    harness_version: str
    at: datetime
    representative: bool = True
    gpu_fraction: float | None = None
    digest: str | None = None
    serving_endpoint: str | None = None
    imperfect_cases: tuple[tuple[str, int, int], ...] = ()
    measured_think_mode: ThinkMode | None = None
    # The row's own id, so a WRITER can record which measurement a decision
    # rested on. Never a resolution input - a probe's identity says nothing
    # about what it measured - and carried here only so a declaration does not
    # have to re-query the log for a row this read already holds.
    probe_id: str | None = None


@dataclass(frozen=True)
class EarnedExplanation:
    """The earned verdict plus the evidence behind it, for a reader.

    Attributes:
        earned: Exactly what :attr:`EffectiveVerdicts.earned` would be for the
            same inputs - the two must never disagree, which is why both come
            from the same pure resolver AND take the same early-out (a tag whose
            ceiling is already ``deterministic`` with no thinking-with-tools
            claim resolves without touching the log).
        thinking_with_tools: Exactly what
            :attr:`EffectiveVerdicts.thinking_with_tools` would be, under the
            same never-disagree rule and off the same pure resolver. The
            EFFECTIVE value, not the declaration: a card that showed the claim
            where the turn uses the fold would announce a capability the model
            is not being given, and this axis fails as silent empty output
            rather than as an error - so the over-claim would be invisible
            everywhere except in the blank answers.
        thinking_with_tools_measured: What a deep run ESTABLISHED about the
            reasoning case, with no declaration folded in - True, False, or
            None for "nothing usable measured it here". Reported separately
            from the effective value above because the two answer different
            questions and a reader that conflated them would describe the
            ordinary earn-then-declare order backwards: effective is False
            whenever nothing DECLARES the capability, however well the check
            went. This is the field a console needs to say what declaring
            would do. See :func:`_measured_reasoning_case`.

            None is "NOT ESTABLISHED HERE", never "measured absent", and it
            covers one case beyond a missing verdict: the early-out below
            resolves a deterministic-ceilinged tag with no claim WITHOUT
            reading the log at all, so its measurement is unknown rather than
            missing. A reader must therefore word None as the rule that
            decides ("a check here settles it") and never as an assertion
            about what was measured.
        reason: Why ``earned`` sits below the declared ceiling, or None when
            nothing clamped it.
        fast: The latest FAST probe bound to the installed weights, if any.
        deep: The latest DEEP probe bound to the installed weights, if any.
        latest_any_weights: The latest probe at each depth REGARDLESS of the
            weights it measured, ordered fast then deep and carrying each
            row's own ``digest``. Where ``fast``/``deep`` above answer "what
            has been measured on what is installed", this answers "what was
            measured last, on anything" - and the difference between the two is
            how a reader spots verdicts a re-pull left behind. Untouched by the
            resolution: no verdict here binds to the installed weights, so
            none of it may raise or lower what a model is allowed to do.
            Empty whenever the log was not read at all - the early-out above, a
            failed read, and weights the runtime exposes no digest for, where
            there would be nothing to compare a row against. Empty therefore
            means "nothing to say", never "nothing exists".
    """

    earned: AgencyMode
    reason: EarnedReason | None
    fast: ProbeView | None
    deep: ProbeView | None
    latest_any_weights: tuple[ProbeView, ...] = ()
    thinking_with_tools: bool = False
    thinking_with_tools_measured: bool | None = None


@dataclass(frozen=True)
class EffectiveVerdicts:
    """What a turn may assume about a model: declaration and measurement folded.

    Attributes:
        earned: The EARNED axis for :func:`app.services.agency.resolve_tool_offer`.
            Never None; the resolver still ``min()``s it against the declared
            ceiling, so nothing returned here can escalate a turn.
        thinking_with_tools: The effective reason-and-call-tools capability:
            the stored declaration, revoked when a still-bound measurement
            contradicts it (or, under ``AGENCY_REQUIRE_VERIFICATION``, not yet
            proven for the installed weights).
    """

    earned: AgencyMode
    thinking_with_tools: bool


def declared_thinking_with_tools(policy: PolicyView | None = None) -> bool:
    """The DECLARED reason-and-call-tools claim: a stored operator grant, or nothing.

    One rule in one place, because two callers need it and they must not
    disagree: the planner uses it for the IO-free default it carries when a
    turn never touches the verdict log, and :func:`effective_verdicts` uses it
    as the value a measurement may revoke.

    A grant is a DECLARATION and nothing more - the starting point
    :func:`_resolve_thinking_with_tools` works from, never the answer. It is
    evidence-gated in every regime, because it is a click somebody made and
    this axis fails as silent EMPTY OUTPUT (ollama/ollama#10976). See the
    resolution in :func:`effective_verdicts`.

    Nothing else can make the claim, and that is structural rather than an
    omission: an optimistic default here returns blank answers with nothing
    anywhere reporting a fault, so the capability has to be asked for by name
    and then proven.

    Args:
        policy: The folded, digest-gated declarations for this tag, or None.

    Returns:
        True when an operator declared the capability outright.
    """
    return policy is not None and policy.thinking_with_tools is True


def _grant_needs_evidence(policy_grant: bool | None) -> bool:
    """Say whether a stored thinking-with-tools grant has to be measured first.

    Always, when there is one. A grant is a click somebody made, and this axis
    fails as silent EMPTY OUTPUT (ollama/ollama#10976) rather than as an error,
    so it reaches a turn only on a passing deep verdict bound to the installed
    weights. A stored ``False`` never reaches here: it revokes on its own,
    through the ``not declared`` early-out.

    Kept as a named function rather than inlined at its two call sites: the
    resolution and the reader share it, so a card and a turn cannot come to
    different answers about whether a click needs evidence.

    Args:
        policy_grant: The folded, digest-gated stored value, or None.

    Returns:
        True when a grant is in force and unproven.
    """
    return policy_grant is True


def _binding_sibling(
    settings: Settings, *, probe_model: str, policy: PolicyView | None
) -> str | None:
    """Name the tag whose weights a reasoning-with-tools verdict binds to.

    One copy, because BOTH readers here need the identical answer and a second
    inline one is how a card and a turn come to disagree about which sibling was
    measured.

    The resolution itself is not this module's:
    :func:`~app.services.model_resolution.thinking_sibling_for` is the one ranking
    (an environment override, then a stored operator pairing), and the verify
    job DRIVES the tag it names. That shared
    call is the whole point - the tag a run measured and the tag a verdict binds
    to have to be resolved the same way, or the binding never matches and the
    capability is withheld with nothing anywhere explaining it.

    Resolved for the PROBE KEY rather than for the requested tag, because that
    is the tag a run was recorded against. The two are now the same model under
    at most two spellings - every tag is its own probe key, canonicalized to
    the name the runtime lists - so ``policy``, which describes the tag the
    caller ASKED about, describes this one too.

    That was not always so. While a pair's two members shared one probe row,
    asking about the thinking member resolved it against the chat member's key
    and the caller's pairing belonged to a different tag; feeding one tag's
    pairing into another's resolution made one probe row bind to two different
    siblings depending on which member a turn happened to route to, and the
    reasoning verdict flipped with the turn type. The per-tag key removed the
    case rather than the guard removing the risk.

    Args:
        settings: The active settings (the environment rung).
        probe_model: The tag verdicts are recorded against, in the spelling
            the runtime lists it under.
        policy: The folded declarations for that tag, or None.

    Returns:
        The thinking tag, or None when nothing pairs this one.
    """
    return thinking_sibling_for(
        settings,
        probe_model,
        declared=None if policy is None else policy.thinking_model,
    )


def _grace_floor(ceiling: AgencyMode) -> AgencyMode:
    """How far a declaration alone carries a model: assisted, at most.

    ``assisted`` is the automatic FAST check's cap and the no-regression mode -
    tools offered, the deterministic pre-fetch and the ambient knowledge base
    still running - so a deployment that never ran a check is exactly where it
    was before tools existed, plus an offer its model may decline. ``agentic``
    withdraws the pre-fetch and hands the turn over, which is a claim about how
    these weights behave in a loop, and a declaration is not a measurement of
    that however deliberately it was made.
    """
    return min((ceiling, "assisted"), key=_RANK.__getitem__)


def _capped(earned: AgencyMode, ceiling: AgencyMode) -> AgencyMode:
    """Hold ``earned`` at ``ceiling``, for the paths that can exceed it.

    The hardened regime builds the earned axis from the best certificate the
    log holds, which is a fact about the WEIGHTS and knows nothing about the
    ceiling resolved beside it. That is harmless for the decision -
    :func:`~app.services.agency.resolve_agency` mins the two anyway - and wrong
    for a reader: a tag with a DEEP agentic verdict would render
    a card reading ceiling ``assisted``, earned ``agentic``, effective
    ``assisted``, which is a row arguing with itself. Capped at the resolution
    point rather than in display code, so the number the page shows is the
    number the turn used.
    """
    return min((earned, ceiling), key=_RANK.__getitem__)


def _imperfect_cases(tallies: object) -> tuple[tuple[str, int, int], ...]:
    """Read the cases a probe row did not pass every sample of.

    Reads the ``cases["tallies"]`` map the verify job writes
    (:func:`app.repositories.model_probes.row_from_summary`), which is a plain
    JSON column and therefore not a shape this module may assume. Anything
    that does not parse as ``name -> [passed, run]`` is SKIPPED rather than
    raising: the column is written by a job whose payload may widen, rows
    predating tallies carry none at all, and no explanation is worth failing a
    page load over.

    Args:
        tallies: The row's ``cases["tallies"]`` value, whatever it turned out
            to be.

    Returns:
        Each imperfect case as ``(name, passed, run)``, in the order the row
        recorded them - a caller that cares about order sorts for itself.
    """
    if not isinstance(tallies, dict):
        return ()
    found: list[tuple[str, int, int]] = []
    for name, counts in tallies.items():
        if not isinstance(name, str) or not isinstance(counts, (list, tuple)) or len(counts) != 2:
            continue
        passed, run = counts
        # bool is an int in Python, and a [false, true] pair would otherwise
        # read as a case that failed its one sample.
        if isinstance(passed, bool) or isinstance(run, bool):
            continue
        if not isinstance(passed, int) or not isinstance(run, int):
            continue
        # A run of zero samples measured nothing, so it cannot have failed;
        # it is also the only value that would divide by zero downstream.
        if run <= 0 or passed >= run:
            continue
        found.append((name, passed, run))
    return tuple(found)


def _think_mode_of(value: object) -> ThinkMode | None:
    """Read a stored think mode out of the probe row's JSON, or None.

    Args:
        value: The row's ``cases["measured_think_mode"]``, whatever it was.

    Returns:
        The mode when it is one this build knows, else None.
    """
    return value if isinstance(value, str) and value in KNOWN_THINK_MODES else None  # type: ignore[return-value]


def _facts_of(row: ModelProbe | None) -> _ProbeFacts | None:
    """Extract the verdict facts from a probe row, ignoring malformed rows."""
    if row is None:
        return None
    if row.earned_agency not in _RANK:
        logger.warning(
            "Ignoring model_probes row %s: unknown verdict %r", row.id, row.earned_agency
        )
        return None
    cases = row.cases if isinstance(row.cases, dict) else {}
    thinking_digest = cases.get("thinking_digest")
    residency = cases.get("residency")
    fraction = residency.get("gpu_fraction") if isinstance(residency, dict) else None
    return _ProbeFacts(
        earned=row.earned_agency,  # type: ignore[arg-type]  # membership checked above
        thinking_with_tools=bool(row.thinking_with_tools),
        thinking_digest=thinking_digest if isinstance(thinking_digest, str) else None,
        harness_version=row.harness_version,
        at=row.created_at,
        tokens_per_second=row.tokens_per_second,
        vram_bytes=row.vram_bytes,
        runtime_version=row.runtime_version,
        serving_endpoint=row.serving_endpoint,
        # Only an explicit False withholds trust: a row written before the
        # marker existed says nothing about residency, and reading its silence
        # as "not representative" would retroactively void every verdict the
        # log already holds.
        representative=cases.get("representative") is not False,
        gpu_fraction=float(fraction) if isinstance(fraction, (int, float)) else None,
        digest=row.digest,
        imperfect_cases=_imperfect_cases(cases.get("tallies")),
        # A plain JSON column, so an unrecognized value is DROPPED rather than
        # trusted into a typed slot - the same rule the policy store applies to
        # its own string columns, and it matters more here because this value
        # can force a turn's think flag.
        measured_think_mode=_think_mode_of(cases.get("measured_think_mode")),
        probe_id=str(row.id),
    )


def _supersede_stale_fast(probes: _ProbePair) -> _ProbePair:
    """Drop a FAST verdict that a newer DEEP run has re-measured.

    A DEEP run re-runs every FAST case (plus the guards, at three samples), so
    a DEEP row at least as new as the FAST one supersedes it outright - both
    its certificate and its clamp. Without this, one transient guard failure
    (a provider hiccup scores as a case failure) would clamp a model forever,
    unfixable by the very DEEP re-check that proves it healthy. The reverse
    never holds: FAST measures a subset, so a newer FAST row supersedes
    nothing about a DEEP verdict (though its own failure still clamps).
    """
    if probes.fast is not None and probes.deep is not None and probes.deep.at >= probes.fast.at:
        return _ProbePair(fast=None, deep=probes.deep)
    return probes


async def _latest_probes(
    sessionmaker: async_sessionmaker[AsyncSession],
    provider_name: str,
    model: str,
    digest: str,
) -> _ProbePair:
    """Return the latest digest-bound probe facts per depth, cached."""
    key = (provider_name, model, digest)
    now = time.monotonic()
    hit = _probe_cache.get(key)
    if hit is not None and now - hit[0] < _TTL_SECONDS:
        return hit[1]
    async with sessionmaker() as session:
        repo = ModelProbeRepository(session)
        fast = _facts_of(await repo.latest(provider_name, model, digest=digest, depth="fast"))
        deep = _facts_of(await repo.latest(provider_name, model, digest=digest, depth="deep"))
    pair = _supersede_stale_fast(_ProbePair(fast=fast, deep=deep))
    _probe_cache[key] = (now, pair)
    return pair


def endpoint_certifies(measured: str | None, serving: str | None) -> bool:
    """Say whether a probe row was measured on the host serving right now.

    Public because a READER needs the identical answer: a card that discounted
    a verdict without saying so is the confusion this axis would otherwise
    create, and a second copy of the rule in display code is how the page and
    the turn start disagreeing about one model.

    Deliberately strict on BOTH sides, and neither None is a near-miss:

    - A row with no endpoint was measured somewhere nobody recorded. Reading
      that as "here" is exactly the false certificate the column exists to
      refuse, and it is the shape every row written before the column has.
    - A CURRENT endpoint of None means the running deployment cannot say which
      host it serves from, so there is nothing to match against. A backend that
      names no host can hold no certificate - which points the safe way, and
      costs nothing on the two providers that ship.

    Only the POSITIVE direction consults this. A failure verdict clamps
    whatever host produced it: a model that failed a guard somewhere is a model
    that failed a guard, and invalidation is allowed to travel where
    certification is not.
    """
    return measured is not None and measured == serving


def _resolve_earned(
    *,
    ceiling: AgencyMode,
    require_verification: bool,
    digest: str | None,
    probes: _ProbePair,
    current_harness: str,
    serving_endpoint: str | None,
) -> tuple[AgencyMode, EarnedReason | None]:
    """Resolve the EARNED axis from the probe facts (pure; see module doc).

    Both regimes build from the same certificates - a verdict at its depth's
    cap, from the running harness, on a representative run - and differ only in
    what they credit without one. Strict credits nothing (``deterministic``).
    The grace credits the reviewed declaration up to ``assisted``, which is the
    automatic FAST check's own cap: offering tools alongside the deterministic
    pre-fetch is the no-regression mode, so granting it on a reviewed
    declaration costs a deployment nothing it had before. Above ``assisted``
    the two regimes agree - handing the turn to the model needs a DEEP
    certificate bound to these weights, because that is the only thing that has
    ever measured whether these weights can drive a loop.

    Then failure clamps apply to both, identically.

    Two asymmetries keep invalidation pointing the safe way only:

    - A certificate is trusted for AT MOST its depth's cap, whatever the row
      claims - the log is machine-writable, and the hardened regime must not
      accept a hand-inserted ``fast``/``agentic`` row as an agentic proof.
    - A certificate must carry the RUNNING harness's version (edited cases
      change what a verdict measured, voiding what old cases earned), but
      failure evidence clamps regardless of version - refusing to un-clamp on
      a harness edit is the conservative direction.
    - A certificate must come from a REPRESENTATIVE run. A probe the model
      spilled off the GPU for still clamps if it failed a guard - a spill does
      not make a model's tool calls better - but it cannot be the proof that
      lifts one, because it measured a machine the deployment does not serve
      from. Same asymmetry, same direction.
    - A certificate must name the SERVING HOST this deployment talks to now.
      The literal version of the point above: a verdict earned against a
      rented pod describes that pod's hardware and that pod's runtime, and
      must not arm a laptop it never ran on. A failure from any host still
      clamps, so a model that broke somewhere is not silently rehabilitated by
      moving the deployment.

    Returns:
        The earned mode, and why it sits below ``ceiling`` (None when it does
        not). The reason is an EXPLANATION only - no caller may feed it back
        in as an input, or a report would become part of the policy it
        describes.
    """
    if digest is None:
        # Nothing to bind a verdict to. A hardened deployment cannot call
        # unbindable weights verified; the grace keeps its own floor, but not
        # the rung above it - a certificate is digest-bound, so weights the
        # runtime cannot identify can never earn one, whatever is declared.
        if require_verification:
            return "deterministic", EarnedReason(code="no_digest")
        floor = _grace_floor(ceiling)
        return floor, (None if _RANK[floor] >= _RANK[ceiling] else EarnedReason(code="no_digest"))
    by_depth = [("fast", probes.fast), ("deep", probes.deep)]
    clamps = [
        (facts.earned, depth, facts.at)
        for depth, facts in by_depth
        if facts is not None and _RANK[facts.earned] < _RANK[_DEPTH_CAP[depth]]
    ]
    certs = [
        min((facts.earned, _DEPTH_CAP[depth]), key=_RANK.__getitem__)
        for depth, facts in by_depth
        if facts is not None
        and facts.harness_version == current_harness
        and facts.representative
        and endpoint_certifies(facts.serving_endpoint, serving_endpoint)
    ]
    certified = max(certs, key=_RANK.__getitem__) if certs else "deterministic"
    if require_verification:
        earned = certified
    else:
        # The grace floor, or a certificate that reaches higher - never the
        # declaration on its own. A FAST certificate can only ever match the
        # floor, so in practice this is "assisted, unless a DEEP run passed".
        earned = max((_grace_floor(ceiling), _capped(certified, ceiling)), key=_RANK.__getitem__)
    for clamped, _depth, _at in clamps:
        if _RANK[clamped] < _RANK[earned]:
            earned = clamped
    if _RANK[earned] >= _RANK[ceiling]:
        return earned, None  # nothing below the declaration to explain
    evidence = [entry for entry in clamps if _RANK[entry[0]] <= _RANK[earned]]
    if evidence:
        # Failure evidence outranks "not proven" as the blame: something was
        # measured and it failed, which is the actionable fact. Both can hold
        # at once under the hardened regime, where a failing row is also not a
        # certificate.
        worst, depth, at = min(evidence, key=lambda entry: _RANK[entry[0]])
        return earned, EarnedReason(code="failed_case", depth=depth, probe_at=at)
    # A certificate that would have reached higher, discounted for naming
    # another host - or naming none. Reported as its own kind of "not proven"
    # because the fix is different: the check has to be run HERE, not run at
    # all. The fold above is untouched by this branch; it explains a decision
    # already made and can neither raise nor lower it.
    discounted = [
        (min((facts.earned, _DEPTH_CAP[depth]), key=_RANK.__getitem__), depth, facts.at)
        for depth, facts in by_depth
        if facts is not None
        and facts.harness_version == current_harness
        and facts.representative
        and not endpoint_certifies(facts.serving_endpoint, serving_endpoint)
    ]
    reaching = [entry for entry in discounted if _RANK[entry[0]] > _RANK[earned]]
    if reaching:
        # The strongest one: it is the verdict a re-check here would recover,
        # and naming the weaker of two would understate what is being missed.
        _mode, depth, at = max(reaching, key=lambda entry: _RANK[entry[0]])
        return earned, EarnedReason(code="endpoint_mismatch", depth=depth, probe_at=at)
    # Nothing measured, and the answer is under the declaration: the hardened
    # regime has no certificate at all, and the grace has none reaching past
    # its floor. Both are the same sentence to a reader - "no check has proven
    # this much for the installed weights" - so both carry the same code.
    return earned, EarnedReason(code="unverified_weights")


def _resolve_thinking_with_tools(
    *,
    declared: bool,
    require_verification: bool,
    digest: str | None,
    deep: _ProbeFacts | None,
    installed_thinking_digest: str | None,
    current_harness: str,
    serving_endpoint: str | None,
) -> bool:
    """Resolve the effective reason-and-call-tools capability (pure).

    Only a DEEP run exercises the reasoning-with-tools case, and it drives the
    pair's THINKING tag - so its verdict binds to those weights via the row's
    recorded ``thinking_digest`` (both None when nothing pairs the tag, where
    the case ran on the chat tag bound by the row's own digest). A verdict
    measured against weights no longer installed is void: it neither proves
    nor revokes anything about the current ones. As with the earned axis, a
    POSITIVE verdict additionally requires the running harness's version (an
    edited case voids what the old one earned), a representative run, and the
    host this deployment serves from, while a negative revokes regardless -
    invalidation only ever points the safe way.

    ``require_verification`` is TRUE on every resolution path that decides a
    turn: a grant is the only thing that can claim this axis, and a grant is
    evidence-gated in every regime (:func:`_grant_needs_evidence`), so
    ``AGENCY_REQUIRE_VERIFICATION`` no longer moves it. The ungated arm is kept
    because :func:`_measured_reasoning_case` runs BOTH arms deliberately, to
    separate what a probe measured from what the gate then made of it - a
    console needs the measurement on its own to say what declaring would do.
    """
    if not declared:
        return False  # only an explicit, earned declaration ever enables it
    if digest is None or deep is None:
        # No measurement to consult: the ungated answer keeps the declaration,
        # the gated one treats unproven as false.
        return not require_verification
    bound = deep.thinking_digest == installed_thinking_digest
    if require_verification:
        return (
            deep.thinking_with_tools
            and bound
            and deep.harness_version == current_harness
            and deep.representative
            and endpoint_certifies(deep.serving_endpoint, serving_endpoint)
        )
    # Grace: only a still-bound negative revokes the declaration.
    return not (bound and not deep.thinking_with_tools)


def _measured_reasoning_case(
    *,
    digest: str | None,
    deep: _ProbeFacts | None,
    installed_thinking_digest: str | None,
    current_harness: str,
    serving_endpoint: str | None,
) -> bool | None:
    """What a DEEP run established about the reasoning case, independent of any claim.

    The question a CONSOLE has to answer before it can describe what declaring
    this axis would do: "if a grant were stored right now, would a turn act on
    it?" That is NOT answerable from the effective value, and the difference is
    the bug this exists to kill. ``thinking_with_tools`` resolves False the
    moment nothing DECLARES the capability - :func:`_resolve_thinking_with_tools`
    short-circuits on ``not declared`` before it ever reads the log - so a
    reader inferring "the check failed" from a False effective value gets the
    ordinary earn-then-declare order exactly backwards: the deep run passed,
    nobody has declared yet, and granting it would take effect immediately.

    Answered by asking the REAL resolver twice with the claim held true, rather
    than by restating its conditions. That is what keeps this honest: every
    binding rule (digest, harness, representativeness, serving host, and the
    pair's own thinking weights) lives in one function, and a change there
    moves this answer with it.

    Args:
        digest: The installed weights the verdicts were read against.
        deep: The latest deep probe bound to those weights, if any.
        installed_thinking_digest: The pair's thinking weights right now.
        current_harness: The running harness fingerprint.
        serving_endpoint: The host this deployment serves from.

    Returns:
        True when a certificate establishes the capability, so a stored grant
        would take effect at once. False when a still-bound run measured the
        opposite, so a grant would be inert until a fresh check disagrees. None
        when nothing usable was measured - no deep run, or one this deployment
        cannot count (another host, an edited harness, a spilled run) - which
        is "not established here", never "established false".
    """
    common = {
        "digest": digest,
        "deep": deep,
        "installed_thinking_digest": installed_thinking_digest,
        "current_harness": current_harness,
        "serving_endpoint": serving_endpoint,
    }
    if _resolve_thinking_with_tools(declared=True, require_verification=True, **common):
        # A certificate that certifies in EVERY regime: the strictest reading
        # already says yes, so a grant needs nothing further.
        return True
    if not _resolve_thinking_with_tools(declared=True, require_verification=False, **common):
        # Even the forgiving reading refuses, which happens on exactly one
        # input: a run bound to these weights that measured the capability
        # ABSENT. That is evidence, not silence.
        return False
    return None


# The running harness's case fingerprint, computed once per process: verdict
# rows carry the version they were measured under, and only a match certifies.
_current_harness: str | None = None


def _running_harness() -> str:
    """Return (and cache) the running harness's version fingerprint."""
    global _current_harness
    if _current_harness is None:
        _current_harness = harness_version()
    return _current_harness


async def effective_verdicts(
    *,
    provider: ModelProvider,
    provider_name: str,
    model: str,
    settings: Settings,
    require_verification: bool,
    sessionmaker: async_sessionmaker[AsyncSession],
    effective_ceiling: AgencyMode,
    policy: PolicyView | None = None,
) -> EffectiveVerdicts:
    """Resolve the measured half of the agency inputs for ``model``.

    Every tag is its own probe key, in the spelling the runtime lists it under
    (:func:`~app.services.model_identity.installed_name_of`), so a verdict is a
    fact about specific weights on this server and nothing else.

    ``thinking_with_tools`` stays False unless an OPERATOR declared it, and
    that is structural rather than an oversight: the value starts from a
    DECLARATION that a measurement may only revoke
    (:func:`_resolve_thinking_with_tools`), and nothing else declares it.
    Defaulting it optimistically is the one mistake this axis cannot absorb -
    reasoning plus tools fails as EMPTY OUTPUT rather than as an error
    (ollama/ollama#10976), so an unproven reasoner would return blank answers
    with nothing reporting a fault.

    The console's writer checks that the weights are installed and
    digest-identifiable, and nothing more, which is deliberate: a remote server
    whose whole inventory nobody has described is exactly the deployment that
    needs the verb. It changes nothing about the safety of the default, because
    a stored grant is evidence-gated (see :func:`_grant_needs_evidence`) and
    reaches a turn only on a deep certificate. Unless an operator (or the
    environment) pairs the tag, that certificate binds against a thinking
    digest of None on both sides.

    IO failures degrade by regime, and never upward: the default holds at the
    grace floor (a read that failed is holding no certificate, so it must not
    outrank a healthy turn), while a hardened deployment fails closed to
    ``deterministic``. Either way the failure is logged, never swallowed into
    a different answer silently.

    Args:
        provider: The model seam, for the installed-digest read (cached).
        provider_name: The ``MODEL_PROVIDER`` selector value probe rows key on.
        model: The tag the turn would route to (chat tag or pair sibling).
        settings: The active settings, for the thinking-sibling resolution -
            the environment override is its strongest rung, and a lookup that
            skipped it would bind a certificate to a tag the deployment told it
            not to route to.
        require_verification: The ``AGENCY_REQUIRE_VERIFICATION`` setting.
        sessionmaker: Session factory for the probe-log read (cached).
        effective_ceiling: The ceiling the caller already resolved for this tag
            (:func:`app.services.model_facts.effective_agency_ceiling`), which
            depends on a runtime report and a stored declaration. Passed in
            rather than re-derived here on purpose: the chat turn has already
            paid for that report, and a second derivation is a second chance for
            the page, the turn and this lookup to disagree about one model.
        policy: The folded, digest-gated declarations for this tag, or None.
            Read for its ``thinking_with_tools`` grant and its stored PAIRING -
            the ceiling it may also carry has already been folded into
            ``effective_ceiling`` by the caller, and reading that twice here
            would be a second place for the page and the turn to disagree.

    Returns:
        The effective earned agency and thinking-with-tools values.
    """
    ceiling = effective_ceiling
    # The two sources are kept apart from the folded value below, because what
    # decides how much evidence the claim needs is not WHAT was claimed but how
    # much of it the stored row is responsible for - see the resolution at the
    # end, and the degraded answer on the failure path.
    policy_grant = policy.thinking_with_tools if policy is not None else None
    declared_ttw = declared_thinking_with_tools(policy)
    if ceiling == "deterministic" and not declared_ttw:
        # Nothing a probe could change: a tag floored by its ceiling with no
        # thinking-with-tools claim to revoke resolves without touching the log.
        # This used to floor every UNDESCRIBED tag too, by testing whether one
        # was described rather than the ceiling - which is how a tool-capable
        # model on an undescribed server could never consult its own verdict.
        return EffectiveVerdicts(earned=ceiling, thinking_with_tools=declared_ttw)
    # Every tag is its own probe key, canonicalized to the installed spelling
    # below. Nothing pairs it but the environment or an operator, so the
    # resolution beneath answers None when neither did.
    probe_model = model
    thinking_tag = _binding_sibling(settings, probe_model=probe_model, policy=policy)
    try:
        # Read through the SEAM, not from settings: the provider is what
        # actually talks to a host, so its own answer cannot drift from the URL
        # a request is served over. IO-free, and inside this try because a
        # third-party implementation is free to raise where the shipped ones
        # cannot - and a raise here must degrade like any other failed read.
        serving_endpoint = provider.serving_endpoint()
        digests = await installed_digests(provider, provider_name)
        # See :func:`~app.services.model_identity.installed_name_of`: verdicts
        # are written under the name the runtime lists, so the read has to ask
        # under that name too.
        probe_model = installed_name_of(probe_model, digests) or probe_model
        digest = digest_of(probe_model, digests)
        thinking_digest = digest_of(thinking_tag, digests) if thinking_tag else None
        probes = (
            await _latest_probes(sessionmaker, provider_name, probe_model, digest)
            if digest is not None
            else _ProbePair(fast=None, deep=None)
        )
    except Exception:
        logger.warning(
            "Earned-verdict lookup failed for %r (%s); %s",
            model,
            provider_name,
            "failing closed under AGENCY_REQUIRE_VERIFICATION"
            if require_verification
            else "holding at the grace floor",
            exc_info=True,
        )
        if require_verification:
            return EffectiveVerdicts(earned="deterministic", thinking_with_tools=False)
        # The GRACE FLOOR, never the ceiling. A failed read knows nothing about
        # the installed weights, so it cannot be holding a certificate - and a
        # broken listing or a database blip must not be worth MORE autonomy
        # than a healthy turn gets. Returning the ceiling here would make a
        # timeout on /api/tags the one reliable way to reach agentic.
        #
        # The same rule, applied to the other axis. A stored GRANT does not
        # survive an unverifiable moment: it is evidence-gated on the healthy
        # path, so honouring it here would make the failure the one way to get
        # it ungated - and this axis fails as think:true plus tools reaching a
        # runtime that answers with NOTHING (ollama/ollama#10976), which no
        # test downstream would catch. So the answer degrades to the
        # CATALOG-only claim, which keeps its own default-regime grace. A
        # stored DENIAL survives, because revocation is the safe direction and
        # the safe direction survives everything.
        return EffectiveVerdicts(
            earned=_grace_floor(ceiling),
            thinking_with_tools=False,
        )
    # The reason is dropped here on purpose: the chat path resolves privilege,
    # and an explanation must never become one of its inputs.
    earned, _reason = _resolve_earned(
        ceiling=ceiling,
        require_verification=require_verification,
        digest=digest,
        probes=probes,
        current_harness=_running_harness(),
        serving_endpoint=serving_endpoint,
    )
    return EffectiveVerdicts(
        earned=_capped(earned, ceiling),
        thinking_with_tools=_resolve_thinking_with_tools(
            declared=declared_ttw,
            # A grant is evidence-gated in BOTH regimes: it is a click
            # somebody made, and this axis fails as silent EMPTY OUTPUT
            # (ollama/ollama#10976), so it has to be measured. A stored
            # DENIAL needs no evidence; it takes the `not declared`
            # early-out below on its own.
            #
            # Gated on what the grant is DOING, not on its existence - see
            # :func:`_grant_needs_evidence`, which the reader shares so the
            # card and the turn apply one rule.
            require_verification=(require_verification or _grant_needs_evidence(policy_grant)),
            digest=digest,
            deep=probes.deep,
            installed_thinking_digest=thinking_digest,
            current_harness=_running_harness(),
            serving_endpoint=serving_endpoint,
        ),
    )


@dataclass(frozen=True)
class MeasuredThinkMode:
    """What a check watched a tag do with the think flag, and which row said so.

    Attributes:
        mode: The measured mode, bound to the installed weights and measured by
            the running harness.
        probe_id: The measurement row it came from, which the confirm endpoint
            records as the evidence a stored declaration rested on.
        digest: The weights it was measured against - the same digest the read
            was bound to, carried so a writer stores the binding it actually
            relied on rather than re-deriving it a second time.
    """

    mode: ThinkMode
    probe_id: str
    digest: str


async def measured_think_for(
    *,
    provider_name: str,
    model: str,
    digests: dict[str, str],
    sessionmaker: async_sessionmaker[AsyncSession],
    health: ReadHealth | None = None,
) -> MeasuredThinkMode | None:
    """Return what a check MEASURED about ``model``'s think flag, or None.

    The pre-confirm half of the think-mode story, and the reason it exists at
    all: the policy store is written only when an operator confirms, and until
    they do, a tag whose weights were measured to reason regardless of the flag
    would keep being routed on its declaration - which sends ``think:false``
    and puts the raw ``<think>`` block in front of a reader. This read is what
    lets the ROUTING correct itself immediately while the DECLARATION still
    waits for a human (see :func:`app.services.model_facts.effective_think_mode`
    for the one direction it is allowed to move).

    The variant for a caller that already listed the provider, exactly as
    :func:`app.services.model_policy.policy_for` is - and paired with
    :func:`measured_think_mode` for the same reason. A caller holding a listing
    must not be made to take a second one: the chat turn lists ONCE and hands
    the snapshot around, which is a property with a test on it.

    It reads through the same short-TTL probe cache the earned verdict uses,
    deliberately: a second UNCACHED path would put a query on every turn. It
    does not always hit the same cache ENTRY, and the reason is the first
    difference below - the verdict path keys on the governing entry's chat tag,
    this one keys on the routed tag, so a pair-routed thinking turn fills a
    second key. Two keys, two cheap indexed reads per window; the alternative
    is answering about the wrong weights.

    Three differences from :func:`effective_verdicts`, each with a reason:

    - It keys on the TAG ITSELF, never on the governing entry's chat tag. A
      pair's two members share one probe row and emphatically do not share
      a think mode - the instruct sibling never reasons and the thinking one
      always does - so reading the entry's tag would answer about the wrong
      weights, which is the exact mistake this measurement exists to catch.
    - It reads the newest row across BOTH depths rather than preferring deep.
      Every depth runs the thinking group identically, so newer is simply
      truer here; there is no fuller measurement to prefer.
    - It does NOT require a representative run. Where the model sat decides
      whether a throughput number describes the deployment; it has nothing to
      do with which channel reasoning came out on, and withholding a CPU-run
      measurement would leave the leak uncorrected on exactly the machines
      that spill.

    The running harness's version IS required. Adding or editing a thinking
    case changes what "measured always" meant, and this value changes a turn.

    Args:
        provider_name: The ``MODEL_PROVIDER`` selector value rows key on.
        model: The tag a turn would route to.
        digests: The installed name -> digest map the caller already read.
        sessionmaker: Session factory for the probe-log read (cached).
        health: A caller-owned record marked when the read fails, exactly as
            :func:`app.services.model_policy.policy_for` takes one and for the
            same reason - a caller resolving many models needs to know the
            store is unreachable before it asks about the rest.

    Returns:
        The measurement and the row behind it, or None when nothing measured
        the installed weights - which is also the answer when the read failed.
        Degrading to None is the safe direction here and the only one
        available: None reproduces the behaviour that shipped before this
        measurement existed.
    """
    if not model:
        return None
    try:
        # The tag as the runtime spells it, then the tag as asked for. The
        # first is what every store keys on (app.services.model_identity); the
        # second is what the verify job actually wrote, since it records the
        # tag it was handed. They are the same string on any deployment whose
        # configuration spells its tags the way the runtime lists them, and the
        # second read only happens when they are not.
        installed = installed_name_of(model, digests)
        digest = digest_of(model, digests)
        if digest is None:
            return None
        candidates = [name for name in (installed, model) if name]
        found: _ProbeFacts | None = None
        for name in dict.fromkeys(candidates):
            probes = await _latest_probes(sessionmaker, provider_name, name, digest)
            for facts in (probes.fast, probes.deep):
                if facts is None or facts.measured_think_mode is None:
                    continue
                if facts.harness_version != _running_harness():
                    continue
                if found is None or facts.at > found.at:
                    found = facts
            if found is not None:
                break
    except Exception:
        if health is not None:
            health.mark()
        logger.warning(
            "Measured think-mode lookup failed for %r (%s); routing on the declaration alone",
            model,
            provider_name,
            exc_info=True,
        )
        return None
    if found is None or found.probe_id is None or found.measured_think_mode is None:
        return None
    return MeasuredThinkMode(mode=found.measured_think_mode, probe_id=found.probe_id, digest=digest)


async def measured_think_mode(
    *,
    provider: ModelProvider,
    provider_name: str,
    model: str,
    sessionmaker: async_sessionmaker[AsyncSession],
) -> MeasuredThinkMode | None:
    """Resolve the measured think mode for ``model``, listing the provider itself.

    The variant for a caller that holds no listing of its own - the warm hook
    and the background title job. Both the listing and the probe rows are
    cached, so a steady-state caller pays neither.

    Args:
        provider: The model seam, for the installed-digest read (cached).
        provider_name: The ``MODEL_PROVIDER`` selector value rows key on.
        model: The tag a turn would route to.
        sessionmaker: Session factory for the probe-log read (cached).

    Returns:
        The measurement and the row behind it, or None - see
        :func:`measured_think_for`.
    """
    if not model:
        return None
    try:
        digests = await installed_digests(provider, provider_name)
    except Exception:
        logger.warning(
            "Could not list %s to measure the think mode for %r; routing on the declaration alone",
            provider_name,
            model,
            exc_info=True,
        )
        return None
    return await measured_think_for(
        provider_name=provider_name, model=model, digests=digests, sessionmaker=sessionmaker
    )


async def _read_for_reader(
    repo: ModelProbeRepository,
    *,
    provider_name: str,
    model: str,
    digest: str,
) -> tuple[_ProbePair, tuple[_ProbeFacts | None, ...]]:
    """Read both halves a reader needs: the bound verdicts, and the newest ones.

    Uncached on purpose (see :func:`explain_earned`), and no more queries than
    the two reads it replaces in the ordinary case. The UNBOUND read comes
    first, and when the newest verdict at a depth already carries the installed
    digest it IS the digest-bound one - the same row a filtered query would
    return - so no second query is made. Only a depth whose newest verdict
    measured OTHER weights costs the extra read, which is exactly the state
    worth spending a query on.

    The branch is on the ROW, never on the facts parsed out of it. A row the
    parser rejects (an ``earned_agency`` this build does not know) yields no
    facts while still occupying the newest slot, so branching on the facts
    would read it as "no rows at that depth" and skip the filtered query - and
    a valid bound verdict underneath it would vanish, leaving this reader
    disagreeing with the turn about what the model earned.

    Args:
        repo: The measurement log, bound to a caller-owned session.
        provider_name: The ``MODEL_PROVIDER`` selector value rows key on.
        model: The governing entry's chat tag, which verdicts are recorded
            against.
        digest: The weights installed right now.

    Returns:
        The digest-bound pair (superseding applied, exactly as the chat path
        resolves it), and the newest verdict per depth in :data:`_DEPTHS`
        order whatever weights it measured.
    """
    bound: dict[str, _ProbeFacts | None] = {}
    newest: list[_ProbeFacts | None] = []
    for depth in _DEPTHS:
        row = await repo.latest(provider_name, model, depth=depth)
        latest = _facts_of(row)
        newest.append(latest)
        if row is None or row.digest == digest:
            # No row at all, or the newest one already measured these weights:
            # either way the filtered query has the same answer.
            bound[depth] = latest
            continue
        bound[depth] = _facts_of(
            await repo.latest(provider_name, model, digest=digest, depth=depth)
        )
    pair = _ProbePair(fast=bound["fast"], deep=bound["deep"])
    return _supersede_stale_fast(pair), tuple(newest)


def _view_of(depth: str, facts: _ProbeFacts | None) -> ProbeView | None:
    """Project one depth's probe facts into the public reader shape."""
    if facts is None:
        return None
    return ProbeView(
        depth=depth,
        earned=facts.earned,
        thinking_with_tools=facts.thinking_with_tools,
        tokens_per_second=facts.tokens_per_second,
        vram_bytes=facts.vram_bytes,
        runtime_version=facts.runtime_version,
        harness_version=facts.harness_version,
        at=facts.at,
        representative=facts.representative,
        gpu_fraction=facts.gpu_fraction,
        digest=facts.digest,
        serving_endpoint=facts.serving_endpoint,
        imperfect_cases=facts.imperfect_cases,
        measured_think_mode=facts.measured_think_mode,
        probe_id=facts.probe_id,
    )


async def explain_earned(
    *,
    provider_name: str,
    model: str,
    settings: Settings,
    require_verification: bool,
    sessionmaker: async_sessionmaker[AsyncSession],
    digests: dict[str, str],
    serving_endpoint: str | None,
    effective_ceiling: AgencyMode,
    policy: PolicyView | None = None,
) -> EarnedExplanation:
    """Resolve the earned verdict for ``model`` AND the evidence behind it.

    The reporting sibling of :func:`effective_verdicts`: same inputs, same
    pure resolver, same early-outs, plus the reason and the probe rows the
    answer rests on. Three deliberate differences:

    - **It reads the probe log fresh**, bypassing the short-TTL cache the chat
      path relies on. A management view is a request-time join whose whole
      value is being current: showing a verdict from a re-pull five minutes
      ago, or omitting a probe that just finished, is the one failure it
      cannot afford. The chat path pays a query per turn to avoid; a reader
      pays one per page load and gains truth.
    - **It never touches the probe cache or the digest cache** - neither reads
      one nor writes one - so a page load can neither be served something the
      chat path pinned minutes ago nor pin anything onto later turns. That is
      why ``digests`` is required rather than optional: the caller's snapshot
      is the only digest source, and the freshness guarantee holds
      unconditionally instead of only on the paths that happened to pass one.
    - **It also reports what the digest-bound reads hide.** The chat path wants
      only verdicts bound to the installed weights; a reader additionally needs
      to know that verdicts EXIST which those weights outran, because otherwise
      a re-pull erases a model's measured facts with no explanation at all.
      ``latest_any_weights`` carries them, digest included, and the comparison
      is left to the caller (see the module docstring).

    Failures degrade exactly as the chat lookup does (the grace holds at its
    floor, the hardened regime fails closed), with no probe evidence to show
    either way.

    The reasoning-with-tools axis IS resolved here, which an earlier revision
    of this docstring deferred - correctly, at the time, because no probe had
    yet bound a verdict to the pair's thinking weights. The verify flow now
    records that binding (``thinking_digest`` on the probe row), so the same
    pure resolver the chat path uses can answer here too, and the two are held
    to the never-disagree rule the ``earned`` axis already lives under. The
    alternative - a card reporting the DECLARATION where the turn uses the fold
    - is the one shape this axis cannot afford, because a granted-but-unearned
    claim fails as an EMPTY answer rather than as an error.

    Args:
        provider_name: The ``MODEL_PROVIDER`` selector value probe rows key on.
        model: The tag being explained.
        settings: The active settings, taken for exactly the reason
            :func:`effective_verdicts` takes them - the thinking sibling this
            reader binds against has to be the one the turn routes to, and its
            strongest rung is the environment.
        require_verification: The ``AGENCY_REQUIRE_VERIFICATION`` setting.
        sessionmaker: Session factory for the probe-log read.
        digests: The installed name -> digest map the caller already read. A
            grid explains many tags from ONE provider listing, so passing it
            through costs no round trip per row and guarantees every row is
            explained against the same snapshot the grid displays.
        serving_endpoint: The host the deployment serves from right now, from
            :meth:`~app.providers.base.ModelProvider.serving_endpoint`. Taken
            as a parameter for the same reason ``digests`` is: this function
            holds no provider, and every row of a grid must be explained
            against ONE reading of where the models are being served. Required
            with no default, because a defaulted None would silently mean "no
            certificate reaches this deployment" on any caller that forgot it.
        effective_ceiling: The ceiling the caller already resolved, exactly as
            :func:`effective_verdicts` takes it - and for the same reason: a row
            that reported one ceiling and was explained against another would be
            a card arguing with itself.
        policy: The folded, digest-gated declarations for this tag, or None -
            taken for exactly the same reason and read for exactly the same
            axes as on :func:`effective_verdicts`. It reaches the early-out,
            which is the whole reason it is here: a grant that made the two
            functions take different early-outs would make them disagree about
            a model, which is the one thing this pair may never do.

    Returns:
        The :class:`EarnedExplanation` for ``model``.
    """
    ceiling = effective_ceiling
    # Kept apart from the folded value for the same reason as in the
    # resolution: what decides how much evidence a claim needs is not what was
    # claimed but how much of it the STORED row is responsible for.
    policy_grant = policy.thinking_with_tools if policy is not None else None
    declared_ttw = declared_thinking_with_tools(policy)
    if ceiling == "deterministic" and not declared_ttw:
        # The same early-out :func:`effective_verdicts` takes, so the two can
        # never disagree: a tag floored by its ceiling with nothing a probe
        # could change resolves without reading the log. Nothing is declared,
        # so the effective claim is False by the resolver's own first line.
        return EarnedExplanation(
            earned=ceiling, reason=None, fast=None, deep=None, thinking_with_tools=False
        )
    # Every tag is its own probe key, canonicalized to the installed
    # spelling so this reader asks the log the same question the chat path does;
    # see :func:`effective_verdicts` and
    # :func:`~app.services.model_identity.installed_name_of`.
    probe_model = installed_name_of(model, digests) or model
    thinking_tag = _binding_sibling(settings, probe_model=probe_model, policy=policy)
    thinking_digest = digest_of(thinking_tag, digests) if thinking_tag else None
    try:
        digest = digest_of(probe_model, digests)
        if digest is None:
            # Nothing to compare a row's weights against, so the unbound read
            # would be a query whose answer no caller could use.
            probes, newest = _ProbePair(fast=None, deep=None), (None, None)
        else:
            async with sessionmaker() as session:
                probes, newest = await _read_for_reader(
                    ModelProbeRepository(session),
                    provider_name=provider_name,
                    model=probe_model,
                    digest=digest,
                )
    except Exception:
        logger.warning(
            "Earned-verdict explanation failed for %r (%s); reporting the %s outcome",
            model,
            provider_name,
            "hardened" if require_verification else "grace-floor",
            exc_info=True,
        )
        # The same floor the chat lookup degrades to, for the same reason and so
        # the card cannot report an autonomy the turn would refuse.
        earned: AgencyMode = "deterministic" if require_verification else _grace_floor(ceiling)
        return EarnedExplanation(
            earned=earned,
            reason=None,
            fast=None,
            deep=None,
            # The chat lookup's degraded answer, verbatim: a stored GRANT does
            # not survive an unverifiable moment. It is evidence-gated on the
            # healthy path, so honouring it here would make the failure the one
            # way to get it ungated - and there is no other source left to
            # honour, since a grant is the only thing that can claim this axis.
            thinking_with_tools=False,
        )
    earned, reason = _resolve_earned(
        ceiling=ceiling,
        require_verification=require_verification,
        digest=digest,
        probes=probes,
        current_harness=_running_harness(),
        serving_endpoint=serving_endpoint,
    )
    return EarnedExplanation(
        earned=_capped(earned, ceiling),
        reason=reason,
        fast=_view_of("fast", probes.fast),
        deep=_view_of("deep", probes.deep),
        latest_any_weights=tuple(
            view
            for depth, facts in zip(_DEPTHS, newest, strict=True)
            if (view := _view_of(depth, facts)) is not None
        ),
        # The SAME resolver, off the same digest-bound deep row, under the same
        # evidence gate - so this reader and the chat path cannot report
        # different capabilities for one model.
        thinking_with_tools=_resolve_thinking_with_tools(
            declared=declared_ttw,
            require_verification=(require_verification or _grant_needs_evidence(policy_grant)),
            digest=digest,
            deep=probes.deep,
            installed_thinking_digest=thinking_digest,
            current_harness=_running_harness(),
            serving_endpoint=serving_endpoint,
        ),
        # The measurement on its own, with no claim folded in - what a console
        # needs to say what DECLARING would do, which the effective value above
        # cannot answer.
        thinking_with_tools_measured=_measured_reasoning_case(
            digest=digest,
            deep=probes.deep,
            installed_thinking_digest=thinking_digest,
            current_harness=_running_harness(),
            serving_endpoint=serving_endpoint,
        ),
    )
