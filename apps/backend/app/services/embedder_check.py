"""The embedder-check harness: the score bands an embedder actually produces.

The tool check earns a chat model its agency; this is the same idea one seam
over. Retrieval's two relevance floors - ``RAG_MIN_SCORE`` and
``RAG_REL_MARGIN`` - are calibrated numbers on a similarity scale that belongs
to ONE embedder and ONE query wrapper. Swap either and the floors say nothing:
the 2026-07-09 grounding bug was exactly that, unprefixed queries turning tiny
boilerplate chunks into similarity hubs while genuinely relevant passages fell
under a floor calibrated for a different scale. Nothing in the deployment
notices, because a wrong floor is not an error - it is a citation that should
not be there, or an answer that quietly lost its grounding.

So this module MEASURES the scale. A fixed corpus of triplets - a query, a
passage that answers it, a passage that does not - is embedded through the
:class:`~app.providers.base.ModelProvider` seam, every pair is scored, and the
result is two bands: what a related passage scores, and what an unrelated one
does.

**The verdict is then taken on the floors COMPOSED, never on either one alone.**
:func:`app.services.rag_service.retrieve` applies ``min_score`` as a coarse
gate and then drops anything more than ``rel_margin`` below the top surviving
hit, so neither floor ever acts by itself. :func:`compose_gate` simulates
exactly that per query and asks the two questions a user of retrieval would
ask: was the real answer retrieved, and did any noise come with it. The
per-floor band checks are kept BESIDE it as diagnostics - when the composed
gate fails they say which floor to look at - but a band check failing on its
own is not a failure. The first live run (2026-08-06, qwen3-embedding:0.6b)
had the noise peak sitting above ``min_score`` while the composed gate leaked
nothing whatsoever, because the margin was doing the work. A harness that
reported that as broken would have sent somebody to change a healthy
configuration.

From the bands the floors are also PROPOSED where nothing declares them, or
where the composed gate fails. A proposal is REPORTED, never applied, and it
carries a caveat that has to travel with the numbers: this corpus is written as
clean question-and-answer text and scores systematically HIGHER than production
prose, so a floor derived here transplants downward badly. The same live run
proposed 0.51 against a production calibration that had measured real relevant
chunks at 0.40-0.60 - a floor that would have excluded half of them.

The corpus is deliberately general - factual, procedural, conversational,
definitional - because the platform is general-purpose. A corpus drawn from one
domain would measure how well an embedder does that domain, which is not the
question; the question is how far apart it holds "answers this" from "does not".

**What counts as noise is a corpus decision, and it is not "everything else".**
A query's noise pool is every passage in the corpus except its own answers AND
except every other triplet's SUPPORTING passage. Supporting passages are
written as near-paraphrases of their own triplet's topic, which makes them the
most semi-related text here and the least honest noise: the first live run's
noise peak was a supporting passage answering a different triplet's query at
0.389, a number about how the corpus was written rather than about the
embedder. Other triplets' RELATED passages stay in the pool deliberately - a
real knowledge base holds documents about adjacent topics, and a noise pool
with no topical adjacency at all would measure an unrealistically easy problem
and derive floors too low for the corpus it is meant to protect. The residual
adjacency between the account-and-policy triplets is therefore ACCEPTED and
visible: every triplet reports which passage crowded it hardest, so a run that
starts leaking says which pair did it.

Two things this harness deliberately is NOT. It is not a quality ranking - a
wide separation is a fact about a scale, not a score to compare embedders on -
and it is not a retrieval benchmark: it never touches the vector store, never
ingests anything, and reads no tenant data. It is pure in the same sense
:mod:`app.services.tool_check` is: it drives a provider and returns numbers.
Persistence, target resolution and re-warming live in
:mod:`app.services.embedder_verify`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Final, Literal

from app.providers.base import EmbedOptions, ModelProvider
from app.services.known_models import quirks_for

# What ``model_probes.depth`` records for a run of THIS harness. A value the
# agency resolver has never heard of, which is the point: the tool-check reads
# are filtered to {"fast", "deep"}, so an embedder row cannot reach them even
# by accident. See :func:`app.services.earned.effective_verdicts`.
PROBE_DEPTH: Final[str] = "embedder"

# How far the WEAKEST related passage must sit above an absolute floor for that
# floor to be called sound. A floor level with the weakest related score drops
# that passage on any run-to-run jitter, which is a floor that only looks
# calibrated.
_RELATED_HEADROOM: Final[float] = 0.03

# The mirror of it: how far the floor must sit above the STRONGEST unrelated
# passage. Same reasoning in the other direction - a floor level with the noise
# peak admits noise the moment anything moves.
_UNRELATED_CLEARANCE: Final[float] = 0.03

# Held back from the widest margin the composed gate could survive, when one is
# proposed. A margin exactly at the point where noise starts being retrieved is
# a margin that leaks on the first score that moves.
_MARGIN_HEADROOM: Final[float] = 0.02

# The narrowest relative margin worth proposing. Below this the margin has
# stopped being a relevance rule and become a second absolute floor pinned to
# whatever happened to rank first.
_MIN_REL_MARGIN: Final[float] = 0.02

# Similarity scores are reported to this many decimals, and the verdicts are
# taken on the REPORTED numbers - so the figure a reader checks the arithmetic
# against is the figure the verdict used.
_SCORE_PLACES: Final[int] = 4

# Proposed floors are rounded to this many decimals - the form a human types
# into an environment file. It shapes a stored number, so it is fingerprinted
# with the rest.
_PROPOSAL_PLACES: Final[int] = 2

# EDIT THESE STRINGS WHENEVER THE RULE THEY NAME CHANGES.
#
# Every other behavior-bearing input to :func:`harness_version` is a constant or
# the corpus text, and changing one changes the fingerprint on its own. These
# three rules live in CODE, so nothing about editing them moves a hash - and a
# fingerprint that does not move is a promise that two rows measured the same
# thing. That promise has already been broken once: the noise-pool rule was
# narrowed (foreign supporting passages excluded, the unrelated band going from
# 141 scores to 120) while the corpus text stayed identical, and two runs of
# demonstrably different measurements were recorded under one fingerprint.
#
# The strings are never parsed and never compared to behaviour. They are a hand
# maintained declaration, and the only thing standing between a changed rule and
# a stale verdict that still looks current.
_CODE_RULES: Final[dict[str, str]] = {
    "noise_pool": "corpus minus own answers minus foreign supporting",
    "composed_gate": "min_score coarse gate, then top-survivor minus rel_margin",
    "proposal": (
        "min_score = noise peak + clearance rounded up; "
        "rel_margin = tightest top-to-noise room minus headroom"
    ),
}

# Embedding requests are chunked at this size, mirroring
# :class:`~app.providers.rag.embedder.ProviderEmbedder`: the corpus fits in one
# request today, and a corpus that grows must not silently start building an
# oversized one.
_BATCH: Final[int] = 16

PrefixSource = Literal["known_models", "none"]

# The band diagnostics. These DECOMPOSE the floors - each one asks what a
# single floor does to a single band - and they are informational by design:
# retrieval never applies either floor alone, so a band check failing is a
# pointer to where a composed failure would come from, never a failure itself.
# The verdict that decides is :class:`ComposedGate`.
CheckName = Literal[
    "min_score_admits_related",
    "min_score_excludes_unrelated",
]


@dataclass(frozen=True)
class TripletSpec:
    """One probe triplet: a question, an answer to it, and a distractor.

    Attributes:
        name: Stable identifier, used as the passage id prefix and in reports.
        query: The retrieval query, phrased as a person would ask it.
        related: A passage that genuinely answers ``query``. Another triplet's
            related passage IS noise for this one, and stays in the noise pool
            on purpose - see the module docstring on why a pool with no topical
            adjacency measures an unrealistically easy problem.
        unrelated: A passage from a plainly different subject.
        supporting: A second passage that is genuinely relevant but says less -
            the kind of hit the RELATIVE margin decides the fate of. Present on
            a subset, because the margin can only be measured where a query has
            more than one true answer.

            A supporting passage is scored ONLY against its own query. It is
            written as a near-paraphrase of another passage's topic, which
            makes it the most semi-related text in the corpus and no kind of
            noise at all: measured live 2026-08-06, the noise peak was
            ``password-reset/supporting`` answering the idempotency query at
            0.389, which said more about how the corpus was written than about
            the embedder.
    """

    name: str
    query: str
    related: str
    unrelated: str
    supporting: str | None = None


# The corpus. Deliberately broad - a factual lookup, two procedures, a
# conversational ask, a definition, a troubleshooting question, a how-to and a
# historical date - and deliberately small: every query is scored against every
# passage, so this is 8 queries against 19 passages in both prefix arms, which
# is seconds of embedding rather than minutes of generation.
#
# Editing ANY field changes :func:`harness_version`, which is what correctly
# invalidates the floors an older corpus validated.
CORPUS: tuple[TripletSpec, ...] = (
    TripletSpec(
        name="light-travel-time",
        query="How long does light from the Sun take to reach Earth?",
        related=(
            "Sunlight crosses the roughly 150 million kilometres to Earth in about "
            "eight minutes and twenty seconds."
        ),
        unrelated=(
            "A sourdough starter stays lively when it is fed equal weights of flour "
            "and water once a day."
        ),
    ),
    TripletSpec(
        name="password-reset",
        query="How do I reset a password I have forgotten?",
        related=(
            "To reset a forgotten password, choose 'Forgot password' on the sign-in "
            "screen, enter the address on the account, and open the emailed link "
            "within one hour."
        ),
        supporting=(
            "A reset link expires after an hour, and requesting a second one invalidates the first."
        ),
        unrelated=(
            "The alpine ibex climbs near-vertical dam walls to lick mineral salts from the stone."
        ),
    ),
    TripletSpec(
        name="office-hours",
        query="What time does the office close on a Friday?",
        related=(
            "The office closes at four in the afternoon on Fridays and at six on "
            "every other weekday."
        ),
        unrelated=("A compiler usually runs register allocation after instruction selection."),
    ),
    TripletSpec(
        name="return-window",
        query="How many days do I have to send an item back?",
        related="Items may be returned within thirty days of delivery for a full refund.",
        supporting=(
            "Refunds reach the original payment method within five working days of "
            "the return arriving back."
        ),
        unrelated="A double bass is tuned in fourths, unlike the rest of the string family.",
    ),
    TripletSpec(
        name="idempotent-request",
        query="What does it mean for a request to be idempotent?",
        related=(
            "An idempotent request leaves the same result whether it is sent once or "
            "many times, so a client can safely retry after a timeout."
        ),
        unrelated="Rye bread keeps longer than wheat bread because it is more acidic.",
    ),
    TripletSpec(
        name="laptop-fan",
        query="Why does my laptop fan run all the time?",
        related=(
            "A fan that never settles usually means a background process is pinning "
            "the processor, or the vents are clogged with dust."
        ),
        supporting=(
            "Blocked vents raise the internal temperature, and the fan compensates by "
            "running faster for longer."
        ),
        unrelated=("The armistice was signed in a railway carriage parked in a French forest."),
    ),
    TripletSpec(
        name="lentil-soaking",
        query="Do dried lentils need soaking before they are cooked?",
        related=(
            "Dried lentils need no soaking: rinse them and simmer for twenty to "
            "thirty minutes until they are tender."
        ),
        unrelated=(
            "TCP slow start doubles the congestion window each round trip until a loss is seen."
        ),
    ),
    TripletSpec(
        name="atlantic-cable",
        query="When was the first telegraph cable laid across the Atlantic?",
        related=(
            "The first transatlantic telegraph cable was completed in 1858, though it "
            "failed after a few weeks of service."
        ),
        unrelated=(
            "Set the saddle height so the knee stays slightly bent at the bottom of "
            "the pedal stroke."
        ),
    ),
)


@dataclass(frozen=True)
class DeclaredFloors:
    """The floors something already declares for an embedder.

    Attributes:
        min_score: The absolute relevance floor, when declared.
        rel_margin: The relative margin, when declared.
        source: Where they came from - ``known_models`` for a seed entry,
            ``none`` when nothing describes this tag.
    """

    min_score: float | None = None
    rel_margin: float | None = None
    source: Literal["known_models", "none"] = "none"


@dataclass(frozen=True)
class PrefixChoice:
    """Which query wrapper governs a run, and which source supplied it."""

    prefix: str
    source: PrefixSource


@dataclass(frozen=True)
class ScoreBand:
    """The range one family of similarity scores occupied.

    Attributes:
        count: How many scores went into it (0 leaves the rest None).
        minimum: The lowest score.
        maximum: The highest score.
        mean: The arithmetic mean.
    """

    count: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None


@dataclass(frozen=True)
class TripletMeasurement:
    """What one triplet scored under one prefix arm.

    Attributes:
        name: The triplet's identifier.
        related: The similarity between the query and its own answer.
        supporting: The similarity to the weaker true answer, when the triplet
            carries one.
        best_unrelated: The highest similarity the query reached against any
            passage in its NOISE POOL - across the corpus, not just its own
            distractor.
        best_unrelated_passage: Which passage that was. This is the corpus's
            own cross-talk report: a run whose composed gate starts leaking
            names the pair that did it, rather than leaving somebody to guess
            which two triplets drifted together.
        scores: Every passage's score for this query, keyed by passage id. The
            composed gate needs the whole ranking, including passages no band
            counts - retrieval ranks against everything stored, and the top hit
            it takes its relative floor from could be any of them.
        noise_pool: Which of those ids count as noise for this query.
    """

    name: str
    related: float
    supporting: float | None
    best_unrelated: float
    best_unrelated_passage: str
    scores: dict[str, float] = field(default_factory=dict)
    noise_pool: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ArmMeasurement:
    """One prefix arm's whole reading: its bands, its gap, its triplets.

    Attributes:
        related: The band genuinely relevant passages occupied.
        supporting: The band the weaker true answers occupied.
        unrelated: The band everything else occupied.
        separation: ``related.minimum - unrelated.maximum`` - the width of the
            corridor a floor has to sit in. Negative means the bands overlap
            and NO absolute floor separates them.
        triplets: The per-triplet detail behind the bands.
    """

    related: ScoreBand
    supporting: ScoreBand
    unrelated: ScoreBand
    separation: float
    triplets: tuple[TripletMeasurement, ...]


@dataclass(frozen=True)
class PrefixComparison:
    """Whether the instruct wrapper earned its place on this embedder.

    Attributes:
        source: Which source supplied the prefix under test.
        prefix: The wrapper itself, so a report can show what was measured.
        wrapped_separation: The separation the wrapped queries produced.
        raw_separation: The separation the same queries produced unwrapped.
        delta: ``wrapped_separation - raw_separation``.
        improves_separation: True when wrapping widened the corridor. False on
            an asymmetric embedder is a finding, not a formality: the prefix and
            the floors are calibrated together, so a wrapper that does not help
            means the floors beside it were calibrated on a scale nobody is
            using.
        degenerate: True when the prefix is empty, which makes the two arms the
            same measurement and ``delta`` necessarily zero.
    """

    source: PrefixSource
    prefix: str
    wrapped_separation: float
    raw_separation: float
    delta: float
    improves_separation: bool
    degenerate: bool


@dataclass(frozen=True)
class QueryGate:
    """What the composed floors would actually retrieve for one query.

    Attributes:
        name: The triplet's identifier.
        top_score: The best-scoring passage of any kind - what the relative
            margin measures down from.
        top_passage: Which passage that was. Not always the right answer, and
            when it is not, the margin's floor is being set by something else.
        effective_floor: The score a passage had to reach to survive BOTH
            floors: ``max(min_score, top_score - rel_margin)``.
        related_retrieved: Whether the query's real answer came back. False is
            the worse of the two failures - the deployment silently lost its
            grounding on this question.
        supporting_retrieved: Whether the weaker true answer came back, or None
            when the triplet has none. Informational: pruning weak context is
            what the margin is FOR.
        noise_retrieved: Passages from the noise pool that survived both floors,
            in descending score order. Non-empty is a leak.
        noise_clearance: How far the strongest noise passage sat BELOW the
            effective floor. Negative on a leak; a small positive number is a
            gate that holds today and would not survive much drift.
    """

    name: str
    top_score: float
    top_passage: str
    effective_floor: float
    related_retrieved: bool
    supporting_retrieved: bool | None
    noise_retrieved: tuple[str, ...]
    noise_clearance: float


@dataclass(frozen=True)
class ComposedGate:
    """The verdict that decides: what the floors do when applied TOGETHER.

    Retrieval never applies ``min_score`` or ``rel_margin`` alone, so neither
    band check is a verdict on its own. This one simulates
    :func:`app.services.rag_service.retrieve`'s dense-only path exactly - the
    coarse floor first, then the relative margin measured down from the top
    surviving hit - and reports what a user of retrieval would notice.

    Attributes:
        min_score: The absolute floor that was composed.
        rel_margin: The relative margin that was composed.
        queries: The per-query simulation, in corpus order.
        leaks: ``triplet: passage`` for every noise passage that was retrieved.
        related_lost: Triplets whose real answer the floors excluded.
        supporting_pruned: Triplets whose weaker true answer the margin pruned.
            A reported TRADE, never a failure: this corpus's supporting
            passages are deliberately weak, and pruning them is the documented
            job of a relative margin.
        holds: True when no noise was retrieved and no real answer was lost.
        detail: One human-readable line summarising the above.
    """

    min_score: float
    rel_margin: float
    queries: tuple[QueryGate, ...]
    leaks: tuple[str, ...]
    related_lost: tuple[str, ...]
    supporting_pruned: tuple[str, ...]
    holds: bool
    detail: str


@dataclass(frozen=True)
class FloorCheck:
    """One declared floor, measured against one band - a DIAGNOSTIC, not a verdict.

    Informational by construction: retrieval composes the two floors, so what a
    single floor does to a single band never decides anything. These exist to
    say WHERE a composed failure comes from, and a band check failing beside a
    composed gate that holds is an ordinary, healthy reading.

    Attributes:
        name: Which property was measured.
        passed: Whether the measurement supports the declaration.
        detail: One human-readable line carrying the numbers behind it.
    """

    name: CheckName
    passed: bool
    detail: str


@dataclass(frozen=True)
class FloorProposal:
    """Floors derived from the measured bands, for a human to apply or refuse.

    Attributes:
        min_score: The proposed absolute floor, or None when the bands overlap
            too far for any floor to separate them.
        rel_margin: The proposed relative margin, or None when the corpus
            measured no supporting passage to derive one from.
        derivation: The rule, stated with the numbers it was applied to.
        validates: Whether the proposed pair holds when COMPOSED - derived and
            then simulated by one code path, so a proposal can never be
            recommended on a rule the gate would reject.
        caveat: Why these numbers must not be copied into configuration
            unexamined. A separate field rather than a sentence buried in the
            derivation, so nothing can render the numbers without it.
    """

    min_score: float | None
    rel_margin: float | None
    derivation: str
    validates: bool
    caveat: str = ""


@dataclass(frozen=True)
class FloorsVerdict:
    """What the run concluded about the floors for this embedder.

    Attributes:
        declared: What was already declared, and by whom.
        composed: What the declared pair would actually retrieve - the verdict.
            None when fewer than both floors are declared, since retrieval
            applies both and composing against a value nobody declared would
            report a configuration that does not exist.
        checks: Band DIAGNOSTICS for the declared absolute floor. They never
            decide; see :func:`validate_floors`.
        hold: The composed gate's answer, or None when there was nothing to
            compose. Not derived from ``checks`` - a band check failing beside
            a gate that holds is an ordinary healthy reading.
        proposal: Present whenever ``hold`` is not True - an undeclared pair, a
            half-declared one, or a composed gate that failed. A verdict that
            says only "these do not work" leaves a reader with nothing to do.
    """

    declared: DeclaredFloors
    composed: ComposedGate | None
    checks: tuple[FloorCheck, ...]
    hold: bool | None
    proposal: FloorProposal | None


@dataclass
class EmbedderCheckSummary:
    """Everything one embedder-check run measured.

    Attributes:
        model: The embedder tag the run drove.
        depth: What the measurement log records this as. Fixed to
            :data:`PROBE_DEPTH` - the agency resolver reads only ``fast`` and
            ``deep``, so this value is what keeps an embedder row out of a
            decision it says nothing about.
        prefix: The wrapped-versus-raw comparison.
        wrapped: The arm run with the query prefix applied - the one retrieval
            actually uses, and therefore the one the floors are judged on.
        raw: The same corpus with unwrapped queries, for the comparison.
        floors: The validation or the proposal.
        harness_version: :func:`harness_version` at run time.
        digest: The weights the tag pointed at. None here by design - the
            harness stays pure, so the runner captures it at persist time and
            binds the reading to weights rather than to a mutable tag.
    """

    model: str
    depth: str
    prefix: PrefixComparison
    wrapped: ArmMeasurement
    raw: ArmMeasurement
    floors: FloorsVerdict
    harness_version: str
    digest: str | None = None


def harness_version() -> str:
    """Fingerprint everything that decides what a reading MEANS.

    A stored reading says "these bands, this verdict" - statements about the
    corpus that produced them, the constants that judged them, and the rules
    that defined them. Editing a triplet, moving a headroom, or narrowing what
    counts as noise all change what a verdict MEASURED, so the version has to
    change with any of them. Computed from live module state rather than frozen
    at import, so a test that replaces the corpus demonstrably changes the
    answer.

    Three of those inputs are RULES rather than values - the noise-pool
    definition, the composed-gate simulation, and the proposal derivation - and
    they live in code, where nothing about editing them would move a hash. They
    reach the payload through :data:`_CODE_RULES`, a hand-maintained
    declaration that must be edited alongside the rule it names. That is a
    weaker guarantee than the rest of this payload carries and it is the honest
    one available: the alternative is a fingerprint that silently certifies a
    measurement nobody made any more, which is what happened when the noise
    pool was narrowed under an unchanged corpus.

    Two kinds of thing are deliberately NOT in here, so nobody adds them later
    thinking it was an oversight:

    - **Per-run INPUTS** - the query prefix under test, the floors the seed
      declares, the floors the deployment has configured. Each one is recorded
      on the row itself, so a reading already says what it was judged against;
      folding them into the version would invalidate every stored verdict the
      moment somebody edited an environment file, having measured nothing new.
    - **:data:`_BATCH`**, which splits the embedding requests. It changes how
      many round trips a run makes and nothing about what comes back, so
      hashing it would retire verdicts for a performance knob.

    Returns:
        A short stable hex digest of every behavior-bearing constant and rule.
    """
    payload = json.dumps(
        {
            "corpus": [
                {
                    "name": spec.name,
                    "query": spec.query,
                    "related": spec.related,
                    "supporting": spec.supporting,
                    "unrelated": spec.unrelated,
                }
                for spec in CORPUS
            ],
            "related_headroom": _RELATED_HEADROOM,
            "unrelated_clearance": _UNRELATED_CLEARANCE,
            "margin_headroom": _MARGIN_HEADROOM,
            "min_rel_margin": _MIN_REL_MARGIN,
            "score_places": _SCORE_PLACES,
            "proposal_places": _PROPOSAL_PLACES,
            # The rules that live in code rather than in a constant. Without
            # them a narrowed noise pool, a re-derived proposal or a changed
            # gate simulation all leave the fingerprint untouched - which is
            # exactly how two rows measuring differently-defined bands came to
            # share one.
            "code_rules": _CODE_RULES,
            # The rules that live in code rather than in a constant. Without
            # them a narrowed noise pool, a re-derived proposal or a changed
            # gate simulation all leave the fingerprint untouched - which is
            # exactly how two rows measuring differently-defined bands came to
            # share one.
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def declared_floors(model: str) -> DeclaredFloors:
    """Read the floors the quirk seed declares for ``model``.

    The seed is this harness's first consumer of ``min_score``/``rel_margin``:
    they were written there as forward data, alongside the ``query_prefix`` they
    are only meaningful next to (a floor calibrated on one prefix scale says
    nothing on another).

    Args:
        model: The embedder tag, as configured or as the runtime lists it.

    Returns:
        What is declared, and whether anything is. A partial entry - a prefix
        but no floors - reports ``none`` rather than a source with nothing
        under it.
    """
    entry = quirks_for(model)
    if entry is None or (entry.min_score is None and entry.rel_margin is None):
        return DeclaredFloors()
    return DeclaredFloors(
        min_score=entry.min_score, rel_margin=entry.rel_margin, source="known_models"
    )


def resolve_query_prefix(model: str) -> PrefixChoice:
    """Decide which query wrapper governs a run of the harness, and say whose.

    One source: the quirk seed (:mod:`app.services.known_models`), which
    records the instruct wrapper an asymmetric embedder's own model card
    prescribes, beside the provenance for it. A tag the seed does not describe
    embeds its queries RAW, and the run reports source ``none`` with the
    comparison calling itself degenerate.

    The wrapper is a property of the WEIGHTS - it is what those weights were
    trained against - so there is nothing for a per-deployment file to say
    about it. A second home for the same string was the only way the harness
    and live retrieval could ever have measured different similarity scales,
    and one of them would have been validating floors nobody queries on.

    Args:
        model: The embedder tag under test.

    Returns:
        The governing prefix and the source that supplied it.
    """
    seeded = quirks_for(model)
    if seeded is not None and seeded.query_prefix:
        return PrefixChoice(prefix=seeded.query_prefix, source="known_models")
    return PrefixChoice(prefix="", source="none")


def _cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity, computed rather than assumed.

    Nothing at the seam promises unit-length vectors - that is a property of a
    particular backend's implementation, not of
    :meth:`~app.providers.base.ModelProvider.embed` - so the norms are divided
    out here. A zero vector scores 0.0 against everything rather than dividing
    by zero.

    Raises:
        ValueError: The two vectors have different dimensions, which means the
            provider answered inconsistently and no score would mean anything.
    """
    if len(left) != len(right):
        raise ValueError(
            f"The embedder returned vectors of different sizes ({len(left)} and "
            f"{len(right)}), so no similarity between them is meaningful."
        )
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _band(scores: list[float]) -> ScoreBand:
    """Summarize a family of scores; an empty family reports only its count."""
    if not scores:
        return ScoreBand(count=0)
    return ScoreBand(
        count=len(scores),
        minimum=min(scores),
        maximum=max(scores),
        mean=round(sum(scores) / len(scores), _SCORE_PLACES),
    )


async def _embed_all(
    provider: ModelProvider, texts: list[str], *, model: str, options: EmbedOptions | None
) -> list[list[float]]:
    """Embed every text in order, in batches.

    Raises:
        ValueError: The provider returned a different number of vectors than
            it was given texts - the one failure that would silently misalign
            every score in the run.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH):
        batch = texts[start : start + _BATCH]
        vectors.extend(await provider.embed(batch, model=model, options=options))
    if len(vectors) != len(texts):
        raise ValueError(
            f"The embedder returned {len(vectors)} vectors for {len(texts)} texts, "
            "so the scores could not be matched to the passages they came from."
        )
    return vectors


def _passage_catalogue() -> tuple[list[str], list[str], dict[str, set[str]]]:
    """Flatten the corpus into passage ids, texts, and each query's noise pool.

    The pool is the corpus MINUS this query's own answers and MINUS every other
    triplet's supporting passage - see the module docstring for why a
    near-paraphrase written for a different query is not honest noise, and why
    other triplets' related passages nevertheless stay in.

    Returns:
        The passage ids in order, their texts in the same order, and a map from
        triplet name to the ids that count as noise for that query.
    """
    ids: list[str] = []
    texts: list[str] = []
    own_answers: dict[str, set[str]] = {}
    supporting_ids: set[str] = set()
    for spec in CORPUS:
        own: set[str] = set()
        for label, text in (
            ("related", spec.related),
            ("supporting", spec.supporting),
            ("unrelated", spec.unrelated),
        ):
            if text is None:
                continue
            passage_id = f"{spec.name}/{label}"
            ids.append(passage_id)
            texts.append(text)
            if label != "unrelated":
                own.add(passage_id)
            if label == "supporting":
                supporting_ids.add(passage_id)
        own_answers[spec.name] = own
    noise = {
        spec.name: {
            passage_id
            for passage_id in ids
            if passage_id not in own_answers[spec.name] and passage_id not in supporting_ids
        }
        for spec in CORPUS
    }
    return ids, texts, noise


def _measure_arm(
    query_vectors: list[list[float]],
    passage_vectors: list[list[float]],
    passage_ids: list[str],
    noise_pools: dict[str, set[str]],
) -> ArmMeasurement:
    """Score every query against every passage and roll the result into bands.

    Retrieval ranks a query against everything stored, so a query is measured
    against the whole corpus - but only the passages in its NOISE POOL count
    toward the unrelated band (see :func:`_passage_catalogue`). Every score is
    kept per triplet regardless, because the composed gate has to know what the
    top hit was even when the top hit is a passage no band counts.
    """
    related_scores: list[float] = []
    supporting_scores: list[float] = []
    unrelated_scores: list[float] = []
    triplets: list[TripletMeasurement] = []
    by_id = dict(zip(passage_ids, passage_vectors, strict=True))
    for spec, query_vector in zip(CORPUS, query_vectors, strict=True):
        pool = noise_pools[spec.name]
        scores = {
            passage_id: round(_cosine(query_vector, vector), _SCORE_PLACES)
            for passage_id, vector in by_id.items()
        }
        related = scores[f"{spec.name}/related"]
        related_scores.append(related)
        supporting: float | None = None
        if spec.supporting is not None:
            supporting = scores[f"{spec.name}/supporting"]
            supporting_scores.append(supporting)
        noise = [(score, passage_id) for passage_id, score in scores.items() if passage_id in pool]
        unrelated_scores.extend(score for score, _ in noise)
        best_score, best_id = max(noise)
        triplets.append(
            TripletMeasurement(
                name=spec.name,
                related=related,
                supporting=supporting,
                best_unrelated=best_score,
                best_unrelated_passage=best_id,
                scores=scores,
                noise_pool=frozenset(pool),
            )
        )
    related_band = _band(related_scores)
    unrelated_band = _band(unrelated_scores)
    separation = round(
        (related_band.minimum or 0.0) - (unrelated_band.maximum or 0.0), _SCORE_PLACES
    )
    return ArmMeasurement(
        related=related_band,
        supporting=_band(supporting_scores),
        unrelated=unrelated_band,
        separation=separation,
        triplets=tuple(triplets),
    )


def compose_gate(arm: ArmMeasurement, *, min_score: float, rel_margin: float) -> ComposedGate:
    """Simulate what these two floors, applied TOGETHER, would retrieve.

    This is the verdict. It mirrors
    :func:`app.services.rag_service.retrieve`'s dense-only path exactly, per
    query: score every passage, keep those at or above ``min_score``, take the
    best survivor, then drop anything more than ``rel_margin`` below it. Two
    things are then asked, and they are the two things a user of retrieval
    would notice:

    - **Was the real answer retrieved?** Losing it is the worse failure - the
      deployment answers from nothing and nobody is told.
    - **Did any noise come with it?** A retrieved noise passage is a passage
      the answer may cite.

    A supporting passage the margin prunes is neither: it is the margin doing
    its documented job on a corpus whose supporting passages are deliberately
    weak, and it is reported as a TRADE.

    Two deliberate simplifications, both in the safe direction. The top-k cut
    is NOT simulated - it is a per-request cap rather than a relevance decision,
    and a noise passage that survives the floors has been admitted by them
    whether or not a limit later trims it. Nor is the RERANKER path, which
    skips the margin entirely: the floors are what this harness is about, and
    the dense-only path is the one they govern (and the one every tight-VRAM
    tier actually runs).

    Args:
        arm: The measured arm to simulate on, normally the wrapped one.
        min_score: The absolute floor to compose.
        rel_margin: The relative margin to compose.

    Returns:
        The :class:`ComposedGate` verdict, per query and in summary.
    """
    gates: list[QueryGate] = []
    leaks: list[str] = []
    related_lost: list[str] = []
    pruned: list[str] = []
    for triplet in arm.triplets:
        ranked = sorted(triplet.scores.items(), key=lambda item: (-item[1], item[0]))
        top_passage, top_score = ranked[0]
        survivors = [(name, score) for name, score in ranked if score >= min_score]
        if survivors and rel_margin > 0:
            # The floor is measured from the best SURVIVOR, exactly as
            # rag_service does - not from the best score overall, which may
            # have been cut by min_score a line earlier.
            floor = round(survivors[0][1] - rel_margin, _SCORE_PLACES)
            retrieved = [(name, score) for name, score in survivors if score >= floor]
        else:
            floor = min_score
            retrieved = survivors
        effective_floor = round(max(min_score, floor), _SCORE_PLACES)
        kept = {name for name, _score in retrieved}
        related_id = f"{triplet.name}/related"
        supporting_id = f"{triplet.name}/supporting"
        noise_in = tuple(name for name, _score in retrieved if name in triplet.noise_pool)
        related_ok = related_id in kept
        supporting_ok = None if triplet.supporting is None else supporting_id in kept
        best_noise = max(
            (score for name, score in triplet.scores.items() if name in triplet.noise_pool),
            default=0.0,
        )
        gates.append(
            QueryGate(
                name=triplet.name,
                top_score=top_score,
                top_passage=top_passage,
                effective_floor=effective_floor,
                related_retrieved=related_ok,
                supporting_retrieved=supporting_ok,
                noise_retrieved=noise_in,
                noise_clearance=round(effective_floor - best_noise, _SCORE_PLACES),
            )
        )
        leaks.extend(f"{triplet.name}: {name}" for name in noise_in)
        if not related_ok:
            related_lost.append(triplet.name)
        if supporting_ok is False:
            pruned.append(triplet.name)
    holds = not leaks and not related_lost
    tightest = min((gate.noise_clearance for gate in gates), default=0.0)
    if related_lost:
        detail = (
            f"the floors excluded the real answer for {len(related_lost)} of "
            f"{len(gates)} queries ({', '.join(related_lost)}) - these questions "
            "would be answered without their grounding"
        )
    elif leaks:
        detail = (
            f"{len(leaks)} noise passage(s) survived both floors and would be "
            f"retrievable: {', '.join(leaks)}"
        )
    else:
        detail = (
            f"every real answer was retrieved and no noise came with it across "
            f"{len(gates)} queries; the tightest query cleared its strongest "
            f"noise by {tightest}"
        )
    if pruned:
        # Appended to a PASSING line as readily as a failing one: it is a fact
        # about what the margin costs, never a fault.
        detail += (
            f". Trade: the margin prunes the weaker supporting passage on "
            f"{len(pruned)} query/queries ({', '.join(pruned)}), which is what a "
            "relative margin is for"
        )
    return ComposedGate(
        min_score=min_score,
        rel_margin=rel_margin,
        queries=tuple(gates),
        leaks=tuple(leaks),
        related_lost=tuple(related_lost),
        supporting_pruned=tuple(pruned),
        holds=holds,
        detail=detail,
    )


def validate_floors(
    arm: ArmMeasurement, *, min_score: float | None, rel_margin: float | None = None
) -> tuple[FloorCheck, ...]:
    """Measure the declared floors against the bands - DIAGNOSTICS, not a verdict.

    Two properties, each a distinct way an absolute floor goes wrong:

    - ``min_score_admits_related``: the weakest genuinely relevant passage sits
      at least :data:`_RELATED_HEADROOM` above the floor. A floor above it
      silently drops true answers, which reads as a model that "did not find
      anything" rather than as a misconfiguration.
    - ``min_score_excludes_unrelated``: the strongest noise passage sits at
      least :data:`_UNRELATED_CLEARANCE` BELOW it.

    Neither decides anything, and the second one is why. Retrieval composes the
    floors, so noise above ``min_score`` is only a problem if it also survives
    the relative margin - which on the first live run it did not, on any query.
    These lines say WHERE a composed failure comes from; :func:`compose_gate`
    says WHETHER there is one.

    The relative margin has no band check of its own. Its effect is entirely
    relational - which passages survive relative to which top hit - so there is
    nothing about a single band it can be measured against, and the composed
    gate is its whole test.

    Args:
        arm: The measured arm to judge, normally the wrapped one.
        min_score: The absolute floor to measure, or None to skip both checks.
        rel_margin: Accepted and unused, so callers can pass a whole floor pair;
            see above for why the margin has no band check.

    Returns:
        One check per measurable property, in a stable order.
    """
    del rel_margin  # relational, not a band property - see the docstring
    checks: list[FloorCheck] = []
    related_min = arm.related.minimum
    unrelated_max = arm.unrelated.maximum
    if min_score is not None and related_min is not None:
        headroom = round(related_min - min_score, _SCORE_PLACES)
        checks.append(
            FloorCheck(
                name="min_score_admits_related",
                passed=headroom >= _RELATED_HEADROOM,
                detail=(
                    f"the weakest related passage scored {related_min}, "
                    f"{headroom} above the floor {min_score} "
                    f"(at least {_RELATED_HEADROOM} is wanted)"
                ),
            )
        )
    if min_score is not None and unrelated_max is not None:
        clearance = round(min_score - unrelated_max, _SCORE_PLACES)
        checks.append(
            FloorCheck(
                name="min_score_excludes_unrelated",
                passed=clearance >= _UNRELATED_CLEARANCE,
                detail=(
                    f"the strongest noise passage scored {unrelated_max}, "
                    f"{clearance} below the floor {min_score} "
                    f"(at least {_UNRELATED_CLEARANCE} is wanted); informational - "
                    "the composed gate decides whether this actually leaks"
                ),
            )
        )
    return tuple(checks)


# Appended to every proposal, verbatim. The first live run derived 0.51 from a
# corpus whose own production calibration had measured real relevant chunks at
# 0.40-0.60 - a floor that would have excluded half of them. The numbers are
# still worth reporting; what must never travel without them is why they cannot
# be pasted into configuration.
_TRANSPLANT_CAVEAT = (
    "These numbers describe THIS corpus. Its passages are purpose-written "
    "question-and-answer text and score systematically higher than production "
    "prose, so a floor derived here transplants DOWNWARD badly: a live "
    "calibration on real content measured genuinely relevant chunks at "
    "0.40-0.60, which a corpus-derived floor can sit above. Read it as an "
    "upper bound to compare a real calibration against, never as a value to "
    "copy into RAG_MIN_SCORE."
)


def propose_floors(arm: ArmMeasurement) -> FloorProposal:
    """Derive a floor pair from the measured bands, and check its own work.

    The derivation, stated once and applied literally:

    - ``min_score`` is the LOWEST floor that still clears this corpus's noise:
      the noise peak plus :data:`_UNRELATED_CLEARANCE`, rounded UP so the
      rounding cannot land back inside the clearance. Deliberately the
      conservative end of the corridor rather than its midpoint, and the reason
      is the caveat every proposal carries: this corpus scores HIGH, so a floor
      placed mid-corridor here sits above scores that real content produces.
      Erring low costs some noise the relative margin then has to catch; erring
      high costs grounding, silently. Only the second failure is invisible.
      The floor must still leave :data:`_RELATED_HEADROOM` under the weakest
      related passage, and yields NO proposal when it cannot - on such a corpus
      no absolute floor separates the two bands, and inventing one would hide
      that.
    - ``rel_margin`` is the WIDEST margin the composed gate survives: the
      smallest per-query distance from the top hit down to that query's
      strongest noise, less :data:`_MARGIN_HEADROOM`. Widest rather than
      narrowest because a wider margin keeps more corroborating context, and
      the binding constraint is the only one that matters - it must not reach
      down far enough to admit noise on any query.

    The pair is then run back through :func:`compose_gate` - the same
    simulation a declared pair faces - so ``validates`` reports what the
    proposal would actually retrieve rather than asserting that the arithmetic
    was done right.

    Args:
        arm: The measured arm to derive from, normally the wrapped one.

    Returns:
        The proposal, with its derivation spelled out in numbers.
    """
    related_min = arm.related.minimum
    unrelated_max = arm.unrelated.maximum
    reasons: list[str] = []
    min_score: float | None = None
    if related_min is None or unrelated_max is None:
        reasons.append("the corpus produced no scores to derive an absolute floor from")
    elif arm.separation < _RELATED_HEADROOM + _UNRELATED_CLEARANCE:
        reasons.append(
            f"the related band bottoms out at {related_min} while noise peaks at "
            f"{unrelated_max}, a corridor of {arm.separation} - too narrow for any "
            f"floor to separate them (at least "
            f"{round(_RELATED_HEADROOM + _UNRELATED_CLEARANCE, _SCORE_PLACES)} is wanted)"
        )
    else:
        # Rounded UP: rounding to the two decimals a human types must not undo
        # the clearance the derivation just guaranteed.
        scale = 10**_PROPOSAL_PLACES
        min_score = math.ceil((unrelated_max + _UNRELATED_CLEARANCE) * scale) / scale
        reasons.append(
            f"min_score {min_score} is the LOWEST floor clearing this corpus's noise "
            f"peak {unrelated_max} by {_UNRELATED_CLEARANCE} (the conservative end of "
            f"the corridor up to {related_min}, deliberately - see the caveat)"
        )
    rel_margin: float | None = None
    # The distance each query can afford to reach down before it starts
    # retrieving its own noise. The tightest one binds.
    room = [
        round(gate_top - best_noise, _SCORE_PLACES)
        for gate_top, best_noise in (
            (
                triplet.scores.get(f"{triplet.name}/related", triplet.related),
                max(
                    (score for name, score in triplet.scores.items() if name in triplet.noise_pool),
                    default=0.0,
                ),
            )
            for triplet in arm.triplets
        )
    ]
    if room and min_score is not None:
        widest = round(min(room) - _MARGIN_HEADROOM, _PROPOSAL_PLACES)
        if widest >= _MIN_REL_MARGIN:
            rel_margin = widest
            reasons.append(
                f"rel_margin {rel_margin} is the widest margin no query leaks at - the "
                f"tightest query sits {min(room)} above its own strongest noise, less "
                f"{_MARGIN_HEADROOM}"
            )
        else:
            reasons.append(
                f"no relative margin is proposed: the tightest query sits only "
                f"{min(room)} above its own strongest noise, so any usable margin "
                "would reach into it"
            )
    else:
        reasons.append("no absolute floor was derived, so no margin is derived beside it")
    if min_score is not None and rel_margin is not None:
        gate = compose_gate(arm, min_score=min_score, rel_margin=rel_margin)
        validates = gate.holds
        if not validates:
            reasons.append(f"the proposal does NOT hold when composed: {gate.detail}")
    else:
        validates = False
    return FloorProposal(
        min_score=min_score,
        rel_margin=rel_margin,
        derivation="; ".join(reasons),
        validates=validates,
        caveat=_TRANSPLANT_CAVEAT,
    )


def judge_floors(arm: ArmMeasurement, declared: DeclaredFloors) -> FloorsVerdict:
    """Compose what is declared, diagnose the bands beside it, propose where needed.

    ``hold`` is the COMPOSED gate's answer and nothing else. The band checks
    ride along as diagnostics and never decide: the first live run had a band
    check failing on a deployment whose composed gate leaked nothing at all,
    and reporting that as broken would have sent somebody to change a healthy
    configuration.

    A proposal accompanies a failed composed gate deliberately. A verdict that
    says only "these floors do not work on this embedder" leaves a reader with
    numbers to distrust and nothing to replace them with, and the bands that
    failed are the same bands a proposal is derived from.

    A pair with only ONE floor declared cannot be composed - retrieval applies
    both - so it is reported as unjudged with a proposal beside it, rather than
    composed against a value nobody declared.

    Args:
        arm: The measured arm to judge, normally the wrapped one.
        declared: What the seed declares for this tag.

    Returns:
        The verdict: the composed gate, the diagnostics, and a proposal when
        one is warranted.
    """
    checks = validate_floors(arm, min_score=declared.min_score)
    composed: ComposedGate | None = None
    if declared.min_score is not None and declared.rel_margin is not None:
        composed = compose_gate(arm, min_score=declared.min_score, rel_margin=declared.rel_margin)
    hold = composed.holds if composed is not None else None
    proposal = propose_floors(arm) if hold is not True else None
    return FloorsVerdict(
        declared=declared,
        composed=composed,
        checks=checks,
        hold=hold,
        proposal=proposal,
    )


async def run_embedder_check(
    provider: ModelProvider,
    *,
    model: str,
    query_prefix: str,
    prefix_source: PrefixSource,
    options: EmbedOptions | None = None,
) -> EmbedderCheckSummary:
    """Measure ``model``'s score bands and judge the floors against them.

    Three embedding passes, in one shape: every passage RAW (documents always
    embed raw, so measuring them any other way would measure a pipeline nobody
    runs), every query WRAPPED in the prefix under test, and every query raw
    again for the comparison. The wrapped arm is the one the floors are judged
    on, because it is the one :meth:`ProviderEmbedder.embed_query
    <app.providers.rag.embedder.ProviderEmbedder.embed_query>` produces on a
    live retrieval.

    Args:
        provider: The model provider to drive (the real one, or a fake).
        model: The exact embedder tag to measure.
        query_prefix: The instruct wrapper under test; "" runs both arms the
            same and marks the comparison degenerate.
        prefix_source: Which source supplied that prefix, carried through to
            the report so a reader can tell a seeded wrapper from none at all.
        options: Placement options for the embed calls - normally the same
            device pin, so the probe lands where retrieval's own embedding work
            lands.

    Returns:
        The :class:`EmbedderCheckSummary` for this run.

    Raises:
        ValueError: The provider answered with a vector count or a dimension
            that no score could be matched to.
        ProviderUnavailableError: The embedding backend is unreachable, or the
            model is not pulled. Deliberately propagated: a run that could not
            embed measured nothing, and a band derived from a failure would
            invalidate correct floors.
    """
    passage_ids, passage_texts, noise_pools = _passage_catalogue()
    queries = [spec.query for spec in CORPUS]
    passage_vectors = await _embed_all(provider, passage_texts, model=model, options=options)
    wrapped_vectors = await _embed_all(
        provider, [f"{query_prefix}{query}" for query in queries], model=model, options=options
    )
    raw_vectors = await _embed_all(provider, queries, model=model, options=options)
    wrapped = _measure_arm(wrapped_vectors, passage_vectors, passage_ids, noise_pools)
    raw = _measure_arm(raw_vectors, passage_vectors, passage_ids, noise_pools)
    delta = round(wrapped.separation - raw.separation, _SCORE_PLACES)
    comparison = PrefixComparison(
        source=prefix_source,
        prefix=query_prefix,
        wrapped_separation=wrapped.separation,
        raw_separation=raw.separation,
        delta=delta,
        improves_separation=delta > 0,
        degenerate=not query_prefix,
    )
    return EmbedderCheckSummary(
        model=model,
        depth=PROBE_DEPTH,
        prefix=comparison,
        wrapped=wrapped,
        raw=raw,
        floors=judge_floors(wrapped, declared_floors(model)),
        harness_version=harness_version(),
        # Digest deliberately unset: the runner captures it at persist time
        # (the harness stays pure), binding the reading to the weights it saw.
        digest=None,
    )
