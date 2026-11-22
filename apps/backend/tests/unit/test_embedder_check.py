"""Pure-service tests for the embedder-check harness.

Every test drives the harness with an embedder whose vectors are BUILT, not
hashed: each triplet gets its own three orthogonal axes, so the cosine between
any query and any passage is a number the test wrote down rather than one it
had to discover. That is what makes the band arithmetic, the floors validation
and the prefix comparison assertable to the fourth decimal - a random or
hash-derived embedder would only let a test check that numbers exist.

The layout, per triplet i:

- axis ``3i`` is the QUERY direction,
- axis ``3i+1`` is the direction its true answers lean off into,
- axis ``3i+2`` is the direction its distractor leans off into.

A wrapped query is the pure query axis, so its cosine with a passage is that
passage's own leading coefficient. A raw query is tilted toward the distractor
axis, which is how the corpus models an unwrapped query on an asymmetric
embedder: the true answer drops a little and the noise climbs a lot.
"""

from __future__ import annotations

import math

import pytest

from app.providers.base import EmbedOptions
from app.providers.fake_provider import FakeProvider
from app.services import embedder_check
from app.services.embedder_check import (
    CORPUS,
    PROBE_DEPTH,
    ArmMeasurement,
    ScoreBand,
    TripletMeasurement,
    TripletSpec,
    compose_gate,
    declared_floors,
    harness_version,
    propose_floors,
    resolve_query_prefix,
    run_embedder_check,
)

# The seed tag whose floors (0.35 / 0.10) this harness is the first consumer of.
SEEDED = "qwen3-embedding:0.6b"
UNSEEDED = "some-other-embedder:1b"
PREFIX = "Instruct: retrieve passages\nQuery:"

DIM = 3 * len(CORPUS)

# The healthy embedder's shape. Related passages start at BASE_RELATED and
# climb by STEP per triplet, distractors start at BASE_UNRELATED and fall by
# the same - so the bands have real spread and their edges belong to a
# NAMEABLE triplet rather than being one repeated number.
BASE_RELATED = 0.56
BASE_UNRELATED = 0.30
STEP = 0.01
# The three triplets that carry a supporting passage, in corpus order, drop
# this far below their own related passage. The widest drop is what a proposed
# relative margin has to clear.
SUPPORT_DROPS = (0.04, 0.06, 0.08)
# How much of a raw (unwrapped) query still points at the query axis; the rest
# leans into the distractor axis.
RAW_ALONG = 0.97
# How hard each supporting passage crowds the NEXT triplet's query. Set above
# every distractor's score, mirroring the live run where a supporting passage
# answering somebody else's question WAS the noise peak - so a harness that
# counted foreign supporting passages as noise would report this number as the
# unrelated band's maximum.
SUPPORT_BLEED = 0.45

SUPPORTING_INDEXES = tuple(
    index for index, spec in enumerate(CORPUS) if spec.supporting is not None
)


def unit(index: int, along: float, leaning: int, *, bleed: float = 0.0) -> list[float]:
    """A unit vector in triplet ``index``'s own plane, optionally leaking into the next.

    Args:
        index: Which triplet's axes to build in.
        along: The coefficient on the query axis - which IS the cosine against
            that triplet's wrapped query.
        leaning: 1 for the true-answer axis, 2 for the distractor axis.
        bleed: A component on the NEXT triplet's query axis, which is exactly
            the cosine this passage scores against that unrelated query. The
            live corpus does this for real - a supporting passage answering a
            different triplet's question was the whole noise peak - so the fake
            has to be able to, or nothing here can tell the noise-pool rule
            from a no-op.

    Returns:
        A unit vector, zero outside this triplet's three axes and (when
        ``bleed`` is set) the next triplet's query axis.
    """
    vector = [0.0] * DIM
    vector[3 * index] = along
    if bleed:
        vector[3 * ((index + 1) % len(CORPUS))] = bleed
    vector[3 * index + leaning] = math.sqrt(max(0.0, 1.0 - along * along - bleed * bleed))
    return vector


class ScriptedEmbedder(FakeProvider):
    """An embedder with hand-built vectors, keyed by the exact text embedded.

    Wrapped and raw queries map to DIFFERENT vectors, which is what makes a
    harness that silently dropped the prefix visible: the wrapped arm would
    come back holding the raw arm's numbers.
    """

    def __init__(
        self,
        *,
        prefix: str = PREFIX,
        related: float = BASE_RELATED,
        unrelated: float = BASE_UNRELATED,
        step: float = STEP,
        support_drops: tuple[float, ...] = SUPPORT_DROPS,
        raw_along: float = RAW_ALONG,
        support_bleed: float = SUPPORT_BLEED,
    ) -> None:
        super().__init__()
        self.calls: list[tuple[str | None, EmbedOptions | None, list[str]]] = []
        self.related = [round(related + step * index, 6) for index in range(len(CORPUS))]
        self.unrelated = [round(unrelated - step * index, 6) for index in range(len(CORPUS))]
        self.raw_along = raw_along
        drops = dict(zip(SUPPORTING_INDEXES, support_drops, strict=True))
        self.supporting = {
            index: round(self.related[index] - drop, 6) for index, drop in drops.items()
        }
        table: dict[str, list[float]] = {}
        for index, spec in enumerate(CORPUS):
            table[spec.related] = unit(index, self.related[index], 1)
            table[spec.unrelated] = unit(index, self.unrelated[index], 2)
            if spec.supporting is not None:
                # Bled into the NEXT triplet's query on purpose: a supporting
                # passage that crowds somebody else's question is the exact
                # shape the live corpus produced, and the only shape that can
                # tell the noise-pool rule apart from a no-op.
                table[spec.supporting] = unit(index, self.supporting[index], 1, bleed=support_bleed)
            # The raw key is written FIRST so an empty prefix (where the two
            # keys are the same string) leaves the wrapped vector in place -
            # the degenerate case the harness reports as such.
            table[spec.query] = unit(index, raw_along, 2)
            table[f"{prefix}{spec.query}"] = unit(index, 1.0, 1)
        self._table = table

    async def embed(self, texts, *, model=None, options=None):
        self.calls.append((model, options, list(texts)))
        return [self._table[text] for text in texts]

    def expected_related(self, index: int, *, wrapped: bool = True) -> float:
        """The cosine this embedder produces between a query and its own answer."""
        value = self.related[index]
        return round(value if wrapped else self.raw_along * value, 4)

    def expected_unrelated(self, index: int, *, wrapped: bool = True) -> float:
        """The cosine between a query and its own distractor."""
        value = self.unrelated[index]
        if wrapped:
            return round(value, 4)
        across = math.sqrt(1.0 - self.raw_along**2)
        return round(self.raw_along * value + across * math.sqrt(1.0 - value * value), 4)


async def check(provider: ScriptedEmbedder, *, model: str = SEEDED, prefix: str = PREFIX):
    """Run the harness the way the deployment runner does."""
    return await run_embedder_check(
        provider,
        model=model,
        query_prefix=prefix,
        prefix_source="catalog" if prefix else "none",
    )


def measured(
    name: str, *, related: float, noise: float, supporting: float | None = None
) -> TripletMeasurement:
    """One measured triplet with an explicit score map, as an arm builds it.

    The composed gate ranks the whole score map, so a hand-built triplet has to
    carry one - a fixture that only set the band summaries would let the gate
    simulate a ranking the test never described.
    """
    scores = {f"{name}/related": related, f"{name}/unrelated": noise}
    if supporting is not None:
        scores[f"{name}/supporting"] = supporting
    return TripletMeasurement(
        name=name,
        related=related,
        supporting=supporting,
        best_unrelated=noise,
        best_unrelated_passage=f"{name}/unrelated",
        scores=scores,
        noise_pool=frozenset({f"{name}/unrelated"}),
    )


def arm(*triplets: TripletMeasurement) -> ArmMeasurement:
    """An arm assembled from hand-built triplets, bands derived as the harness does."""
    related = [t.related for t in triplets]
    supporting = [t.supporting for t in triplets if t.supporting is not None]
    noise = [t.best_unrelated for t in triplets]

    def band(values: list[float]) -> ScoreBand:
        if not values:
            return ScoreBand(count=0)
        return ScoreBand(
            count=len(values),
            minimum=min(values),
            maximum=max(values),
            mean=round(sum(values) / len(values), 4),
        )

    return ArmMeasurement(
        related=band(related),
        supporting=band(supporting),
        unrelated=band(noise),
        separation=round(min(related) - max(noise), 4),
        triplets=triplets,
    )


def test_the_corpus_is_varied_and_carries_measurable_margins() -> None:
    # The corpus is the harness: a duplicated query or a triplet whose
    # "unrelated" passage answers it would move every band silently.
    assert 6 <= len(CORPUS) <= 10
    names = [spec.name for spec in CORPUS]
    assert len(set(names)) == len(names)
    texts = [
        text
        for spec in CORPUS
        for text in (spec.query, spec.related, spec.unrelated, spec.supporting)
        if text is not None
    ]
    # Two identical passages would embed identically, so one of them would
    # score as another query's noise at exactly the strength of a true answer.
    assert len(set(texts)) == len(texts)
    # The relative margin can only be measured where a query has more than one
    # true answer, so at least one triplet has to carry a second.
    assert len(SUPPORTING_INDEXES) >= 2


async def test_bands_are_measured_from_the_vectors_the_embedder_returned() -> None:
    provider = ScriptedEmbedder()
    summary = await check(provider)

    assert summary.model == SEEDED
    assert summary.depth == PROBE_DEPTH
    wrapped = summary.wrapped
    assert wrapped.related.count == len(CORPUS)
    assert wrapped.related.minimum == pytest.approx(provider.expected_related(0))
    assert wrapped.related.maximum == pytest.approx(provider.expected_related(len(CORPUS) - 1))
    # The noise pool is the whole corpus minus each query's own answers, so it
    # is far larger than one distractor per query - and its peak is the one
    # distractor that shares a plane with its query.
    assert wrapped.unrelated.count > len(CORPUS)
    assert wrapped.unrelated.maximum == pytest.approx(provider.expected_unrelated(0))
    assert wrapped.separation == pytest.approx(
        provider.expected_related(0) - provider.expected_unrelated(0)
    )
    # Every triplet reports which passage crowded it hardest, so a failing band
    # names a text rather than only a number.
    first = wrapped.triplets[0]
    assert first.name == CORPUS[0].name
    assert first.best_unrelated_passage == f"{CORPUS[0].name}/unrelated"
    assert first.supporting is None
    seeded_index = SUPPORTING_INDEXES[0]
    assert wrapped.triplets[seeded_index].supporting == pytest.approx(
        provider.supporting[seeded_index]
    )


async def test_documents_embed_raw_and_the_device_pin_reaches_the_seam() -> None:
    provider = ScriptedEmbedder()
    await run_embedder_check(
        provider,
        model=SEEDED,
        query_prefix=PREFIX,
        prefix_source="catalog",
        options=EmbedOptions(device="cpu"),
    )
    embedded = [text for _model, _options, texts in provider.calls for text in texts]
    # Passages go in verbatim: the live pipeline embeds documents raw, so a
    # harness that wrapped them would measure a scale nothing produces.
    assert CORPUS[0].related in embedded
    assert f"{PREFIX}{CORPUS[0].related}" not in embedded
    assert all(
        options is not None and options.device == "cpu" for _m, options, _t in provider.calls
    )
    assert {model for model, _options, _texts in provider.calls} == {SEEDED}


async def test_the_wrapped_arm_actually_applies_the_prefix() -> None:
    # The mutation this pins: drop the wrap in run_embedder_check and the
    # wrapped arm silently becomes a second raw arm - same separation, no
    # delta, and a floors verdict taken on a scale live retrieval never sees.
    provider = ScriptedEmbedder()
    summary = await check(provider)

    assert f"{PREFIX}{CORPUS[0].query}" in [
        text for _m, _o, texts in provider.calls for text in texts
    ]
    assert summary.wrapped.separation == pytest.approx(
        provider.expected_related(0) - provider.expected_unrelated(0)
    )
    assert summary.raw.separation == pytest.approx(
        provider.expected_related(0, wrapped=False) - provider.expected_unrelated(0, wrapped=False)
    )
    assert summary.raw.separation < summary.wrapped.separation
    assert summary.prefix.improves_separation is True
    assert summary.prefix.delta == pytest.approx(
        summary.wrapped.separation - summary.raw.separation
    )
    assert summary.prefix.degenerate is False
    assert summary.prefix.source == "catalog"


async def test_no_prefix_makes_the_comparison_degenerate_rather_than_flattering() -> None:
    provider = ScriptedEmbedder(prefix="")
    summary = await check(provider, prefix="")

    assert summary.prefix.degenerate is True
    assert summary.prefix.delta == 0.0
    # Not "the prefix helped": there was no prefix, and reporting a tie as an
    # improvement would recommend a wrapper that does not exist.
    assert summary.prefix.improves_separation is False
    assert summary.wrapped.separation == summary.raw.separation


async def test_a_foreign_supporting_passage_is_not_counted_as_noise() -> None:
    # The corpus fix, driven end to end. Every supporting passage here crowds
    # the NEXT triplet's query at 0.45 - harder than any distractor - which is
    # the shape the live run produced and mistook for the embedder's noise
    # floor. Counting those as noise inflates the unrelated band, widens the
    # apparent corridor, and pushes every derived floor up with it.
    provider = ScriptedEmbedder()
    summary = await check(provider)

    # The band peaks at the strongest DISTRACTOR, not at the bleed.
    assert summary.wrapped.unrelated.maximum == pytest.approx(provider.expected_unrelated(0))
    assert summary.wrapped.unrelated.maximum < SUPPORT_BLEED
    # And no triplet names somebody else's supporting passage as its crowder.
    crowders = {triplet.best_unrelated_passage for triplet in summary.wrapped.triplets}
    assert not any(name.endswith("/supporting") for name in crowders)
    # The pool still holds the other triplets' RELATED passages, deliberately:
    # a pool with no topical adjacency measures an unrealistically easy problem.
    pooled = summary.wrapped.triplets[0].noise_pool
    assert f"{CORPUS[1].name}/related" in pooled
    assert f"{CORPUS[1].name}/supporting" not in pooled
    assert f"{CORPUS[0].name}/unrelated" in pooled


async def test_healthy_floors_hold_when_composed() -> None:
    provider = ScriptedEmbedder()
    summary = await check(provider)

    floors = summary.floors
    assert (floors.declared.min_score, floors.declared.rel_margin) == (0.35, 0.10)
    assert floors.declared.source == "known_models"
    composed = floors.composed
    assert composed is not None
    assert composed.holds is True
    assert composed.leaks == ()
    assert composed.related_lost == ()
    # Every real answer came back and nothing else did.
    assert all(gate.related_retrieved for gate in composed.queries)
    assert len(composed.queries) == len(CORPUS)
    # hold IS the composed answer, and nothing is proposed over a gate that holds.
    assert floors.hold is True
    assert floors.proposal is None


async def test_noise_above_the_floor_is_a_diagnostic_when_the_margin_catches_it() -> None:
    # The finding that reshaped this harness. On the first live run the noise
    # peak sat ABOVE min_score while the composed gate leaked nothing at all,
    # because the relative margin was doing the work. A verdict taken on the
    # band check alone would have called a healthy deployment broken and sent
    # somebody to change a floor that was fine.
    provider = ScriptedEmbedder(unrelated=0.40)
    summary = await check(provider)

    excludes = next(c for c in summary.floors.checks if c.name == "min_score_excludes_unrelated")
    assert excludes.passed is False  # 0.40 noise against a 0.35 floor
    assert "informational" in excludes.detail
    composed = summary.floors.composed
    assert composed is not None
    # The margin floor is 0.56 - 0.10 = 0.46, comfortably above the 0.40 noise.
    assert composed.leaks == ()
    assert composed.holds is True
    assert summary.floors.hold is True
    assert summary.floors.proposal is None


async def test_noise_the_margin_cannot_catch_fails_the_composed_gate() -> None:
    # The mutation the check above must not swallow: push the noise up until it
    # survives BOTH floors and the gate has to fail. Top hit 0.56, margin floor
    # 0.46, noise 0.50 - through the coarse gate and through the margin.
    provider = ScriptedEmbedder(unrelated=0.50)
    summary = await check(provider)

    composed = summary.floors.composed
    assert composed is not None
    assert composed.holds is False
    assert composed.leaks  # named, so a reader knows which passage got through
    assert "light-travel-time/unrelated" in composed.leaks[0]
    assert summary.floors.hold is False
    # A failing gate always comes with something to do about it.
    assert summary.floors.proposal is not None


async def test_a_floor_above_the_related_band_loses_the_answer_outright() -> None:
    # The worse failure of the two, and the silent one: the floor excludes the
    # real answer, so the question is answered with no grounding and nothing
    # anywhere reports it.
    provider = ScriptedEmbedder(related=0.30, unrelated=0.20)
    summary = await check(provider)

    composed = summary.floors.composed
    assert composed is not None
    assert composed.holds is False
    assert CORPUS[0].name in composed.related_lost
    assert "excluded the real answer" in composed.detail
    assert summary.floors.hold is False


async def test_a_pruned_supporting_passage_is_a_trade_not_a_failure() -> None:
    # Re-scoped deliberately. The margin drops hits far below the top result -
    # that is its documented job - and this corpus's supporting passages are
    # written weak. Reporting the pruning as a FAILURE would condemn a margin
    # for working, on evidence the corpus manufactured.
    provider = ScriptedEmbedder(support_drops=(0.04, 0.06, 0.20))
    summary = await check(provider)

    composed = summary.floors.composed
    assert composed is not None
    assert composed.supporting_pruned == (CORPUS[SUPPORTING_INDEXES[2]].name,)
    assert "Trade:" in composed.detail
    # Pruning changes nothing about the verdict.
    assert composed.holds is True
    assert summary.floors.hold is True
    assert summary.floors.proposal is None


async def test_an_unknown_embedder_gets_a_conservative_proposal_that_composes() -> None:
    provider = ScriptedEmbedder()
    summary = await check(provider, model=UNSEEDED)

    floors = summary.floors
    assert floors.declared.source == "none"
    assert floors.checks == ()  # nothing was claimed, so nothing was measured against
    assert floors.composed is None  # nothing declared, so nothing to compose
    assert floors.hold is None
    proposal = floors.proposal
    assert proposal is not None
    # The LOWEST floor clearing the noise peak, not the midpoint of the
    # corridor: this corpus scores high, and a mid-corridor floor derived here
    # sits above scores real content produces.
    noise_peak = summary.wrapped.unrelated.maximum
    assert proposal.min_score == pytest.approx(math.ceil((noise_peak + 0.03) * 100) / 100)
    assert proposal.min_score < (noise_peak + summary.wrapped.related.minimum) / 2
    # The widest margin no query leaks at, less the headroom.
    assert proposal.rel_margin is not None
    assert proposal.validates is True
    assert "LOWEST floor" in proposal.derivation
    assert "widest margin no query leaks at" in proposal.derivation


async def test_every_proposal_carries_the_transplant_caveat() -> None:
    # The live run proposed 0.51 against a production calibration that had
    # measured real relevant chunks at 0.40-0.60. The numbers are still worth
    # reporting; what must never travel without them is why they cannot be
    # pasted into configuration.
    summary = await check(ScriptedEmbedder(), model=UNSEEDED)
    proposal = summary.floors.proposal
    assert proposal is not None
    assert "0.40-0.60" in proposal.caveat
    assert "upper bound" in proposal.caveat
    assert "RAG_MIN_SCORE" in proposal.caveat


async def test_overlapping_bands_refuse_a_floor_instead_of_inventing_one() -> None:
    # No absolute floor separates bands that overlap, and proposing one anyway
    # would hide the only finding this run has.
    provider = ScriptedEmbedder(related=0.40, unrelated=0.44)
    summary = await check(provider, model=UNSEEDED)

    proposal = summary.floors.proposal
    assert proposal is not None
    assert proposal.min_score is None
    assert proposal.validates is False
    assert "too narrow" in proposal.derivation


def test_the_margin_measures_down_from_whatever_ranks_first_including_noise() -> None:
    # The pathological case the simulation exists to expose, and one no band
    # check can see: when a noise passage outranks the real answer, the
    # relative margin is measured down from the NOISE. The gate then keeps the
    # noise and prunes the answer - both failures at once, on floors whose
    # bands look unremarkable.
    measurement = arm(measured("outranked", related=0.50, noise=0.90))
    gate = compose_gate(measurement, min_score=0.45, rel_margin=0.10)
    query = gate.queries[0]
    assert query.top_passage == "outranked/unrelated"
    assert query.effective_floor == pytest.approx(0.80)
    assert query.related_retrieved is False
    assert query.noise_retrieved == ("outranked/unrelated",)
    assert gate.holds is False
    # Everything under the floor is cut, so nothing at all comes back.
    empty = compose_gate(measurement, min_score=0.95, rel_margin=0.10)
    assert empty.queries[0].related_retrieved is False
    assert empty.queries[0].noise_retrieved == ()


def test_a_gate_reports_how_much_room_it_had_left() -> None:
    # A gate that holds by 0.002 and one that holds by 0.2 are different
    # deployments, and only the number says which is which.
    tight = compose_gate(
        arm(measured("t", related=0.60, noise=0.49)), min_score=0.35, rel_margin=0.1
    )
    assert tight.holds is True
    assert tight.queries[0].noise_clearance == pytest.approx(0.01)
    roomy = compose_gate(
        arm(measured("t", related=0.60, noise=0.20)), min_score=0.35, rel_margin=0.1
    )
    assert roomy.queries[0].noise_clearance == pytest.approx(0.30)


def test_a_proposal_with_no_absolute_floor_does_not_validate() -> None:
    # Without a floor nothing holds noise out at all, so a margin that would
    # pass on its own must not present as a usable pair.
    proposal = propose_floors(arm(measured("t", related=0.40, supporting=0.38, noise=0.44)))
    assert proposal.min_score is None
    assert proposal.validates is False


def test_declared_floors_come_from_the_quirk_seed() -> None:
    seeded = declared_floors(SEEDED)
    assert (seeded.min_score, seeded.rel_margin, seeded.source) == (0.35, 0.10, "known_models")
    # A family entry that carries a prefix but no floors declares no floors:
    # reporting a source with nothing under it would read as a validated pair.
    family = declared_floors("qwen3-embedding:4b")
    assert (family.min_score, family.rel_margin, family.source) == (None, None, "none")
    assert declared_floors(UNSEEDED).source == "none"


def test_a_seeded_tag_gets_its_wrapper_and_an_unseeded_tag_gets_none() -> None:
    # The wrapper is a property of the WEIGHTS (what they were trained
    # against), not of a deployment - so the quirk seed is the only source
    # left to resolve_query_prefix(model) now that the catalog is gone.
    seeded = resolve_query_prefix(SEEDED)
    assert seeded.source == "known_models"
    assert seeded.prefix.startswith("Instruct:")
    # A tag the seed does not describe embeds its queries RAW.
    unseeded = resolve_query_prefix(UNSEEDED)
    assert (unseeded.prefix, unseeded.source) == ("", "none")


def test_the_harness_version_tracks_the_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    # Verdicts are stored against this fingerprint, so an edited corpus has to
    # invalidate what the old one validated.
    before = harness_version()
    monkeypatch.setattr(
        embedder_check,
        "CORPUS",
        (*CORPUS, TripletSpec(name="extra", query="q", related="r", unrelated="u")),
    )
    assert harness_version() != before
    assert len(harness_version()) == 12


def test_the_harness_version_tracks_the_rules_that_live_in_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The defect this pins, found against a live run: the noise-pool rule was
    # narrowed (foreign supporting passages excluded, the unrelated band going
    # from 141 scores to 120) while the corpus text stayed identical, so the
    # fingerprint did not move and two rows measuring differently-defined bands
    # were recorded under one version. Nothing about editing a rule expressed
    # in code moves a hash - only the declaration beside it can.
    before = harness_version()
    for rule in ("noise_pool", "composed_gate", "proposal"):
        monkeypatch.setitem(embedder_check._CODE_RULES, rule, "something else entirely")
        assert harness_version() != before, (
            f"editing the {rule!r} rule must invalidate old verdicts, and only "
            "_CODE_RULES can make it"
        )
        monkeypatch.undo()


def test_the_harness_version_tracks_every_constant_that_shapes_a_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The docstring promises "every behavior-bearing constant", and the promise
    # is only worth what the least-watched constant is worth: a threshold that
    # moves a verdict, or a rounding step that moves a stored proposal, has to
    # invalidate the readings taken under the old value.
    before = harness_version()
    for name, value in (
        ("_RELATED_HEADROOM", 0.11),
        ("_UNRELATED_CLEARANCE", 0.12),
        ("_MARGIN_HEADROOM", 0.13),
        ("_MIN_REL_MARGIN", 0.14),
        ("_SCORE_PLACES", 5),
        ("_PROPOSAL_PLACES", 3),
    ):
        monkeypatch.setattr(embedder_check, name, value)
        assert harness_version() != before, f"{name} shapes a stored number and must be hashed"
        monkeypatch.undo()


async def test_a_short_answer_from_the_embedder_is_an_error_not_a_silent_misalignment() -> None:
    class ShortProvider(ScriptedEmbedder):
        async def embed(self, texts, *, model=None, options=None):
            vectors = await super().embed(texts, model=model, options=options)
            return vectors[:-1]

    with pytest.raises(ValueError, match="could not be matched"):
        await check(ShortProvider())


async def test_mismatched_dimensions_are_an_error_not_a_meaningless_score() -> None:
    # A backend answering with two different vector sizes has produced nothing
    # any similarity could be taken between; scoring it anyway would publish a
    # number with no meaning and a floors verdict resting on it.
    class RaggedProvider(ScriptedEmbedder):
        async def embed(self, texts, *, model=None, options=None):
            vectors = await super().embed(texts, model=model, options=options)
            return [vector[:-1] if index else vector for index, vector in enumerate(vectors)]

    with pytest.raises(ValueError, match="different sizes"):
        await check(RaggedProvider())


async def test_a_zero_vector_scores_zero_rather_than_dividing_by_zero() -> None:
    class ZeroProvider(ScriptedEmbedder):
        async def embed(self, texts, *, model=None, options=None):
            vectors = await super().embed(texts, model=model, options=options)
            return [[0.0] * DIM for _ in vectors]

    summary = await check(ZeroProvider(), model=UNSEEDED)
    assert summary.wrapped.related.maximum == 0.0
    assert summary.wrapped.unrelated.maximum == 0.0
    # Bands that sit on top of each other separate nothing, and the proposal
    # says so rather than inventing a floor between two identical numbers.
    assert summary.floors.proposal is not None
    assert summary.floors.proposal.min_score is None
