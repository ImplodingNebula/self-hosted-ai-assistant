"""The stranded-vectors detector: what the report says, and how often it asks.

The store is faked here rather than run in memory because two of the rules
under test are about the CALLS made (an enumeration that fails, and an
enumeration that must not happen twice), which a counting double states
directly and a real store only implies.

The active embedder is an ARGUMENT rather than something this module looks up,
and every test passes it explicitly. That is the whole point of the seam: the
tag retrieval queries with comes from the resolution chain
(:mod:`app.services.embedding_resolution`), so a report resolving it a second
way here would be able to judge a corpus against an embedder no turn uses -
which is a verdict about a deployment that does not exist.
"""

from __future__ import annotations

import asyncio

import pytest

from app.providers.rag.base import (
    ChunkPoint,
    CollectionKind,
    DocFingerprint,
    ScoredChunk,
    StoredInventory,
    VectorSpace,
    VectorStore,
    VectorStoreUnavailableError,
)
from app.services.vector_spaces import (
    _CACHE_TTL_SECONDS,
    clear_vector_space_cache,
    observed_embed_model,
    vector_space_report,
)

ACTIVE = "embed:0.6b"
OLD = "embed:legacy"


@pytest.fixture(autouse=True)
def _fresh_caches():
    clear_vector_space_cache()
    yield
    clear_vector_space_cache()


def space(name: str, kind: CollectionKind, *, points: int, models: set[str]) -> VectorSpace:
    return VectorSpace(
        name=name,
        kind=kind,
        slug=name.split("__", 1)[-1],
        points=points,
        embed_models=frozenset(models),
    )


class CountingStore(VectorStore):
    """A VectorStore that only enumerates, and counts how often it was asked.

    Every other seam method raises: a test that reaches one has wandered off
    the path under test, and should say so loudly rather than pass.
    """

    def __init__(self, spaces: list[VectorSpace] | None = None, *, fail: bool = False) -> None:
        self.spaces = spaces or []
        self.fail = fail
        self.calls = 0

    async def list_spaces(self) -> list[VectorSpace]:
        self.calls += 1
        if self.fail:
            raise VectorStoreUnavailableError("The vector store is unavailable.")
        return list(self.spaces)

    async def ping(self) -> None:
        raise AssertionError("not part of the space report")

    async def ensure_ready(self, space: str, dim: int) -> None:
        raise AssertionError("not part of the space report")

    async def upsert(
        self, kind: CollectionKind, space: str, tenant_id: str, points: list[ChunkPoint]
    ) -> int:
        raise AssertionError("not part of the space report")

    async def query(self, *args: object, **kwargs: object) -> list[ScoredChunk]:
        raise AssertionError("not part of the space report")

    async def count(self, *args: object, **kwargs: object) -> int:
        raise AssertionError("not part of the space report")

    async def doc_fingerprints(self, *args: object, **kwargs: object) -> dict[str, DocFingerprint]:
        raise AssertionError("not part of the space report")

    async def delete_doc(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("not part of the space report")

    async def delete_record(self, *args: object, **kwargs: object) -> int:
        raise AssertionError("not part of the space report")

    async def list_kb_sources(self, *args: object, **kwargs: object) -> StoredInventory:
        raise AssertionError("not part of the space report")

    async def delete_kb_source(self, *args: object, **kwargs: object) -> int:
        raise AssertionError("not part of the space report")

    async def delete_session(self, tenant_id: str, session_id: str) -> int:
        raise AssertionError("not part of the space report")

    async def sweep_expired(self, now: float) -> int:
        raise AssertionError("not part of the space report")

    async def aclose(self) -> None:
        return None


async def test_empty_store_is_not_stranded() -> None:
    # Nothing stored is nothing lost. The alarm fires on EVIDENCE that points
    # exist out of reach, never on their absence - a fresh deployment must not
    # greet its first user with a warning about vectors it never wrote.
    report = await vector_space_report(store=CountingStore([]), active_model=ACTIVE)
    assert report.known is True
    assert report.stranded is False
    assert report.stranded_points == 0
    assert report.active_model == ACTIVE
    assert report.message is None
    assert report.spaces == []


async def test_empty_collections_are_not_stranded() -> None:
    # Collections created but never written to (ensure_ready makes both of a
    # space) hold nothing, so there is nothing to be out of reach.
    store = CountingStore(
        [
            space("kb_chunks__embed_legacy", "kb", points=0, models=set()),
            space("session_chunks__embed_legacy", "session", points=0, models=set()),
        ]
    )
    report = await vector_space_report(store=store, active_model=ACTIVE)
    assert report.stranded is False
    assert all(row.readable for row in report.spaces)


async def test_matching_provenance_is_not_stranded() -> None:
    # Points written by the model in use are exactly what a healthy store looks
    # like; flagging them would make the alarm permanent and therefore useless.
    store = CountingStore([space("kb_chunks__embed_0_6b", "kb", points=40, models={ACTIVE})])
    report = await vector_space_report(store=store, active_model=ACTIVE)
    assert report.stranded is False
    assert report.stranded_points == 0
    assert report.message is None
    assert report.spaces[0].readable is True


async def test_foreign_provenance_with_nothing_readable_is_the_alarm() -> None:
    # The real emergency: the old collection holds everything, the active one
    # holds nothing, and retrieval therefore returns nothing at all.
    store = CountingStore(
        [
            space("kb_chunks__embed_legacy", "kb", points=137, models={OLD}),
            space("kb_chunks__embed_0_6b", "kb", points=0, models=set()),
        ]
    )
    report = await vector_space_report(store=store, active_model=ACTIVE)
    assert report.stranded is True
    assert report.stranded_points == 137
    assert report.readable_points == 0
    assert report.stranded_models == [OLD]
    # The text is shown verbatim, so it has to name both models.
    assert report.message is not None
    assert OLD in report.message
    assert ACTIVE in report.message
    assert "cannot be searched" in report.message
    # But NOT the count. list_spaces is tenant-free, so 137 is summed across
    # every tenant, and this same sentence is shown to one user on a chat turn.
    # The number stays on the payload, where the caller decides who sees it.
    assert "137" not in report.message
    stale = next(row for row in report.spaces if row.name == "kb_chunks__embed_legacy")
    assert stale.readable is False
    assert stale.embed_models == [OLD]


async def test_leftovers_beside_a_working_corpus_are_cleanup_not_an_alarm() -> None:
    # The state a deployment lands in after following the alarm's OWN advice:
    # embedder changed, everything re-ingested, old collection still sitting
    # there because nothing removes it. Calling that "stored knowledge cannot be
    # searched" is false, never clears, and - riding a standing notice - would
    # print onto every reloaded answer forever.
    store = CountingStore(
        [
            space("kb_chunks__embed_legacy", "kb", points=137, models={OLD}),
            space("kb_chunks__embed_0_6b", "kb", points=500, models={ACTIVE}),
        ]
    )
    report = await vector_space_report(store=store, active_model=ACTIVE)
    assert report.stranded is False
    # The facts are all still reported - only the VERDICT differs.
    assert report.stranded_points == 137
    assert report.readable_points == 500
    assert report.stranded_models == [OLD]
    assert report.message is not None
    assert "cannot be searched" not in report.message
    assert "Retrieval is working" in report.message
    # Operator-facing only (stranded is False, so no turn shows it), so this
    # shape is free to state the numbers an operator needs to act.
    assert "137" in report.message
    assert "500" in report.message
    # And it must not imply an action the app does not offer: nothing here
    # deletes a collection, so the only honest instruction points at the store.
    assert "only be removed directly against the vector store" in report.message
    # One abandoned collection, so the singular reads naturally.
    assert "in a collection written by" in report.message


async def test_several_abandoned_collections_read_as_plural() -> None:
    # Two embedder changes leave two collections behind, and the counts sum
    # across both - so "a collection written by A, B" would be wrong twice over.
    older = "embed:older-still"
    store = CountingStore(
        [
            space("kb_chunks__embed_legacy", "kb", points=100, models={OLD}),
            space("kb_chunks__embed_older_still", "kb", points=37, models={older}),
            space("kb_chunks__embed_0_6b", "kb", points=500, models={ACTIVE}),
        ]
    )
    report = await vector_space_report(store=store, active_model=ACTIVE)
    assert report.stranded is False
    assert report.stranded_points == 137
    assert report.stranded_models == [OLD, older]
    assert report.message is not None
    assert "in 2 collections written by" in report.message
    # Tags read as English, not as a truncated comma list.
    assert f"{OLD} and {older}" in report.message


async def test_session_only_stranding_is_reported_but_does_not_alarm() -> None:
    # Session points carry a TTL and the sweep clears them within hours, so a
    # stranded session space fixes itself before anyone could act on it.
    # Alarming on it would cry wolf on every embedder change.
    store = CountingStore(
        [space("session_chunks__embed_legacy", "session", points=9, models={OLD})]
    )
    report = await vector_space_report(store=store, active_model=ACTIVE)
    assert report.stranded is False
    assert report.stranded_points == 0
    assert report.stranded_session_points == 9
    assert report.message is None
    # Still visible in the detail, so a reader wondering where a conversation's
    # fetched pages went can see them.
    assert report.spaces[0].readable is False


async def test_unenumerable_store_is_unknown_not_fine() -> None:
    report = await vector_space_report(store=CountingStore(fail=True), active_model=ACTIVE)
    assert report.known is False
    assert report.stranded is False
    assert report.message is not None
    assert "unknown" in report.message


async def test_a_slow_store_degrades_to_unknown(monkeypatch) -> None:
    # A black-holed store - connection accepted, answer never sent - must not be
    # able to hold the page it merely annotates. The store's own client timeout
    # is 30 seconds; this report gives up far sooner and says "unknown".
    class HangingStore(CountingStore):
        async def list_spaces(self):
            self.calls += 1
            await asyncio.sleep(60)
            raise AssertionError("the timeout should have fired")

    monkeypatch.setattr("app.services.vector_spaces._ENUMERATION_TIMEOUT_SECONDS", 0.05)
    report = await vector_space_report(store=HangingStore(), active_model=ACTIVE)
    assert report.known is False
    assert report.stranded is False
    assert "unknown" in (report.message or "")


async def test_an_unresolved_embedder_is_unknown_but_still_lists_what_is_stored() -> None:
    # The resolution chain answered NOTHING - no workspace choice, no corpus it
    # could read, and either no installed embedder or several. Saying "all
    # fine" would be a guess about a deployment that cannot retrieve at all.
    # But the STORE answered perfectly well, and an unresolved embedder is
    # exactly when an operator needs to see what is actually in it - so the
    # inventory is kept even though the verdict is unknown.
    store = CountingStore([space("kb_chunks__embed_legacy", "kb", points=5, models={OLD})])
    report = await vector_space_report(store=store, active_model=None)
    assert report.known is False
    assert report.stranded is False
    assert report.active_model is None
    assert store.calls == 1
    assert [row.name for row in report.spaces] == ["kb_chunks__embed_legacy"]
    assert report.spaces[0].points == 5
    assert report.spaces[0].embed_models == [OLD]
    # Nothing is readable when there is nothing to read it WITH - a statement
    # about the missing configuration, not a verdict on the points.
    assert report.spaces[0].readable is False
    assert "nothing to compare" in (report.message or "")


async def test_points_without_provenance_are_not_called_stranded() -> None:
    # A sample that yielded no embed_model at all is "cannot tell", and cannot
    # be reported as "these are lost".
    store = CountingStore([space("kb_chunks__mystery", "kb", points=12, models=set())])
    report = await vector_space_report(store=store, active_model=ACTIVE)
    assert report.stranded is False
    assert report.spaces[0].readable is False


async def test_second_read_is_served_from_cache() -> None:
    # The whole point of the cache: the turn-side consultation must not turn
    # into a Qdrant round trip per collection per turn.
    store = CountingStore([space("kb_chunks__embed_legacy", "kb", points=3, models={OLD})])

    first = await vector_space_report(store=store, active_model=ACTIVE)
    second = await vector_space_report(store=store, active_model=ACTIVE)
    assert store.calls == 1
    assert second == first


async def test_refresh_recomputes_immediately() -> None:
    store = CountingStore([space("kb_chunks__embed_legacy", "kb", points=3, models={OLD})])

    await vector_space_report(store=store, active_model=ACTIVE)
    store.spaces = [space("kb_chunks__embed_0_6b", "kb", points=3, models={ACTIVE})]
    report = await vector_space_report(store=store, active_model=ACTIVE, refresh=True)
    assert store.calls == 2
    assert report.stranded is False


async def test_cache_expires_after_the_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    store = CountingStore([space("kb_chunks__embed_legacy", "kb", points=3, models={OLD})])

    clock = {"now": 1_000.0}
    monkeypatch.setattr("app.services.vector_spaces.time.monotonic", lambda: clock["now"])
    await vector_space_report(store=store, active_model=ACTIVE)
    clock["now"] += _CACHE_TTL_SECONDS - 1
    await vector_space_report(store=store, active_model=ACTIVE)
    assert store.calls == 1
    clock["now"] += 2
    await vector_space_report(store=store, active_model=ACTIVE)
    assert store.calls == 2


async def test_a_changed_embedder_is_re_judged_off_the_same_enumeration() -> None:
    # The verdict must not outlive the model it was computed against. An
    # embedder can move under a living process - a workspace names one, or a
    # second gets installed - and a cached ALL-CLEAR would keep the alarm
    # silent through exactly the minutes it exists for. What is cached is the
    # ENUMERATION, so the new judgement is recomputed and still costs no round
    # trip; caching the verdict instead would have bought one at the price of
    # the other.
    store = CountingStore([space("kb_chunks__embed_0_6b", "kb", points=8, models={ACTIVE})])
    healthy = await vector_space_report(store=store, active_model=ACTIVE)
    assert healthy.stranded is False

    report = await vector_space_report(store=store, active_model="embed:brand-new")
    assert store.calls == 1
    assert report.active_model == "embed:brand-new"
    assert report.stranded is True


async def test_the_verdict_follows_the_embedder_the_caller_resolved() -> None:
    # The alarm is only meaningful against the embedder in USE, and which one
    # that is comes from the resolution chain rather than from anything this
    # module can see. One corpus, two answers: judged against a tag nothing
    # wrote it looks stranded, and judged against the tag the chain actually
    # settled on it is healthy. Resolve it a second way here and a workspace
    # that named its own embedder would be told its corpus is lost.
    store = CountingStore([space("kb_chunks__chosen", "kb", points=8, models={"chosen:1b"})])
    mismatched = await vector_space_report(store=store, active_model=ACTIVE)
    assert mismatched.stranded is True
    resolved = await vector_space_report(store=store, active_model="chosen:1b")
    assert resolved.active_model == "chosen:1b"
    assert resolved.stranded is False


async def test_the_observed_embedder_is_the_busiest_persistent_space() -> None:
    # The evidence rung of the embedding resolution chain. Session collections
    # are swept within hours, so a corpus that expires by tomorrow is not
    # something to configure a deployment from; among the persistent ones the
    # fullest wins, because an abandoned collection sits beside a live one until
    # somebody removes it by hand.
    store = CountingStore(
        [
            space("kb_chunks__embed_legacy", "kb", points=3, models={OLD}),
            space("kb_chunks__embed_0_6b", "kb", points=90, models={ACTIVE}),
            space("session_chunks__embed_new", "session", points=900, models={"embed:session"}),
        ]
    )
    assert await observed_embed_model(store) == ACTIVE


async def test_an_unreachable_store_observes_nothing() -> None:
    # Non-blocking on failure: the rung answers "nothing", so the chain falls
    # through to the next one. Raising here would take down every request that
    # merely wanted to know who embeds.
    assert await observed_embed_model(CountingStore(fail=True)) is None
    assert await observed_embed_model(CountingStore([])) is None
    # A non-empty collection whose points carry no provenance says nothing
    # either - "I cannot tell" is not an answer to configure from.
    anonymous = CountingStore([space("kb_chunks__mystery", "kb", points=7, models=set())])
    assert await observed_embed_model(anonymous) is None
