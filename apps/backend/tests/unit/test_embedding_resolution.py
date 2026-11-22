"""The embedding resolution chain: which embedder, from which rung, and why.

Five rungs, and every one of them gets a test that names the model the
resolution fell THROUGH to rather than a mock's call count - so a reordering
shows up as the wrong tag, not as a vague failure.

    stored workspace choice
      > the model this deployment's own vectors were written by
      > EXACTLY ONE installed embedding-capable tag
      > SEVERAL installed  -> refuse
      > none

The fourth rung is a GUARD, not a fallback. Choosing between two installed
embedders would be a ranking, and the wrong pick writes vectors into a space
the other model can never read - a corpus split in half with nothing raising.
Refusing costs a deployment one settings click and cannot corrupt anything, so
the refusal has its own test and its own mutation note.

The store and the provider are both driven for real: the store answers the same
``list_spaces`` seam method the stranded-vectors alarm reads, and the inventory
rung goes through the actual capability probe rather than being handed a list.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.providers.base import ModelCapabilities, ModelInfo
from app.providers.fake_provider import FakeProvider
from app.providers.rag.base import VectorSpace, VectorStoreUnavailableError
from app.schemas.settings import WorkspaceDefaults
from app.services.embedding_resolution import (
    AMBIGUOUS_DETAIL,
    UNCONFIGURED_DETAIL,
    config_for,
    installed_embedders,
    resolve_embedding,
)
from app.services.model_facts import clear_facts_cache
from app.services.vector_spaces import clear_vector_space_cache, vector_space_report

OBSERVED_EMBEDDER = "observed-embed:4b"
CHOSEN_EMBEDDER = "chosen-embed:8b"
INSTALLED_EMBEDDER = "installed-embed:0.6b"
SECOND_EMBEDDER = "another-embed:2b"
CHAT_TAG = "chat:4b"

# The runtime's report for each ROLE the inventory rung has to tell apart.
EMBED_ONLY = ModelCapabilities(embedding=True, completion=False)
CHAT_ONLY = ModelCapabilities(embedding=False, completion=True)
# A tag that answers both. Everywhere else in the platform this is a CHAT
# model, and the inventory rung has to agree or one tag becomes a candidate for
# two roles.
BOTH = ModelCapabilities(embedding=True, completion=True)


@pytest.fixture(autouse=True)
def _fresh_caches():
    clear_vector_space_cache()
    clear_facts_cache()
    yield
    clear_vector_space_cache()
    clear_facts_cache()


def make_settings(**overrides) -> Settings:
    """Hermetic settings; ``_env_file=None`` keeps a local .env out of it."""
    return Settings(_env_file=None, **overrides)


def server(reports: dict[str, ModelCapabilities]) -> FakeProvider:
    """A model server holding exactly these tags, each with its own report.

    A real inventory is a set of tags with DIFFERENT capabilities, which is
    precisely what the inventory rung has to read - so listing and per-tag
    report are seeded together and always agree.

    Args:
        reports: Tag name to the capability report the runtime gives for it.
            Insertion order becomes the server's listing order.

    Returns:
        A :class:`FakeProvider` holding exactly those tags.
    """
    return FakeProvider(
        models=[ModelInfo(name=name) for name in reports],
        capabilities_by_model=dict(reports),
    )


class _Store:
    """A store that answers the space enumeration and refuses everything else."""

    def __init__(self, spaces: list[VectorSpace] | None = None, *, fail: bool = False) -> None:
        self.spaces = spaces or []
        self.fail = fail
        self.calls = 0

    async def list_spaces(self) -> list[VectorSpace]:
        self.calls += 1
        if self.fail:
            raise VectorStoreUnavailableError("The vector store is unavailable.")
        return list(self.spaces)

    def __getattr__(self, name: str):  # pragma: no cover - guard, never called
        raise AssertionError(f"resolution must not call {name!r} on the store")


def corpus(model: str, *, points: int = 20) -> _Store:
    """A store holding one persistent collection written by ``model``."""
    return _Store(
        [
            VectorSpace(
                name=f"kb_chunks__{model}",
                kind="kb",
                slug=model,
                points=points,
                embed_models=frozenset({model}),
            )
        ]
    )


# --------------------------------------------------------------------------
# The five rungs, in order
# --------------------------------------------------------------------------


async def test_a_stored_choice_beats_every_rung_beneath_it() -> None:
    # The top rung, and the only one anybody WRITES. Staged against a corpus
    # AND an unambiguous inventory that both name something else, so a
    # reordering names the tag it fell through to.
    choice = await resolve_embedding(
        settings=make_settings(),
        stored=WorkspaceDefaults(embedding_model=CHOSEN_EMBEDDER),
        store=corpus(OBSERVED_EMBEDDER),
        provider=server({INSTALLED_EMBEDDER: EMBED_ONLY}),
        provider_name="fake",
    )
    assert (choice.model, choice.source) == (CHOSEN_EMBEDDER, "workspace")


async def test_the_corpus_answers_when_nothing_is_stored() -> None:
    # The evidence rung outranks the inventory deliberately: an upgraded
    # deployment that never opened a settings page keeps embedding with the
    # model its own passages were built by, rather than starting a second
    # vector space beside them because some other tag happens to be installed.
    choice = await resolve_embedding(
        settings=make_settings(),
        stored=WorkspaceDefaults(),
        store=corpus(OBSERVED_EMBEDDER),
        provider=server({INSTALLED_EMBEDDER: EMBED_ONLY}),
        provider_name="fake",
    )
    assert (choice.model, choice.source) == (OBSERVED_EMBEDDER, "observed")


async def test_the_only_installed_embedder_needs_nobody_to_choose_it() -> None:
    # The rung that makes a fresh box work with no setup beyond pulling an
    # embedder. One candidate is not a choice anybody has to make - an
    # inventory is observed reality, not curation - and the source says
    # "installed" so a surface can explain where the answer came from.
    choice = await resolve_embedding(
        settings=make_settings(),
        store=_Store(),
        provider=server({INSTALLED_EMBEDDER: EMBED_ONLY, CHAT_TAG: CHAT_ONLY}),
        provider_name="fake",
    )
    assert (choice.model, choice.source) == (INSTALLED_EMBEDDER, "installed")
    assert choice.candidates == (INSTALLED_EMBEDDER,)


async def test_several_installed_embedders_are_refused_rather_than_ranked() -> None:
    """Two candidates and nothing selecting one is a REFUSAL, never a guess.

    This is the guard the rung exists for. Picking either would be a ranking
    the platform has no basis for, and the wrong pick is not a slow answer -
    it writes into a vector space the other model can never read, so half the
    corpus becomes unreachable with nothing raising anywhere. A settings click
    fixes an honest refusal; nothing cheaply fixes a corpus split in two.

    MUTATION: make ``resolve_embedding`` return ``found[0]`` where it now
    returns the ambiguous choice (i.e. change ``if len(found) == 1`` to
    ``if found``) and this fails on the tuple - the config is no longer None
    and the source reads "installed".
    """
    choice = await resolve_embedding(
        settings=make_settings(),
        store=_Store(),
        provider=server({INSTALLED_EMBEDDER: EMBED_ONLY, SECOND_EMBEDDER: EMBED_ONLY}),
        provider_name="fake",
    )
    assert (choice.config, choice.source) == (None, "ambiguous")
    # The refusal NAMES what there was to choose between, so nobody has to go
    # and look - and sorted, for the reason installed_embedders is sorted.
    assert choice.candidates == (SECOND_EMBEDDER, INSTALLED_EMBEDDER)
    assert choice.detail == AMBIGUOUS_DETAIL
    # Two failures with two different fixes: "nothing can embed" sends someone
    # to pull a model, "several can" sends them to the workspace defaults. The
    # sentences must not be interchangeable.
    assert choice.detail != UNCONFIGURED_DETAIL


async def test_a_server_holding_no_embedder_resolves_to_nothing() -> None:
    # The bottom rung. None is the honest "retrieval is unavailable" answer,
    # and the sentence the surfaces show names the workspace defaults - never a
    # hardware profile or a committed file, neither of which decides this.
    choice = await resolve_embedding(
        settings=make_settings(),
        store=_Store(),
        provider=server({CHAT_TAG: CHAT_ONLY}),
        provider_name="fake",
    )
    assert (choice.config, choice.source) == (None, "none")
    assert choice.candidates == ()
    assert choice.detail == UNCONFIGURED_DETAIL
    assert "workspace defaults" in UNCONFIGURED_DETAIL
    assert "MODEL_TIER" not in UNCONFIGURED_DETAIL
    assert "models.yaml" not in UNCONFIGURED_DETAIL


# --------------------------------------------------------------------------
# installed_embedders: what counts as a candidate, and in what order
# --------------------------------------------------------------------------


async def test_only_a_tag_that_embeds_and_cannot_chat_is_a_candidate() -> None:
    """A model that can also hold a conversation is a CHAT model everywhere.

    The "and not completion" half is the precedence every other surface
    applies, and dropping it does real damage in both directions: a
    dual-capability tag would be offered as an embedder while the manager
    lists it under chat, and - worse - a box holding one real embedder beside
    one dual-capability chat model would resolve as AMBIGUOUS and lose
    retrieval outright.

    MUTATION: drop ``and not report.completion`` from ``installed_embedders``
    and the dual tag joins the tuple, failing both assertions below.
    """
    provider = server({INSTALLED_EMBEDDER: EMBED_ONLY, "dual:7b": BOTH, CHAT_TAG: CHAT_ONLY})
    assert await installed_embedders(provider, "fake") == (INSTALLED_EMBEDDER,)

    # And the consequence, driven through the chain: one candidate, so the box
    # resolves rather than refusing.
    choice = await resolve_embedding(
        settings=make_settings(), store=_Store(), provider=provider, provider_name="fake"
    )
    assert (choice.model, choice.source) == (INSTALLED_EMBEDDER, "installed")


async def test_candidates_come_back_sorted_whatever_order_the_server_listed() -> None:
    """The order decides which vector space a deployment writes into.

    A model server is free to list its inventory in any order it likes - by
    install time, by size, by nothing at all. Left unsorted, a restart that
    reordered the listing would move the single-candidate answer on a box that
    later gained a second embedder, and silently re-point a corpus. Sorting is
    what makes the rung a function of WHAT is installed rather than of when.

    MUTATION: drop the ``sorted`` in ``installed_embedders`` and the tuple
    comes back in the fake's declaration order, failing the equality.
    """
    forwards = server({"zeta:1b": EMBED_ONLY, "alpha:1b": EMBED_ONLY})
    backwards = server({"alpha:1b": EMBED_ONLY, "zeta:1b": EMBED_ONLY})
    assert await installed_embedders(forwards, "fake") == ("alpha:1b", "zeta:1b")
    assert await installed_embedders(backwards, "fake") == ("alpha:1b", "zeta:1b")


async def test_a_tag_whose_probe_failed_is_not_counted_as_an_embedder() -> None:
    # "The runtime would not say" is not "these weights embed". Counting an
    # unreadable tag would turn a half-answering server into an ambiguous one
    # and take retrieval down with it.
    class HalfAnswering(FakeProvider):
        async def capabilities(self, model: str):
            if model == "silent:1b":
                raise RuntimeError("the runtime would not say")
            return await super().capabilities(model)

    provider = HalfAnswering(
        models=[ModelInfo(name=INSTALLED_EMBEDDER), ModelInfo(name="silent:1b")],
        capabilities_by_model={INSTALLED_EMBEDDER: EMBED_ONLY},
    )
    assert await installed_embedders(provider, "fake") == (INSTALLED_EMBEDDER,)


async def test_a_server_that_cannot_be_listed_skips_the_rung() -> None:
    # A fall-through, never a failure: a resolution that raised here would fail
    # every request that merely wanted to know who embeds.
    class DeadProvider(FakeProvider):
        async def list_models(self):
            raise RuntimeError("the model server is unreachable")

    assert await installed_embedders(DeadProvider(), "fake") == ()
    choice = await resolve_embedding(
        settings=make_settings(), store=_Store(), provider=DeadProvider(), provider_name="fake"
    )
    assert (choice.config, choice.source) == (None, "none")


async def test_given_candidates_are_used_verbatim_and_skip_the_probe() -> None:
    # The management view has already probed the whole inventory once. Passing
    # its reading in is what makes the page and a turn resolve against ONE
    # reading of the server rather than two takens seconds apart.
    class Untouchable(FakeProvider):
        async def list_models(self):
            raise AssertionError("the provider must not be consulted")

    choice = await resolve_embedding(
        settings=make_settings(),
        store=_Store(),
        provider=Untouchable(),
        provider_name="fake",
        candidates=[INSTALLED_EMBEDDER],
    )
    assert (choice.model, choice.source) == (INSTALLED_EMBEDDER, "installed")


# --------------------------------------------------------------------------
# The rungs a caller can decline to supply
# --------------------------------------------------------------------------


async def test_an_unreachable_store_falls_through_instead_of_raising() -> None:
    # The evidence rung is best-effort by contract: a store that is down costs
    # the answer nothing but its own rung.
    choice = await resolve_embedding(
        settings=make_settings(),
        store=_Store(fail=True),
        provider=server({INSTALLED_EMBEDDER: EMBED_ONLY}),
        provider_name="fake",
    )
    assert (choice.model, choice.source) == (INSTALLED_EMBEDDER, "installed")


async def test_a_caller_with_no_store_or_provider_resolves_the_stored_answer_only() -> None:
    # A boot check or a maintenance probe holds neither seam, and skips both
    # rungs rather than inventing one. A workspace choice still answers.
    stored = await resolve_embedding(
        settings=make_settings(), stored=WorkspaceDefaults(embedding_model=CHOSEN_EMBEDDER)
    )
    assert (stored.model, stored.source) == (CHOSEN_EMBEDDER, "workspace")
    bare = await resolve_embedding(settings=make_settings())
    assert (bare.config, bare.source) == (None, "none")


async def test_a_refresh_re_reads_the_store_even_when_a_choice_short_circuits_it() -> None:
    # An operator surface asks for a fresh reading because a SECOND reader -
    # the stranded-vectors verdict, and through it the delete guard - is about
    # to judge the same inventory. Refreshing inside the rung that consults it
    # loses that whenever a stored choice answers first, leaving the guard on a
    # reading up to fifteen minutes old: a corpus ingested since the page was
    # drawn would be invisible to it, and deleting its embedder would be
    # allowed. Move the refresh back inside the rung and the count stays 1.
    store = corpus(OBSERVED_EMBEDDER)
    await resolve_embedding(settings=make_settings(), store=store)
    assert store.calls == 1
    await resolve_embedding(
        settings=make_settings(),
        stored=WorkspaceDefaults(embedding_model=CHOSEN_EMBEDDER),
        store=store,
        refresh=True,
    )
    assert store.calls == 2


async def test_one_refresh_serves_both_readers() -> None:
    # And exactly one re-read, not one each. Two enumerations would not merely
    # cost a round trip: the embedder would be resolved from one reading of the
    # corpus and the corpus judged against another, which is two answers
    # wearing one face.
    store = corpus(OBSERVED_EMBEDDER)
    choice = await resolve_embedding(settings=make_settings(), store=store, refresh=True)
    report = await vector_space_report(store=store, refresh=False, active_model=choice.model)
    assert store.calls == 1
    assert report.active_model == OBSERVED_EMBEDDER
    assert report.stranded is False


# --------------------------------------------------------------------------
# What describes the MODEL rather than the deployment
# --------------------------------------------------------------------------


async def test_placement_comes_from_the_setting_alone() -> None:
    # Where the work RUNS is a fact about the machine - how much card there is
    # and what else wants it - so EMBEDDING_DEVICE decides it outright rather
    # than arbitrating with whoever chose the tag.
    store = corpus(OBSERVED_EMBEDDER)
    pinned = await resolve_embedding(settings=make_settings(), store=store)
    assert pinned.config is not None and pinned.config.device == "cpu"
    freed = await resolve_embedding(settings=make_settings(embedding_device="auto"), store=store)
    assert freed.config is not None and freed.config.device == "auto"


def test_the_query_wrapper_comes_from_the_seed_whoever_chose_the_tag() -> None:
    # The instruct wrapper is a property of the WEIGHTS - it is what they were
    # trained against - so it rides the tag rather than the rung that picked
    # it. Unknown dim is a legitimate answer: the store learns a collection's
    # width from the first vectors written into it.
    seeded = config_for("qwen3-embedding:0.6b", make_settings())
    assert seeded.query_prefix.startswith("Instruct:")
    assert seeded.dim is None
    # A tag nothing describes embeds its queries RAW rather than borrowing a
    # neighbour's wrapper, which would measure on a scale nobody queries on.
    assert config_for(OBSERVED_EMBEDDER, make_settings()).query_prefix == ""


def test_a_dimension_a_caller_already_read_is_carried_through() -> None:
    # A cross-check rather than something retrieval needs told, so it is
    # optional - but when the runtime did report a width it must reach the
    # config rather than being dropped on the floor.
    assert config_for(OBSERVED_EMBEDDER, make_settings(), dim=2560).dim == 2560
