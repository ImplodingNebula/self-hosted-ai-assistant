"""Direct tests for the get_embedder/get_reranker dependency bodies.

The embedder dependency is where the resolution chain meets a real request: it
opens a session for the workspace rung, hands the store to the evidence rung
and the provider to the inventory rung, and turns whatever comes back into the
one object that DEFINES the vector space. So the cases here are driven through
the dependency rather than through ``resolve_embedding`` - the wiring is the
subject, and a rung the dependency forgot to pass would still resolve perfectly
in a unit test of the chain.

``get_reranker`` is the opposite shape and stays that way deliberately: it
reads deployment configuration alone, opens no session, and the workspace
toggle that can withhold a wired sidecar is applied downstream by callers that
already hold the tenant's defaults.
"""

from __future__ import annotations

import types

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import get_embedder, get_reranker
from app.core.config import Settings
from app.main import create_app
from app.providers.base import ModelCapabilities, ModelInfo
from app.providers.fake_provider import FakeProvider
from app.providers.rag.base import VectorSpace, VectorStore, VectorStoreUnavailableError
from app.providers.rag.embedder import FakeEmbedder, ProviderEmbedder
from app.providers.rag.reranker import InfinityReranker, NoopReranker
from app.schemas.settings import WorkspaceDefaultsPatch
from app.services.auth.principal import seeded_principal
from app.services.model_facts import clear_facts_cache
from app.services.vector_spaces import clear_vector_space_cache
from app.services.workspace_defaults import apply_defaults

TENANT = seeded_principal()

CHOSEN = "chosen:1b"
OBSERVED = "observed:2b"
INSTALLED = "installed-embed:0.6b"
SECOND = "another-embed:2b"
CHAT = "chat:4b"

EMBED_ONLY = ModelCapabilities(embedding=True, completion=False)
CHAT_ONLY = ModelCapabilities(embedding=False, completion=True)


@pytest.fixture(autouse=True)
def _fresh_caches():
    clear_vector_space_cache()
    clear_facts_cache()
    yield
    clear_vector_space_cache()
    clear_facts_cache()


class _SpaceStore:
    """A vector store that answers the space enumeration and nothing else.

    Only :meth:`list_spaces` is ever reached from the resolution chain, so
    every other seam method asserts rather than pretending: a dependency that
    started touching the store for anything else should fail loudly here.
    """

    spaces: list[VectorSpace] = []

    async def list_spaces(self) -> list[VectorSpace]:
        return list(self.spaces)

    def __getattr__(self, name: str):  # pragma: no cover - guard, never called
        raise AssertionError(f"the embedder dependency must not call {name!r} on the store")


class CorpusStore(_SpaceStore):
    """A store holding one persistent collection written by a named embedder."""

    def __init__(self, model: str) -> None:
        self.spaces = [
            VectorSpace(
                name=f"kb_chunks__{model}",
                kind="kb",
                slug=model,
                points=12,
                embed_models=frozenset({model}),
            )
        ]


class BrokenStore(_SpaceStore):
    """A store that cannot be enumerated at all."""

    async def list_spaces(self) -> list[VectorSpace]:
        raise VectorStoreUnavailableError("The vector store is unavailable.")


EMPTY_STORE: VectorStore = _SpaceStore()  # type: ignore[assignment]


def make_settings(**overrides) -> Settings:
    """Hermetic settings on the fake provider, so no local .env reaches these."""
    return Settings(_env_file=None, model_provider="fake", **overrides)


def server(reports: dict[str, ModelCapabilities]) -> FakeProvider:
    """A model server holding exactly these tags, each with its own report."""
    return FakeProvider(
        models=[ModelInfo(name=name) for name in reports],
        capabilities_by_model=dict(reports),
    )


def make_request() -> types.SimpleNamespace:
    """A Request substitute: both deps only ever touch ``request.app.state``."""
    return types.SimpleNamespace(app=create_app())


async def embedder_for(
    settings: Settings,
    db_sessionmaker: async_sessionmaker[AsyncSession],
    *,
    provider: FakeProvider | None = None,
    store: VectorStore = EMPTY_STORE,
    request: types.SimpleNamespace | None = None,
):
    """Resolve the embedder exactly as the dependency graph would."""
    return await get_embedder(
        request or make_request(),
        settings,
        provider if provider is not None else server({}),
        db_sessionmaker,
        TENANT,
        store,
    )


async def store_embedder(db_sessionmaker, tag: str) -> None:
    """Write the workspace's embedding choice through the real writer."""
    async with db_sessionmaker() as session:
        await apply_defaults(session, TENANT, WorkspaceDefaultsPatch(embedding_model=tag))
        await session.commit()


# --------------------------------------------------------------------------
# get_embedder: every rung, reached through the dependency
# --------------------------------------------------------------------------


async def test_the_workspace_choice_outranks_the_corpus_and_the_inventory(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # The rung order, driven through the real store and a real listing: what a
    # workspace SAID beats both what its vectors show and what happens to be
    # installed. Break the precedence and this names the model it fell to.
    await store_embedder(db_sessionmaker, CHOSEN)
    embedder = await embedder_for(
        make_settings(),
        db_sessionmaker,
        provider=server({INSTALLED: EMBED_ONLY}),
        store=CorpusStore(OBSERVED),
    )
    assert isinstance(embedder, ProviderEmbedder)
    assert embedder.model == CHOSEN


async def test_the_observed_corpus_answers_when_nothing_is_stored(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # Nothing stored, so the deployment's own passages answer - ahead of the
    # inventory, which holds a different embedder entirely. This is the rung
    # that keeps an upgraded deployment reading the corpus it already built.
    embedder = await embedder_for(
        make_settings(),
        db_sessionmaker,
        provider=server({INSTALLED: EMBED_ONLY}),
        store=CorpusStore(OBSERVED),
    )
    assert isinstance(embedder, ProviderEmbedder)
    assert embedder.model == OBSERVED


async def test_the_only_installed_embedder_answers_a_fresh_deployment(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # A box with nothing stored and nothing ingested: one embedder installed is
    # not a choice anybody has to make, and the dependency has to actually PASS
    # the provider for that rung to exist at all.
    embedder = await embedder_for(
        make_settings(), db_sessionmaker, provider=server({INSTALLED: EMBED_ONLY, CHAT: CHAT_ONLY})
    )
    assert isinstance(embedder, ProviderEmbedder)
    assert embedder.model == INSTALLED


async def test_two_installed_embedders_leave_retrieval_unconfigured(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # The refusal has to reach the REQUEST, not just the resolver: guessing
    # here would write vectors into a space the other model can never read,
    # and None is what makes the surfaces say so instead.
    embedder = await embedder_for(
        make_settings(),
        db_sessionmaker,
        provider=server({INSTALLED: EMBED_ONLY, SECOND: EMBED_ONLY}),
    )
    assert embedder is None


async def test_nothing_that_can_embed_yields_no_embedder(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    embedder = await embedder_for(
        make_settings(), db_sessionmaker, provider=server({CHAT: CHAT_ONLY})
    )
    assert embedder is None


async def test_a_store_that_cannot_be_enumerated_falls_through(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # The evidence rung is best-effort: a store that raises must fall THROUGH
    # to the inventory, never fail the request that was only asking who embeds.
    embedder = await embedder_for(
        make_settings(),
        db_sessionmaker,
        provider=server({INSTALLED: EMBED_ONLY}),
        store=BrokenStore(),
    )
    assert isinstance(embedder, ProviderEmbedder)
    assert embedder.model == INSTALLED


async def test_the_batch_size_and_placement_come_from_settings(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # Both are deployment configuration the dependency has to carry into the
    # object it builds: a builtin default here would ignore a tuned batch, and
    # a placement read from anywhere else would put ingestion on a card this
    # machine keeps it off.
    settings = make_settings(rag_embed_batch=7, embedding_device="cpu")
    embedder = await embedder_for(
        settings, db_sessionmaker, provider=server({INSTALLED: EMBED_ONLY})
    )
    assert isinstance(embedder, ProviderEmbedder)
    assert embedder._batch_size == 7  # noqa: SLF001 - the wiring is the subject
    assert embedder._options.device == "cpu"  # noqa: SLF001 - same


async def test_the_state_override_wins_before_anything_is_resolved(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # A preset instance short-circuits the whole chain, which is what lets a
    # test inject a FakeEmbedder. Staged against a provider that would raise if
    # consulted, so this cannot pass by resolving to the same thing.
    class Untouchable(FakeProvider):
        async def list_models(self):
            raise AssertionError("the override must be returned before any resolution")

    request = make_request()
    preset = FakeEmbedder(dim=8)
    request.app.state.embedder = preset
    resolved = await embedder_for(
        make_settings(), db_sessionmaker, provider=Untouchable(), request=request
    )
    assert resolved is preset


# --------------------------------------------------------------------------
# get_reranker: the URL is the gate, and the only one
# --------------------------------------------------------------------------


async def test_no_url_means_dense_only() -> None:
    assert await get_reranker(make_request(), make_settings(reranker_url=None)) is None


async def test_a_url_and_a_model_build_the_sidecar_client_once() -> None:
    # Cached on app state: the client holds a connection pool, and rebuilding
    # it per request would open one per turn.
    settings = make_settings(reranker_url="http://reranker:7997")
    request = make_request()
    first = await get_reranker(request, settings)
    assert isinstance(first, InfinityReranker)
    assert request.app.state.reranker is first
    assert await get_reranker(request, settings) is first


async def test_the_sidecar_serves_the_model_the_deployment_named() -> None:
    # The tag reaches the client rather than being defaulted inside it: the
    # sidecar was started with a `--model-id` and a rerank call has to carry
    # the matching name, or every request 404s against a live service.
    settings = make_settings(reranker_url="http://reranker:7997", reranker_model="bge-base")
    reranker = await get_reranker(make_request(), settings)
    assert isinstance(reranker, InfinityReranker)
    assert reranker._model == "bge-base"  # noqa: SLF001 - the wiring is the subject


async def test_a_blank_model_name_never_reaches_the_sidecar_client() -> None:
    # Compose forwards a key it cannot resolve as an EMPTY STRING, and a rerank
    # call has to carry a model name - so a blank one would 404 against a live
    # sidecar on every turn, quietly, since reranking degrades rather than
    # fails. The settings validator is what prevents it, and the dependency is
    # where it would show, so the two are asserted together.
    settings = make_settings(reranker_url="http://reranker:7997", reranker_model="   ")
    reranker = await get_reranker(make_request(), settings)
    assert isinstance(reranker, InfinityReranker)
    assert reranker._model == "bge-reranker-v2-m3"  # noqa: SLF001 - same


async def test_the_reranker_state_override_wins() -> None:
    settings = make_settings(reranker_url="http://reranker:7997")
    request = make_request()
    preset = NoopReranker()
    request.app.state.reranker = preset
    assert await get_reranker(request, settings) is preset
