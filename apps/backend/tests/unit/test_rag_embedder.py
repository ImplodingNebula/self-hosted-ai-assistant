"""Unit tests for ProviderEmbedder batching and FakeEmbedder determinism."""

import math

import pytest

from app.providers.base import EmbedOptions, ModelProvider, ProviderUnavailableError
from app.providers.fake_provider import FakeProvider
from app.providers.rag.embedder import FakeEmbedder, ProviderEmbedder
from app.services.model_resolution import EmbeddingConfig


class RecordingProvider(FakeProvider):
    """Records every embed call so batching behavior is observable."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_calls: list[tuple[list[str], str | None, EmbedOptions | None]] = []

    async def embed(self, texts, *, model=None, options=None):  # type: ignore[override]
        self.embed_calls.append((list(texts), model, options))
        return [[0.1, 0.2] for _ in texts]


class FailingProvider(FakeProvider):
    async def embed(self, texts, *, model=None, options=None):  # type: ignore[override]
        raise ProviderUnavailableError("The model server is unavailable.")


CONFIG = EmbeddingConfig(model="qwen3-embedding:0.6b", dim=1024, device="cpu")


async def test_provider_embedder_batches_and_preserves_order() -> None:
    provider = RecordingProvider()
    embedder = ProviderEmbedder(provider, CONFIG, batch_size=2)
    texts = [f"text {i}" for i in range(5)]
    vectors = await embedder.embed_texts(texts)
    assert len(vectors) == 5
    assert [len(call[0]) for call in provider.embed_calls] == [2, 2, 1]
    assert provider.embed_calls[0][0] == ["text 0", "text 1"]


async def test_provider_embedder_passes_catalog_model_and_device() -> None:
    provider = RecordingProvider()
    embedder = ProviderEmbedder(provider, CONFIG, batch_size=8)
    await embedder.embed_texts(["hello"])
    _, model, options = provider.embed_calls[0]
    assert model == "qwen3-embedding:0.6b"
    assert options is not None and options.device == "cpu"
    assert embedder.model == "qwen3-embedding:0.6b"
    assert embedder.dim == 1024


async def test_provider_embedder_propagates_seam_errors() -> None:
    embedder = ProviderEmbedder(FailingProvider(), CONFIG)
    with pytest.raises(ProviderUnavailableError):
        await embedder.embed_texts(["hello"])


async def test_embed_query_returns_single_vector() -> None:
    embedder = ProviderEmbedder(RecordingProvider(), CONFIG)
    vector = await embedder.embed_query("hello")
    assert vector == [0.1, 0.2]


PREFIXED_CONFIG = EmbeddingConfig(
    model="qwen3-embedding:0.6b",
    dim=1024,
    device="cpu",
    query_prefix="Instruct: retrieve\nQuery:",
)


async def test_embed_query_applies_catalog_prefix_to_queries_only() -> None:
    # Asymmetric retrieval: the instruct wrapper goes on QUERIES exactly as
    # configured, while documents (embed_texts) stay raw - stored vectors
    # remain valid when a prefix is introduced.
    provider = RecordingProvider()
    embedder = ProviderEmbedder(provider, PREFIXED_CONFIG)
    await embedder.embed_query("say hi")
    await embedder.embed_texts(["say hi"])
    query_call, document_call = provider.embed_calls
    assert query_call[0] == ["Instruct: retrieve\nQuery:say hi"]
    assert document_call[0] == ["say hi"]


async def test_embed_query_without_prefix_stays_raw() -> None:
    provider = RecordingProvider()
    embedder = ProviderEmbedder(provider, CONFIG)
    await embedder.embed_query("say hi")
    assert provider.embed_calls[0][0] == ["say hi"]


async def test_fake_embedder_is_deterministic_and_unit_length() -> None:
    embedder = FakeEmbedder(dim=32)
    [first] = await embedder.embed_texts(["same text"])
    [second] = await embedder.embed_texts(["same text"])
    assert first == second
    assert len(first) == 32
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-9)


async def test_fake_embedder_separates_different_texts() -> None:
    embedder = FakeEmbedder(dim=32)
    [a, b] = await embedder.embed_texts(["refund policy", "shipping regions"])
    cosine = sum(x * y for x, y in zip(a, b, strict=True))
    assert cosine < 0.9


async def test_fake_embedder_extends_beyond_one_digest() -> None:
    embedder = FakeEmbedder(dim=100)
    [vector] = await embedder.embed_texts(["long dim"])
    assert len(vector) == 100


def test_fake_provider_satisfies_seam() -> None:
    # Guards the ABC contract: FakeProvider must stay instantiable as the
    # embed signature evolves.
    assert isinstance(FakeProvider(), ModelProvider)
