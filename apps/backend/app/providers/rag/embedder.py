"""Embedder implementations: the ModelProvider-backed one and the test fake."""

from __future__ import annotations

import hashlib
import math

from app.providers.base import EmbedOptions, ModelProvider
from app.providers.rag.base import Embedder
from app.services.model_resolution import EmbeddingConfig


class ProviderEmbedder(Embedder):
    """Embeds through the ModelProvider seam using the resolved embedding model.

    The embedding tag is always passed explicitly - the provider's own default
    model is the CHAT model, which would silently produce garbage vectors.
    Requests are split into ``batch_size`` groups so a large ingest neither
    builds an oversized request nor hogs the embedding backend in one burst.
    """

    def __init__(
        self,
        provider: ModelProvider,
        config: EmbeddingConfig,
        *,
        batch_size: int = 32,
    ) -> None:
        self._provider = provider
        self._config = config
        self._batch_size = max(1, batch_size)
        self._options = EmbedOptions(device=config.device)

    @property
    def model(self) -> str:
        """The embedding model tag the resolution chain settled on."""
        return self._config.model

    @property
    def dim(self) -> int | None:
        """The vector dimension, when something states it."""
        return self._config.dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` in batches, preserving order.

        Raises:
            ProviderUnavailableError: Propagated from the provider when the
                embedding backend is unreachable or the model is not pulled.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(
                await self._provider.embed(batch, model=self._config.model, options=self._options)
            )
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed one QUERY, applying the embedder's asymmetric instruct prefix.

        Retrieval-tuned embedders (the qwen3-embedding family, bge, e5) are
        trained with an instruction wrapper on the query side while documents
        embed raw. Skipping the wrapper collapses the similarity scale - tiny
        boilerplate chunks outscore genuinely relevant passages for short
        queries (verified live 2026-07-09) - so the prefix is applied here, at
        the single query-side entrypoint, never in :meth:`embed_texts` (the
        document path). Stored document vectors stay valid either way.
        """
        vectors = await self.embed_texts([f"{self._config.query_prefix}{text}"])
        return vectors[0]


class FakeEmbedder(Embedder):
    """Deterministic hash-based embedder for tests.

    Identical text always maps to the identical unit-length vector (cosine
    similarity 1.0), and different texts land far apart - enough structure to
    test ranking, thresholds, and storage without any model. See
    docs/TESTING.md.
    """

    def __init__(self, dim: int = 32, model: str = "fake-embed") -> None:
        self._dim = max(1, dim)
        self._model = model

    @property
    def model(self) -> str:
        """The synthetic model tag stamped on produced vectors."""
        return self._model

    @property
    def dim(self) -> int:
        """The configured vector dimension."""
        return self._dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return a unit-normalized hash vector per text."""
        return [self._hash_vector(text) for text in texts]

    def _hash_vector(self, text: str) -> list[float]:
        """Derive ``dim`` floats from sha256, extended and unit-normalized."""
        raw: list[float] = []
        counter = 0
        while len(raw) < self._dim:
            digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
            # Center on zero so vectors spread across the whole sphere instead
            # of crowding the positive orthant (which would inflate cosines).
            raw.extend(byte / 127.5 - 1.0 for byte in digest)
            counter += 1
        vector = raw[: self._dim]
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            vector = [1.0] + [0.0] * (self._dim - 1)
            norm = 1.0
        return [v / norm for v in vector]
