"""Reranker implementations: passthrough and the Infinity sidecar client.

Reranking is optional per deployment: dense scores pass through unchanged where
no sidecar is wired, and where one is (``RERANKER_URL``) the top dense hits are
re-scored by the model that sidecar was started with (``RERANKER_MODEL``);
Ollama has no rerank endpoint, so it runs beside the model server rather than
inside it. Selection happens in the services layer - callers never construct a
concrete reranker directly.
"""

from __future__ import annotations

import logging

import httpx

from app.providers.rag.base import Reranker, ScoredChunk

logger = logging.getLogger(__name__)


class NoopReranker(Reranker):
    """Keeps the dense ranking; used when no reranker sidecar is wired."""

    async def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Return the first ``top_k`` chunks unchanged."""
        return chunks[:top_k]


class InfinityReranker(Reranker):
    """Re-scores hits via an Infinity sidecar's ``/rerank`` endpoint.

    Degrades, never breaks: any sidecar failure logs a warning and returns the
    dense ranking - a missing reranker must cost quality, not availability.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the client for an Infinity server at ``base_url``.

        Args:
            base_url: Root URL of the Infinity sidecar.
            model: Reranker model id as the sidecar serves it.
            timeout: Read timeout in seconds for rerank calls.
            client: Optional externally-owned HTTP client (tests inject a mock
                transport). When omitted, one is created lazily and owned;
                call :meth:`aclose` on shutdown.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the owned HTTP client (no-op for an injected one)."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared client, creating it lazily on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Return ``top_k`` chunks re-ordered by the sidecar's relevance scores.

        The sidecar's score replaces the dense score on the returned chunks so
        downstream thresholds and citations report what actually ranked them.
        """
        if not chunks:
            return []
        payload = {
            "model": self._model,
            "query": query,
            "documents": [chunk.text for chunk in chunks],
        }
        try:
            resp = await self._get_client().post(f"{self._base_url}/rerank", json=payload)
            resp.raise_for_status()
            results = resp.json().get("results")
            ranked: list[ScoredChunk] = []
            seen: set[int] = set()
            for item in results or []:
                index = item.get("index")
                score = item.get("relevance_score")
                if not isinstance(index, int) or not 0 <= index < len(chunks):
                    raise ValueError(f"rerank result index out of range: {index!r}")
                if index in seen:
                    # A repeated index would duplicate the passage in the
                    # grounding block and its citation downstream.
                    raise ValueError(f"rerank result index repeated: {index}")
                seen.add(index)
                ranked.append(
                    chunks[index].model_copy(
                        update={"score": float(score) if score is not None else 0.0}
                    )
                )
            if not ranked:
                raise ValueError("rerank response carried no results")
            ranked.sort(key=lambda c: c.score, reverse=True)
            return ranked[:top_k]
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            logger.warning(
                "Reranker unavailable or returned a malformed response; keeping the dense ranking",
                exc_info=True,
            )
            return chunks[:top_k]
