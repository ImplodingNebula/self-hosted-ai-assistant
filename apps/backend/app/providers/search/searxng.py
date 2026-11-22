"""SearXNG search provider (+ a deterministic fake for tests).

SearXNG is a self-hosted metasearch engine aggregating public engines behind
one JSON API (free, no account or API key). This client hits its ``/search``
endpoint with ``format=json`` and maps each result to a
:class:`~app.providers.search.base.SearchResult`. It mirrors the RAG reranker's
httpx client management: an injectable client for tests, a lazily-owned one
otherwise, released via :meth:`aclose` at shutdown.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging

import httpx

from app.providers.search.base import (
    SearchProvider,
    SearchProviderUnavailableError,
    SearchResult,
)

logger = logging.getLogger(__name__)


class SearXNGSearchProvider(SearchProvider):
    """Queries a SearXNG sidecar's JSON API.

    Raises :class:`SearchProviderUnavailableError` on any transport or parse
    failure so the caller decides whether to degrade (auto mode) or surface it
    (on mode). An empty list means "the engine found nothing", which is
    deliberately distinct from "the engine is down".
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 12.0,
        retry_backoff: float = 0.25,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the client for a SearXNG server at ``base_url``.

        Args:
            base_url: Root URL of the SearXNG sidecar.
            timeout: Read timeout in seconds. It MUST stay at or above SearXNG's
                own ``outgoing.request_timeout`` (10s in infra/searxng/
                settings.yml) plus a little overhead: SearXNG holds the response
                open while it waits on its slowest engine, so a client ceiling
                below that wait would cut off slow-but-successful engines - the
                cause of the "first (cold) search fails, second works" symptom.
            retry_backoff: Seconds to pause before the single retry on a
                transient transport fault (see :meth:`_get_results_with_retry`);
                0 disables the pause (tests).
            client: Optional externally-owned HTTP client (tests inject a mock
                transport). When omitted, one is created lazily and owned; call
                :meth:`aclose` on shutdown.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._retry_backoff = retry_backoff
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

    async def search(self, query: str, *, limit: int) -> list[SearchResult]:
        """Return up to ``limit`` SearXNG results for ``query``."""
        results = await self._get_results_with_retry({"q": query, "format": "json"})
        mapped: list[SearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            title = item.get("title")
            if not isinstance(url, str) or not url.strip():
                continue
            if not isinstance(title, str) or not title.strip():
                continue
            snippet = item.get("content")
            mapped.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet if isinstance(snippet, str) and snippet.strip() else None,
                )
            )
            if len(mapped) >= limit:
                break
        return mapped

    async def _fetch_results(self, params: dict[str, str]) -> list[object]:
        """Run one SearXNG query and return its raw ``results`` list.

        Lets the underlying error propagate on failure - an httpx error on a
        transport/status fault, or ``ValueError`` on a body that is not JSON or
        lacks a ``results`` list - so :meth:`_get_results_with_retry` can decide
        which are worth a second attempt.
        """
        resp = await self._get_client().get(f"{self._base_url}/search", params=params)
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise ValueError("SearXNG response is missing a 'results' list")
        return results

    async def _get_results_with_retry(self, params: dict[str, str]) -> list[object]:
        """Fetch the results list, retrying once on a transient cold-start fault.

        The first search after boot (or a long idle) commonly fails on
        connection setup - an empty connection pool, DNS not yet resolvable
        ([Errno -3]), or a cold uwsgi worker - then succeeds immediately after
        ("first search fails, second works"). A single retry folds that recovery
        into one call so the user never sees it. Two failures are deliberately
        NOT retried: a read timeout (the engines are genuinely slow; a retry
        would just pay the full wait again, and the client ``timeout`` is sized
        above SearXNG's own engine wait to cover the honest case) and an HTTP
        status error (e.g. 403 when the json format is disabled - a
        configuration fault a retry cannot fix). An empty result list is a
        successful "found nothing" and never reaches this path.
        """
        for attempt in range(2):
            try:
                return await self._fetch_results(params)
            except httpx.ReadTimeout as exc:
                raise SearchProviderUnavailableError("The search backend timed out.") from exc
            except httpx.HTTPStatusError as exc:
                # SearXNG returns 403 for format=json when its settings.yml has
                # not enabled the json format: a configuration fault, not a
                # transient one, so surface it without retrying.
                raise SearchProviderUnavailableError("The search backend is unavailable.") from exc
            except (httpx.TransportError, ValueError, TypeError) as exc:
                if attempt == 0:
                    logger.debug("SearXNG search failed (attempt 1), retrying once: %s", exc)
                    if self._retry_backoff > 0:
                        await asyncio.sleep(self._retry_backoff)
                    continue
                raise SearchProviderUnavailableError("The search backend is unavailable.") from exc
            except httpx.HTTPError as exc:
                # Any other httpx error (bad URL, redirect loop, stream error):
                # not transient, so map to the seam error without retrying.
                raise SearchProviderUnavailableError("The search backend is unavailable.") from exc
        # Unreachable: attempt 1 above either returns or raises.
        raise SearchProviderUnavailableError("The search backend is unavailable.")


class FakeSearchProvider(SearchProvider):
    """Deterministic search provider for tests and offline development.

    Returns a fixed (or query-derived) list of results with no network, so
    web-search chat turns are reproducible. Records the last call so tests can
    assert on routing. See docs/TESTING.md.
    """

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        """Create a fake that returns ``results`` (or synthetic ones) per query."""
        self._results = results
        self.last_query: str | None = None
        self.last_limit: int | None = None

    async def search(self, query: str, *, limit: int) -> list[SearchResult]:
        """Return the scripted results, or synthetic ones derived from ``query``."""
        self.last_query = query
        self.last_limit = limit
        if self._results is not None:
            return self._results[:limit]
        digest = hashlib.sha256(query.encode()).hexdigest()[:8]
        return [
            SearchResult(
                title=f"Result {index + 1} for {query}",
                url=f"https://example.test/{digest}/{index + 1}",
                snippet=f"Synthetic web snippet {index + 1} about {query}.",
            )
            for index in range(min(max(limit, 0), 3))
        ]
