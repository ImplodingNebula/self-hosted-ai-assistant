"""Web-search seam: the SearchProvider abstraction and its value types.

Mirrors the RAG seam (app/providers/rag/base.py): an ABC with one abstract
async method, a pydantic value type, and a seam-level exception the chat service
catches instead of any backend-specific error (httpx, ...). The first impl is a
SearXNG client; any metasearch engine can replace it behind this boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class SearchProviderUnavailableError(Exception):
    """Raised when the search backend cannot be reached or answers abnormally.

    Seam-level type: the chat service catches this (not httpx), keeping the
    boundary engine-agnostic. An empty result list means "found nothing" and is
    NOT an error; this signals the engine itself is unavailable.
    """


class SearchResult(BaseModel):
    """One search-engine hit.

    Carries engine metadata only. Fetching and extracting the full result page
    is a separate, later concern, so v1 grounds on ``snippet`` (the engine's
    query-focused excerpt). ``url`` always travels so a citation can link back
    to the source.
    """

    title: str
    url: str
    snippet: str | None = None


class SearchProvider(ABC):
    """The web-search seam. First impl: :class:`SearXNGSearchProvider`."""

    @abstractmethod
    async def search(self, query: str, *, limit: int) -> list[SearchResult]:
        """Return up to ``limit`` results for ``query`` (fewer, or empty, is fine).

        Raises:
            SearchProviderUnavailableError: The backend is unreachable or
                returned a malformed response.
        """
        raise NotImplementedError
