"""Fetch web pages and extract readable text for grounding.

Search snippets describe a page; they are often the site's boilerplate meta
description ("Breaking News, Latest News and Videos") rather than its content,
so grounding on them alone answers broad questions poorly. This module fetches
the actual result pages and extracts their readable text, which the RAG pipeline
then chunks, ranks, and cites. Extraction itself lives in :mod:`app.parsing.html`
(an EPUB chapter is the same problem: markup that has to become prose); it drops
page chrome (navigation, headers, footers, banners) and, when a page marks a
semantic <main> region, keeps only that - a lightweight readability pass, all with
the stdlib parser and no third-party dependency, that stops the article body from
being buried under menus (or truncated away under the char cap). Fetching sits
behind the :class:`PageFetcher` seam so a richer extractor can drop in later
without touching callers. Pages that build their content with client-side JavaScript
(many news landing pages) expose little text to a static fetch, so grounding on
them stays thin - an accepted limit of a local-first fetcher that runs no browser.

Web pages are attacker-influenceable, so fetching is defensive: http(s) only, a
host guard that rejects any non-globally-routable address (SSRF) re-applied to
every redirect hop (redirects are followed by hand, not by httpx, and capped),
an HTML content-type check, and hard byte/char caps. Every failure degrades to
``None`` so the caller falls back to the snippet - fetching never raises into a
turn.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from abc import ABC, abstractmethod
from urllib.parse import urljoin, urlsplit

import httpx

# Re-exported: extraction moved to the parsing seam, because an EPUB chapter is
# HTML in a zip and needs exactly the same pass. This module keeps the name so the
# fetcher's own callers (and its tests) still find it where it has always been.
from app.parsing.html import extract_text

__all__ = ["HttpPageFetcher", "PageFetcher", "extract_text"]

logger = logging.getLogger(__name__)

# A browser-ish UA: some sites return an empty body or 403 to unknown agents.
_USER_AGENT = "Mozilla/5.0 (compatible; AI-Chatbot-Platform/1.0; +local)"

# Redirects are followed manually (see fetch_text) so the SSRF host guard runs on
# EVERY hop, not just the original URL. This caps how many hops are chased.
_MAX_REDIRECTS = 3


class PageFetcher(ABC):
    """Fetches a web page and returns its readable text (or None)."""

    @abstractmethod
    async def fetch_text(self, url: str) -> str | None:
        """Return the page's extracted text, or None when it cannot be used."""


async def host_is_public(host: str) -> bool:
    """True when every address ``host`` resolves to is a globally-routable IP.

    SSRF guard: a search result (or a redirect it triggers) could point at an
    internal address (link-local metadata endpoints, loopback admin panels,
    RFC1918 or shared/CGN hosts). Every resolved address must be globally
    reachable (``is_global``), which rejects private, loopback, link-local,
    shared/carrier-grade-NAT (100.64.0.0/10), reserved, multicast, and
    unspecified ranges in one check that does not hinge on the interpreter patch
    level; IPv4-mapped IPv6 forms are unwrapped first so a mapped internal
    address cannot slip through. Resolving up front blocks the common case; it
    does not defeat DNS-rebinding (a re-resolve between check and request), an
    accepted limitation for a best-effort grounding fetch.
    """
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        return False
    if not infos:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None:
            ip = mapped
        if not ip.is_global:
            return False
    return True


class HttpPageFetcher(PageFetcher):
    """Fetches pages over HTTP(S) and extracts their text with the stdlib parser.

    Mirrors the search/reranker client management: an injectable client for
    tests, a lazily-owned one otherwise, released via :meth:`aclose`.
    """

    def __init__(
        self,
        *,
        timeout: float = 5.0,
        max_bytes: int = 2_000_000,
        max_chars: int = 8000,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure fetch bounds and the optional injected HTTP client."""
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._max_chars = max_chars
        self._client = client
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the owned HTTP client (no-op for an injected one)."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared client, creating it lazily on first use.

        ``follow_redirects`` is deliberately off: redirects are chased by hand in
        :meth:`fetch_text` so the SSRF host guard re-runs on every hop. httpx's
        own redirect following would jump to an attacker-chosen internal host
        without re-checking it.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
                headers={"User-Agent": _USER_AGENT},
            )
        return self._client

    async def fetch_text(self, url: str) -> str | None:
        """Fetch ``url`` and return its extracted text, or None on any problem.

        Redirects are followed manually, up to :data:`_MAX_REDIRECTS`, with the
        scheme allowlist and :func:`host_is_public` guard re-applied to every
        hop - so a public page cannot redirect the fetch onto an internal host.
        """
        current = url
        try:
            for _ in range(_MAX_REDIRECTS + 1):
                parts = urlsplit(current)
                if parts.scheme not in ("http", "https") or not parts.hostname:
                    return None
                if not await host_is_public(parts.hostname):
                    logger.debug("Skipping non-public host in web fetch: %s", parts.hostname)
                    return None
                async with self._get_client().stream("GET", current) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location")
                        if not location:
                            return None
                        # Resolve a relative Location against the current URL; the
                        # loop re-validates the new host before the next request.
                        current = urljoin(current, location)
                        continue
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "").lower()
                    if "html" not in content_type:
                        return None
                    body = bytearray()
                    async for chunk in resp.aiter_bytes():
                        body.extend(chunk)
                        if len(body) >= self._max_bytes:
                            break
                    html = bytes(body).decode(resp.encoding or "utf-8", errors="replace")
                    text = extract_text(html, max_chars=self._max_chars)
                    return text or None
            logger.debug("Web fetch exceeded redirect cap for %s", url)
            return None
        except (httpx.HTTPError, ValueError, OSError) as exc:
            logger.debug("Web page fetch failed for %s: %s", url, exc)
            return None


class FakePageFetcher(PageFetcher):
    """Deterministic fetcher for tests and offline development.

    Returns scripted text per URL (or a default), and records the URLs it was
    asked for. No network. See docs/TESTING.md.
    """

    def __init__(self, pages: dict[str, str] | None = None, *, default: str | None = None) -> None:
        """Create a fake returning ``pages[url]`` (falling back to ``default``)."""
        self._pages = pages or {}
        self._default = default
        self.fetched: list[str] = []

    async def fetch_text(self, url: str) -> str | None:
        """Return the scripted text for ``url`` (or the default)."""
        self.fetched.append(url)
        return self._pages.get(url, self._default)
