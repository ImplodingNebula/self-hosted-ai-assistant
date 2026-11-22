"""Tests for search_and_ingest: mapping results into the RAG session scope."""

import asyncio
import time
from unittest import mock

from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.providers.rag.embedder import FakeEmbedder
from app.providers.search.base import SearchResult
from app.providers.search.fetcher import FakePageFetcher, PageFetcher
from app.providers.search.searxng import FakeSearchProvider
from app.services import rag_service, web_search
from tests.conftest import make_memory_store


def _tenant() -> TenantContext:
    return TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")


def _settings(**over) -> Settings:
    return Settings(_env_file=None, **over)


async def test_lands_results_in_session_scope_with_url() -> None:
    store = make_memory_store()
    embedder = FakeEmbedder(dim=32)
    query = "latest python release"
    provider = FakeSearchProvider(
        results=[
            # Snippet equals the query so the hash embedder retrieves it (cosine 1).
            SearchResult(title="Downloads", url="https://python.org/dl", snippet=query),
            SearchResult(title="News", url="https://python.org/news", snippet="unrelated"),
        ]
    )
    count = await web_search.search_and_ingest(
        query=query,
        provider=provider,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-1",
    )
    assert count == 2
    hits = await rag_service.retrieve(
        query=query,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-1",
    )
    assert hits
    assert hits[0].source_type == "web"
    assert hits[0].url == "https://python.org/dl"
    await store.aclose()


async def test_dedupes_by_url() -> None:
    store = make_memory_store()
    provider = FakeSearchProvider(
        results=[
            SearchResult(title="A", url="https://same.test/x", snippet="alpha"),
            SearchResult(title="B", url="https://same.test/x", snippet="beta"),
        ]
    )
    count = await web_search.search_and_ingest(
        query="q",
        provider=provider,
        tenant=_tenant(),
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=_settings(),
        session_id="c",
    )
    assert count == 1
    await store.aclose()


async def test_empty_results_ingest_nothing() -> None:
    store = make_memory_store()
    provider = FakeSearchProvider(results=[])
    count = await web_search.search_and_ingest(
        query="q",
        provider=provider,
        tenant=_tenant(),
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=_settings(),
        session_id="c",
    )
    assert count == 0
    await store.aclose()


async def test_snippet_falls_back_to_title() -> None:
    store = make_memory_store()
    title = "Fallback title used as text"
    provider = FakeSearchProvider(
        results=[SearchResult(title=title, url="https://t.test/1", snippet=None)]
    )
    count = await web_search.search_and_ingest(
        query=title,
        provider=provider,
        tenant=_tenant(),
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=_settings(),
        session_id="c",
    )
    assert count == 1
    hits = await rag_service.retrieve(
        query=title,
        tenant=_tenant(),
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=_settings(),
        session_id="c",
    )
    assert hits and hits[0].text == title
    await store.aclose()


async def test_oversized_fields_degrade_instead_of_raising() -> None:
    # Untrusted engine output larger than the ingest caps must not raise a
    # ValidationError (which would abort the chat turn); the over-long url is
    # dropped (not a clickable link) but the result is still ingested and cited.
    store = make_memory_store()
    embedder = FakeEmbedder(dim=32)
    query = "latest breaking news"
    provider = FakeSearchProvider(
        results=[
            SearchResult(
                title="T" * 900,
                url="https://example.test/" + "a" * 3000,
                snippet=query,
            )
        ]
    )
    count = await web_search.search_and_ingest(
        query=query,
        provider=provider,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-big",
    )
    assert count == 1
    hits = await rag_service.retrieve(
        query=query,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-big",
    )
    assert hits
    assert hits[0].url is None  # over-long url dropped, but the result survived
    await store.aclose()


async def test_non_http_scheme_url_is_dropped() -> None:
    # A non-http(s) result url (javascript:/data:/magnet:) must never reach the
    # frontend as a clickable citation; it is dropped while the result is kept.
    store = make_memory_store()
    embedder = FakeEmbedder(dim=32)
    query = "today's headline"
    provider = FakeSearchProvider(
        results=[SearchResult(title="X", url="javascript:alert(1)", snippet=query)]
    )
    count = await web_search.search_and_ingest(
        query=query,
        provider=provider,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-js",
    )
    assert count == 1
    hits = await rag_service.retrieve(
        query=query,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-js",
    )
    assert hits and hits[0].url is None
    await store.aclose()


async def test_top_k_limits_the_search() -> None:
    store = make_memory_store()
    provider = FakeSearchProvider()  # synthesizes up to 3 results
    await web_search.search_and_ingest(
        query="anything",
        provider=provider,
        tenant=_tenant(),
        embedder=FakeEmbedder(dim=32),
        store=store,
        settings=_settings(web_search_top_k=2),
        session_id="c",
    )
    assert provider.last_limit == 2
    await store.aclose()


async def test_grounds_on_fetched_page_text_over_snippet() -> None:
    # With a fetcher wired, the ingested text is the page body, not the search
    # snippet (which is often the site's boilerplate meta description). The hash
    # embedder scores 1.0 only on identical text, so retrieval uses the page body
    # verbatim: a hit proves the page text (not the "site tagline" snippet) landed.
    store = make_memory_store()
    embedder = FakeEmbedder(dim=32)
    page_body = "Team A won 3-1 to take the trophy in the final."
    provider = FakeSearchProvider(
        results=[
            SearchResult(title="Match report", url="https://sport.test/1", snippet="site tagline"),
        ]
    )
    fetcher = FakePageFetcher(pages={"https://sport.test/1": page_body})
    count = await web_search.search_and_ingest(
        query="who won the final",
        provider=provider,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-fetch",
        fetcher=fetcher,
    )
    assert count == 1
    assert fetcher.fetched == ["https://sport.test/1"]
    hits = await rag_service.retrieve(
        query=page_body,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-fetch",
    )
    assert hits
    assert hits[0].text == page_body  # page body ingested, not "site tagline"
    await store.aclose()


async def test_fetch_failure_falls_back_to_snippet() -> None:
    # The fetcher returns None (fetch failed / empty); the result is still
    # ingested on its snippet rather than dropped.
    store = make_memory_store()
    embedder = FakeEmbedder(dim=32)
    query = "latest release notes"
    provider = FakeSearchProvider(
        results=[SearchResult(title="R", url="https://x.test/1", snippet=query)]
    )
    fetcher = FakePageFetcher(default=None)  # every fetch fails
    count = await web_search.search_and_ingest(
        query=query,
        provider=provider,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-fail",
        fetcher=fetcher,
    )
    assert count == 1
    assert fetcher.fetched == ["https://x.test/1"]
    hits = await rag_service.retrieve(
        query=query,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-fail",
    )
    assert hits and hits[0].text == query  # snippet text survived
    await store.aclose()


async def test_fetch_disabled_stays_snippet_only() -> None:
    # WEB_FETCH_PAGES=false: the fetcher is never called even when supplied, and
    # grounding stays on the snippet (the lower-latency small-system path).
    store = make_memory_store()
    embedder = FakeEmbedder(dim=32)
    query = "current standings"
    provider = FakeSearchProvider(
        results=[SearchResult(title="R", url="https://x.test/1", snippet=query)]
    )
    fetcher = FakePageFetcher(pages={"https://x.test/1": "FULL PAGE BODY"})
    count = await web_search.search_and_ingest(
        query=query,
        provider=provider,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(web_fetch_pages=False),
        session_id="conv-off",
        fetcher=fetcher,
    )
    assert count == 1
    assert fetcher.fetched == []  # disabled -> not fetched
    hits = await rag_service.retrieve(
        query=query,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(web_fetch_pages=False),
        session_id="conv-off",
    )
    assert hits and hits[0].text == query  # snippet, not the page body
    await store.aclose()


async def test_raising_fetcher_falls_back_to_snippet() -> None:
    # A fetcher that RAISES must not abort the turn: the per-page failure is
    # swallowed and the result is ingested on its snippet.
    class _BoomFetcher(PageFetcher):
        async def fetch_text(self, url: str) -> str | None:
            raise RuntimeError("boom")

    store = make_memory_store()
    embedder = FakeEmbedder(dim=32)
    query = "some fresh query"
    provider = FakeSearchProvider(
        results=[SearchResult(title="R", url="https://x.test/1", snippet=query)]
    )
    count = await web_search.search_and_ingest(
        query=query,
        provider=provider,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-boom",
        fetcher=_BoomFetcher(),
    )
    assert count == 1  # raise swallowed -> snippet used
    hits = await rag_service.retrieve(
        query=query,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-boom",
    )
    assert hits and hits[0].text == query
    await store.aclose()


async def test_page_fetch_concurrency_is_bounded() -> None:
    # Concurrent fetches are capped by web_fetch_concurrency; the peak in-flight
    # count never exceeds it (a resource-safety bound on the fetch fan-out).
    class _CountingFetcher(PageFetcher):
        def __init__(self) -> None:
            self.live = 0
            self.peak = 0

        async def fetch_text(self, url: str) -> str | None:
            self.live += 1
            self.peak = max(self.peak, self.live)
            await asyncio.sleep(0.02)  # hold the slot so overlaps accumulate
            self.live -= 1
            return url

    store = make_memory_store()
    fetcher = _CountingFetcher()
    provider = FakeSearchProvider(
        results=[
            SearchResult(title=f"T{i}", url=f"https://x.test/{i}", snippet="s") for i in range(8)
        ]
    )
    await web_search.search_and_ingest(
        query="q",
        provider=provider,
        tenant=_tenant(),
        embedder=FakeEmbedder(dim=32),
        store=store,
        # max_pages raised past the result count so all 8 fetch (the page cap
        # is tested separately); only the concurrency bound is under test here.
        settings=_settings(web_fetch_concurrency=3, web_fetch_max_pages=8),
        session_id="conv-conc",
        fetcher=fetcher,
    )
    assert fetcher.peak == 3  # 8 pages, bound of 3 -> at most 3 in flight
    await store.aclose()


async def test_fetches_only_top_pages_rest_use_snippets() -> None:
    # web_fetch_max_pages caps full-page fetches to the TOP results; the tail
    # is still ingested (and retrievable) on its snippet.
    store = make_memory_store()
    embedder = FakeEmbedder(dim=32)
    tail_snippet = "tail result snippet"
    provider = FakeSearchProvider(
        results=[
            SearchResult(title="T0", url="https://x.test/0", snippet="s0"),
            SearchResult(title="T1", url="https://x.test/1", snippet="s1"),
            SearchResult(title="T2", url="https://x.test/2", snippet=tail_snippet),
            SearchResult(title="T3", url="https://x.test/3", snippet="s3"),
        ]
    )
    fetcher = FakePageFetcher(default="FULL PAGE BODY")
    count = await web_search.search_and_ingest(
        query="q",
        provider=provider,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(web_fetch_max_pages=2),
        session_id="conv-cap",
        fetcher=fetcher,
    )
    assert count == 4  # every result ingested...
    assert fetcher.fetched == ["https://x.test/0", "https://x.test/1"]  # ...top 2 fetched
    hits = await rag_service.retrieve(
        query=tail_snippet,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-cap",
    )
    assert hits and hits[0].text == tail_snippet  # tail grounded on its snippet
    await store.aclose()


async def test_repeat_search_reuses_recent_session_results() -> None:
    # A page already ingested into this session within the reuse window is
    # served from the store: the second search neither re-fetches nor
    # re-ingests it, yet it still counts as retrievable. The page body differs
    # from the snippet, marking the stored ingest as full-page grounded.
    store = make_memory_store()
    embedder = FakeEmbedder(dim=32)
    query = "latest python release"
    page_body = query + " with the full release notes"
    provider = FakeSearchProvider(
        results=[SearchResult(title="D", url="https://python.org/dl", snippet=query)]
    )
    fetcher = FakePageFetcher(pages={"https://python.org/dl": page_body})

    async def run() -> int:
        return await web_search.search_and_ingest(
            query=query,
            provider=provider,
            tenant=_tenant(),
            embedder=embedder,
            store=store,
            settings=_settings(),
            session_id="conv-cache",
            fetcher=fetcher,
        )

    assert await run() == 1
    assert fetcher.fetched == ["https://python.org/dl"]
    assert await run() == 1  # still counted: the stored chunks serve the turn
    assert fetcher.fetched == ["https://python.org/dl"]  # no second fetch
    hits = await rag_service.retrieve(
        query=page_body,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-cache",
    )
    assert hits
    await store.aclose()


async def test_snippet_only_ingest_is_refetched_within_window() -> None:
    # Turn 1's page fetch fails, so the doc lands on its thin snippet. Turn 2
    # (same window, fetch now healthy) must NOT reuse the snippet-grounded
    # ingest: the fetch is retried and the full page replaces it.
    store = make_memory_store()
    embedder = FakeEmbedder(dim=32)
    query = "latest python release"
    url = "https://python.org/dl"
    page_body = query + " with the full release notes body"
    provider = FakeSearchProvider(results=[SearchResult(title="D", url=url, snippet=query)])

    async def run(fetcher: FakePageFetcher) -> int:
        return await web_search.search_and_ingest(
            query=query,
            provider=provider,
            tenant=_tenant(),
            embedder=embedder,
            store=store,
            settings=_settings(),
            session_id="conv-thin",
            fetcher=fetcher,
        )

    assert await run(FakePageFetcher(default=None)) == 1  # fetch fails -> snippet
    healthy = FakePageFetcher(pages={url: page_body})
    assert await run(healthy) == 1
    assert healthy.fetched == [url]  # retried despite the reuse window
    hits = await rag_service.retrieve(
        query=page_body,
        tenant=_tenant(),
        embedder=embedder,
        store=store,
        settings=_settings(),
        session_id="conv-thin",
    )
    assert hits and hits[0].text == page_body  # full page replaced the snippet
    await store.aclose()


async def test_stale_result_outside_window_is_refetched() -> None:
    # The reuse window is a strict recency check: a page ingested BEFORE the
    # horizon is re-fetched, never silently served stale for the whole session.
    store = make_memory_store()
    embedder = FakeEmbedder(dim=32)
    query = "latest python release"
    url = "https://python.org/dl"
    provider = FakeSearchProvider(results=[SearchResult(title="D", url=url, snippet=query)])
    fetcher = FakePageFetcher(pages={url: query + " full page"})

    async def run() -> int:
        return await web_search.search_and_ingest(
            query=query,
            provider=provider,
            tenant=_tenant(),
            embedder=embedder,
            store=store,
            settings=_settings(),
            session_id="conv-stale",
            fetcher=fetcher,
        )

    an_hour_ago = time.time() - 3600
    with mock.patch("app.services.rag_service.time") as frozen:
        # Back-date the ingest-side clock only: created_at lands an hour ago,
        # far past the 15-minute reuse horizon (ttl_at stays in the future).
        frozen.time.return_value = an_hour_ago
        assert await run() == 1
    assert len(fetcher.fetched) == 1
    assert await run() == 1
    assert len(fetcher.fetched) == 2  # outside the window -> fetched again
    await store.aclose()


async def test_reuse_window_zero_disables_the_cache() -> None:
    # WEB_RESULT_CACHE_MINUTES=0 restores the always-refetch behavior (the
    # unchanged-content ingest skip still avoids re-embedding downstream).
    store = make_memory_store()
    embedder = FakeEmbedder(dim=32)
    query = "latest python release"
    provider = FakeSearchProvider(
        results=[SearchResult(title="D", url="https://python.org/dl", snippet=query)]
    )
    fetcher = FakePageFetcher(pages={"https://python.org/dl": query})

    async def run() -> int:
        return await web_search.search_and_ingest(
            query=query,
            provider=provider,
            tenant=_tenant(),
            embedder=embedder,
            store=store,
            settings=_settings(web_result_cache_minutes=0),
            session_id="conv-nocache",
            fetcher=fetcher,
        )

    assert await run() == 1
    assert await run() == 1
    assert fetcher.fetched == ["https://python.org/dl", "https://python.org/dl"]
    await store.aclose()
