"""Tests for the SearXNG search provider and the test fake."""

import httpx
import pytest

from app.core.errors import ServiceUnavailableError
from app.main import _warm_search_bg
from app.providers.search.base import SearchProviderUnavailableError, SearchResult
from app.providers.search.searxng import FakeSearchProvider, SearXNGSearchProvider
from app.services.warmup import warm_search


def _mock_provider(handler) -> SearXNGSearchProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    # retry_backoff=0 so the one cold-start retry does not sleep in tests.
    return SearXNGSearchProvider("http://searxng:8080", retry_backoff=0.0, client=client)


async def test_maps_results_to_search_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert request.url.params["format"] == "json"
        assert request.url.params["q"] == "python releases"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://python.org/downloads",
                        "title": "Downloads",
                        "content": "The latest Python release is documented here.",
                        "engine": "duckduckgo",
                    }
                ]
            },
        )

    provider = _mock_provider(handler)
    results = await provider.search("python releases", limit=5)
    assert results == [
        SearchResult(
            title="Downloads",
            url="https://python.org/downloads",
            snippet="The latest Python release is documented here.",
        )
    ]
    await provider.aclose()


async def test_limit_truncates_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": f"https://ex.test/{i}", "title": f"R{i}", "content": f"c{i}"}
                    for i in range(10)
                ]
            },
        )

    provider = _mock_provider(handler)
    results = await provider.search("q", limit=3)
    assert len(results) == 3
    await provider.aclose()


async def test_skips_results_missing_url_or_title() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": "", "title": "no url"},
                    {"url": "https://ex.test/1", "title": ""},
                    {"title": "no url key", "content": "x"},
                    {"url": "https://ex.test/ok", "title": "keep", "content": "kept"},
                ]
            },
        )

    provider = _mock_provider(handler)
    results = await provider.search("q", limit=10)
    assert [r.url for r in results] == ["https://ex.test/ok"]
    await provider.aclose()


async def test_missing_content_yields_none_snippet() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"url": "https://ex.test/1", "title": "T"}]})

    provider = _mock_provider(handler)
    results = await provider.search("q", limit=5)
    assert results[0].snippet is None
    await provider.aclose()


async def test_403_json_disabled_raises_seam_error() -> None:
    # SearXNG returns 403 for format=json until its settings.yml enables the
    # json format - a common misconfiguration that must surface as the seam
    # error (not a raw httpx error), so the caller can degrade.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Forbidden")

    provider = _mock_provider(handler)
    with pytest.raises(SearchProviderUnavailableError):
        await provider.search("q", limit=5)
    await provider.aclose()


async def test_connection_failure_raises_seam_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = _mock_provider(handler)
    with pytest.raises(SearchProviderUnavailableError):
        await provider.search("q", limit=5)
    await provider.aclose()


async def test_malformed_body_raises_seam_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = _mock_provider(handler)
    with pytest.raises(SearchProviderUnavailableError):
        await provider.search("q", limit=5)
    await provider.aclose()


# --- retry-once on transient cold-start faults ---


async def test_retry_recovers_from_transient_first_failure() -> None:
    # The "first search fails, second works" case: a cold connection fault on
    # attempt 1 is retried once, and the second attempt's results are returned.
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("cold connection")
        return httpx.Response(
            200,
            json={"results": [{"url": "https://ex.test/ok", "title": "T", "content": "c"}]},
        )

    provider = _mock_provider(handler)
    results = await provider.search("q", limit=5)
    assert calls["n"] == 2
    assert [r.url for r in results] == ["https://ex.test/ok"]
    await provider.aclose()


async def test_read_timeout_is_not_retried() -> None:
    # A read timeout means the engines are genuinely slow; retrying would just
    # pay the full wait again, so it fails fast on the first attempt.
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("engines slow")

    provider = _mock_provider(handler)
    with pytest.raises(SearchProviderUnavailableError):
        await provider.search("q", limit=5)
    assert calls["n"] == 1
    await provider.aclose()


async def test_http_status_error_is_not_retried() -> None:
    # A 403 (json format disabled) is a configuration fault, not transient, so
    # it is surfaced without a wasted retry.
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, text="Forbidden")

    provider = _mock_provider(handler)
    with pytest.raises(SearchProviderUnavailableError):
        await provider.search("q", limit=5)
    assert calls["n"] == 1
    await provider.aclose()


async def test_malformed_body_is_retried_once_then_fails() -> None:
    # A malformed body could be a transient partial response during boot, so it
    # is retried once; a persistently bad shape then maps to the seam error.
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = _mock_provider(handler)
    with pytest.raises(SearchProviderUnavailableError):
        await provider.search("q", limit=5)
    assert calls["n"] == 2
    await provider.aclose()


async def test_empty_results_return_cleanly() -> None:
    # An empty results list is a successful "found nothing", never an error and
    # never retried.
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"results": []})

    provider = _mock_provider(handler)
    assert await provider.search("q", limit=5) == []
    assert calls["n"] == 1
    await provider.aclose()


# --- warm_search (service) ---


async def test_warm_search_returns_timing_and_count() -> None:
    took_ms, count = await warm_search(FakeSearchProvider())
    assert isinstance(took_ms, int)
    # FakeSearchProvider synthesizes one result for limit=1 (the warm query).
    assert count == 1


async def test_warm_search_maps_unavailable_to_service_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    provider = _mock_provider(handler)
    with pytest.raises(ServiceUnavailableError):
        await warm_search(provider)
    await provider.aclose()


# --- _warm_search_bg (background lifespan task) ---


async def test_warm_search_bg_swallows_failure() -> None:
    # Best-effort: a failing warm must NEVER raise out of the background task,
    # or a cold SearXNG would crash startup.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    provider = _mock_provider(handler)
    await _warm_search_bg(provider)  # must not raise
    await provider.aclose()


async def test_warm_search_bg_success_path() -> None:
    # A healthy provider warms without raising.
    await _warm_search_bg(FakeSearchProvider())


# --- FakeSearchProvider ---


async def test_fake_returns_scripted_results_and_records_call() -> None:
    scripted = [SearchResult(title="T", url="https://ex.test/1", snippet="s")]
    fake = FakeSearchProvider(results=scripted)
    results = await fake.search("hello", limit=5)
    assert results == scripted
    assert fake.last_query == "hello"
    assert fake.last_limit == 5


async def test_fake_synthesizes_deterministic_results() -> None:
    fake = FakeSearchProvider()
    first = await fake.search("same query", limit=2)
    second = await fake.search("same query", limit=2)
    assert first == second
    assert len(first) == 2
    assert all(r.url.startswith("https://example.test/") for r in first)


async def test_fake_respects_limit() -> None:
    fake = FakeSearchProvider()
    assert len(await fake.search("q", limit=1)) == 1
    assert len(await fake.search("q", limit=0)) == 0
