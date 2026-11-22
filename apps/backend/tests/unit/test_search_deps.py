"""Tests for the search-provider and page-fetcher deps (config-gated, override-first)."""

from types import SimpleNamespace

from app.api.deps import get_page_fetcher, get_search_provider
from app.core.config import Settings
from app.providers.search.fetcher import FakePageFetcher, HttpPageFetcher
from app.providers.search.searxng import FakeSearchProvider, SearXNGSearchProvider


def _request() -> SimpleNamespace:
    """A minimal stand-in exposing request.app.state, as the dep reads."""
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


def _settings(**over) -> Settings:
    return Settings(_env_file=None, **over)


def test_none_when_url_unset() -> None:
    # No SEARXNG_URL -> web search is off, the turn stays fully local.
    assert get_search_provider(_request(), _settings(searxng_url=None)) is None


def test_constructs_and_caches_when_url_set() -> None:
    req = _request()
    settings = _settings(searxng_url="http://searxng:8080")
    provider = get_search_provider(req, settings)
    assert isinstance(provider, SearXNGSearchProvider)
    # Cached on app.state so the httpx client is shared across requests.
    assert req.app.state.search_provider is provider
    assert get_search_provider(req, settings) is provider


def test_override_wins() -> None:
    req = _request()
    fake = FakeSearchProvider()
    req.app.state.search_provider = fake
    # An injected provider (tests) is returned even when a URL is configured.
    assert get_search_provider(req, _settings(searxng_url="http://searxng:8080")) is fake


def test_page_fetcher_none_when_search_unconfigured() -> None:
    # No search backend -> nothing to fetch, so no fetcher is built.
    assert get_page_fetcher(_request(), _settings(searxng_url=None)) is None


def test_page_fetcher_none_when_disabled() -> None:
    # WEB_FETCH_PAGES=false keeps the snippet-only path even with search wired.
    settings = _settings(searxng_url="http://searxng:8080", web_fetch_pages=False)
    assert get_page_fetcher(_request(), settings) is None


def test_page_fetcher_constructs_and_caches() -> None:
    req = _request()
    settings = _settings(searxng_url="http://searxng:8080")
    fetcher = get_page_fetcher(req, settings)
    assert isinstance(fetcher, HttpPageFetcher)
    # Cached on app.state so the httpx client is shared across requests.
    assert req.app.state.page_fetcher is fetcher
    assert get_page_fetcher(req, settings) is fetcher


def test_page_fetcher_override_wins() -> None:
    req = _request()
    fake = FakePageFetcher()
    req.app.state.page_fetcher = fake
    # An injected fetcher (tests) is returned even when search is configured.
    assert get_page_fetcher(req, _settings(searxng_url="http://searxng:8080")) is fake
