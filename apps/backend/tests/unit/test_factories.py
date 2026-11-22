"""The provider factory honours the deployed timeout configuration.

Why this is worth a file of its own: the Ollama client's read timeout is the
ceiling on a COLD MODEL LOAD, and a client that hangs up mid-load makes Ollama
abort the load entirely. The factory passing the setting through - and the
setting outlasting the server's own OLLAMA_LOAD_TIMEOUT - is what stops a warm
from being killed by its own warmer, so both facts are pinned here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.factories import build_model_provider, build_search_provider
from app.providers.search.searxng import SearXNGSearchProvider

# The BASE compose file, which is where the ollama services set their runtime
# environment. It lived in the development override until the production
# layering was finished; a production stack does not read that file, so the
# server was silently running its built-in 5-minute load ceiling there.
_BASE_COMPOSE = Path(__file__).resolve().parents[4] / "infra" / "docker-compose.yml"


def _make_settings(**overrides: object) -> Settings:
    """Ollama-provider settings, hermetic from the developer's own .env."""
    return Settings(model_provider="ollama", _env_file=None, **overrides)


def test_the_factory_hands_the_configured_read_timeout_to_the_client() -> None:
    provider = build_model_provider(_make_settings(ollama_read_timeout_seconds=123.0))
    assert provider._timeout.read == 123.0  # noqa: SLF001 - the wiring is the subject


def test_the_client_read_timeout_outlasts_the_server_load_timeout() -> None:
    # The compose default for the SERVER's give-up ceiling, read from the file
    # that sets it, so the two cannot drift apart silently: the backend must be
    # the one still listening when Ollama reports a load it gave up on.
    text = _BASE_COMPOSE.read_text(encoding="utf-8")
    match = re.search(r"OLLAMA_LOAD_TIMEOUT:-(\d+)m", text)
    assert match is not None, "the ollama services no longer default OLLAMA_LOAD_TIMEOUT"
    server_ceiling_seconds = int(match.group(1)) * 60
    assert Settings(_env_file=None).ollama_read_timeout_seconds > server_ceiling_seconds


def _search_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_the_default_search_provider_is_built_from_the_url() -> None:
    provider = build_search_provider(_search_settings(searxng_url="http://searxng:8080"))

    assert isinstance(provider, SearXNGSearchProvider)


def test_an_unset_url_still_yields_no_provider() -> None:
    # The pre-existing contract, unchanged by the new selector: a deployment
    # that simply never set the URL keeps behaving exactly as it did.
    assert build_search_provider(_search_settings(searxng_url=None)) is None
    assert build_search_provider(_search_settings(searxng_url="")) is None


def test_selecting_none_refuses_even_with_a_url_configured() -> None:
    # The whole point of naming the seam: "off" is now a stated choice rather
    # than something inferred from a blank URL somebody may have blanked by
    # accident.
    settings = _search_settings(search_provider="none", searxng_url="http://searxng:8080")

    assert build_search_provider(settings) is None


def test_an_unknown_search_provider_fails_the_boot() -> None:
    # A Literal, like MODEL_PROVIDER: a typo must be loud, not silently
    # disable web search for the life of the deployment.
    with pytest.raises(ValueError, match="search_provider"):
        _search_settings(search_provider="searxnb")
