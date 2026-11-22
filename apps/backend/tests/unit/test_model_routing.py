"""Unit tests for route_around_missing: one row per line of its decision table.

The chat-level tests (test_chat_service.py) pin the substitution end to end
through the app; these drive the router directly, because the interesting
cases are the ones a working deployment never reaches - an empty server, a
server holding only embedders, a metadata endpoint that will not answer, a box
where nobody has chosen a chat model at all - and each of them has to be told
apart from the others.
"""

from __future__ import annotations

import pytest

from app.providers.base import (
    ModelCapabilities,
    ModelInfo,
    ModelProvider,
    ProviderUnavailableError,
)
from app.providers.fake_provider import FakeProvider
from app.services.model_facts import clear_facts_cache
from app.services.model_resolution import ChatModelPair
from app.services.model_routing import (
    InstalledNameMatch,
    NoChatModelConfiguredError,
    match_installed_name,
    route_around_missing,
)

PAIR = ChatModelPair(default="qwen3:4b-instruct", thinking="qwen3:4b")

# What a deployment that has chosen nothing resolves to. The blank default is a
# first-run STATE rather than a fault, and it takes a different branch from
# every configured tag below - so it gets its own pair rather than borrowing
# one whose sibling would rescue it before the branch is ever reached.
UNCONFIGURED = ChatModelPair(default="", thinking=None)


@pytest.fixture(autouse=True)
def _fresh_facts_cache():
    clear_facts_cache()
    yield
    clear_facts_cache()


class RoutingProvider(FakeProvider):
    """A provider with a scripted inventory and per-tag capability reports.

    ``caps`` maps a tag to its report; a tag missing from the map raises, which
    is how a probe FAILURE (the runtime would not say) is staged - the facts
    service turns any exception into None.
    """

    def __init__(self, installed: list[str], caps: dict[str, ModelCapabilities]) -> None:
        super().__init__()
        self._installed = installed
        self._caps = caps
        self.listed = 0
        self.probed: list[str] = []

    async def list_models(self) -> list[ModelInfo]:
        self.listed += 1
        return [ModelInfo(name=name) for name in self._installed]

    async def capabilities(self, model: str) -> ModelCapabilities:
        self.probed.append(model)
        if model not in self._caps:
            raise ProviderUnavailableError("The model server is unavailable.")
        return self._caps[model]


def chat_caps() -> ModelCapabilities:
    return ModelCapabilities(completion=True, tools=True, context_length=131072)


def embed_caps() -> ModelCapabilities:
    return ModelCapabilities(embedding=True)


async def route(
    provider: ModelProvider,
    model: str,
    *,
    think: bool = False,
    pair: ChatModelPair = PAIR,
    **kwargs,
):
    """Drive the router with the provider name every test here shares.

    Args:
        provider: The scripted provider.
        model: The tag the configuration resolved to, or "" for none.
        think: Whether this turn resolved to thinking.
        pair: The turn's resolved chat pair.
        **kwargs: Passed through (``listing``).

    Returns:
        The ``(model, note)`` pair the router answers with.
    """
    return await route_around_missing(provider, "fake", pair, model, think, **kwargs)


# --- row 1: the model is installed ---


async def test_installed_model_is_left_alone_and_never_probed() -> None:
    provider = RoutingProvider(["qwen3:4b-instruct"], {"qwen3:4b-instruct": chat_caps()})
    assert await route(provider, "qwen3:4b-instruct") == ("qwen3:4b-instruct", None)
    # The happy path must cost no metadata round trip at all.
    assert provider.probed == []


async def test_bare_tag_matches_an_installed_latest() -> None:
    provider = RoutingProvider(["mistral:latest"], {"mistral:latest": chat_caps()})
    assert await route(provider, "mistral") == ("mistral", None)


# --- row 2: the pair partner is installed ---


async def test_pair_partner_wins_over_any_other_installed_model() -> None:
    # The unrelated tag is listed FIRST and can chat, so this dies if the
    # general fallback is ever consulted before the pair.
    provider = RoutingProvider(
        ["other:8b", "qwen3:4b"], {"other:8b": chat_caps(), "qwen3:4b": chat_caps()}
    )
    model, note = await route(provider, "qwen3:4b-instruct")
    assert model == "qwen3:4b"
    assert note is not None
    assert "qwen3:4b-instruct" in note
    assert "Download the configured model" in note
    assert provider.probed == []


# --- row 3: the first chat-capable tag the server lists ---


async def test_falls_back_to_the_first_chat_capable_tag_in_listing_order() -> None:
    # An embedder first (skipped), then TWO chat models: the earlier one wins,
    # which is what pins listing order rather than any notion of "better".
    provider = RoutingProvider(
        ["nomic-embed:latest", "llama3.1:8b", "llama3.2:3b"],
        {
            "nomic-embed:latest": embed_caps(),
            "llama3.1:8b": chat_caps(),
            "llama3.2:3b": chat_caps(),
        },
    )
    model, note = await route(provider, "qwen3:4b-instruct")
    assert model == "llama3.1:8b"
    assert note is not None
    assert "qwen3:4b-instruct" in note and "llama3.1:8b" in note
    # Stops at the first match: the third tag is never asked about.
    assert provider.probed == ["nomic-embed:latest", "llama3.1:8b"]


async def test_a_failed_probe_inside_the_scan_is_skipped_not_believed() -> None:
    # The first tag will not answer. Reading that as "cannot chat" would be
    # indistinguishable from the runtime saying so, and the model AFTER it can
    # chat - so the scan must carry on rather than fall through to the raise.
    provider = RoutingProvider(["quiet:1b", "llama3.1:8b"], {"llama3.1:8b": chat_caps()})
    model, _ = await route(provider, "qwen3:4b-instruct")
    assert model == "llama3.1:8b"


# --- row 3, the first-run half: nothing was ever configured ---
#
# The same substitution and a DIFFERENT note. There is no missing tag to
# install and no mistake to correct here, only a choice nobody has made, so a
# note naming a configured model would send a reader hunting for a setting they
# never wrote.


async def test_an_unconfigured_deployment_answers_on_the_first_chat_model() -> None:
    provider = RoutingProvider(
        ["nomic-embed:latest", "llama3.1:8b"],
        {"nomic-embed:latest": embed_caps(), "llama3.1:8b": chat_caps()},
    )
    model, note = await route(provider, "", pair=UNCONFIGURED)
    assert model == "llama3.1:8b"
    assert note is not None
    assert "No default chat model is configured" in note
    assert "llama3.1:8b" in note


async def test_the_first_run_note_is_not_the_substitution_note() -> None:
    # Both branches substitute the same tag, so only the WORDING tells a reader
    # which of the two situations they are in - and the first-run one must not
    # claim anything is missing or misconfigured.
    provider = RoutingProvider(["llama3.1:8b"], {"llama3.1:8b": chat_caps()})
    _, unconfigured = await route(provider, "", pair=UNCONFIGURED)
    clear_facts_cache()
    _, substituted = await route(provider, "qwen3:4b-instruct")
    assert unconfigured is not None and substituted is not None
    assert unconfigured != substituted
    assert "not installed" not in unconfigured


async def test_an_unconfigured_deployment_on_an_empty_server_raises() -> None:
    provider = RoutingProvider([], {})
    with pytest.raises(NoChatModelConfiguredError) as excinfo:
        await route(provider, "", pair=UNCONFIGURED)
    assert "Pull a chat model" in str(excinfo.value)


async def test_an_unconfigured_deployment_on_an_embedder_only_server_raises() -> None:
    provider = RoutingProvider(["nomic-embed:latest"], {"nomic-embed:latest": embed_caps()})
    with pytest.raises(NoChatModelConfiguredError):
        await route(provider, "", pair=UNCONFIGURED)


async def test_an_unconfigured_deployment_raises_when_no_probe_answers() -> None:
    # The one place a blank tag is treated differently from a configured one:
    # a configured tag is KEPT when the inventory is unreadable (row 6 below),
    # because the provider's own error tells a better story than a guess. There
    # is nothing to keep here - handing the runtime an empty tag is not a turn -
    # so the typed first-run error is the honest answer instead.
    provider = RoutingProvider(["mystery:1b", "mystery:3b"], {})
    with pytest.raises(NoChatModelConfiguredError):
        await route(provider, "", pair=UNCONFIGURED)


async def test_an_unconfigured_deployment_raises_when_the_server_cannot_be_listed() -> None:
    class Unlistable(FakeProvider):
        async def list_models(self) -> list[ModelInfo]:
            raise ProviderUnavailableError("The model server is unavailable.")

    with pytest.raises(NoChatModelConfiguredError):
        await route(Unlistable(), "", pair=UNCONFIGURED)


# --- row 4: nothing is installed ---


async def test_an_empty_server_raises_the_typed_first_run_error() -> None:
    provider = RoutingProvider([], {})
    with pytest.raises(NoChatModelConfiguredError) as excinfo:
        await route(provider, "qwen3:4b-instruct")
    assert "Pull a chat model" in str(excinfo.value)


# --- row 5: the inventory is known and nothing in it can chat ---


async def test_an_embedder_only_server_raises() -> None:
    provider = RoutingProvider(
        ["nomic-embed:latest", "bge-m3:latest"],
        {"nomic-embed:latest": embed_caps(), "bge-m3:latest": embed_caps()},
    )
    with pytest.raises(NoChatModelConfiguredError):
        await route(provider, "qwen3:4b-instruct")


# --- row 6: the inventory is known but nothing about it is ---


async def test_every_probe_failing_keeps_the_model_and_says_nothing() -> None:
    # Unknown is not evidence. Refusing the turn here would replace the
    # provider's own precise failure with a guess about configuration.
    provider = RoutingProvider(["mystery:1b", "mystery:3b"], {})
    assert await route(provider, "qwen3:4b-instruct") == ("qwen3:4b-instruct", None)


# --- the listing itself ---


async def test_a_handed_over_listing_is_not_fetched_again() -> None:
    provider = RoutingProvider(["llama3.1:8b"], {"llama3.1:8b": chat_caps()})
    listing = await provider.list_models()
    provider.listed = 0
    model, _ = await route(provider, "qwen3:4b-instruct", listing=listing)
    assert model == "llama3.1:8b"
    assert provider.listed == 0


async def test_an_unlistable_server_keeps_the_model() -> None:
    class Unlistable(FakeProvider):
        async def list_models(self) -> list[ModelInfo]:
            raise ProviderUnavailableError("The model server is unavailable.")

    assert await route(Unlistable(), "qwen3:4b-instruct") == ("qwen3:4b-instruct", None)


# --- match_installed_name: the typed-name resolver ---
#
# Pure, so these are a truth table rather than a scenario suite. The three
# outcomes are told apart by FIELD, and most tests assert the whole dataclass:
# a resolution that also carried candidates, or an ambiguity that also named a
# model, would be a caller trap that a one-field assertion walks past.

INSTALLED = ("qwen3:4b-instruct", "qwen3:4b", "qwen3-embedding:0.6b")


def test_an_exact_tag_resolves_to_itself() -> None:
    assert match_installed_name("qwen3:4b-instruct", INSTALLED) == InstalledNameMatch(
        model="qwen3:4b-instruct"
    )


def test_an_exact_tag_beats_the_longer_tags_containing_it() -> None:
    # "qwen3:4b" is installed AND is a substring of "qwen3:4b-instruct".
    # Reading it as a fragment would make an exact name the one spelling that
    # cannot be resolved.
    assert match_installed_name("qwen3:4b", INSTALLED) == InstalledNameMatch(model="qwen3:4b")


def test_a_bare_name_matches_the_installed_latest_tag() -> None:
    # The registry convention: an un-versioned name IS the ":latest" one.
    assert match_installed_name("mistral", ("mistral:latest", "qwen3:4b")) == InstalledNameMatch(
        model="mistral:latest"
    )


def test_a_latest_tag_matches_the_installed_bare_name() -> None:
    # The other direction, which the one-sided membership helper does not
    # cover: a provider that lists a bare name must still answer a request
    # spelling it ":latest", or the equivalence holds for one spelling only.
    assert match_installed_name("mistral:latest", ("mistral", "qwen3:4b")) == InstalledNameMatch(
        model="mistral"
    )


def test_the_namespace_is_equivalent_too() -> None:
    assert match_installed_name("library/mistral:latest", ("mistral",)) == InstalledNameMatch(
        model="mistral"
    )


def test_an_exact_tag_in_another_case_still_resolves() -> None:
    # The runtime lowercases a library name while resolving it, so case is not
    # identity. Without this, typing the right name in the wrong case would
    # fall through to the fragment rule and be refused as ambiguous.
    assert match_installed_name("QWEN3:4B", INSTALLED) == InstalledNameMatch(model="qwen3:4b")


def test_a_unique_fragment_resolves() -> None:
    assert match_installed_name("instruct", INSTALLED) == InstalledNameMatch(
        model="qwen3:4b-instruct", by_fragment=True
    )


def test_a_fragment_is_matched_case_insensitively() -> None:
    assert match_installed_name("EMBEDDING", INSTALLED) == InstalledNameMatch(
        model="qwen3-embedding:0.6b", by_fragment=True
    )


def test_only_a_fragment_win_is_flagged_as_inferred() -> None:
    # The flag is what lets a caller guard the INFERRED half without touching
    # the exact one. Every non-fragment outcome must leave it false, or a
    # caller keying on it would treat a name somebody spelled out in full as
    # something this module guessed at.
    assert match_installed_name("instruct", INSTALLED).by_fragment is True
    assert match_installed_name("qwen3:4b", INSTALLED).by_fragment is False
    assert match_installed_name("QWEN3:4B", INSTALLED).by_fragment is False
    assert match_installed_name("mistral", ("mistral:latest",)).by_fragment is False
    assert match_installed_name("qwen3", INSTALLED).by_fragment is False
    assert match_installed_name("llama", INSTALLED).by_fragment is False


def test_an_ambiguous_fragment_returns_every_candidate() -> None:
    match = match_installed_name("qwen3", INSTALLED)
    assert match.model is None
    # All three, in listing order - the caller prints them and the person
    # picks. Dropping any would hide a legitimate answer from them.
    assert match.candidates == INSTALLED


def test_an_unmatched_name_returns_neither() -> None:
    # Distinct from ambiguity by the EMPTY candidate tuple, which is the only
    # thing that lets the caller answer 404 rather than 400.
    assert match_installed_name("llama", INSTALLED) == InstalledNameMatch()


def test_an_empty_inventory_matches_nothing() -> None:
    assert match_installed_name("qwen3", ()) == InstalledNameMatch()


def test_a_blank_name_matches_nothing() -> None:
    # Guarded explicitly: the empty string is a substring of every tag, so the
    # fragment rule alone would call a blank name ambiguous with the whole
    # inventory - a refusal that describes the wrong problem.
    assert match_installed_name("   ", INSTALLED) == InstalledNameMatch()


def test_one_identity_spelled_two_ways_is_not_guessed_through() -> None:
    # A server listing both "qwen3" and "qwen3:latest" is naming one model
    # twice. Picking either is probably right and definitely a guess.
    match = match_installed_name("qwen3", ("qwen3", "qwen3:latest"))
    assert match.model is None
    assert match.candidates == ("qwen3", "qwen3:latest")
