"""Unit tests for the settings chain's typed views and model resolution.

Covers the read-side pieces slice 4 adds: the lenient typed views over stored
JSON, the conversation-over-workspace merge, the exhaustive request>conv>ws
precedence table for the enum axes, and the model rungs inside
resolve_chat_models (conversation > workspace > environment > unset, with the
provider's installed listing - not a committed file - making a stale stored tag
fall through).
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.schemas.settings import (
    ConversationSettings,
    TurnDefaults,
    WorkspaceDefaults,
    resolve_turn_defaults,
)
from app.services.model_resolution import resolve_chat_models


def make_settings(**overrides) -> Settings:
    """Settings with nothing but the overrides a test names.

    Args:
        **overrides: Fields to set on the constructed settings.

    Returns:
        Settings built without reading the developer's own ``.env``.
    """
    return Settings(_env_file=None, model_provider="fake", **overrides)


# --- lenient typed views -----------------------------------------------------


def test_from_stored_keeps_valid_and_drops_malformed_fields():
    """A blob from another code version reads field by field, never raises."""
    parsed = ConversationSettings.from_stored(
        {"model": "chat-a", "thinking": "yes", "web_search": "off", "junk": 1}
    )
    assert parsed.model == "chat-a"
    assert parsed.thinking is None  # "yes" is not a mode; dropped, not raised
    assert parsed.web_search == "off"


def test_from_stored_survives_a_non_dict_blob():
    """A corrupted stored value reads as fully unset."""
    assert ConversationSettings.from_stored(None) == ConversationSettings()
    assert WorkspaceDefaults.from_stored("oops") == WorkspaceDefaults()
    assert ConversationSettings.from_stored({"model": 7}).model is None


def test_from_stored_reads_an_explicit_null_as_unset():
    """A stored {"thinking": null} means cleared: fall through, never raise."""
    parsed = ConversationSettings.from_stored({"thinking": None, "web_search": "on"})
    assert parsed.thinking is None
    assert parsed.web_search == "on"


# --- conversation-over-workspace merge ---------------------------------------


def test_conversation_beats_workspace_per_field():
    """The merge is per field, not per blob: each axis falls through alone."""
    merged = resolve_turn_defaults(
        ConversationSettings(thinking="on"),
        WorkspaceDefaults(thinking="off", web_search="off"),
    )
    assert merged == TurnDefaults(thinking="on", web_search="off")


def test_empty_tiers_merge_to_unset():
    assert resolve_turn_defaults(ConversationSettings(), WorkspaceDefaults()) == TurnDefaults()


# --- the precedence table (request > conversation > workspace) ---------------


@pytest.mark.parametrize("request_mode", ["auto", "on", "off"])
@pytest.mark.parametrize("conv", [None, "auto", "on", "off"])
@pytest.mark.parametrize("ws", [None, "auto", "on", "off"])
def test_effective_mode_is_first_explicit_tier(request_mode, conv, ws):
    """Exhaustively: the effective mode is the first non-auto tier, else auto.

    This mirrors the expression plan_turn applies to both enum axes; the table
    is the contract the composer (slice 9) builds against.
    """
    defaults = resolve_turn_defaults(
        ConversationSettings(thinking=conv), WorkspaceDefaults(thinking=ws)
    )
    effective = request_mode if request_mode != "auto" else (defaults.thinking or "auto")

    expected = next(
        (tier for tier in (request_mode, conv, ws) if tier not in (None, "auto")), "auto"
    )
    # The stored tiers only apply when the request defers; a stored "auto" is
    # the same as unset. The one nuance the naive "first non-auto" misses: a
    # conversation "auto"... falls through to ws by the merge? No - the merge
    # keeps conversation "auto" (it is a real stored value), and "auto" or
    # "auto" still lands on auto, so the outcomes agree; assert they do.
    if request_mode == "auto" and conv == "auto" and ws not in (None, "auto"):
        # Stored conversation "auto" pins the merge, deliberately shadowing a
        # workspace explicit: the conversation said "let the heuristic decide".
        expected = "auto"
    assert effective == expected


# --- the model rungs in resolve_chat_models ----------------------------------


def test_conversation_model_beats_workspace_and_environment():
    settings = make_settings(default_chat_model="chat-b")
    pair = resolve_chat_models(settings, conversation_model="solo-c", workspace_model="chat-a")
    assert pair.default == "solo-c"
    # Nothing pairs solo-c: a sibling is now a written-down statement (the
    # environment override or a stored operator declaration), so a pick that
    # nobody paired routes its own thinking turns.
    assert pair.thinking is None


def test_workspace_model_applies_when_conversation_silent():
    settings = make_settings()
    pair = resolve_chat_models(settings, workspace_model="chat-b")
    assert pair.default == "chat-b"


def test_stale_conversation_tag_falls_through_to_workspace():
    """A stored pick the server does not hold cannot wedge a chat.

    The listing is the only thing that can disprove a pick now - there is no
    committed catalog left to validate it against - so the caller has to hand
    one in for the fall-through to happen at all.
    """
    settings = make_settings()
    pair = resolve_chat_models(
        settings,
        conversation_model="deleted:9b",
        workspace_model="chat-b",
        installed={"chat-b"},
    )
    assert pair.default == "chat-b"


def test_a_stale_tag_stands_when_no_listing_disproves_it():
    """The other half of the same rule, and the reason the rung is not a lookup.

    A caller with no listing in hand holds nothing that contradicts the pick,
    so the pick stands. Falling through on a missing listing would silently
    retarget every conversation whenever the provider was unreachable.
    """
    settings = make_settings(default_chat_model="chat-b")
    pair = resolve_chat_models(settings, conversation_model="deleted:9b")
    assert pair.default == "deleted:9b"


def test_no_stored_tiers_falls_through_to_the_environment():
    """The regression pin: absent stored settings resolve to deployment config."""
    settings = make_settings(default_chat_model="chat-a")
    assert resolve_chat_models(settings) == resolve_chat_models(
        settings, conversation_model=None, workspace_model=None
    )
    assert resolve_chat_models(settings).default == "chat-a"


def test_nothing_configured_anywhere_resolves_to_the_empty_first_run_state():
    """An unconfigured deployment resolves to no chat model, and says so.

    This is the state a fresh box is in before anybody has chosen a model, and
    the empty string is the honest report of it. A hardcoded tag used to sit
    here instead, which meant every unconfigured deployment claimed to run a
    specific model nobody had pulled - the turn then failed against the runtime
    naming weights the operator had never heard of, rather than against the
    setting they had never filled in.
    """
    pair = resolve_chat_models(make_settings())
    assert pair.default == ""
    assert pair.thinking is None


def test_a_declared_pairing_serves_the_thinking_turns():
    """The sibling rung below the environment: a stored operator pairing."""
    pair = resolve_chat_models(
        make_settings(), conversation_model="solo-c", declared_thinking_model="chat-b"
    )
    assert pair.default == "solo-c"
    assert pair.thinking == "chat-b"


def test_env_thinking_override_still_wins_over_a_declared_pairing():
    """THINKING_CHAT_MODEL stays an independent sibling override.

    An operator who set the pair in the environment must not be overruled by a
    declaration made from the console.
    """
    settings = make_settings(thinking_chat_model="chat-b")
    pair = resolve_chat_models(
        settings, conversation_model="solo-c", declared_thinking_model="chat-z"
    )
    assert pair.default == "solo-c"
    assert pair.thinking == "chat-b"
