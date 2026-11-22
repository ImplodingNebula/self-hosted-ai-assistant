"""Which model serves a turn, and which partner serves its reasoning.

Two pure resolutions, and both answer CONFIGURATION questions - what this
deployment runs, and how it pairs two tags - from the places a deployment
records an answer. Nothing here describes a model, so there is no fixture to
build: the inputs are a settings object, two stored strings and, where the
caller has one, the provider's listing.

    resolve_chat_models:  conversation pick > workspace default > environment
                          > "" (the first-run state)
    resolve_thinking_sibling:  THINKING_CHAT_MODEL > stored declaration > None

The rules that cost the most when they break, and therefore get the most
pointed tests here:

- **The empty default is deliberate.** It is this deployment saying it has not
  chosen a chat model, and the routing rescue answers the turn meanwhile. A
  literal here would be a guess at somebody else's inventory, and it used to be
  exactly that.
- **A stored pick falls through only when a listing DISPROVES it.** No listing
  disproves nothing, so the pick stands; a listing that does not hold it means
  a deleted or mistyped pick falls to the next rung rather than wedging the
  conversation.
- **``THINKING_CHAT_MODEL`` reads ``""`` as unset and ``"none"`` as disabled.**
  Compose forwards an unset key as an empty string, so reading empty as "off"
  would silently strip pair-routing from every containerized deployment.
- **A model is not its own sibling.** ``thinking_sibling_for`` self-excludes
  alias-aware, because both sides are written-down configuration and one model
  spelled two ways is one model.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services.model_resolution import (
    KNOWN_THINK_MODES,
    ChatModelPair,
    resolve_chat_models,
    resolve_thinking_sibling,
    thinking_sibling_for,
)

CHAT = "qwen3:4b-instruct"
SIBLING = "qwen3:4b"
PICK = "llama3.1:8b"
OTHER_PICK = "llama3.2:3b"


def make_settings(**overrides: object) -> Settings:
    """Hermetic settings; ``_env_file=None`` keeps a local .env out of it."""
    return Settings(_env_file=None, **overrides)


# --------------------------------------------------------------------------
# resolve_chat_models: the chain, and its first-run floor
# --------------------------------------------------------------------------


def test_nothing_configured_anywhere_resolves_to_the_empty_default() -> None:
    """A box nobody has configured says so, rather than naming a tag at random.

    This is a STATE, not a fault: the first turn against a server holding one
    chat model adopts it, the routing rescue answers the turn meanwhile, and a
    server holding nothing that can chat raises a typed error naming the fix.
    A hardcoded literal here would be this codebase guessing at an inventory it
    has never seen - which is what it did until the deployment-specific model
    file was retired, and it was wrong on every box that did not run that file's
    own tags.
    """
    assert resolve_chat_models(make_settings()) == ChatModelPair(default="", thinking=None)


def test_the_environment_answers_when_nothing_is_stored() -> None:
    # DEFAULT_CHAT_MODEL is deployment configuration and the bottom rung that
    # can answer at all.
    pair = resolve_chat_models(make_settings(default_chat_model=CHAT))
    assert pair.default == CHAT


def test_a_padded_environment_tag_is_trimmed() -> None:
    # The value goes to a model server as a tag and is compared against a
    # listing by equality, where " qwen3:4b-instruct" matches nothing anywhere.
    assert resolve_chat_models(make_settings(default_chat_model=f"  {CHAT}  ")).default == CHAT


def test_a_stored_workspace_default_beats_the_environment() -> None:
    pair = resolve_chat_models(make_settings(default_chat_model=CHAT), workspace_model=PICK)
    assert pair.default == PICK


def test_the_conversation_pick_beats_the_workspace_default() -> None:
    # The nearest rung wins: a conversation pinned to a model keeps it even
    # where the workspace has named another.
    pair = resolve_chat_models(
        make_settings(default_chat_model=CHAT),
        conversation_model=PICK,
        workspace_model=OTHER_PICK,
    )
    assert pair.default == PICK


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_blank_stored_pick_is_no_pick_at_all(blank: str | None) -> None:
    # An empty string is what a cleared field stores, and it has to read as
    # "nothing chosen" rather than as a tag - a blank tag reaching a model
    # server is a request nothing can serve.
    pair = resolve_chat_models(
        make_settings(default_chat_model=CHAT), conversation_model=blank, workspace_model=blank
    )
    assert pair.default == CHAT


# --------------------------------------------------------------------------
# resolve_chat_models: the installed listing, and only when there IS one
# --------------------------------------------------------------------------


def test_a_stored_pick_the_server_holds_is_honoured() -> None:
    # The whole point of consulting the listing: a pick nothing else describes
    # is a real model on a real server, not a typo to fall through.
    pair = resolve_chat_models(
        make_settings(default_chat_model=CHAT),
        conversation_model=PICK,
        installed={PICK, OTHER_PICK},
    )
    assert pair.default == PICK


def test_a_pick_the_server_does_not_hold_falls_through_to_the_next_rung() -> None:
    """A deleted or mistyped pick must not wedge the conversation.

    The listing is the only thing that can DISPROVE a stored tag, and a model
    deleted from the console is exactly the case: the conversation's pin now
    names weights nobody holds, and every turn on it would fail rather than
    quietly falling to the workspace default.

    MUTATION: drop the ``installed_tag_matches`` test from the loop in
    ``resolve_chat_models`` (accept every non-blank stored tag) and this
    returns the deleted pick instead of the environment's tag.
    """
    pair = resolve_chat_models(
        make_settings(default_chat_model=CHAT),
        conversation_model=PICK,
        installed={OTHER_PICK},
    )
    assert pair.default == CHAT


def test_a_disproved_conversation_pick_falls_to_a_workspace_default_that_holds() -> None:
    # Falls to the NEXT RUNG, not straight to the environment: the workspace
    # default is a real answer and the listing holds it.
    pair = resolve_chat_models(
        make_settings(default_chat_model=CHAT),
        conversation_model=PICK,
        workspace_model=OTHER_PICK,
        installed={OTHER_PICK},
    )
    assert pair.default == OTHER_PICK


def test_a_caller_with_no_listing_disproves_nothing() -> None:
    # ``installed=None`` is "I did not ask", never "the server holds nothing".
    # Treating the two the same would make every listing-free caller - the
    # title job, a background resolve - ignore stored picks outright.
    pair = resolve_chat_models(
        make_settings(default_chat_model=CHAT), conversation_model=PICK, installed=None
    )
    assert pair.default == PICK


def test_an_empty_listing_disproves_every_stored_pick() -> None:
    # The other half of the pair above, and the reason None cannot simply be
    # spelled as an empty set: a server that genuinely holds nothing must
    # disprove the pick, so the pair reaches the routing rescue with the
    # environment's tag rather than one nothing can serve.
    pair = resolve_chat_models(
        make_settings(default_chat_model=CHAT), conversation_model=PICK, installed=set()
    )
    assert pair.default == CHAT


@pytest.mark.parametrize(
    ("stored", "installed"),
    [
        ("llama3.1:8b", {"llama3.1:8b"}),
        ("llama3", {"llama3:latest"}),
    ],
    ids=["exact", "implicit-latest"],
)
def test_a_stored_pick_matches_the_provider_spelling(stored: str, installed: set[str]) -> None:
    # Registry-style backends list an explicit ":latest" nobody types, so the
    # comparison goes through the seam's matcher rather than ==. Without it a
    # perfectly valid stored pick reads as "not installed" and is discarded.
    pair = resolve_chat_models(
        make_settings(default_chat_model=CHAT), conversation_model=stored, installed=installed
    )
    assert pair.default == stored


# --------------------------------------------------------------------------
# resolve_thinking_sibling: environment above declaration, and the sentinel
# --------------------------------------------------------------------------


def test_no_source_names_a_sibling_and_nothing_is_paired() -> None:
    # The single-tag deployment: no pair-routing, so the think FLAG does the
    # work on whatever tag is running.
    assert resolve_thinking_sibling(make_settings(), declared=None) is None


def test_a_stored_declaration_pairs_the_two_tags() -> None:
    # The rung an operator writes from the console, against the installed
    # weights - the only source below the environment.
    assert resolve_thinking_sibling(make_settings(), declared=SIBLING) == SIBLING


def test_the_environment_outranks_a_stored_declaration() -> None:
    """An operator who set the pair in the environment is not overruled by a click.

    MUTATION: read ``declared`` before ``settings.thinking_chat_model`` and the
    stored row wins, so a deployment's own configuration silently stops
    applying the moment anybody declares a pairing from the console.
    """
    settings = make_settings(thinking_chat_model="reasoner:8b")
    assert resolve_thinking_sibling(settings, declared=SIBLING) == "reasoner:8b"


@pytest.mark.parametrize("sentinel", ["none", "NONE", "None"])
def test_the_none_sentinel_switches_pair_routing_off(sentinel: str) -> None:
    # The environment is the only source that can turn pair-routing OFF, and
    # the sentinel is a literal rather than a blank precisely because blank
    # already means something else. Case-insensitive: it is typed by hand into
    # a .env file.
    settings = make_settings(default_chat_model=CHAT, thinking_chat_model=sentinel)
    assert resolve_chat_models(settings) == ChatModelPair(default=CHAT, thinking=None)


def test_the_kill_switch_beats_a_stored_declaration_too() -> None:
    """A declaration is a preference about two tags; the sentinel is a policy.

    Letting a stored row override it would make the kill switch unreachable
    from the place it is set - an operator would edit their environment,
    redeploy, and find pair-routing still on with nothing saying why.
    """
    settings = make_settings(thinking_chat_model="none")
    assert resolve_thinking_sibling(settings, declared=SIBLING) is None


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_environment_value_reads_as_unset_not_as_disabled(
    blank: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compose forwards a key it cannot resolve as an EMPTY STRING.

    So blank has to mean "unset" - reading it as "disabled" would strip
    pair-routing from every containerized deployment that never set the key,
    and silently: such a deployment still answers every turn, just never with
    the reasoning checkpoint. It is the contrast with the ``none`` sentinel
    that makes the sentinel a literal rather than an absence.

    Driven through the ENVIRONMENT because that is the shape a deployment
    actually produces, and because the layer that does the work is the settings
    validator rather than the resolver: ``Settings`` normalizes a blank key to
    None before the resolver ever sees it, and the resolver's own blank branch
    then agrees with it. Both agreeing is why no single edit to
    ``model_resolution`` alone can fail this - stated so nobody reads it as a
    resolver guard. What it does pin is the end-to-end contract, which is what
    a future "treat blank as no pair" simplification would break.
    """
    monkeypatch.setenv("THINKING_CHAT_MODEL", blank)
    settings = Settings(_env_file=None)
    assert settings.thinking_chat_model is None
    assert resolve_thinking_sibling(settings, declared=SIBLING) == SIBLING


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_blank_declaration_is_nothing_declared(blank: str | None) -> None:
    # What a cleared form field sends. It must read as "nothing declared"
    # rather than as "no sibling" - opposite answers, and only one of them is
    # what an empty field means, since the environment above still gets to
    # speak.
    settings = make_settings(thinking_chat_model="reasoner:8b")
    assert resolve_thinking_sibling(settings, declared=blank) == "reasoner:8b"
    assert resolve_thinking_sibling(make_settings(), declared=blank) is None


def test_a_padded_sibling_is_trimmed_on_both_rungs() -> None:
    # Same reason as the chat tag: this is a model name, and a padded one
    # reaches the runtime as a tag nothing holds while also slipping past every
    # equality check made against it - including the self-exclusion below.
    padded = make_settings(thinking_chat_model=f" {SIBLING} ")
    assert resolve_thinking_sibling(padded, declared=None) == SIBLING
    assert resolve_thinking_sibling(make_settings(), declared=f"\t{SIBLING}\n") == SIBLING


def test_the_pair_travels_on_the_resolved_chat_models() -> None:
    # The turn reads the sibling through resolve_chat_models, so the two have
    # to be one answer: a second ranking in display code is precisely how a
    # card comes to name a partner the turn ignores.
    pair = resolve_chat_models(
        make_settings(default_chat_model=CHAT), declared_thinking_model=SIBLING
    )
    assert pair == ChatModelPair(default=CHAT, thinking=SIBLING)


# --------------------------------------------------------------------------
# thinking_sibling_for: the same ranking, self-excluded
# --------------------------------------------------------------------------


def test_a_model_is_never_its_own_thinking_sibling() -> None:
    """ "No pair" is what keeps a single-model deployment on the flag path.

    A tag paired with itself is not merely redundant: pair-routing would send
    a thinking turn to the model already running and report a partner that does
    not exist, where "no sibling" correctly leaves the think flag to do the
    work.
    """
    settings = make_settings(thinking_chat_model=SIBLING)
    assert thinking_sibling_for(settings, SIBLING, declared=None) is None
    assert thinking_sibling_for(settings, CHAT, declared=None) == SIBLING


@pytest.mark.parametrize(
    ("configured", "requested"),
    [
        ("qwen3:4b", "library/qwen3:4b"),
        ("library/qwen3:4b", "qwen3:4b"),
        ("qwen3", "qwen3:latest"),
        ("qwen3:latest", "qwen3"),
    ],
    ids=["namespaced-request", "namespaced-config", "implicit-latest", "explicit-latest"],
)
def test_the_self_exclusion_is_alias_aware(configured: str, requested: str) -> None:
    """One model spelled two ways is one model, and both sides are configuration.

    A one-directional match fails OPEN for one of the two spellings, which is
    the wrong direction for an exclusion: a deployment spelling its pair
    ``library/qwen3:4b`` against a request for ``qwen3:4b`` would pair a tag
    with itself under a second name and route every thinking turn to the model
    already loaded.

    MUTATION: compare with ``sibling == model`` instead of
    ``sibling in tag_aliases(model)`` and every row here comes back paired.
    """
    settings = make_settings(thinking_chat_model=configured)
    assert thinking_sibling_for(settings, requested, declared=None) is None


@pytest.mark.parametrize(
    ("default", "sibling"),
    [
        ("qwen3", "qwen3:latest"),
        ("library/qwen3:4b", "qwen3:4b"),
    ],
)
def test_the_pair_resolution_self_excludes_like_every_other_surface(
    default: str, sibling: str
) -> None:
    """An env pair spelling one model two ways reads as "no pair" for the turn too.

    resolve_chat_models used to call the raw ranking directly, skipping the
    alias-aware self-exclusion every other surface applies - so a deployment
    with ``DEFAULT_CHAT_MODEL=qwen3`` and ``THINKING_CHAT_MODEL=qwen3:latest``
    got a pair of two spellings of one model, and thinking turns routed to a
    "second" tag that is the same weights, churning the warm model for nothing.

    MUTATION: build the pair with ``resolve_thinking_sibling`` instead of
    ``thinking_sibling_for`` and both rows come back paired.
    """
    settings = make_settings(default_chat_model=default, thinking_chat_model=sibling)
    assert resolve_chat_models(settings) == ChatModelPair(default=default, thinking=None)


def test_the_declaration_rung_is_self_excluded_as_well() -> None:
    # The exclusion sits below BOTH sources, or a console pairing could name
    # the model itself and the environment path alone would be protected.
    assert thinking_sibling_for(make_settings(), CHAT, declared=CHAT) is None
    assert thinking_sibling_for(make_settings(), CHAT, declared=SIBLING) == SIBLING


def test_a_caller_holding_no_store_reads_as_nothing_paired() -> None:
    # ``declared=None`` is the honest answer for a surface with no store to
    # read, and it must not fall back to naming some other tag.
    assert thinking_sibling_for(make_settings(), CHAT, declared=None) is None


# --------------------------------------------------------------------------
# The shared capability vocabulary
# --------------------------------------------------------------------------


def test_unknown_is_a_resolver_answer_and_never_a_stored_one() -> None:
    # KNOWN_THINK_MODES is the membership set every store validates an untyped
    # value against (a JSON column, a string column, the confirm endpoint).
    # "unknown" is what the resolvers answer when no source has one, so a row
    # or a probe carrying it would be a declaration that nothing is declared.
    assert frozenset({"always", "hybrid", "never"}) == KNOWN_THINK_MODES
    assert "unknown" not in KNOWN_THINK_MODES
