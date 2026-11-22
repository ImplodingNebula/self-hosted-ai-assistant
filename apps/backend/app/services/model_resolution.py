"""Which model serves a turn, and which partner serves its reasoning.

Nothing here describes a model. It answers the two CONFIGURATION questions -
what this deployment runs, and how it pairs two tags - from the places a
deployment actually records an answer: a conversation's pin, the workspace
defaults, the environment, and a stored operator declaration.

What a model IS comes from somewhere else entirely, and from four sources that
cannot rot: the runtime's own capability report, the code-shipped quirk seed
(:mod:`app.services.known_models`), the digest-bound policy declarations
(:mod:`app.services.model_policy`), and the measured probe log
(:mod:`app.services.earned`). A committed file listing per-model facts used to
sit above all of them; it was removed because a written-down inventory is only
ever as current as the day somebody last edited it, while every one of those
four is either measured on the weights in front of it or versioned with the
code that reads it.

The capability vocabulary those four sources speak lives here rather than with
any one of them, because all four and the schemas need it and none of them owns
it.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from pydantic import BaseModel

from app.core.config import Settings
from app.providers.base import installed_tag_matches, tag_aliases

# THINKING_CHAT_MODEL sentinel that disables pair-routing outright. It cannot
# be the empty string: compose forwards an unset key as "", so empty must mean
# "unset", never "disabled".
_THINKING_DISABLED = "none"

# How a model treats the think request flag. "always" models reason on every
# turn and cannot be told not to (the Qwen3 2507 thinking checkpoints); "never"
# models reject or ignore the flag (instruct checkpoints); "hybrid" models
# genuinely toggle per request; "unknown" is what the resolvers answer when no
# source describes the tag.
ThinkMode = Literal["always", "hybrid", "never", "unknown"]

# The think modes anything may MEASURE or DECLARE, as a runtime membership set.
# "unknown" is excluded deliberately: it is the resolvers' answer when no source
# has one, never a statement anybody stores - a row or a probe carrying it would
# be a declaration that nothing is declared. Shared by every store that has to
# validate an untyped value read back out of a JSON column or a string column
# (app.services.model_policy, app.services.earned, the confirm endpoint), so
# there is one list of what those three consider a legal think mode.
KNOWN_THINK_MODES: frozenset[str] = frozenset({"always", "hybrid", "never"})

# The strongest agency mode a model may attempt (its declared ceiling). Ordered
# least to most autonomous; the resolver only ever clamps DOWN this scale. See
# app.services.agency.
AgencyMode = Literal["deterministic", "assisted", "agentic"]

# Whether a chat template can take a tool list at all - a MECHANICAL fact about
# the template, distinct from the policy question of how much the model may
# drive (AgencyMode). "none" means the model is never offered tools.
ToolsMode = Literal["native", "none"]


class ChatModelPair(NamedTuple):
    """The resolved chat tags for a turn: the default and its thinking sibling.

    Attributes:
        default: The tag normal (non-thinking) turns request. EMPTY on a
            deployment where nothing has chosen one yet - see
            :func:`resolve_chat_models` for why that is a state rather than a
            fault.
        thinking: The always-thinking pair sibling, or None when pair-routing
            is off (one tag that toggles, or an explicit disable).
    """

    default: str
    thinking: str | None


class EmbeddingConfig(BaseModel):
    """The embedding setup a retrieval runs through.

    Attributes:
        model: The embedding model tag (defines the vector space).
        dim: Vector dimension, when something states it.
        device: Where inference should run; "cpu" keeps a small embedder off
            the GPU so it never evicts the chat model on a tight card.
        query_prefix: Instruction prepended to retrieval QUERIES only
            (documents embed raw). Retrieval-tuned embedders are asymmetric;
            skipping their prescribed query wrapper collapses the similarity
            scale the relevance floors are calibrated against.
    """

    model: str
    dim: int | None = None
    device: Literal["auto", "cpu"] = "auto"
    query_prefix: str = ""


def resolve_chat_models(
    settings: Settings,
    *,
    conversation_model: str | None = None,
    workspace_model: str | None = None,
    installed: set[str] | None = None,
    declared_thinking_model: str | None = None,
) -> ChatModelPair:
    """Resolve the chat pair: conversation > workspace > environment > unset.

    The stored rungs come first: a per-conversation model pick, then the
    workspace default. Each is honoured unless the caller has the provider's
    ``installed`` listing in hand AND that listing says the server does not
    hold the tag - a deleted or mistyped pick then falls through to the next
    rung instead of wedging the conversation. A caller with no listing has
    nothing that could disprove a stored pick, so it stands.

    Below them, ``DEFAULT_CHAT_MODEL`` is deployment configuration and wins
    over nothing further, because there is nothing further. When no rung
    answers, the default is the EMPTY string, and that is the honest answer on
    a box nobody has configured yet: this deployment has not chosen a chat
    model. It is not a failure state and it does not stay long - the first turn
    against a server holding exactly one chat model adopts that model as the
    workspace default (``chat_service._seed_single_model_default``), the
    routing rescue answers the turn meanwhile
    (:func:`app.services.model_routing.route_around_missing`), and a server
    holding nothing that can chat raises the typed
    :class:`~app.services.model_routing.NoChatModelConfiguredError` that names
    the fix. A hardcoded literal here would be a guess at somebody else's
    inventory, and it used to be exactly that.

    Args:
        settings: The active application settings (the environment rung).
        conversation_model: The conversation's stored model pick, if any.
        workspace_model: The workspace's stored default model, if any.
        installed: Model names as the provider reports them, when the caller
            has a listing in hand; None disproves nothing.
        declared_thinking_model: The stored operator pairing for the tag this
            call resolves to, when the caller has read one. Only meaningful
            once the default is known, so a caller resolves once to learn the
            tag, reads the declaration for it, and resolves again. That second
            call is pure and the read behind it is cached, so it costs nothing
            on a warm path.

    Returns:
        The resolved :class:`ChatModelPair`.
    """
    chained: str | None = None
    for stored in (conversation_model, workspace_model):
        tag = (stored or "").strip()
        if not tag:
            continue
        if installed is None or installed_tag_matches(tag, installed):
            chained = tag
            break
    default = chained or (settings.default_chat_model or "").strip()
    # Through thinking_sibling_for, never resolve_thinking_sibling directly:
    # only the tag-keyed form self-excludes alias-aware, and an env pair that
    # spells one model two ways (`qwen3` / `qwen3:latest`) must read as "no
    # pair" here exactly as it does on every card and certificate - two
    # spellings in one pair would route thinking turns to a "second" tag that
    # is the same weights. An empty default excludes nothing (no aliases).
    return ChatModelPair(
        default=default,
        thinking=thinking_sibling_for(settings, default, declared=declared_thinking_model),
    )


def resolve_thinking_sibling(settings: Settings, *, declared: str | None) -> str | None:
    """Rank the two sources of a thinking sibling into the one that applies.

    Pure, and the single copy of the ranking - but every consumer, the turn
    through :func:`resolve_chat_models` included, reaches it via
    :func:`thinking_sibling_for`, which adds the alias-aware self-exclusion on
    top. A second copy of this ranking in display code is precisely how a card
    comes to name a partner the turn ignores.

    The order, strongest first:

    1. ``THINKING_CHAT_MODEL``. Deployment configuration, and the only source
       that can also switch pair-routing OFF (the literal ``none``). An
       operator who set it in the environment must not be overruled by a click.
    2. A stored operator declaration, made from the console against the
       installed weights.

    Nothing else answers. A pairing is a statement about how this deployment
    wants two tags used, so it lives where this deployment records things.

    Args:
        settings: The active settings (the environment override).
        declared: The stored operator pairing for this tag, if any.

    Returns:
        The tag that should serve thinking turns, or None for no pair-routing.
    """
    raw = settings.thinking_chat_model
    if raw is not None:
        # Empty reads as UNSET, never as "disabled": compose forwards an unset
        # key as an empty string, and reading that as off would silently strip
        # pair-routing from every containerized deployment.
        stripped = raw.strip()
        if stripped:
            # The TRIMMED value, like the rung below it: this is a model name,
            # and a padded one reaches the runtime as a tag nothing holds while
            # also slipping past any equality check made against it.
            return None if stripped.lower() == _THINKING_DISABLED else stripped
    return (declared or "").strip() or None


def thinking_sibling_for(settings: Settings, model: str, *, declared: str | None) -> str | None:
    """Name the tag that would serve ``model``'s thinking turns, as a turn routes it.

    :func:`resolve_thinking_sibling` holds the ranking; this is the form for a
    caller that holds ONE TAG rather than a resolved pair, and it exists so the
    two surfaces a reasoning-with-tools certificate passes through cannot
    resolve that tag differently. The verify job drives the sibling to measure
    the reasoning case (:mod:`app.services.model_verify`) and the earned
    resolution binds the verdict back to those weights
    (:mod:`app.services.earned`); resolve them apart and the binding never
    matches, so the capability is withheld with nothing anywhere saying why.

    Args:
        settings: The active settings (the environment override).
        model: The tag whose thinking turns are being placed.
        declared: The stored operator pairing for ``model``, when the caller
            has read one. A caller holding no store passes None, which reads as
            "nothing is paired".

    Returns:
        The sibling tag, or None when nothing pairs this one. A resolution that
        names ``model`` ITSELF also answers None: a model is not its own
        sibling, and "no pair" is what keeps a single-model deployment on the
        path where the think flag does the work.
    """
    sibling = resolve_thinking_sibling(settings, declared=declared)
    # Alias-aware, because both sides here are written-down configuration: an
    # environment override spelling the pair `qwen3:latest` against a request
    # for `qwen3` would otherwise pair a tag with itself under a second name.
    if sibling is None or sibling in tag_aliases(model):
        return None
    return sibling
