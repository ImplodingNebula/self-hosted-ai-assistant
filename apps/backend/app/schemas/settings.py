"""Typed views over the stored settings JSON (conversation and workspace).

``conversations.settings`` and ``workspace_settings.settings`` are free JSON
columns; these models are the narrow typed windows the resolution chain reads
through. Reading is LENIENT by design: a stored blob may have been written by
another code version, and a chat turn must never fail on it - a malformed
field is dropped (falling through to the next tier of the chain), never
raised. Writing is the opposite: :class:`_SettingsPatch` rejects unknown keys
and malformed values (422 at the API boundary), because a write that silently
stored a typo would surface later as a mysteriously ignored setting. The same
three field names ride the wire in ``ChatRequest``, the stored JSON, and the
composer controls, so the chain never translates vocabulary between tiers.

Agency is deliberately absent: it is a developer affordance, never a stored
default (a stored escalation would be a privilege-escalation surface).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

# The request-axis vocabulary shared with ChatRequest: "auto" means "no
# explicit choice here - defer to the next tier".
SettingMode = Literal["auto", "on", "off"]

# The smallest context ceiling an operator may set. A thinking turn already
# reserves 2048 tokens for its answer, so a ceiling at that number leaves
# literally nothing for the system prompt and the question - every turn would
# assemble a prompt the runtime then truncates from the head, which is the
# exact failure the always-a-number window rule exists to prevent. 4096 is the
# runtime's own VRAM-scaled default and the smallest window this platform has
# ever run a real turn in. Pinned against the reserve by a test, so lowering
# the reserve cannot silently make this floor unsafe.
MIN_CONTEXT_CEILING = 4096
# The largest, and no longer a formality. This bound used to be close to
# decorative: the ceiling only ever clamped DOWN, so a number above every real
# window was merely inert garbage. That stopped being true when the window for
# a tag NOTHING declares became the ceiling itself rather than something the
# ceiling trimmed - there, the number typed here IS what a turn asks the model
# server to allocate, and this bound is the only thing between a fat-fingered
# zero and a request for a gigabyte of KV cache. The model's own native window
# still bounds the request underneath it, which is what keeps a merely
# generous value harmless; this catches the absurd ones.
MAX_CONTEXT_CEILING = 1_048_576

# The longest workspace persona. Nothing caps the SYSTEM_PROMPT env var, and
# deliberately so - that one is deployment configuration, edited on the box by
# whoever runs it. This one arrives from a textarea over HTTP, lands in a JSON
# column every chat turn reads, and is prepended to every prompt, so it needs a
# bound. 8000 characters is about 2000 tokens at the budget's four-chars-a-token
# approximation: half of MIN_CONTEXT_CEILING, the smallest window an operator
# may pin. A persona longer than that could crowd the question out of the
# smallest permitted window entirely, which no amount of retrieval budgeting
# downstream can repair.
MAX_SYSTEM_PROMPT_CHARS = 8000


def _as_prompt(value: object) -> object:
    """Trim a prompt and read a blank one as unset; pass anything else through.

    Runs BEFORE validation so the length bound applies to the trimmed text
    rather than to the whitespace around it, and - more importantly - so a
    whitespace-only textarea is indistinguishable from an empty one at every
    later layer.

    That indistinguishability is the point. ``SYSTEM_PROMPT=""`` disables the
    system prompt outright, injection defenses and tool guidance included, and
    that power stays with the environment: a person editing the box's config,
    not a text box in a browser. Mapping blank to ``None`` here makes the
    difference UNREPRESENTABLE in the stored blob - the workspace field can
    say "use this persona" or "I have nothing to say", never "silence the
    prompt" - so no downstream reader has to remember the distinction.

    Args:
        value: Whatever the caller supplied for the field.

    Returns:
        The trimmed text, None for a blank string, or ``value`` untouched when
        it is not a string (so a wrong TYPE still fails validation loudly).
    """
    if isinstance(value, str):
        return value.strip() or None
    return value


# The one spelling of the field, shared by the read view and the patch so the
# two cannot normalize differently - a stored value that was trimmed on write
# but not on read is how "why does my prompt have a blank first line" happens.
WorkspacePrompt = Annotated[str | None, BeforeValidator(_as_prompt)]


class _StoredSettings(BaseModel):
    """Shared lenient-read behaviour for typed views over stored JSON."""

    @classmethod
    def from_stored(cls, raw: object) -> Self:
        """Build the typed view, dropping anything malformed field by field.

        A stored ``thinking: "yes"`` reads as absent rather than raising, so
        the chain falls through to the next tier - the same never-500 stance
        the citation reader takes on stored metadata.
        """
        if not isinstance(raw, dict):
            return cls()
        values = {}
        for name in cls.model_fields:
            if name not in raw:
                continue
            try:
                candidate = cls.model_validate({name: raw[name]})
            except ValidationError:
                continue
            values[name] = getattr(candidate, name)
        return cls(**values)


class ConversationSettings(_StoredSettings):
    """Per-conversation defaults (the composer's controls persist here).

    Attributes:
        model: The chat tag this conversation prefers, or None.
        thinking: The conversation's default thinking mode, or None.
        web_search: The conversation's default web-search mode, or None.
    """

    model: str | None = None
    thinking: SettingMode | None = None
    web_search: SettingMode | None = None


class WorkspaceDefaults(_StoredSettings):
    """Workspace-wide defaults (the console's Workspace defaults page writes these).

    The three request axes :class:`ConversationSettings` also carries, plus the
    three a conversation has no business holding: the two budget clamps are
    DEPLOYMENT-shaped choices about how much machine a turn may spend, and the
    persona is who the assistant IS for everyone here, so all three exist once
    per workspace rather than per thread.

    Attributes:
        model: A chat tag every turn defaults to, or None.
        embedding_model: The tag retrieval embeds with, or None to let the
            deployment answer. Unlike the chat default this one is not a
            preference: the embedding model DEFINES the vector space, so
            changing it points every query at a fresh, empty collection and
            leaves the old one full and unread. Stored here because the
            alternative - one answer shipped identically to every machine
            - could not describe a deployment that had moved on.
        thinking: The workspace's default thinking mode, or None.
        web_search: The workspace's default web-search mode, or None.
        system_prompt: The persona every turn opens with, replacing the
            built-in one and any ``SYSTEM_PROMPT`` the environment set, or None
            to leave the persona to the tiers below. Blank reads as None; the
            tool-guidance layer is appended to it either way - see
            :func:`app.services.prompts.compose_system_prompt`.
        context_ceiling: The largest context window any turn may request, in
            tokens, or None to leave the size to the deployment. It NARROWS a
            window the weights themselves grant and never widens it; below
            that it is the deployment's affordance itself and moves the window
            in both directions - see
            :func:`app.services.model_facts.effective_num_ctx`.
        reranker: False forces retrieval to stay dense-only even where the
            deployment wired a reranker sidecar; True and None both mean "use
            the sidecar when there is one". True never CONJURES a reranker -
            the sidecar is deployment configuration - so the two permissive
            values differ only in what the toggle remembers.
    """

    model: str | None = None
    embedding_model: str | None = None
    thinking: SettingMode | None = None
    web_search: SettingMode | None = None
    # Bounded on the READ side for the same reason the ceiling below is: a
    # persona is not a preference that can harmlessly fall through, it is
    # prepended to every prompt. A blob written by a code version with a wider
    # bound reads as unset here - the deployment's own persona, which is a
    # working turn - rather than as a wall of text eating the window.
    system_prompt: WorkspacePrompt = Field(default=None, max_length=MAX_SYSTEM_PROMPT_CHARS)
    # Bounded on the READ side too, which the three older fields have no need
    # of. Their malformed values fall through to the next tier harmlessly,
    # whereas an out-of-range ceiling is actively hostile in BOTH directions
    # now: a stored ``1`` would clamp every turn to one token, and on a tag
    # nothing declares a stored ``10**9`` is a window the turn would actually
    # ask for. Bounded here, the lenient reader DROPS it and the deployment
    # falls back to its configured default window - the answer it would have
    # given had nothing been stored at all.
    context_ceiling: int | None = Field(
        default=None, ge=MIN_CONTEXT_CEILING, le=MAX_CONTEXT_CEILING
    )
    reranker: bool | None = None


class _SettingsPatch(BaseModel):
    """A per-field merge over a stored settings blob, never a whole-blob write.

    Three states per field, told apart by presence: a field absent from the
    request leaves the stored value alone, an explicit ``null`` clears it, and
    a present value overwrites. Unknown keys are rejected
    (``extra="forbid"``) - stored settings are read leniently, but accepting a
    typo on write would store a key nothing reads.

    Shared by both scopes so the tri-state cannot drift apart between them: a
    conversation PATCH and a workspace PATCH mean the same thing by "absent"
    and by "null", and a reader of one has learned the other.
    """

    model_config = ConfigDict(extra="forbid")

    def merged_into(self, stored: object) -> dict:
        """Return a new settings dict with this patch applied over ``stored``.

        Only fields the request actually carried change anything; stored keys
        this model does not know (written by another code version) ride along
        untouched. Returns a NEW dict on purpose: the JSON column has no
        mutable wrapper, so only a reassigned object marks the row dirty -
        an in-place update would read back correctly and silently never persist.

        Args:
            stored: The row's current settings value (any JSON).

        Returns:
            The merged settings dict to assign back to the column.
        """
        merged = dict(stored) if isinstance(stored, dict) else {}
        for name in self.model_fields_set:
            value = getattr(self, name)
            if value is None:
                merged.pop(name, None)
            else:
                merged[name] = value
        return merged


class ConversationSettingsPatch(_SettingsPatch):
    """The settings half of a conversation PATCH.

    A stored ``"auto"`` on the enum axes is deliberately DISTINCT from a
    clear: it pins the heuristic for this thread even where a workspace
    default says on/off (the resolution chain lets a conversation ``"auto"``
    shadow the workspace tier by design). The composer never writes it - its
    Auto is a ``null`` clear - but the API keeps the tri-state expressible.

    Attributes:
        model: The chat tag to pin, or null to unpin. Not validated
            against the inventory here: the read side falls through on a tag
            the server does not hold, so a stored unknown is harmless.
        thinking: The conversation's default thinking mode, or null to clear.
        web_search: The conversation's default web-search mode, or null to clear.
    """

    model: str | None = Field(default=None, min_length=1, max_length=200)
    thinking: SettingMode | None = None
    web_search: SettingMode | None = None


class WorkspaceDefaultsPatch(_SettingsPatch):
    """The body of the workspace-defaults PATCH: the same per-field merge.

    Every field this accepts is one some live read path already consults, and
    that is the rule the page is held to - a stored default nothing reads is
    a control that lies about having done something.

    The model tag's SHAPE is checked here (length) and its grammar in the
    service, through the same parser the management verbs use; its EXISTENCE
    deliberately is not. An operator may set a default ahead of pulling the
    weights, a remote server's inventory changes without this deployment
    hearing about it, and the resolution chain already falls through a tag it
    cannot honour - so refusing here would trade a working fall-through for a
    validation failure against a listing that may be stale by the time the
    turn runs.

    Attributes:
        model: The workspace's default chat tag, or null to clear.
        embedding_model: The tag retrieval embeds with, or null to clear and
            let the deployment answer. Clearing is not the same as never
            having set it only in what it leaves behind: whichever tag wrote
            the stored passages goes on being observable, and the resolution
            falls back to it.
        thinking: The workspace's default thinking mode, or null to clear.
        web_search: The workspace's default web-search mode, or null to clear.
        system_prompt: The persona every turn opens with, or null to clear.
            A blank or whitespace-only value IS a clear, deliberately: the
            environment's ``SYSTEM_PROMPT=""`` kill switch has no workspace
            twin, so the emptied textarea means "fall back", never "silence
            the system prompt".
        context_ceiling: The largest window every turn is held to, or null to
            clear. Bounded at BOTH ends deliberately: never below the room a
            turn needs for its own answer, and never above a plausible window,
            because for a tag nothing declares this number is not a trim on
            somebody else's window - it is the one the turn requests.
        reranker: False to force dense-only retrieval, True to use the
            deployment's sidecar when it has one, null to clear.
    """

    model: str | None = Field(default=None, min_length=1, max_length=200)
    embedding_model: str | None = Field(default=None, min_length=1, max_length=200)
    thinking: SettingMode | None = None
    web_search: SettingMode | None = None
    system_prompt: WorkspacePrompt = Field(default=None, max_length=MAX_SYSTEM_PROMPT_CHARS)
    context_ceiling: int | None = Field(
        default=None, ge=MIN_CONTEXT_CEILING, le=MAX_CONTEXT_CEILING
    )
    reranker: bool | None = None


@dataclass(frozen=True)
class TurnDefaults:
    """The merged conversation-over-workspace defaults one turn plans against.

    Request-excluded on purpose: the request is arbitrated inside the planner
    (an explicit request value always wins), so this value carries only the
    stored rungs. The MODEL axis is deliberately absent: a model tag needs
    checking against the inventory per rung (a conversation tag the server
    does not hold must fall through to the workspace one), so
    ``resolve_chat_models`` takes the two tags separately.
    """

    thinking: SettingMode | None = None
    web_search: SettingMode | None = None


def resolve_turn_defaults(
    conversation: ConversationSettings, workspace: WorkspaceDefaults
) -> TurnDefaults:
    """Merge the stored enum tiers per field: conversation beats workspace.

    Args:
        conversation: The typed view of this conversation's settings.
        workspace: The typed view of the tenant's workspace defaults.

    Returns:
        The per-field merge the planner consumes.
    """
    return TurnDefaults(
        thinking=(
            conversation.thinking if conversation.thinking is not None else workspace.thinking
        ),
        web_search=(
            conversation.web_search if conversation.web_search is not None else workspace.web_search
        ),
    )
