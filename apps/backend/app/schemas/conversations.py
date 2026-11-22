"""HTTP schemas for the conversations REST API - the frozen wire contract.

DTOs are deliberately separate from the DB models: UUIDs go out as strings,
and server-internal columns (``tenant_id``, ``user_id``) never leave the
backend. ``meta`` is server-internal too, save for a handful of keys deliberately
re-exposed through dedicated DTOs so a reloaded transcript matches what the live
stream showed: ``citations`` (its sources), ``attachments`` (the files a turn
carried), ``quote`` (the passage a question was replying to), ``notices`` (the
standing notes qualifying a reply), ``tool_steps`` (the tool-activity trace),
``usage`` (what a finished reply took to write),
and the ``interrupted`` flag (a reply cut short by a disconnect or provider
error mid-stream). The ``from_*`` constructors are the single mapping point
between the two shapes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.schemas.attachments import AttachmentOut
from app.schemas.common import UtcDatetime
from app.schemas.settings import ConversationSettingsPatch

if TYPE_CHECKING:
    from app.models import Attachment, Conversation, Message

logger = logging.getLogger(__name__)


class CitationOut(BaseModel):
    """One stored citation, as persisted in the message's metadata.

    Mirrors the SSE ``citation`` event shape so a reloaded transcript renders
    sources exactly like the live stream did.
    """

    id: str
    index: int
    source: str
    snippet: str
    score: float | None = None
    title: str | None = None
    doc_id: str | None = None
    source_type: str | None = None
    url: str | None = None


class MessageAttachmentOut(BaseModel):
    """One file a turn carried or produced, as recorded on its message.

    A reference, not the live attachment row: the composer's tray is cleared once
    a message is sent, so this is what tells a reloaded transcript which files the
    question arrived with. It survives the attachment's own deletion by design -
    the turn really did carry that file, and a transcript that quietly forgot it
    would misrepresent what the answer was given.

    An ASSISTANT message carries the same shape for a file one of its tools
    generated, which is what ``origin`` distinguishes: ``upload``/``paste`` came
    from a person, ``generated`` came from a tool the assistant ran. It defaults
    to ``upload`` so every snapshot written before the field existed - all of
    them user turns - still parses and reads correctly.
    """

    id: str
    filename: str
    content_type: str
    size_bytes: int
    origin: str = "upload"


class MessageQuoteOut(BaseModel):
    """The passage a question was replying to, as recorded on it.

    ``text`` is what the user quoted, stored verbatim: it survives the message it
    came from being edited or deleted, which is why the chip can still be
    rendered afterwards. ``message_id`` is provenance - where the passage was
    taken from - and may be null, or may name a row that no longer exists, so a
    client must not depend on resolving it.
    """

    message_id: str | None = None
    text: str


class NoticeOut(BaseModel):
    """One standing notice explaining the answer it is stored with.

    Mirrors the SSE ``status`` event shape for the standing codes
    (``model_substituted``, ``web_search_unavailable``, ``web_search_no_results``,
    ``attachment_unavailable``), so a reloaded transcript still says why a reply
    came from a substitute model, or ignored a file it was handed.
    """

    code: str
    message: str


class ToolStepOut(BaseModel):
    """One tool the assistant ran this turn, as stored on the message.

    Mirrors the ``tool_call`` + ``tool_result`` SSE pair the client renders live,
    so a reloaded transcript can replay the same activity trace: the call and its
    arguments, a short display summary of what came back, whether it errored, and
    any narration the model streamed before it. ``result`` is null when the tool
    never returned one (the turn was cut off mid-run, or the call was declined).
    ``approved``/``denied`` mark a side-effecting call the user authorized or
    refused at the confirmation prompt, so the reloaded card carries the same badge
    the live view showed; both are false for an ordinary (read-only) tool call.
    """

    id: str
    name: str
    arguments: dict = Field(default_factory=dict)
    narration: str | None = None
    result: str | None = None
    is_error: bool = False
    approved: bool = False
    denied: bool = False
    count: int | None = None


def _citations_from_meta(meta: dict) -> list[CitationOut]:
    """Leniently parse stored citations; malformed entries are dropped.

    Rows written by other code versions must never 500 a transcript read -
    a lost citation is cosmetic, a broken conversation reload is not.
    """
    citations: list[CitationOut] = []
    for entry in meta.get("citations") or []:
        try:
            citations.append(CitationOut.model_validate(entry))
        except ValidationError:
            logger.warning("Dropping malformed stored citation: %r", entry)
    return citations


def _attachments_from_meta(meta: dict) -> list[MessageAttachmentOut]:
    """Leniently parse the files stored on a message; malformed entries are dropped."""
    attachments: list[MessageAttachmentOut] = []
    for entry in meta.get("attachments") or []:
        try:
            attachments.append(MessageAttachmentOut.model_validate(entry))
        except ValidationError:
            logger.warning("Dropping malformed stored attachment reference: %r", entry)
    return attachments


def _quote_from_meta(meta: dict) -> MessageQuoteOut | None:
    """Leniently parse the quote stored on a message; a malformed one is dropped.

    Same rule as every other stored-meta reader here: a blob written by another
    code version costs a chip, never the whole transcript.
    """
    entry = meta.get("quote")
    if entry is None:
        return None
    try:
        return MessageQuoteOut.model_validate(entry)
    except ValidationError:
        logger.warning("Dropping malformed stored quote: %r", entry)
        return None


def _notices_from_meta(meta: dict) -> list[NoticeOut]:
    """Leniently parse the notices stored on a message; malformed entries are dropped."""
    notices: list[NoticeOut] = []
    for entry in meta.get("notices") or []:
        try:
            notices.append(NoticeOut.model_validate(entry))
        except ValidationError:
            logger.warning("Dropping malformed stored notice: %r", entry)
    return notices


def _tool_steps_from_meta(meta: dict) -> list[ToolStepOut]:
    """Leniently parse the tool trace stored on a message; malformed entries drop.

    A lost step is cosmetic (the answer and its citations still stand); a broken
    conversation reload is not - so a step written by another code version must
    never 500 a transcript read.
    """
    steps: list[ToolStepOut] = []
    for entry in meta.get("tool_steps") or []:
        try:
            steps.append(ToolStepOut.model_validate(entry))
        except ValidationError:
            logger.warning("Dropping malformed stored tool step: %r", entry)
    return steps


class MessageUsageOut(BaseModel):
    """What a stored reply took to write: the figures its live stream closed on.

    ``completion_tokens`` covers every round of the turn; ``generation_ms`` is
    the runtime's own writing time for them, null when it measured none. The
    same object the ``done`` event carried as ``turn_usage``.
    """

    completion_tokens: int = Field(gt=0)
    generation_ms: int | None = Field(default=None, gt=0)


def _usage_from_meta(meta: dict) -> MessageUsageOut | None:
    """Leniently parse the stored figures; anything malformed reads as none.

    Same rule as every other stored-meta reader here. The footer simply shows
    no figures for such a reply, which is also what every reply saved before
    the key existed shows.
    """
    entry = meta.get("usage")
    if entry is None:
        return None
    try:
        return MessageUsageOut.model_validate(entry)
    except ValidationError:
        logger.warning("Dropping malformed stored usage: %r", entry)
        return None


class MessageOut(BaseModel):
    """One persisted chat message as exposed to clients."""

    id: str
    role: str
    content: str
    thinking: str | None = None
    model: str | None = None
    citations: list[CitationOut] = Field(default_factory=list)
    # Files this turn carried; user turns only. Recorded on the message because
    # the composer's tray is emptied the moment it is sent.
    attachments: list[MessageAttachmentOut] = Field(default_factory=list)
    # The passage this question was replying to; user turns only, null when the
    # turn quoted nothing.
    quote: MessageQuoteOut | None = None
    # Standing notices qualifying this reply (a substituted model, a fruitless
    # search, a file that could not be used); assistant turns only.
    notices: list[NoticeOut] = Field(default_factory=list)
    # The tools this turn ran, in order, with any narration - the activity trace,
    # so a reloaded transcript replays what the live stream showed; assistant only.
    tool_steps: list[ToolStepOut] = Field(default_factory=list)
    # True when the reply was cut short (client disconnect or a provider error
    # mid-stream), so a reloaded transcript can badge it as incomplete.
    interrupted: bool = False
    # What the reply took to write (tokens, the runtime's writing time);
    # assistant turns only, null on a cut-short reply, on one whose provider
    # reported no count, and on every reply saved before this was recorded.
    usage: MessageUsageOut | None = None
    created_at: UtcDatetime

    @classmethod
    def from_entity(cls, message: Message) -> MessageOut:
        """Map a ``Message`` row onto the wire shape (UUID id becomes str)."""
        meta = message.meta or {}
        return cls(
            id=str(message.id),
            role=message.role,
            content=message.content,
            thinking=message.thinking,
            model=message.model,
            citations=_citations_from_meta(meta),
            attachments=_attachments_from_meta(meta),
            quote=_quote_from_meta(meta),
            notices=_notices_from_meta(meta),
            tool_steps=_tool_steps_from_meta(meta),
            interrupted=bool(meta.get("interrupted")),
            usage=_usage_from_meta(meta),
            created_at=message.created_at,
        )


class ConversationSummary(BaseModel):
    """Sidebar-sized view of a conversation: no settings, no messages."""

    id: str
    title: str | None = None
    # When the thread was pinned, or null when it is not. The sidebar groups on
    # its presence and orders the group by its value, newest pin first.
    pinned_at: UtcDatetime | None = None
    created_at: UtcDatetime
    updated_at: UtcDatetime

    @classmethod
    def from_entity(cls, conversation: Conversation) -> ConversationSummary:
        """Map a ``Conversation`` row onto the summary wire shape."""
        return cls(
            id=str(conversation.id),
            title=conversation.title,
            pinned_at=conversation.pinned_at,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )


class ConversationListResponse(BaseModel):
    """One page of conversation summaries plus the tenant-wide total."""

    items: list[ConversationSummary]
    total: int


class ConversationDetail(BaseModel):
    """Full conversation view: summary fields, settings, messages, and its files."""

    id: str
    title: str | None = None
    created_at: UtcDatetime
    updated_at: UtcDatetime
    settings: dict
    messages: list[MessageOut]
    # Every file the conversation holds, sent or not. A client restores the ones
    # no message carries into its composer: they were uploaded and ingested but
    # never sent, and the server still has them, so a reopened conversation
    # should show them rather than pretend they were lost.
    attachments: list[AttachmentOut] = Field(default_factory=list)

    @classmethod
    def from_entities(
        cls,
        conversation: Conversation,
        messages: list[Message],
        attachments: list[Attachment] | None = None,
    ) -> ConversationDetail:
        """Build the detail view from a conversation, its messages, and its files."""
        return cls(
            id=str(conversation.id),
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            settings=conversation.settings,
            messages=[MessageOut.from_entity(m) for m in messages],
            attachments=[AttachmentOut.from_entity(a) for a in attachments or []],
        )


class ConversationUpdate(BaseModel):
    """PATCH body - only fields actually present in the request are applied.

    ``settings`` merges PER FIELD rather than replacing the stored JSON: a
    field absent from the object is left as stored, an explicit ``null``
    clears it, and an unknown key is rejected (422). A control therefore
    writes only its own axis, and the server locks the row for the
    read-merge-write, so saves cannot clobber each other the way a wholesale
    replace silently did. Unknown top-level keys are rejected too - a typo'd
    field name must fail loudly, not no-op. Top-level explicit ``null``
    values are ignored rather than treated as clears - clearing a title is
    not part of the v1 contract.
    """

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=200)
    # Intent, never a timestamp: the client says pinned or not and the server
    # stamps the clock. A client-supplied time would let a skewed (or hostile)
    # clock decide where the row sits in the pinned group.
    pinned: bool | None = None
    settings: ConversationSettingsPatch | None = None


class ConversationContextEstimate(BaseModel):
    """`GET /conversations/{id}/context`: the window's fill before any turn runs.

    The pre-turn sibling of the SSE ``done`` frame's ``context_usage``: what
    the next prompt already opens with (the system prompt plus the capped
    history replay), over the window the named model would be granted. The
    same character-based arithmetic as the turn's own reading, and hedged the
    same way - an estimate, not a tokenizer's count. ``model`` names the tag
    the reading describes, echoed back so a client that switched picks while
    the request was in flight can discard a stale answer.
    """

    prompt_tokens: int
    num_ctx: int
    model: str
