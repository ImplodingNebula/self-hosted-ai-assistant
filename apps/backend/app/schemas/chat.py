"""HTTP/SSE schemas for the chat API - the frozen wire contract.

The chat endpoint streams Server-Sent Events. Each event has a name and a JSON
``data`` payload::

    event: meta         data: {"conversation_id"}                 # first frame: names the thread
    event: token        data: {"delta": "..."}                    # assistant text chunk
    event: thinking     data: {"delta": "..."}                    # reasoning text chunk
    event: status       data: {"message","code"}                  # progress note (e.g. model loading)
    event: citation     data: {"id","source","snippet","score",   # source backing this turn
                               "index","title","doc_id","source_type","url"}
    event: tool_call    data: {"id","name","arguments"}           # model asked for a tool (Phase 5)
    event: tool_result  data: {"id","name","result","is_error",   # tool output (Phase 5); count
                               "denied","count"}                  #   is null when it cannot say
    event: tool_confirmation data: {"id","conversation_id",       # side-effecting call needs the
                               "calls":[{"id","name","arguments"}]} #   user's OK (Phase 5); terminal
    event: error        data: {"message","code"}
    event: done         data: {"message_id","conversation_id","usage":{...},
                               "context_usage":{"prompt_tokens","num_ctx","model"},
                               "turn_usage":{"completion_tokens","generation_ms"}}

The ``done`` payload carries the persisted assistant message id and the
conversation id (new or continued) so clients can address the thread on the
next turn. ``context_usage`` reports how full the model's context window was
on the prompt this turn actually sent - the platform's own estimate of the
prompt, over the window it requested - so a client can show a fill meter and
say when a conversation is running out of room. It is null when the window is
unknown. ``turn_usage`` is what the reply took to write, summed over every
round of the turn (``usage`` is the last round's report alone): the tokens
written and the runtime's own writing time for them, the latter null when the
runtime measured none. The same figures are stored with the reply, so a client
that shows them live shows the same ones after a reload.

``meta`` carries that same conversation id as the FIRST frame of
every stream: a turn that never reaches ``done`` - an interrupted reply, a
dropped connection - must still tell its client which conversation it belongs
to, or the client cannot continue the thread it just started. Comment frames
starting with ":" (e.g. ": keep-alive") are idle heartbeats; SSE clients
ignore them by spec.

``citation`` events stream AFTER the answer tokens and before ``done``.
Retrieved passages are offered to the model up front (behind the
``searching_web``/``retrieving`` status notes), but only the sources the
finished answer references inline by ``index`` marker ([1], [2], ...) are
emitted - and persisted - as citations, so an answer that ignored the offered
context carries no sources. Used sources are renumbered 1..k in order of
first reference (the streamed markers carry the retrieval's offer numbering,
so the first visible source could otherwise be "[4]"); when that rewrite
changed the text, ``done`` carries the final answer in ``content`` and
clients should replace their streamed copy with it. ``url`` is set for web
sources so the client can link to them, and null for local records. Clients
must not assume citation position beyond "before ``done``".

Error ``code`` values: ``provider_unavailable`` (model server unreachable or
failing), ``provider_unauthorized`` (something in front of the model server
refused the request's credentials - a deployment setting, not anything a client
sent, and the message names it), ``model_not_found`` (requested model is not
pulled), ``no_chat_model_configured`` (the model server was reached and holds
no model this deployment can chat with - a first-run state, and the message
names the fix), ``bad_request`` (the model server rejected the request),
``internal_error`` (unexpected backend failure).

Status ``code`` values: ``loading_model`` (the target model is being
loaded into memory; the first delta may take minutes on small GPUs),
``model_substituted`` (the configured model for this turn is not installed, so
another model the server does hold is answering instead - its configured pair
partner where there is one, then the first
chat-capable model the server listed; the message names both models and how to
make the choice permanent), ``retrieving`` (context is being searched for
grounding before generation; the message names what is actually being read -
the files the turn is about, pages just fetched from the web, stored records,
or some combination - so clients must show it rather than assume a fixed
wording),
``searching_web`` (the web is being searched for fresh context, which is then
ingested and retrieved like any other source), ``web_search_unavailable`` (a
search was announced but the search backend could not be reached),
``web_search_no_results`` (the search ran but nothing citeable came back),
``attachment_processing`` (an attached file named by this turn is still being
parsed; the turn is waiting for it), ``attachment_unavailable`` (an attached file
named by this turn could not be used - it failed to parse, it was still
processing when the wait ran out, or it did not fit the model's context window -
and the answer therefore does not draw on it), and ``image_not_seen`` (the turn draws on
an image, but only on the text OCR read out of it, because the picture itself
could not be shown to the model - so an answer about a photograph carrying no
text has nothing to work from). That last one names the images it applies to and
why, and fires ONLY for pictures that were not delivered: a turn showing one
image and skipping another names just the skipped one, and a turn whose pictures
all reached the model does not emit it at all.

Tool events (Phase 5, emitted only for a tool-driving turn): ``tool_running`` (a
tool the model called is executing), ``tool_error`` (a tool the model called
could not run, so the answer is missing what it would have supplied),
``tool_limit_reached`` (the turn's tool budget is spent and the next reply is
final), and ``tools_not_used`` (an agentic turn that LOOKED like it wanted
current or external information was offered tools and answered without calling
one, so the reply is the model's own knowledge, not a live search - anything
time-sensitive in it may be out of date; it is freshness-gated, so an ordinary
creative or reasoning turn carries no such notice).

There is also a ``tool_declined`` standing status: the user refused a
side-effecting action (see ``tool_confirmation`` below), so the answer was
written without it - kept beside the answer for the same reason as the others.

``vectors_stranded`` is the last standing status: stored knowledge exists, but
it was written by a different embedding model than the one now in use, so no
query can reach it and the reply is ungrounded. It names both models. Emitted
only once per turn, and only after a knowledge-base lookup has ALREADY come
back empty - either the automatic pre-fetch or, on a turn where the model
drives its own retrieval, a knowledge search that matched nothing. That is the
moment the distinction between "nothing was on file about that" and
"everything on file is unreadable" is what a reader needs.

``model_substituted``, ``web_search_unavailable``, ``web_search_no_results``,
``attachment_unavailable``, ``image_not_seen``, ``tool_error``,
``tool_limit_reached``, ``tools_not_used``, ``tool_declined`` and
``vectors_stranded`` are STANDING notices: each describes the answer that was
given, so a client keeps it beside that answer rather than clearing it when the
next status arrives. The rest (including ``tool_running``) are transient
progress notes.

The ``tool_confirmation`` event (Phase 5) is the one place a turn does not end
in ``done`` or ``error``: when the model asks for a side-effecting tool, the
turn SUSPENDS and this event carries the pending call(s) for the user to
approve or deny. The stream then ends. The client resumes the SAME turn with a
second request that sets ``ChatRequest.confirmation`` to the decision and the
event's ``id``; an approved turn runs the tool and streams its answer, a denied
one continues without it. Only a client that advertises ``supports_confirmation``
is offered side-effecting tools, so a client that ignores this event is never
suspended by one.

``tool_call`` (the model asked for a tool, with its arguments) and
``tool_result`` (a short summary of what the tool produced; the full result
goes to the model, not the wire) bracket each tool a turn runs, so the client
can show a tool card of what happened during the wait. They are informational -
the answer and its citations still arrive as ``token`` and ``citation`` events -
so a client may ignore them.

A request carries one of FOUR shapes, and the validator on
:class:`ChatRequest` is what keeps them apart:

- a NEW turn (words, files, or both, optionally replying to a ``quote``);
- a confirmation RESUME (``confirmation``, no new words);
- a REGENERATE (``regenerate``, naming a ``conversation_id`` and optionally the
  reply to redo), which answers a stored question again in place of the reply it
  already produced;
- an EDIT (``edit``, naming a stored user message and its new wording), which
  rewrites that question and answers the rewritten one.

The last two stream exactly like a new turn - same events, same order, ``meta``
still first - because each IS a turn, asked again.

The endpoint emits every event above.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.providers.base import Usage


class HistoryTurn(BaseModel):
    """One prior turn of the client-held transcript."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ConfirmationDecision(BaseModel):
    """The user's answer to a side-effecting tool's confirmation (Phase 5).

    Sent to resume a turn that suspended on a ``tool_confirmation`` event.
    ``id`` must match the id that event carried - it is how the backend finds
    the checkpointed turn and guards against a stale or replayed decision.
    ``decision`` either authorizes the pending action(s) (``approve``) or
    refuses them (``deny``), after which the turn continues either way: an
    approved turn runs the tool and answers from its result, a denied one tells
    the model the action was refused and lets it answer without it.
    """

    id: str = Field(min_length=1, max_length=200)
    decision: Literal["approve", "deny"]


# The longest quoted passage a turn may reply to. A quote is a pointer at
# something already on screen, not a second message: past a couple of paragraphs
# it stops being a reference and starts being the turn's whole context, which is
# what the conversation history is for. Bounded here rather than truncated
# because a silently shortened quote would attribute words to a passage that does
# not contain them.
QUOTE_MAX_CHARS = 2000


class MessageEdit(BaseModel):
    """New wording for a question already in the transcript.

    ``message`` replaces the stored text outright - the old wording is not kept
    anywhere - and the turn is then answered from the rewritten question. It
    carries the same constraints a typed message does, because that is what it
    becomes.
    """

    message_id: UUID
    message: str = Field(max_length=32000)


class QuoteRef(BaseModel):
    """A passage the user is replying to, carried on an ordinary turn.

    ``text`` is AUTHORITATIVE: it is what reaches the model and what is stored
    with the message, so quoting a passage still works when the message it came
    from has since been edited away or deleted. ``message_id`` is provenance
    only - it is never resolved against the database, and a client may omit it
    (a quote of a turn that has no server id yet).
    """

    message_id: UUID | None = None
    text: str = Field(min_length=1, max_length=QUOTE_MAX_CHARS)

    @model_validator(mode="after")
    def _quotes_something(self) -> QuoteRef:
        """Reject a quote of nothing but whitespace.

        The length bound alone lets a single space through, which would fence an
        empty block into the prompt and render a blank chip - a turn claiming to
        reply to a passage that says nothing. Not stripped, because the passage
        is stored verbatim and a quoted code block's indentation is part of it.
        """
        if not self.text.strip():
            raise ValueError("a quote must contain something")
        return self


class ChatRequest(BaseModel):
    """A chat turn request from the client.

    ``conversation_id`` continues an existing conversation, whose context is
    rebuilt server-side from persisted messages; unset starts a new one.
    ``history`` is the legacy client-held transcript, kept for back-compat: it
    is used as pre-context only when the request starts a NEW conversation and
    is never persisted. It stays unbounded at the schema layer - the chat
    service truncates to the newest turns, so a long transcript degrades
    gracefully instead of being rejected with a validation error.

    ``attachment_ids`` names the uploaded files THIS turn is about: the turn
    waits for them to finish parsing, draws on them even when the question shares
    no words with their contents ("summarise this"), and says so plainly when one
    could not be used. It is not the only way a file is reachable - an ingested
    file stays in the conversation's retrieval scope, and a later message that
    names it or points at it is read the same way (see
    ``app.services.file_reference``).

    ``message`` may be empty, but only on a turn that carries files, is a
    confirmation resume, is a regenerate, or is an edit. Attaching a file and
    pressing send with nothing typed is a real request - the natural answer to
    "attach the logs and I will take a look" - and the files then ARE the
    message; a resume carries a decision, not words, a regenerate re-asks a
    question the conversation already holds, and an edit carries its words in
    ``edit.message`` instead. A request carrying none of those fields is not a
    request at all.
    """

    message: str = Field(default="", max_length=32000)
    history: list[HistoryTurn] = Field(default_factory=list)
    model: str | None = None
    thinking: Literal["auto", "on", "off"] = "auto"
    # Local-first web search: "auto" consults the web only on freshness cues,
    # "on" always, "off" never (fully local). Only ever fires when the SearXNG
    # sidecar is configured; same shape as ``thinking``.
    web_search: Literal["auto", "on", "off"] = "auto"
    # How much the model may drive this turn (Phase 5). A DEVELOPER/TESTING
    # affordance only - the composer must NOT surface it (how much the model
    # drives is an internal routing decision, unlike the user-legible thinking
    # and web-search switches). It can only clamp DOWN from the model's ceiling
    # and the AGENCY_MODE deployment setting, never raise either; "auto" (the
    # default) adds no clamp of its own.
    agency: Literal["auto", "agentic", "assisted", "deterministic"] = "auto"
    # Validated by the chat service (400 malformed / 404 unknown), not here.
    conversation_id: str | None = None
    # The attachments this turn is about. Bounded so one request cannot ask the
    # store for an unbounded set of sources; validated by the chat service
    # (400 malformed, 404 unknown / not this conversation's), not here.
    attachment_ids: list[str] = Field(default_factory=list, max_length=20)
    # Whether this client understands the confirmation round-trip (Phase 5). Only
    # a client that sets this is offered side-effecting tools at all: a tool that
    # needs confirmation is useless - worse, it silently stalls the turn - to a
    # client that cannot resume it, so the default keeps such tools withheld
    # exactly as before. The Phase 6 composer sets it; the developer/testing
    # affordance and the tests set it explicitly.
    supports_confirmation: bool = False
    # Set to RESUME a turn suspended on a side-effecting tool's confirmation,
    # rather than to start a new one. When present the request carries no new
    # words - ``message`` must be empty and ``conversation_id`` must name the
    # suspended conversation (both enforced below / by the chat service).
    confirmation: ConfirmationDecision | None = None
    # Set to ANSWER A STORED QUESTION AGAIN instead of asking a new one: the
    # reply named by ``message_id`` (or, unset, the conversation's trailing one)
    # is discarded and rewritten from the question that already produced it. It
    # carries nothing of its own - no words, no files, no decision - because
    # everything it needs is already stored; it names only the conversation whose
    # tail it rewrites, and optionally which reply inside it. Enforced below.
    regenerate: bool = False
    # WHICH reply a regenerate redoes. Unset means the newest one, which is the
    # only thing a regenerate could ever mean before this existed. Naming an
    # older reply also discards everything below it - the thread cannot keep
    # answers that were written after a turn that no longer exists in the form
    # they replied to. Meaningful ONLY on a regenerate (enforced below); resolved
    # by the chat service (404 unknown / 400 not a reply), not here.
    message_id: UUID | None = None
    # Set to REWRITE A QUESTION ALREADY ASKED and answer the new wording. The
    # named message must be one of this user's own ``user`` turns; its text is
    # replaced outright and everything after it in the thread is removed, because
    # those turns answered a question that no longer exists. Exclusive with every
    # other shape, and it must name its conversation. Enforced below.
    edit: MessageEdit | None = None
    # A passage the user is replying to, carried alongside an ordinary message.
    # Not a shape of its own - it decorates the NEW-turn shape and nothing else -
    # so it is refused beside a regenerate, an edit or a confirmation, each of
    # which carries no new words for it to attach to.
    quote: QuoteRef | None = None

    @model_validator(mode="after")
    def _has_something_to_answer(self) -> ChatRequest:
        """Keep the four request shapes apart, and refuse anything in between.

        The length check the field used to carry could not express this: whether
        an empty message is valid depends on other fields entirely.

        One rule runs through all four. A field the backend would silently ignore
        is worse than one it refuses, because the client has no way to learn its
        words were dropped - so every shape rejects what it cannot use rather than
        quietly discarding it.

        - A RESUME (``confirmation``) answers a question the backend asked, so it
          must carry no new words: an approve/deny cannot smuggle in a fresh turn
          that skips the tool floors.
        - A REGENERATE re-answers a stored question, so everything it would need
          is already stored; it must name its conversation, because it cannot be
          the first thing a conversation hears.
        - An EDIT supplies the one thing that is NOT already stored - the new
          wording - and nothing else; the files, the history and the ordering all
          come off the row it rewrites. It names its conversation for the same
          reason a regenerate does.
        - ``message_id`` is the regenerate's one modifier, so it is refused
          anywhere else rather than being read as a second way to address a turn.
        - ``quote`` decorates a NEW turn only: the other three carry no fresh
          words for a quotation to be part of.

        The two rewinding shapes reject ANY ``message``, not merely a non-blank
        one: a body of spaces is still words the sender wrote and the backend
        would still throw away, which is the thing this rule exists to stop.
        """
        if self.edit is not None:
            if self.regenerate:
                raise ValueError("an edit and a regenerate are different turns; send one")
            if self.conversation_id is None:
                raise ValueError("an edit must name the conversation it rewrites")
            if self.message:
                raise ValueError("an edit carries its new wording in edit.message")
            if self.attachment_ids:
                raise ValueError(
                    "an edit must not carry attachments; it keeps the files the question had"
                )
            if self.confirmation is not None:
                raise ValueError("an edit must not carry a confirmation decision")
            if self.message_id is not None:
                raise ValueError("an edit names its message in edit.message_id")
            if self.quote is not None:
                raise ValueError("an edit must not carry a quote")
            if not self.edit.message.strip():
                raise ValueError("an edit must not blank the question it rewrites")
            return self
        if self.regenerate:
            if self.conversation_id is None:
                raise ValueError("a regenerate must name the conversation it re-answers")
            if self.message:
                raise ValueError("a regenerate must not carry a new message")
            if self.attachment_ids:
                raise ValueError("a regenerate must not carry attachments")
            if self.confirmation is not None:
                raise ValueError("a regenerate must not carry a confirmation decision")
            if self.quote is not None:
                raise ValueError("a regenerate must not carry a quote")
            return self
        if self.message_id is not None:
            raise ValueError("message_id names the reply a regenerate redoes, and nothing else")
        if self.confirmation is not None:
            if self.message.strip() or self.attachment_ids:
                raise ValueError(
                    "a confirmation resume must not carry a new message or attachments"
                )
            if self.quote is not None:
                raise ValueError("a confirmation resume must not carry a quote")
            return self
        if not self.message.strip() and not self.attachment_ids:
            raise ValueError("message must not be empty unless the turn attaches a file")
        return self


class MetaEvent(BaseModel):
    """`meta` event payload: which conversation this stream belongs to.

    The first frame of every stream, before any token, so a client that never
    reaches `done` - an interrupted turn, a dropped connection - can still
    bind its pane to the conversation the backend created or resolved.

    ``user_message_id`` names the QUESTION this stream is answering: the row a
    new turn just persisted, or the stored one a regenerate or an edit is
    answering again. It lets a client key its optimistic bubbles on the server's
    own ids from the first frame, rather than on their position in the
    transcript. It is optional and may be absent - an older backend does not send
    it at all, and a resume whose checkpoint predates it names no question - so a
    client must stay tolerant of that rather than depend on it.
    """

    conversation_id: str
    user_message_id: str | None = None


class TokenEvent(BaseModel):
    """`token` event payload: an incremental chunk of assistant text."""

    delta: str


class ThinkingEvent(BaseModel):
    """`thinking` event payload: an incremental chunk of reasoning text."""

    delta: str


class StatusEvent(BaseModel):
    """`status` event payload: a human-safe progress note for long waits."""

    message: str
    code: str = "working"


class ToolCallEvent(BaseModel):
    """`tool_call` event payload (Phase 5)."""

    id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class ToolResultEvent(BaseModel):
    """`tool_result` event payload (Phase 5).

    ``denied`` marks a side-effecting call the user refused at the confirmation
    prompt: it ran nothing (``result`` is empty), so the client badges its trace
    card "denied" rather than showing a result.

    ``count`` is how many sources, records or results the tool produced, when
    the tool can say; null when it cannot (a plain-text result, an error, a
    denied call), so the client shows nothing rather than a guess.
    """

    id: str
    name: str
    result: str
    is_error: bool = False
    denied: bool = False
    count: int | None = None


class ToolConfirmationEvent(BaseModel):
    """`tool_confirmation` event payload (Phase 5): the turn is suspended.

    Emitted when the model asked for a side-effecting tool and the turn is
    waiting for the user's OK. It is a TERMINAL event - the stream ends after
    it, with no ``done`` - and the turn resumes only when the client sends the
    decision back (``ChatRequest.confirmation``) carrying this ``id``.
    ``calls`` are the pending actions, each a tool ``name`` and its
    ``arguments``, so the client can show exactly what will run.
    """

    id: str
    conversation_id: str
    calls: list[ToolCallEvent] = Field(default_factory=list)


class FileEvent(BaseModel):
    """`file` event payload: a file one of this turn's tools produced.

    Emitted once per generated file, AFTER its attachment row is committed - so
    an announced file always names a row that is really there. It carries
    metadata only: the bytes went to object storage and never travel the stream.

    Announce-only for now: no route serves an attachment's bytes yet, so a
    client can show that a file was produced and remember its
    ``attachment_id``, but cannot download it until that route lands.

    ``origin`` is always ``generated`` today. It is sent explicitly rather than
    implied by the event name so a client keys on the same field the REST
    attachment shapes carry, and can tell a file the assistant made from one a
    person uploaded without tracking which event delivered it.
    """

    attachment_id: str
    filename: str
    content_type: str
    size_bytes: int
    origin: str


class CitationEvent(BaseModel):
    """`citation` event payload: one retrieved source backing this turn.

    The first four fields are the frozen Phase-1 shape; the rest arrived with
    the Phase-3 implementation, additively. ``index`` matches the [n] markers
    the grounded answer may use inline.
    """

    id: str
    source: str
    snippet: str
    score: float | None = None
    index: int | None = None
    title: str | None = None
    doc_id: str | None = None
    source_type: str | None = None
    url: str | None = None


class ErrorEvent(BaseModel):
    """`error` event payload."""

    message: str
    code: str = "internal_error"


class ContextUsageEvent(BaseModel):
    """How full the context window was on the prompt this turn sent.

    Both numbers come from the backend's own accounting, not the model
    server's: ``prompt_tokens`` is the same estimate the turn's budgets were
    computed against, and ``num_ctx`` is the window the turn requested. A
    client can divide one by the other for a fill meter, but should treat the
    result as an estimate - it is a character-based approximation, not a
    tokenizer's count. ``model`` names the tag the turn actually ran
    (substitutions included), so a client can tell whose window the reading
    describes; None from an older service layer.
    """

    prompt_tokens: int
    num_ctx: int
    model: str | None = None


class TurnUsageEvent(BaseModel):
    """What the reply took to write, over every round of its turn.

    ``completion_tokens`` counts everything the model wrote this turn -
    reasoning and tool-call rounds included. ``generation_ms`` is the
    runtime's own writing time for those tokens (a tool's run time, a wait on
    a confirmation and a model load are not in it), so the two divide into a
    decode speed; it is null when the runtime reported no timing, and a
    client then has a count and no speed.
    """

    completion_tokens: int
    generation_ms: int | None = None


class DoneEvent(BaseModel):
    """`done` event payload: the stream completed and the turn is persisted.

    ``content`` is the final persisted answer, set ONLY when it differs from
    the streamed text (inline citation markers renumbered to match the
    emitted citations); clients should replace their streamed copy with it.
    ``context_usage`` reports how full the model's window was on this turn's
    prompt, and is null when the window is unknown. ``turn_usage`` is the
    turn's own total, null when the provider reported no count; ``usage`` stays
    the provider's report for the final round alone.
    """

    message_id: str
    conversation_id: str | None = None
    usage: Usage | None = None
    content: str | None = None
    context_usage: ContextUsageEvent | None = None
    turn_usage: TurnUsageEvent | None = None
