"""The tool seam: what a tool is, what it produces, and where tools come from.

Routers and services depend only on the value types and ABCs in this module,
never on a concrete tool or transport. A tool that lives in this process
(web search, knowledge search) and a tool that lives behind an MCP server both
implement :class:`Tool`; a :class:`ToolSource` is anything that offers tools -
in-process Python or a remote MCP server alike. The concrete
:class:`~app.services.tools.registry.ToolRegistry` composes several sources into
one catalog. This is what keeps "MCP is a source of tools, not the architecture"
true in code. See docs/ARCHITECTURE.md section 7.

This module holds value types and ABCs ONLY - no I/O, no registry, no concrete
tool - so it can be imported anywhere without pulling in a transport.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field


class ToolError(Exception):
    """Base class for seam-level tool failures.

    Seam-level type: the loop and the registry raise or catch this instead of a
    transport-specific exception (httpx, an MCP client error), keeping the
    boundary source-agnostic.
    """


class ToolNotFoundError(ToolError):
    """Raised when a tool is invoked by a name no source offers."""

    def __init__(self, name: str) -> None:
        """Record the missing tool name and build a readable message."""
        super().__init__(f"No tool named {name!r} is registered")
        self.name = name


class ToolContext(BaseModel):
    """The caller's identity and turn, injected into every tool call.

    This is the ONLY place a tool learns whose data it may touch. A tool's JSON
    Schema deliberately carries no tenant, user, conversation, or session id, so
    another principal's data is not even expressible in a tool call; the executor
    fills this in from the request's verified context, never from anything the
    model produced. The model cannot see or set these fields.

    Frozen on purpose: a tool must not be able to mutate the identity it was
    handed and then reach data it was not scoped to.

    Attributes:
        tenant_id: The tenant every data access is filtered by (always present).
        user_id: The acting user; per-user credentials and user-scoped data key
            on this. Present from the seeded default user even before login.
        conversation_id: The conversation this turn belongs to, when there is one.
        session_id: The ephemeral retrieval scope for this turn's tool results,
            when there is one.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    user_id: str
    conversation_id: str | None = None
    session_id: str | None = None


class ToolSpec(BaseModel):
    """What is advertised to the model about one tool.

    This is the tool's public face: exactly what rides in the prompt when the
    tool is offered. It is metadata only - it never carries identity (see
    :class:`ToolContext`).

    Everything here is rendered into the prompt, and a chat template puts the
    offered tool list in its SYSTEM region - so for a spec built from an
    UNTRUSTED source (a remote MCP listing), every field of it is prose sitting
    in a more privileged position than a tool RESULT gets. Such a source must
    bound its own values BEFORE constructing a spec (``McpToolSource`` caps the
    description and the schema's size, depth and strings, and drops a tool it
    cannot bound). No validator here does it for them: a cap in this value type
    would also silently reshape the in-process tools' hand-written schemas, and
    a seam that quietly rewrites what a caller declared is worse than one that
    states the contract.

    Attributes:
        name: The unique tool name the model calls by.
        description: A short natural-language description of what the tool does
            and when to use it; the model reads this to decide.
        parameters: The JSON Schema for the call arguments. Kept deliberately
            small - a single well-named parameter is the design rule, because a
            small local model asked to fill several structured arguments fills
            them wrong.
        side_effecting: Whether running the tool changes state in the outside
            world (as opposed to a read-only lookup). Defaults to True so an
            un-annotated tool is treated as dangerous: a side-effecting call
            needs confirmation before it runs, and a missing annotation must
            fail closed rather than silently skip that gate.
    """

    name: str
    description: str = ""
    parameters: dict = Field(default_factory=dict)
    side_effecting: bool = True


class ToolResultSource(BaseModel):
    """One attributable source behind a tool result, for the citation flow.

    A tool that read real material (a fetched page, a stored record) returns
    each piece as a source so the generation loop can number it, show it to the
    model as a citable block, and emit a ``citation`` event if the answer
    references it - the same usage-gated flow retrieved context already has.
    Kept as its own value type (not the RAG seam's ScoredChunk) so the tool
    seam stays self-contained: an MCP tool can return sources without the tool
    contract importing retrieval internals.

    Attributes:
        id: The underlying chunk/record id (citation identity).
        doc_id: The source document id; sources sharing it are one citation.
        text: The source text the model reads (the loop caps its length).
        title: Human-readable label for the block header and citation.
        url: Original location, when the source is addressable (a web page).
        source_type: Where it came from ("web", "records", "upload", ...).
        score: Retrieval similarity, when the tool ranked by one.
    """

    id: str
    doc_id: str
    text: str
    title: str | None = None
    url: str | None = None
    source_type: str = "records"
    score: float = 0.0


class ToolRecord(BaseModel):
    """One structured record a tool returned, before it becomes a citable source.

    A tool result carries records when the server returned ``structuredContent``
    matching the records contract (a ``records`` list of ``{id, text, title?,
    url?}``). The generation loop ingests these into the conversation's session
    store - so a later turn can retrieve them - and cites them this turn. A
    result WITHOUT records is handed to the model as plain ``content`` text, so a
    server that does not speak the contract keeps the old, uncited behaviour.

    Kept a value type of the tool seam (not the RAG ``RagIngestItem``) so a
    provider can surface records without importing service-layer schemas; the
    loop, one layer up, maps these onto the ingest request.

    Attributes:
        id: The record's stable id; its citation identity and ingest doc id.
        text: The record's full text, the material the model reads and cites.
        title: Human-readable label for the citation, when the server gave one.
        url: Original location, when the record is addressable.
    """

    id: str
    text: str
    title: str | None = None
    url: str | None = None


class ToolArtifact(BaseModel):
    """One FILE a tool produced, kept out of the model's text entirely.

    A tool that generates a document (a spreadsheet, a rendered PDF) returns its
    bytes here rather than inside :attr:`ToolResult.content`. The separation is
    the whole point: ``content`` is token-budgeted prompt material, and a
    base64 blob dropped into it would evict the prompt HEAD - the system prompt,
    carrying the tool guidance and the injection defense - to feed the model
    characters it cannot read anyway. The loop stores an artifact as an
    attachment and announces it to the CLIENT; the model sees only whatever
    one-line summary the tool wrote in ``content``.

    Frozen because the loop hands the same instance to storage, to the
    attachment row and to the SSE frame, and those three must describe one file.

    Attributes:
        filename: The name the tool suggested, a LABEL and never a path. It is
            untrusted remote input: the host re-validates it before it reaches a
            row, and substitutes a generated name when it cannot be trusted.
        content_type: The declared media type of :attr:`data`.
        data: The decoded file bytes. Never rendered into any prompt.
    """

    model_config = ConfigDict(frozen=True)

    filename: str
    content_type: str
    data: bytes


class ToolResult(BaseModel):
    """The outcome of running one tool, ready to feed back to the model.

    ``content`` is the text that becomes the tool-role message the next model
    pass reads; the loop fences it as quoted data under the same injection
    defenses retrieved context already gets, because a tool result is
    attacker-influenced input, not a trusted instruction. ``is_error`` marks a
    call that could not complete, so the loop can surface an honest notice
    instead of treating the failure text as an answer.

    ``records`` is set when the tool returned structured records (see
    :class:`ToolRecord`); the loop ingests them into session RAG and turns them
    into ``sources``, so a provider never has to reach the ingest pipeline
    itself. When ``sources`` is non-empty (whether the tool built them or the
    loop did from ``records``) the loop IGNORES ``content`` for the happy path
    and renders the sources itself as numbered, citable blocks - the numbering
    must be global to the turn (ambient grounding may already hold [1]..[n]), and
    only the loop knows the turn's citation state. ``content`` still carries the
    no-result and error texts.

    Attributes:
        content: The result text handed back to the model.
        is_error: True when the call failed; ``content`` then explains why.
        sources: The attributable material behind the result, for citations.
        records: Structured records to ingest and cite; the loop turns these
            into ``sources``. Empty for a tool that returned no structured data.
        artifacts: Files the tool produced (see :class:`ToolArtifact`). The loop
            stores each one and announces it to the client; none of it is ever
            rendered into the prompt. Empty for every in-process tool.
        count: How many sources or results the tool produced, for a tool whose text result
            is a countable list but carries no ``sources``. When unset the loop
            falls back to the number of ``sources``; a tool that cannot count
            leaves both empty and the client shows no count.
    """

    content: str
    is_error: bool = False
    sources: list[ToolResultSource] = Field(default_factory=list)
    records: list[ToolRecord] = Field(default_factory=list)
    artifacts: list[ToolArtifact] = Field(default_factory=list)
    count: int | None = None


class Tool(ABC):
    """One callable tool: its advertised spec, and how to run it.

    A concrete tool exposes an immutable :attr:`spec` (what the model sees) and
    an async :meth:`run` (what actually happens). Implementations are in-process
    Python (web search, knowledge search) or thin adapters over an MCP server;
    the seam is the same either way.
    """

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the tool's advertised specification."""
        raise NotImplementedError

    @abstractmethod
    async def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        """Run the tool against validated ``arguments`` for ``context``.

        Args:
            arguments: The call arguments, matching :attr:`ToolSpec.parameters`.
            context: The caller's identity and turn - the sole source of the
                tenant/user the call may act for.

        Returns:
            The tool's result, ready to feed back to the model.
        """
        raise NotImplementedError


class ToolSource(ABC):
    """A source of tools: in-process Python, or a remote MCP server.

    A source lists the tools it currently offers for a given caller. It takes the
    :class:`ToolContext` because a remote source may offer different tools per
    tenant (an MCP server is registered per tenant); an in-process source ignores
    it. The registry composes several sources into one catalog.
    """

    @abstractmethod
    async def list_tools(self, context: ToolContext) -> list[Tool]:
        """Return the tools this source offers ``context``.

        Args:
            context: The caller's identity and turn.

        Returns:
            The tools available from this source, possibly empty.
        """
        raise NotImplementedError
