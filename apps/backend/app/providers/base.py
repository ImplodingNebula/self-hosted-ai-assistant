"""The model-provider seam: the interface every LLM backend implements.

Routers and services depend only on :class:`ModelProvider` and the shared value
types in this module - never on Ollama, vLLM, or any concrete backend. This is
what makes the platform model-agnostic. See docs/ARCHITECTURE.md section 2.1.

This interface is the frozen Phase-1 contract; add to it deliberately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


class ProviderUnavailableError(Exception):
    """Raised when a provider's backing service cannot be reached.

    Seam-level type: routers catch this instead of any backend-specific
    exception (httpx, gRPC, ...), keeping the boundary backend-agnostic.
    """


class ProviderUnauthorizedError(ProviderUnavailableError):
    """Raised when the backing service refused the request's credentials.

    A SUBCLASS rather than a sibling, deliberately: every existing
    ``except ProviderUnavailableError`` keeps catching it, so introducing it
    changes no handler's behaviour, and an unauthorized model server genuinely
    is one no caller can get an answer out of. A caller that can say something
    more useful than "the server is down" - name the credential setting to fill
    in - narrows to this type instead.
    """


class ModelNotFoundError(Exception):
    """Raised when a provider is asked about weights it does not hold.

    Seam-level like :class:`ProviderUnavailableError`, and deliberately NOT a
    subclass of it: the backend answered perfectly well, it simply has no such
    model. A router that conflated the two would report a healthy model server
    as unavailable, sending someone to check the wrong thing entirely.
    """


class ToolCall(BaseModel):
    """A tool invocation requested by the model.

    ``id`` correlates a call with its result across the loop; the provider mints
    one when the runtime does not supply it. ``arguments`` is the parsed JSON
    object the model passed - normalized to a dict at this seam, never a JSON
    string, so services never re-parse a runtime's dialect.
    """

    id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class ImageContent(BaseModel):
    """Raw image bytes riding a message across the seam.

    Bytes plus a media type, never a runtime's encoding: base64-per-message is
    Ollama's wire dialect and belongs in its provider's mapper (a vLLM provider
    would map the same bytes to OpenAI-style content parts instead). JSON
    serialization is pinned to base64 in both directions so a message
    checkpointed to JSON (the confirmation flow) round-trips pixels intact
    rather than mangling them through utf-8.
    """

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    data: bytes
    media_type: str


class ChatMessage(BaseModel):
    """A single message in a conversation.

    ``name`` labels a tool message with the tool that produced it;
    ``tool_call_id`` correlates a tool result with the call it answers; and
    ``tool_calls`` rides on an assistant message that requested tools, so a
    conversation re-sent to the model in the tool loop carries what the model
    asked for. ``images`` carries raw pixels on a user message,
    populated by the chat service only when the routed model is reported
    vision-capable by a live probe (see ``app.services.vision``). Providers map these to their own wire shape (see
    ``OllamaProvider._message_payload``), never by dumping this seam type raw.
    """

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    images: list[ImageContent] | None = None


class Usage(BaseModel):
    """Token accounting for a completed generation.

    ``completion_ms`` is how long the runtime spent WRITING those completion
    tokens, by its own measurement - decode time only, so neither a model load
    nor the prompt's evaluation is in it. A provider that cannot measure it
    leaves it None rather than guessing from a wall clock; the reply then shows
    its token count and no speed.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    completion_ms: float | None = None


class GenerationOptions(BaseModel):
    """Provider-agnostic generation knobs; providers map them to their native fields.

    ``think`` toggles reasoning on thinking-capable models; ``max_tokens`` caps the
    completion length; ``keep_alive`` hints how long the backend should keep the
    model loaded. ``None`` means "provider default" for every field.
    """

    think: bool | None = None
    temperature: float | None = None
    num_ctx: int | None = None
    max_tokens: int | None = None
    keep_alive: str | int | None = None


class EmbedOptions(BaseModel):
    """Provider-agnostic embedding knobs; providers map them to native fields.

    ``device`` places the embedding model: "auto" leaves it to the provider,
    "cpu" keeps it off the GPU entirely - on tight-VRAM tiers a GPU-resident
    embedder would evict the chat model on every retrieval. ``keep_alive``
    matches :class:`GenerationOptions`.
    """

    device: Literal["auto", "cpu"] = "auto"
    keep_alive: str | int | None = None


class Citation(BaseModel):
    """One retrieved source backing the answer (RAG, from Phase 3).

    ``index`` is the 1-based marker the grounding prompt uses ([1], [2], ...)
    so inline references and the citation list line up. ``snippet`` is a
    short verbatim excerpt of the chunk; ``source`` is the human-readable
    label (title when present, else the document id).
    """

    id: str
    index: int
    source: str
    snippet: str
    score: float | None = None
    title: str | None = None
    doc_id: str | None = None
    source_type: str | None = None
    # Clickable source link for web results (source_type "web"); None for
    # records, whose source is the local title/id.
    url: str | None = None


class ToolCallResult(BaseModel):
    """A finished tool call's result, carried on a ``tool_result`` chunk.

    Service-owned like :class:`Citation`: the provider never produces one. ``id``
    and ``name`` echo the :class:`ToolCall` this answers so the UI can pair a call
    with its result. ``content`` is the result text the service already fenced and
    capped; ``is_error`` marks a call that could not complete; ``denied`` marks a
    side-effecting call the user refused (it ran nothing, so ``content`` is empty).
    """

    id: str
    name: str
    content: str
    is_error: bool = False
    # A side-effecting call the user DECLINED at the confirmation prompt: it ran
    # nothing, so ``content`` is empty. Carried so the client can badge the trace
    # card "denied" (symmetric with an approved run), not treat it as a result.
    denied: bool = False
    # How many sources/records/results the tool produced, when it can say; the
    # tool row's trailing "N sources". None means unknown, never zero.
    count: int | None = None


class StoredFile(BaseModel):
    """One file a tool produced this turn, already stored and addressable.

    Service-owned like :class:`Citation` and :class:`ToolCallResult`: a provider
    never produces one. It is the METADATA of a stored attachment and carries no
    bytes at all - the bytes went to object storage before this existed, and
    keeping them out of the chunk is what stops a generated file from ever
    reaching the token-budgeted prompt, the tool trace, or a log line.

    The same shape is announced on the ``file`` SSE event and recorded on the
    assistant message's stored attachment list, so a live stream and a reloaded
    transcript describe the file identically.

    Attributes:
        attachment_id: The stored attachment's id. It addresses a real row from
            the moment this exists, and is what a download route will serve
            from - though NO route serves an attachment's bytes yet, so this
            slice announces files rather than delivering them.
        filename: The name the file is offered under, already host-validated.
        content_type: Its media type, already host-validated.
        size_bytes: The stored size.
        origin: Always ``generated`` today; carried explicitly so a client keys
            on the value rather than on which event delivered it.
    """

    attachment_id: str
    filename: str
    content_type: str
    size_bytes: int
    origin: str


class ToolConfirmation(BaseModel):
    """A side-effecting tool call (or batch) awaiting the user's approval (Phase 5).

    Service-owned like :class:`Citation` and :class:`ToolCallResult`: the provider
    never produces one. When the model asks for a side-effecting tool, the loop
    does NOT run it - it checkpoints the turn, carries this across the seam on a
    ``tool_confirmation`` chunk, and STOPS. The turn is suspended, not finished:
    the client shows the pending action(s), and resumes the same turn by sending
    the decision back with ``id``. ``calls`` are the side-effecting calls this
    confirmation gates - each one an action the user is being asked to authorize.
    """

    id: str
    calls: list[ToolCall] = Field(default_factory=list)


class ContextUsage(BaseModel):
    """How full the model's context window was on the prompt that just ran.

    Service-owned like :class:`Citation` - a provider never produces one. Both
    numbers are the CHAT SERVICE's own accounting: ``prompt_tokens`` is the
    same estimate the turn's budgets acted on (``approx_tokens`` over the
    final assembled prompt), NOT the runtime's reported count, because runtimes
    that cache prompt prefixes report only the tokens they evaluated - a
    number that reads near-zero exactly when the window is fullest. The meter
    must show the number the budgeter trimmed against, or the two would tell
    different stories about the same turn. ``num_ctx`` is the window the turn
    actually requested. ``model`` names the tag the numbers describe - the one
    the turn actually ran, substitutions included - so a reader holding a
    different pick can tell the reading is not about it.
    """

    prompt_tokens: int
    num_ctx: int
    model: str | None = None


class TurnUsage(BaseModel):
    """What a finished reply took to write, over every round of its turn.

    Service-owned like :class:`ContextUsage` - a provider reports one
    generation's :class:`Usage`, and a turn that ran tools is several
    generations. ``completion_tokens`` is the sum of what each round wrote
    (reasoning and tool-call rounds included: the model wrote those too).
    ``generation_ms`` is the sum of the runtime's own writing time for those
    same rounds, so the two divide into a real decode speed: the tools' run
    time, a wait on the user's confirmation and a model load are in neither.
    It is None when any counted round came without a duration - a speed over
    part of the tokens would be a wrong number, not an approximate one.

    The same object is stored with the reply and carried on ``done``, so a
    reloaded turn shows exactly the figures its live stream showed.
    """

    completion_tokens: int
    generation_ms: int | None = None


ChunkType = Literal[
    "token",
    "thinking",
    "status",
    "citation",
    "tool_call",
    "tool_result",
    "tool_confirmation",
    "file",
    "meta",
    "done",
    "error",
]


class ChatChunk(BaseModel):
    """One unit of a streamed chat response, normalized across providers.

    The SSE layer maps each chunk to a wire event (see app.schemas.chat). A
    ``thinking`` chunk carries its reasoning text in the same ``delta`` field as a
    ``token`` chunk; a ``status`` chunk carries a human-safe progress message in
    ``delta`` plus a machine code in ``status_code`` (e.g. a model being loaded
    into memory, so the UI can explain a long time-to-first-token);
    ``error_code`` is set only when ``type == "error"``.

    ``conversation_id``, ``user_message_id`` and ``message_id`` are persistence
    metadata stamped by the chat service - the first two on the opening ``meta``
    chunk (so a stream that never reaches its end still names its thread and the
    question it is answering) and ``conversation_id``/``message_id`` on the
    terminal ``done`` chunk once the assistant reply is saved - internal seam
    plumbing carried on the chunk (instead of a wrapper type) so the SSE layer
    keeps consuming one shape. Providers never set them.
    """

    type: ChunkType = "token"
    delta: str = ""
    tool_call: ToolCall | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    error: str | None = None
    error_code: str | None = None
    status_code: str | None = None
    # Set by the chat service, never by providers (same ownership as
    # conversation_id/message_id below): providers know nothing about
    # retrieval, but the SSE layer keeps consuming one chunk shape.
    citation: Citation | None = None
    # Set by the chat service (never by providers), same ownership as citation:
    # the result of a tool the service ran, carried on a tool_result chunk.
    tool_result: ToolCallResult | None = None
    # Set by the chat service, never by providers; carried on a tool_confirmation
    # chunk. When present the turn is SUSPENDED after this chunk (a side-effecting
    # call needs the user's approval), not finished - no done follows it.
    confirmation: ToolConfirmation | None = None
    # Set by the chat service, never by providers; carried on a `file` chunk. A
    # file a tool produced, already stored: it is emitted only AFTER its row is
    # committed, so an announced file is always one a client can go and fetch.
    file: StoredFile | None = None
    # Set by the chat service, never by providers; see the class docstring.
    conversation_id: str | None = None
    message_id: str | None = None
    # The QUESTION this turn answers, stamped on the opening meta chunk: the row
    # a new turn just persisted, or the stored one a regenerate or an edit
    # re-answers. Carried so a client can key its optimistic bubbles on server
    # ids from the first frame. None on a resumed confirmation, which wrote no
    # question of its own.
    user_message_id: str | None = None
    # Set by the chat service on the terminal done chunk ONLY when the
    # persisted answer differs from the streamed text (citation markers
    # renumbered once the reply's used sources are known); None otherwise.
    content: str | None = None
    # Set by the chat service on the terminal done chunk when the turn's
    # context window is known; None when it is not (a provider that reports
    # no window). Never set by providers - see ContextUsage.
    context_usage: ContextUsage | None = None
    # Set by the chat service on the terminal done chunk when the turn's rounds
    # reported what they wrote; None when the provider reported no count.
    # Never set by providers - theirs is ``usage``, one round's report.
    turn_usage: TurnUsage | None = None


class ModelCapabilities(BaseModel):
    """What a given model can do (drives UI gating and routing).

    These are the RUNTIME's own reports about the weights it holds - the
    serving integration's view, not a human judgement. ``completion`` is what
    separates a model a conversation can run on from one that only embeds;
    ``context_length`` is the native window the weights were trained for,
    which a deployment then clamps DOWN to what its hardware affords (see
    ``app.services.model_facts``). A runtime's ``thinking`` flag reports that
    the serving integration has think plumbing, not that the weights reason -
    a stored declaration outranks it wherever both exist.

    The descriptive fields below are all optional and all default to None,
    which means "this runtime does not say" and never "no". They exist so a
    model server holding tags nothing describes can still be listed
    honestly: what the weights ARE (family, parameter count, how hard they
    were quantized) is a fact only the runtime holds, and a card that shows it
    is a card an operator can act on.

    Attributes:
        vision: The model accepts images.
        tools: The chat template can take a tool list.
        embedding: The model produces embeddings.
        thinking: The serving integration has think plumbing for this model.
        completion: The model can hold a conversation.
        context_length: The weights' native context window, in tokens.
        quantization: How the weights were quantized ("Q4_K_M", "F16", ...).
        parameter_size: The parameter count as the runtime words it ("4.0B").
        family: The architecture family the runtime names ("qwen3", "llama").
        template_thinking: The model's own chat template has reasoning
            plumbing in it - a stronger signal than ``thinking``, which
            describes the SERVER's plumbing and is advertised even for a tag
            whose template can do nothing with it. None when the runtime
            exposes no template to read.
    """

    vision: bool = False
    tools: bool = False
    embedding: bool = False
    thinking: bool = False
    completion: bool = False
    context_length: int | None = None
    quantization: str | None = None
    parameter_size: str | None = None
    family: str | None = None
    template_thinking: bool | None = None


class ModelInfo(BaseModel):
    """Metadata about an available model.

    ``digest`` identifies the WEIGHTS a mutable tag currently points at (the
    2507 split re-pointed a whole tag family) - the measurement log binds
    verdicts to it, and None simply means the runtime does not expose one.
    """

    name: str
    size: int | None = None
    digest: str | None = None
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)


class ManagementCapabilities(BaseModel):
    """Which management VERBS a provider supports, as a property of the backend.

    Distinct from :class:`ModelCapabilities`, which describes one MODEL and is
    answered by a live per-model probe. These are provider-class facts: a
    registry-style backend can download and remove weights on request, while a
    process-per-model runtime loads what it was started with and changes only
    by redeployment. A management UI reads these to decide what it may offer,
    so an unsupported verb reads as "changed via deployment" instead of
    rendering a button that cannot work.

    Every flag defaults to False - the honest floor for a backend that has not
    declared itself.
    """

    can_pull: bool = False
    can_delete: bool = False
    can_list_installed: bool = False
    # Whether the backend can be told to drop a model from memory. A
    # hot-swapping runtime can; a runtime that reserves one model's memory at
    # process start cannot, and says so rather than accepting a request that
    # would change nothing.
    can_unload: bool = False


class ModelResidency(BaseModel):
    """Where a loaded model's weights sit right now: GPU, host memory, or both.

    A model is not simply "loaded" - on a small card a runtime will place part
    of it on the GPU and spill the rest to the CPU, and a model that spilled
    ENTIRELY to the CPU still answers, just an order of magnitude slower. That
    difference is invisible to a byte count of GPU memory alone (zero is
    indistinguishable from "not loaded"), which is why this type carries the
    total as well: only the pair says whether a reading is representative of
    how the deployment actually serves.

    Attributes:
        model: The tag this reading describes, as asked for.
        loaded: The backend reports the model resident right now. False is a
            real answer ("nothing is holding it"), distinct from a None
            residency, which means the backend does not report residency at all.
        total_bytes: Bytes the loaded model occupies across all memory.
        vram_bytes: The part of that sitting in GPU memory.
    """

    model: str
    loaded: bool = False
    total_bytes: int | None = None
    vram_bytes: int | None = None

    @property
    def gpu_fraction(self) -> float | None:
        """Return the share of the model on the GPU, or None when unknowable.

        Clamped to 1.0: a runtime may report a VRAM figure a hair above the
        total (allocator overhead lands in one counter and not the other), and
        a fraction over 100% reads as a bug in whatever displays it.
        """
        if not self.total_bytes or self.vram_bytes is None:
            return None
        return min(self.vram_bytes / self.total_bytes, 1.0)


def installed_tag_matches(tag: str, installed: set[str]) -> bool:
    """Match a configured model tag against a provider's installed names.

    Registry-style backends (Ollama and anything OCI-flavored) list bare names
    with an explicit ``:latest`` suffix, so a configured ``llama3`` must match
    an installed ``llama3:latest``. That naming knowledge lives here at the
    seam - services compare tags only through this helper - so no backend
    naming convention leaks into the service layer.

    Args:
        tag: The configured model tag.
        installed: Model names as reported by :meth:`ModelProvider.list_models`.

    Returns:
        True when ``tag`` refers to an installed model.
    """
    return tag in installed or (":" not in tag and f"{tag}:latest" in installed)


# Registry-style backends leave two things implicit in a name: the publisher
# namespace curated models live under, and the version. Both defaults are the
# backend's naming convention, so they belong at this seam beside
# :func:`installed_tag_matches` rather than in any service.
DEFAULT_NAMESPACE = "library"
DEFAULT_VERSION = "latest"


def tag_aliases(tag: str) -> set[str]:
    """Return every spelling that names the same model under the tag convention.

    The sibling of :func:`installed_tag_matches` for the cases where BOTH sides
    are configuration rather than a provider listing. That function expands only
    its first argument, which is right when comparing a configured tag against
    what a backend reports, and silently one-directional when comparing two
    written-down tags: ``qwen3`` matches a configured ``qwen3:latest``, but a
    request for ``qwen3`` would not match it back. A guard built on the
    one-directional form therefore fails OPEN for one of the two spellings,
    which is the wrong direction for a guard.

    Both implicit defaults are expanded, because a guard has to hold for every
    way the same model can be written down: ``qwen3``, ``qwen3:latest``,
    ``library/qwen3`` and ``library/qwen3:latest`` are one model, and a
    deployment is free to configure whichever it likes.

    Args:
        tag: A model tag as configured or requested.

    Returns:
        Every equivalent spelling, or an empty set for an empty tag. A tag with
        an explicit version that is not ``latest`` still expands across the
        namespace, since only the version is pinned.
    """
    base = tag.removeprefix(f"{DEFAULT_NAMESPACE}/")
    if not base:
        return {tag} if tag else set()
    versions = {base}
    if base.endswith(f":{DEFAULT_VERSION}"):
        versions.add(base.removesuffix(f":{DEFAULT_VERSION}"))
    elif ":" not in base:
        versions.add(f"{base}:{DEFAULT_VERSION}")
    if "/" in base:
        # A name that carries its own publisher is not in the default namespace,
        # and prefixing it would invent a tag nobody could pull.
        return versions
    return versions | {f"{DEFAULT_NAMESPACE}/{version}" for version in versions}


# Ports a URL may state or omit without naming a different address. Only the
# schemes a model server is actually reached over: an entry here FOLDS two
# spellings together, which is the direction that can hand out a certificate
# wrongly, so the table stays as short as the deployments it has to serve.
_DEFAULT_PORTS: dict[str, int] = {"http": 80, "https": 443}


def serving_endpoint_of(url: str | None) -> str | None:
    """Reduce a backend's address to the HOST IDENTITY safe to record forever.

    A measurement is a fact about a machine as much as about a model: decode
    throughput is that host's hardware, and a tool verdict is a property of the
    (model, runtime) pair that host is running. So the probe log has to be able
    to say WHERE a verdict was earned, or a verdict measured against a rented
    cloud pod would go on certifying a laptop that never ran a check.

    What is kept is scheme, host and port, and deliberately nothing else. A
    model-server URL is configuration that routinely carries credentials -
    ``https://user:token@host`` is a valid address and userinfo is the common
    way a hosted proxy is addressed - and this value lands in a database row,
    an API payload and a management page, all of which outlive the URL. Paths
    and query strings go for the weaker version of the same reason: they can
    carry a routing token, and they say nothing about which machine answered.

    Comparison downstream is textual, so equality is only as good as the
    normalization done here - and every spelling this fails to fold is a
    verdict silently voided by an edit that changed nothing. Case is folded, a
    trailing path dropped, the DNS root's trailing dot removed, and a port that
    merely restates its scheme's default (``https://host:443``) dropped so it
    matches the same address written without one.

    Two spellings it still cannot fold, both left alone deliberately:
    ``localhost`` against ``127.0.0.1`` (only name resolution could say, and
    resolving here would put a network call on the chat path), and one host's
    two names. Both fail as a false NEGATIVE - a verdict stops certifying until
    a re-check restores it - which is the direction to fail in. The reverse
    mistake, folding two genuinely different addresses into one, is the one
    that hands a machine a certificate it never earned.

    Not to be confused with :func:`app.core.config.ollama_url_host`, which
    answers a different question for a different consumer: it returns the
    HOSTNAME alone, defaults to ``""``, and exists to decide whether a URL
    points off-box. This one produces a comparable IDENTITY and answers None
    on failure, because "" would compare equal to another failed parse.

    Args:
        url: The backend's configured base URL, or None.

    Returns:
        ``scheme://host`` (plus ``:port`` when the URL named a non-default one),
        or None when the value names no host - an address with no ``//``
        authority, an empty string, or a port that is not a number. None is
        recorded as None ("this deployment did not say"), never guessed, and a
        row carrying None can never certify anything.
    """
    if not url:
        return None
    parsed = urlsplit(url.strip())
    if not parsed.scheme or not parsed.netloc:
        # Bare "host:port" parses with the HOST as the scheme, so requiring an
        # authority is what stops a hostname being recorded as a protocol.
        return None
    try:
        host, port = parsed.hostname, parsed.port
    except ValueError:
        return None  # a non-numeric port: nothing here identifies a machine
    if not host:
        return None
    scheme = parsed.scheme.lower()
    if port == _DEFAULT_PORTS.get(scheme):
        # "https://host:443" and "https://host" are one address; keeping both
        # spellings would void a certificate over an edit that changed nothing.
        port = None
    # A trailing dot is the DNS root label and names the same host.
    host = host.rstrip(".") or host
    # urlsplit strips the brackets an IPv6 literal needs to stay unambiguous.
    authority = f"[{host}]" if ":" in host else host
    return f"{scheme}://{authority}{'' if port is None else f':{port}'}"


class ModelProvider(ABC):
    """The model seam. Implemented first by OllamaProvider; hosted/vLLM later.

    No backend-specific MECHANISM may leak past this interface - Ollama tag
    conventions, Modelfiles, a runtime's own tool-call dialect. Callers above
    the seam must never be able to tell which backend is serving them.

    The line falls between mechanism and DESCRIPTION, which is why
    :class:`ModelCapabilities` carries a model's quantization and parameter
    count. Those are facts about the WEIGHTS that every backend has its own
    word for and any backend can report (Ollama says ``Q4_K_M``, a vLLM
    deployment would say ``AWQ``); reporting them commits a caller to nothing
    and lets a management view describe what is installed. Acting on the value -
    branching on a quantization string, or asking a provider to change one -
    would be the leak this rule forbids.
    """

    @abstractmethod
    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Stream a chat completion as normalized chunks."""
        raise NotImplementedError

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        options: EmbedOptions | None = None,
    ) -> list[list[float]]:
        """Return an embedding vector for each input text."""
        raise NotImplementedError

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List the models available from this provider."""
        raise NotImplementedError

    @abstractmethod
    def pull_model(self, name: str) -> AsyncIterator[dict]:
        """Download a model, yielding the backend's own progress events.

        The events are raw dicts rather than a seam type: what a download
        reports (manifest fetched, N bytes of M, verifying) is genuinely the
        backend's own vocabulary, and inventing a normalized progress record
        here would only be a second dialect to keep in step with the first.
        The service layer maps them onto the wire contract clients see.

        Args:
            name: The exact model tag to download.

        Yields:
            One progress event per line the backend emits.

        Raises:
            ModelNotFoundError: The backend's registry has no such tag.
            ProviderUnavailableError: The backend could not be reached, or
                answered the request with an error.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_model(self, name: str) -> None:
        """Remove a downloaded model from the backend.

        Only ever called where :meth:`management_capabilities` declares
        ``can_delete``: a runtime that loads one model at process start holds
        nothing a request could remove, and says so rather than implementing a
        verb that would have to lie. Callers check that declaration first, so
        an implementation without the verb is free to raise.

        Args:
            name: The exact model tag to remove.

        Raises:
            ModelNotFoundError: The backend does not hold this model.
            ProviderUnavailableError: The backend could not be reached, or
                answered the request with an error.
        """
        raise NotImplementedError

    @abstractmethod
    async def capabilities(self, model: str) -> ModelCapabilities:
        """Report the capabilities of a specific model."""
        raise NotImplementedError

    def management_capabilities(self) -> ManagementCapabilities:
        """Report which management verbs this provider supports.

        Sync and IO-free on purpose: these are facts about the provider CLASS,
        not about any one model, so answering them must never cost a round
        trip. Deliberately a sibling of :meth:`capabilities` rather than a
        field on it - that method probes one model over the network and fails
        for a tag the backend has not downloaded, which is precisely the row a
        management UI most needs to answer "can this be pulled?" about.

        The concrete default declares nothing, so a provider only ever gains a
        verb by saying so.

        Returns:
            The provider's declared :class:`ManagementCapabilities`.
        """
        return ManagementCapabilities()

    def serving_endpoint(self) -> str | None:
        """Identify the HOST this provider serves from, or None when it cannot.

        The third axis a measurement binds to, beside the weights digest and
        the runtime version: a verdict earned against one server must not
        certify the same tag served by another. Throughput is that host's
        hardware, and tool behaviour is a property of the (model, runtime)
        pair the host runs - so a verdict measured on a rented pod says
        nothing about a laptop, and vice versa.

        Sync and IO-free, like :meth:`management_capabilities`: this is
        configuration the provider was built with, and a certificate check on
        the chat path must never cost a round trip to answer it.

        The value must be SAFE TO STORE AND DISPLAY - see
        :func:`serving_endpoint_of`, which is how a URL-addressed backend
        derives one. The default is None, so a provider only ever binds its
        verdicts to a host by saying which one; None means "this deployment
        did not say", and a verdict recorded with None certifies nothing
        rather than certifying everywhere.

        Returns:
            A stable, credential-free identity for the serving host, or None.
        """
        return None

    # Concrete defaults, not abstract: these are OPTIONAL measurement hooks for
    # the probe log (verdicts key on runtime version; observed VRAM feeds the
    # Model Manager's measured-facts grid). A provider that cannot answer
    # returns None and the caller records "unmeasured" - never an estimate.

    async def runtime_version(self) -> str | None:
        """Return the serving runtime's version string, when it exposes one."""
        return None

    async def residency(self, model: str) -> ModelResidency | None:
        """Report where ``model``'s weights currently sit, when the backend says.

        Concrete and None by default: residency is a property of a runtime that
        loads and evicts models on demand, and a backend that reserves one
        model's memory at process start has nothing changing to report. None
        therefore means "this backend does not report residency", which a
        caller must not read as "the model is not on the GPU" - see
        :class:`ModelResidency` for why the distinction is load-bearing.

        Args:
            model: The tag to ask about.

        Returns:
            The reading, or None when the backend reports no residency.
        """
        return None

    async def resident_models(self) -> list[ModelResidency] | None:
        """Report every model the backend holds in memory right now, when it says.

        The whole-runtime view of :meth:`residency`, for a caller that wants to
        show what is loaded rather than to ask about one tag - and a separate
        method rather than a listing built by asking about each configured tag
        in turn, because a runtime can hold a model nothing here
        names, and that is exactly the one worth seeing.

        None and an empty list are different answers, in the same way they are
        for :meth:`residency`: None means this backend does not report residency
        at all, while ``[]`` means it was asked and is holding nothing. A caller
        must not render the first as "nothing loaded".

        Returns:
            One reading per resident model, or None when the backend reports no
            residency.
        """
        return None

    async def resident_vram_bytes(self, model: str) -> int | None:
        """Return ``model``'s currently-resident VRAM in bytes, when known.

        The narrow view of :meth:`residency`, kept because most callers only
        want the one number - and derived from it rather than fetched
        separately, so a backend has exactly one place to answer the question.
        """
        view = await self.residency(model)
        return view.vram_bytes if view is not None else None

    async def unload_model(self, model: str) -> bool:
        """Ask the backend to drop ``model`` from memory now.

        Deliberate eviction, and it is a request rather than a guarantee: on a
        card that fits one model at a time, loading anything evicts something,
        and a caller that wants a clean measurement is far better off choosing
        WHAT gets evicted than racing the runtime's own choice.

        A backend that cannot unload says so - the default returns False rather
        than pretending, and :attr:`ManagementCapabilities.can_unload` lets a
        caller know before it asks. False is never an error: a model that was
        not loaded to begin with is the outcome the caller wanted.

        Args:
            model: The tag to drop from memory.

        Returns:
            True when the backend performed an unload, False when it has no
            such verb or the request did not land.
        """
        return False
