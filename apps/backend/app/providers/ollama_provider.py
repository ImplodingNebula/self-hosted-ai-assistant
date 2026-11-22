"""Ollama-backed ModelProvider (the default). Talks to Ollama's REST API."""

from __future__ import annotations

import json
import re
import secrets
from base64 import b64encode
from collections.abc import AsyncIterator

import httpx

from app.providers.base import (
    ChatChunk,
    ChatMessage,
    EmbedOptions,
    GenerationOptions,
    ManagementCapabilities,
    ModelCapabilities,
    ModelInfo,
    ModelNotFoundError,
    ModelProvider,
    ModelResidency,
    ProviderUnauthorizedError,
    ProviderUnavailableError,
    ToolCall,
    Usage,
    installed_tag_matches,
    serving_endpoint_of,
)


def _redirect_message(location: str) -> str:
    """Build the message for a model server that redirected instead of answering.

    It names the address the server pointed at, because that address IS the
    fix: an HTTP proxy in front of a remote model server commonly answers plain
    HTTP with a permanent redirect to HTTPS, and nothing else in such a
    response says so.
    """
    return (
        f"The model server redirected to '{location}'. Set OLLAMA_URL to the address "
        "the server expects (a cloud proxy usually needs https:// instead of http://)."
    )


def _unauthorized_message(status: int) -> str:
    """Build the message for a model server that refused the request's credentials.

    Carries the status code and nothing else from the response: a proxy that
    rejects a call can echo the credential it was sent back in its own body or
    headers, and this text is shown to a user.
    """
    return (
        f"The model server rejected the request as unauthorized (HTTP {status}). Set "
        "OLLAMA_AUTH_HEADERS to the credentials the server's proxy expects."
    )


_ERROR_DETAIL_MAX_CHARS = 300
"""Ceiling on error text lifted from a response body.

Detail strings travel into ProviderUnavailableError messages and from there
verbatim into SSE error frames, check-case rows, and logs - surfaces sized
for a sentence, not a document.
"""


def _clamp_detail(text: str) -> str:
    """Cap error text at sentence length, marking the cut honestly."""
    stripped = text.strip()
    if len(stripped) <= _ERROR_DETAIL_MAX_CHARS:
        return stripped
    return stripped[:_ERROR_DETAIL_MAX_CHARS] + " [truncated]"


_DOCUMENT_MEDIA_TYPES = frozenset(
    {"text/html", "application/xhtml+xml", "application/xml", "text/xml"}
)
"""Media types whose bodies are documents rather than a sentence about a failure."""


def _declares_document(content_type: str | None) -> bool:
    """Say whether a response DECLARED itself a markup document.

    The header is read as well as the bytes because the two fail in different
    places: a proxy page can open with a comment, an XML declaration, or a
    byte-order mark before any tag the sniff below recognizes, and a body that
    slips past the sniff is dumped whole into a UI. What the sender itself
    called the payload is the cheaper and more reliable of the two signals.
    """
    if not content_type:
        return False
    # "text/html; charset=utf-8" -> "text/html"
    return content_type.split(";", 1)[0].strip().lower() in _DOCUMENT_MEDIA_TYPES


def _looks_like_markup(head: str) -> bool:
    """Say whether a body OPENS like a markup document.

    Kept alongside the declared type rather than replaced by it: an
    intermediary that answers with a page and no content type at all is
    exactly the misconfigured hop this whole function exists for.
    """
    lowered = head.lower()
    return "<html" in lowered or lowered.startswith(("<!doctype", "<?xml"))


def _classify_error_body(text: str, status: int, *, content_type: str | None = None) -> str:
    """Compress a non-JSON error body into something a UI can show.

    A proxy in front of the model server answers with its own error PAGE, not
    Ollama's error shape - Cloudflare's 524 timeout page is ~8KB of markup and
    landed verbatim in the tool-check UI (2026-08-04). A document is summarized
    to the status code plus the page's own title (the one human-readable line
    such a page carries); anything else is passed through under the sentence
    cap, which is itself a ceiling and not a promise about the content.

    A body identified as a document is summarized even when no title can be
    read out of it. Losing a stray sentence that a sender mislabelled as HTML
    is a far smaller cost than the failure this replaces, and the status line
    still names the fix: nothing above this seam can act on markup.

    Args:
        text: The decoded response body.
        status: The HTTP status that carried it.
        content_type: The response's declared content type, when one was sent.

    Returns:
        One terse line, or "" for an empty body (the caller supplies the bare
        status line, so an empty string never reaches a reader).
    """
    stripped = text.strip()
    if not stripped:
        return ""
    if _declares_document(content_type) or _looks_like_markup(stripped[:300]):
        match = re.search(r"<title[^>]*>(.*?)</title>", stripped, re.IGNORECASE | re.DOTALL)
        title = _clamp_detail(re.sub(r"\s+", " ", match.group(1))) if match else ""
        named = f": '{title}'" if title else ""
        return (
            f"HTTP {status}: the reply is an error page from a proxy in front "
            f"of the model server, not a model server answer{named}."
        )
    return _clamp_detail(stripped)


def _byte_field(value: object) -> int | None:
    """Read a byte counter from a runtime payload, ignoring anything else.

    A measurement is reported as measured or not at all: a field the server
    omitted, or sent as something other than a number, becomes None
    ("unmeasured") rather than a zero that would read as a real reading of
    nothing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _eval_ms(value: object) -> float | None:
    """Read the server's writing time (``eval_duration``, nanoseconds) as milliseconds.

    Same rule as :func:`_byte_field`: measured or not at all. Absent, not a
    number, or not positive reads as None, so a reply whose timing the server
    did not report shows no speed rather than an infinite one.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return value / 1_000_000


def _native_context_length(payload: dict) -> int | None:
    """Read the weights' native context window out of an ``/api/show`` payload.

    The key naming it is ARCHITECTURE-scoped (``llama.context_length``,
    ``qwen3.context_length``, ...), so the architecture has to be read first
    and the key built from it. A hardcoded family prefix would answer None for
    every model of every other family - indistinguishable from a runtime that
    reports no window at all, and the caller that sizes a turn's context would
    silently fall back to a guess for exactly the models nobody described.

    None means "the payload does not say": anything absent, non-numeric, or
    not a positive count. A window is a positive integer or it is unknown -
    there is no zero-token model, and a bogus one would size a turn to nothing.
    """
    info = payload.get("model_info")
    if not isinstance(info, dict):
        return None
    arch = info.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return None
    value = info.get(f"{arch}.context_length")
    # bool passes isinstance(int) in Python; a JSON ``true`` here would become
    # a one-token window rather than an unknown one.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


# A Go template ACTION referencing the think plumbing. Ollama renders a Go
# template with fields the server binds for it - ``.IsThinkSet`` (was the flag
# sent at all), ``.Think`` (what it was set to), or ``.Thinking`` (a prior
# turn's reasoning, which only a template separating reasoning from the answer
# ever renders). The leading dot inside ``{{ ... }}`` is what makes this
# Go-specific, and the match is case-sensitive on purpose.
_GO_THINK_ACTION = re.compile(r"\{\{[^{}]*\.(?:IsThinkSet|Think(?:ing)?)\b")

# The same plumbing in a template packaged as Jinja, where the switch is a
# VARIABLE the caller sets rather than a field the server binds. These two
# names cover what the current generation of reasoning models ships with; a
# template using some third name reads as non-reasoning, which is the known
# limit of reading a template at all (see the docstring).
_JINJA_THINK_VARIABLE = re.compile(r"\b(?:enable_thinking|reasoning_content)\b")

# A template that OPENS a reasoning block itself, in either dialect: the marker
# appears inside an output expression rather than in a condition. That is an
# always-reasoning checkpoint whose prompt starts the block for the model, and
# it has no flag to act on - so the actions above miss it entirely.
_THINK_EMISSION = re.compile(r"\{\{[^{}]*<think>")

# Evidence that the template is a CHAT template at all, in either dialect: it
# lays out a conversation rather than substituting one blob. A template with
# none of this is a STUB - ``{{ .Prompt }}`` and nothing else - which says
# nothing about reasoning either way. ``.Prompt`` is deliberately not in the
# list, since it is the stub's whole content.
_TEMPLATE_STRUCTURE = re.compile(
    r"\{\{[^{}]*\.(?:Messages|System|Response|Tools)\b|\bmessages\b|\{%"
)

# A modelfile line handing prompt rendering (and reply parsing) to code built
# into the runtime. When one is present the TEMPLATE is no longer the artifact
# that would act on a think flag, so reading it answers the wrong question.
_BUILTIN_RENDERER = re.compile(r"^\s*(?:RENDERER|PARSER)\s+\S", re.MULTILINE)


def _template_thinking(payload: dict) -> bool | None:
    """Say whether the model's own chat template has reasoning plumbing.

    A DIFFERENT question from the ``thinking`` capability flag, and the
    difference is the whole point of reading it. The flag reports that the
    SERVER has think plumbing for this model, and Ollama advertises it for
    tags whose template cannot do anything with it - a never-reasoning instruct
    checkpoint advertises ``thinking`` while its template only strips reasoning
    out of message history. The template is the artifact that would have to act
    on the flag, so it is the better evidence of the two.

    Judgeability is decided FIRST, and answering None is the whole point of
    that step. A modern tag can ship a STUB template - ``{{ .Prompt }}`` and
    nothing more - with the real prompt rendering built into the runtime and
    named by a ``RENDERER`` line in the modelfile. There is nothing in that
    text to read, so any verdict drawn from it is invented: a stub scanned for
    think plumbing finds none and would report the model unable to reason,
    which is a WRONG answer where None is merely an unknown one. This is not a
    hypothetical shape - it is what the current generation of tags ships.

    The runtime's own ``thinking`` capability is deliberately NOT the fallback
    for those tags, however tempting: it reports the serving integration's
    plumbing rather than the weights' behaviour, and reading it here would
    reintroduce exactly the confusion this function exists to resolve. Unknown
    is the honest answer until something MEASURES the model.

    Past that gate, three positive signals, because templates arrive in two
    dialects and reasoning is driven three ways: a Go action naming the
    server's think fields, a Jinja think VARIABLE, or a template that opens a
    ``<think>`` block in its own output. None of them is the bare string
    ``<think>``, and that exclusion is the load-bearing part. The two tags of
    this project's own dev pair are the case that separates them: the reasoning
    sibling ships a Go template wired to ``.IsThinkSet``, while the instruct
    sibling ships a Jinja template whose ``<think>`` mentions sit in CONDITIONS
    that REMOVE reasoning from history. Matching the marker anywhere would read
    the one model known not to reason as one that does.

    Known limit: this reads a template, it does not execute one, so a Jinja
    template driving reasoning through some variable name not listed above
    reads as non-reasoning. That is tolerable because nothing safety-critical
    keys on the answer - it is a display prior, and the check that MEASURES
    whether reasoning is actually emitted outranks it wherever both exist.

    Returns:
        True when a readable template shows reasoning plumbing, False when a
        readable template shows none, and None when there is nothing to read -
        "the runtime does not say", never "no".
    """
    template = payload.get("template")
    if not isinstance(template, str) or not template.strip():
        return None
    if not _TEMPLATE_STRUCTURE.search(template):
        return None  # a stub; the runtime renders the prompt itself
    modelfile = payload.get("modelfile")
    if isinstance(modelfile, str) and _BUILTIN_RENDERER.search(modelfile):
        return None
    return any(
        pattern.search(template) is not None
        for pattern in (_GO_THINK_ACTION, _JINJA_THINK_VARIABLE, _THINK_EMISSION)
    )


def _detail(payload: dict, key: str) -> str | None:
    """Read one string out of an ``/api/show`` payload's ``details`` map.

    Blank and non-string values collapse to None so a caller never has to tell
    "the runtime said nothing" apart from "the runtime said the empty string".
    """
    details = payload.get("details")
    if not isinstance(details, dict):
        return None
    value = details.get(key)
    return (value.strip() or None) if isinstance(value, str) else None


class OllamaProvider(ModelProvider):
    """Streams chat, embeddings, and model management via the Ollama HTTP API."""

    def __init__(
        self,
        base_url: str,
        default_model: str,
        timeout: float = 300.0,
        client: httpx.AsyncClient | None = None,
        auth_headers: dict[str, str] | None = None,
    ) -> None:
        """Configure the provider for an Ollama server at ``base_url``.

        Args:
            base_url: Root URL of the Ollama server.
            default_model: Model used when a call passes ``model=None``.
            timeout: Read timeout in seconds for generation calls.
            client: Optional externally-owned HTTP client (tests inject a mock
                transport here). When omitted, the provider lazily creates and
                owns one; call :meth:`aclose` on shutdown. An injected client
                carries whatever headers it was built with, so ``auth_headers``
                does not apply to one.
            auth_headers: Headers sent verbatim on every call to the server -
                the credentials a reverse proxy in front of it expects, since
                Ollama itself authenticates nothing. The VALUES ARE SECRETS and
                must never be logged or reported.
        """
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._auth_headers = dict(auth_headers or {})
        # First byte can take minutes on small GPUs: a cold model load (or a
        # CPU-fallback load) alone can run past SIX minutes on a contended
        # host (measured live 2026-07-29), and disconnecting mid-load makes
        # Ollama abort the load entirely - so the read timeout must OUTLAST
        # the server's own OLLAMA_LOAD_TIMEOUT, letting the server report a
        # load it gave up on instead of this client killing a healthy one.
        # The deployed value comes from OLLAMA_READ_TIMEOUT_SECONDS via the
        # factory. Connect/write/pool failures still surface fast.
        self._timeout = httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=10.0)
        self._client = client
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the owned HTTP client (no-op for an injected one)."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared client, creating it lazily on first use.

        Credentials are set on the CLIENT rather than per call, so a request
        added later cannot forget them - the whole point is that every call to
        a protected server carries them. Nothing is passed when none are
        configured, so the unauthenticated path is byte-identical to before.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout, headers=self._auth_headers or None
            )
        return self._client

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Stream `/api/chat`, translating Ollama's NDJSON into ChatChunks.

        Connection failures and non-2xx responses become error chunks (never
        raw httpx exceptions) so the provider seam stays backend-agnostic. A 400
        caused by sending ``think`` to a model without the thinking capability
        is retried once without the flag. A redirect is reported rather than
        followed - following one would turn this POST into a GET - and a
        refused credential gets its own code and its own fix. When the target
        model is not resident in memory, a ``status`` chunk is yielded first so
        clients can explain the long time-to-first-token of a cold load.
        """
        resolved_model = model or self._default_model
        think = options.think if options is not None else None
        if not await self._is_loaded(resolved_model):
            yield ChatChunk(
                type="status",
                delta=(
                    f"Loading model '{resolved_model}' into memory - the first "
                    "reply after a model switch can take a minute or two."
                ),
                status_code="loading_model",
            )
        while True:
            payload = self._chat_payload(messages, resolved_model, tools, options, think)
            try:
                async with self._get_client().stream(
                    "POST", f"{self._base_url}/api/chat", json=payload
                ) as resp:
                    location = self._redirect_location(resp)
                    if location is not None:
                        yield ChatChunk(
                            type="error",
                            error=_redirect_message(location),
                            error_code="provider_unavailable",
                        )
                        return
                    if resp.status_code >= 400:
                        detail = await self._error_detail(resp)
                        if (
                            resp.status_code == 400
                            and think is not None
                            and "does not support thinking" in detail
                        ):
                            # The model cannot reason; retry once without the flag
                            # (think becomes None, so this branch cannot loop).
                            think = None
                            continue
                        yield self._http_error_chunk(resp.status_code, detail, resolved_model)
                        return
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            yield ChatChunk(
                                type="error",
                                error="The model server sent an unreadable response.",
                                error_code="provider_unavailable",
                            )
                            return
                        if data.get("error"):
                            yield ChatChunk(
                                type="error",
                                error=str(data["error"]),
                                error_code="provider_unavailable",
                            )
                            return
                        msg = data.get("message") or {}
                        thinking_delta = msg.get("thinking") or ""
                        if thinking_delta:
                            yield ChatChunk(type="thinking", delta=thinking_delta)
                        content = msg.get("content") or ""
                        if content:
                            yield ChatChunk(type="token", delta=content)
                        # qwen3 on Ollama 0.32.0 emits the whole tool_calls array
                        # on the final (done) chunk - no content precedes it and
                        # there are no deltas to accumulate - but any chunk that
                        # carries tool_calls yields one tool_call chunk per call,
                        # so a runtime that streamed them would still work.
                        for raw_call in msg.get("tool_calls") or []:
                            yield ChatChunk(
                                type="tool_call", tool_call=self._parse_tool_call(raw_call)
                            )
                        if data.get("done"):
                            yield ChatChunk(
                                type="done",
                                finish_reason=data.get("done_reason"),
                                usage=Usage(
                                    prompt_tokens=data.get("prompt_eval_count"),
                                    completion_tokens=data.get("eval_count"),
                                    completion_ms=_eval_ms(data.get("eval_duration")),
                                ),
                            )
                    return
            except (httpx.ConnectError, httpx.ConnectTimeout):
                yield ChatChunk(
                    type="error",
                    error="Cannot reach the model server.",
                    error_code="provider_unavailable",
                )
                return
            except httpx.ReadTimeout:
                yield ChatChunk(
                    type="error",
                    error="The model timed out while generating.",
                    error_code="provider_unavailable",
                )
                return
            except httpx.TransportError:
                # Catch-all base class for the remaining transport failures
                # (protocol/read/write/pool errors, e.g. a runner crash
                # mid-generation) so no raw httpx exception crosses the seam.
                yield ChatChunk(
                    type="error",
                    error="Lost the connection to the model server.",
                    error_code="provider_unavailable",
                )
                return

    async def _is_loaded(self, model: str) -> bool:
        """Best-effort check whether ``model`` is already resident in memory.

        Queries `/api/ps` (instant). Any failure - server down, unexpected
        body - returns True so no spurious "loading" status is emitted and the
        chat call itself surfaces the real error.
        """
        try:
            resp = await self._get_client().get(f"{self._base_url}/api/ps")
            resp.raise_for_status()
            loaded = resp.json().get("models", []) or []
        except Exception:  # noqa: BLE001 - status is best-effort by design
            return True
        names = {m.get("name") for m in loaded} | {m.get("model") for m in loaded}
        return model in names

    def _chat_payload(
        self,
        messages: list[ChatMessage],
        model: str,
        tools: list[dict] | None,
        options: GenerationOptions | None,
        think: bool | None,
    ) -> dict:
        """Build the `/api/chat` request body from the seam-level inputs."""
        payload: dict = {
            "model": model,
            "messages": [self._message_payload(m) for m in messages],
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        # "think" is a TOP-LEVEL field on /api/chat (not inside "options");
        # omit it entirely when unset so non-thinking models are unaffected.
        if think is not None:
            payload["think"] = think
        if options is not None:
            native: dict = {}
            if options.temperature is not None:
                native["temperature"] = options.temperature
            if options.num_ctx is not None:
                native["num_ctx"] = options.num_ctx
            if options.max_tokens is not None:
                native["num_predict"] = options.max_tokens
            if native:
                payload["options"] = native
            if options.keep_alive is not None:
                payload["keep_alive"] = options.keep_alive
        return payload

    @staticmethod
    def _message_payload(message: ChatMessage) -> dict:
        """Map a seam ChatMessage onto Ollama's `/api/chat` message shape.

        An explicit mapper, not ``model_dump``: the seam carries fields Ollama
        does not use, under names it does not use. A tool result's tool name goes
        out as ``tool_name`` (Ollama 0.32.0 also accepts ``name``), while
        ``tool_call_id`` stays seam-side for correlation and never reaches the
        wire; an assistant's requested tool calls are re-serialized into
        Ollama's ``{"function": {...}}`` shape rather than dumped raw; and a
        message's raw images become Ollama's base64 ``images`` list (the seam's
        ``media_type`` stays seam-side - Ollama sniffs the format from the
        bytes).

        Args:
            message: The seam message to serialize.

        Returns:
            The Ollama-shaped message dict.
        """
        payload: dict = {"role": message.role, "content": message.content}
        if message.name is not None:
            payload["tool_name" if message.role == "tool" else "name"] = message.name
        if message.tool_calls:
            payload["tool_calls"] = [
                {"id": call.id, "function": {"name": call.name, "arguments": call.arguments}}
                for call in message.tool_calls
            ]
        if message.images:
            payload["images"] = [b64encode(image.data).decode("ascii") for image in message.images]
        return payload

    @staticmethod
    def _parse_tool_call(raw: dict) -> ToolCall:
        """Parse one Ollama tool call into the seam's :class:`ToolCall`.

        Ollama 0.32.0 emits ``{"id", "function": {"name", "arguments": {...}}}``
        with ``arguments`` already a JSON object; a JSON string is tolerated
        defensively for other runtimes. Ollama supplies the ``id``, but one is
        minted when a runtime omits it so every call stays correlatable.

        Args:
            raw: One entry from a chunk's ``message.tool_calls`` array.

        Returns:
            The normalized tool call.
        """
        function = raw.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        call_id = raw.get("id") or secrets.token_hex(12)
        return ToolCall(id=str(call_id), name=str(function.get("name") or ""), arguments=arguments)

    @staticmethod
    def _redirect_location(resp: httpx.Response) -> str | None:
        """Return where a redirect pointed, or None when the response is not one.

        Redirects are never FOLLOWED, and must not be: the client would turn
        the POST a generation rides on into a GET, so a redirect that appeared
        to work would break chat in a way nothing reports. The address is
        surfaced instead, because it is the fix - a proxy in front of a remote
        model server answering plain HTTP with a redirect to HTTPS is the
        ordinary way this happens, and the default failure text ("cannot reach
        the model server") sends someone to check a server that is perfectly
        healthy.
        """
        if not 300 <= resp.status_code < 400:
            return None
        return resp.headers.get("location") or "an unnamed address"

    @classmethod
    def _raise_for_redirect_or_auth(cls, resp: httpx.Response) -> None:
        """Raise the seam error for the two failures a fronting proxy produces.

        Both are configuration rather than an outage, and neither is explained
        by the response body - a redirect carries none worth reading, and a
        rejecting proxy's may echo the credential it just refused - so each
        becomes a fixed message naming the setting to change and nothing else.

        The best-effort hooks (`/api/ps`, `/api/version`, the unload) need no
        call of their own: they use ``raise_for_status``, which already fails
        on a redirect, and they swallow every failure by design.

        Raises:
            ProviderUnauthorizedError: The server refused the credentials sent.
            ProviderUnavailableError: The server redirected the request.
        """
        location = cls._redirect_location(resp)
        if location is not None:
            raise ProviderUnavailableError(_redirect_message(location))
        if resp.status_code in (401, 403):
            raise ProviderUnauthorizedError(_unauthorized_message(resp.status_code))

    @staticmethod
    async def _error_detail(resp: httpx.Response) -> str:
        """Read a non-2xx response body and extract Ollama's error text.

        Every string this returns is clamped or summarized: it travels into
        seam errors and from there verbatim into SSE frames, check-case rows
        and logs, none of which are sized for a document.
        """
        body = await resp.aread()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("error"):
            return _clamp_detail(str(parsed["error"]))
        return _classify_error_body(
            body.decode("utf-8", errors="replace"),
            resp.status_code,
            content_type=resp.headers.get("content-type"),
        )

    @staticmethod
    def _http_error_chunk(status: int, detail: str, model: str) -> ChatChunk:
        """Translate a non-2xx `/api/chat` response into a normalized error chunk."""
        if status == 404:
            return ChatChunk(
                type="error",
                error=(
                    f"Model '{model}' is not available on the model server. "
                    "Download it from the Model Manager at /console/models."
                ),
                error_code="model_not_found",
            )
        if status in (401, 403):
            # Its own code, not bad_request: nothing about the request is
            # wrong, and the fix is a deployment setting rather than anything
            # the caller can change. The detail is dropped on purpose - see
            # _unauthorized_message.
            return ChatChunk(
                type="error",
                error=_unauthorized_message(status),
                error_code="provider_unauthorized",
            )
        code = "bad_request" if status < 500 else "provider_unavailable"
        return ChatChunk(
            type="error",
            error=detail or f"Model server returned HTTP {status}.",
            error_code=code,
        )

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        options: EmbedOptions | None = None,
    ) -> list[list[float]]:
        """Return embeddings from `/api/embed`.

        Raises:
            ProviderUnavailableError: When the model server cannot be reached,
                rejects the request (including a not-pulled embedding model -
                the actionable detail is in the message), or returns a
                malformed body - the seam-level type, so callers never handle
                httpx or parse exceptions.
        """
        resolved_model = model or self._default_model
        payload: dict = {"model": resolved_model, "input": texts}
        if options is not None:
            if options.device == "cpu":
                # Ollama-specific mapping: num_gpu 0 loads the model with zero
                # layers on the GPU, i.e. CPU-only inference.
                payload["options"] = {"num_gpu": 0}
            if options.keep_alive is not None:
                payload["keep_alive"] = options.keep_alive
        try:
            resp = await self._get_client().post(f"{self._base_url}/api/embed", json=payload)
            self._raise_for_redirect_or_auth(resp)
            if resp.status_code >= 400:
                if resp.status_code == 404:
                    raise ProviderUnavailableError(
                        f"Embedding model '{resolved_model}' is not available on "
                        "the model server. Download it from the Model Manager at "
                        "/console/models."
                    )
                detail = self._error_detail_sync(resp)
                raise ProviderUnavailableError(
                    detail or f"Model server returned HTTP {resp.status_code}."
                )
            embeddings = resp.json().get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise ProviderUnavailableError("The model server returned malformed embeddings.")
            return embeddings
        except (httpx.HTTPError, ValueError, AttributeError, TypeError) as exc:
            raise ProviderUnavailableError("The model server is unavailable.") from exc

    @staticmethod
    def _error_detail_sync(resp: httpx.Response) -> str:
        """Extract Ollama's error text from an already-read non-2xx response.

        The sync sibling of :meth:`_error_detail`, clamping identically - the
        summarization has to live at the seam rather than at each consumer,
        because a body that reaches a caller unclamped is already loose.
        """
        content_type = resp.headers.get("content-type")
        try:
            parsed = resp.json()
        except (json.JSONDecodeError, ValueError):
            return _classify_error_body(resp.text, resp.status_code, content_type=content_type)
        if isinstance(parsed, dict) and parsed.get("error"):
            return _clamp_detail(str(parsed["error"]))
        return _classify_error_body(resp.text, resp.status_code, content_type=content_type)

    async def list_models(self) -> list[ModelInfo]:
        """List locally-available models from `/api/tags`.

        Raises:
            ProviderUnauthorizedError: When a proxy in front of the server
                refused the credentials sent (or the lack of them).
            ProviderUnavailableError: When the model server cannot be reached,
                redirects the request elsewhere, answers with an error status,
                or returns a body that is not the expected shape (e.g. an
                intercepting proxy page) - the seam-level type, so callers
                never handle httpx or parse exceptions.
        """
        try:
            resp = await self._get_client().get(f"{self._base_url}/api/tags")
            self._raise_for_redirect_or_auth(resp)
            resp.raise_for_status()
            models = resp.json().get("models", []) or []
            return [
                ModelInfo(name=m.get("name", ""), size=m.get("size"), digest=m.get("digest"))
                for m in models
            ]
        except (httpx.HTTPError, ValueError, AttributeError, TypeError) as exc:
            # ValueError covers json.JSONDecodeError; Attribute/TypeError cover
            # a 200 body with an unexpected shape.
            raise ProviderUnavailableError("The model server is unavailable.") from exc

    def management_capabilities(self) -> ManagementCapabilities:
        """Declare the registry verbs Ollama supports: pull, delete, list, unload.

        Ollama is a registry-style backend - weights are downloaded and removed
        on request while the server keeps running, and a loaded model can be
        dropped from memory without restarting anything - so all four verbs are
        real here. IO-free: this states what the API offers, never whether the
        server is currently reachable.
        """
        return ManagementCapabilities(
            can_pull=True, can_delete=True, can_list_installed=True, can_unload=True
        )

    def serving_endpoint(self) -> str | None:
        """Identify the Ollama server this provider talks to, credentials stripped.

        Derived from the configured base URL rather than stored separately, so
        a deployment that re-points ``OLLAMA_URL`` at another host is a new
        endpoint from the next probe onwards with nothing to keep in step.
        """
        return serving_endpoint_of(self._base_url)

    async def runtime_version(self) -> str | None:
        """Read the Ollama server version (best-effort; None on any failure).

        The probe log keys verdicts on it - the thinking-plus-tools bug and the
        tool-template fixes were both RUNTIME properties, so a verdict without
        the runtime that produced it answers the wrong question.
        """
        try:
            resp = await self._get_client().get(f"{self._base_url}/api/version")
            resp.raise_for_status()
            version = resp.json().get("version")
            return str(version) if version else None
        except Exception:  # noqa: BLE001 - measurement hooks are best-effort by design
            return None

    async def _loaded_entries(self) -> list[dict] | None:
        """Read `/api/ps` (best-effort): the runtime's own loaded-model list.

        Returns:
            The raw entries, or None when the server could not be asked - which
            the residency readers above this must keep distinct from an empty
            list, since one means "unknown" and the other "holding nothing".
        """
        try:
            resp = await self._get_client().get(f"{self._base_url}/api/ps")
            resp.raise_for_status()
            return resp.json().get("models", []) or []
        except Exception:  # noqa: BLE001 - measurement hooks are best-effort by design
            return None

    async def residency(self, model: str) -> ModelResidency | None:
        """Read where ``model`` currently sits from `/api/ps` (best-effort).

        Ollama reports BOTH numbers per loaded model - ``size`` (everywhere)
        and ``size_vram`` (the GPU's share) - which is what its own `ps` output
        renders as "40%/60% CPU/GPU". Both are carried across the seam: a VRAM
        figure alone cannot tell a model that spilled entirely to the CPU from
        one that is not loaded at all, and those call for opposite reactions.

        Args:
            model: The tag to ask about.

        Returns:
            The reading, a ``loaded=False`` reading when the server answered
            and does not hold it, or None when the server could not be asked.
        """
        loaded = await self._loaded_entries()
        if loaded is None:
            return None
        for entry in loaded:
            names = {n for n in (entry.get("name"), entry.get("model")) if n}
            if installed_tag_matches(model, names):
                return ModelResidency(
                    model=model,
                    loaded=True,
                    total_bytes=_byte_field(entry.get("size")),
                    vram_bytes=_byte_field(entry.get("size_vram")),
                )
        return ModelResidency(model=model, loaded=False)

    async def resident_models(self) -> list[ModelResidency] | None:
        """List everything `/api/ps` reports loaded right now (best-effort).

        The tag is taken from the entry's own ``model`` field (falling back to
        ``name``), so a model nothing here mentions is reported under the
        name the runtime knows it by rather than dropped.

        Returns:
            One reading per loaded model, an empty list when the server answered
            and holds nothing, or None when it could not be asked.
        """
        loaded = await self._loaded_entries()
        if loaded is None:
            return None
        readings: list[ModelResidency] = []
        for entry in loaded:
            name = entry.get("model") or entry.get("name")
            if not name:
                continue
            readings.append(
                ModelResidency(
                    model=str(name),
                    loaded=True,
                    total_bytes=_byte_field(entry.get("size")),
                    vram_bytes=_byte_field(entry.get("size_vram")),
                )
            )
        return readings

    async def unload_model(self, model: str) -> bool:
        """Drop ``model`` from memory now, by asking for a zero keep-alive.

        A zero ``keep_alive`` on an empty generation is how the runtime is told
        to release a model - the same request `ollama stop` makes - so there is
        no separate unload endpoint to call.

        Best-effort like the measurement hooks, and for the same reason: the
        caller is a diagnostic freeing the card, and a server that cannot be
        reached to unload is a problem the caller's NEXT request reports
        properly. It returns False rather than raising so an unload failure
        never becomes the error a caller reports for the work it was clearing
        the way for.

        Args:
            model: The tag to drop from memory.

        Returns:
            True when the server accepted the unload.
        """
        try:
            resp = await self._get_client().post(
                f"{self._base_url}/api/generate",
                json={"model": model, "keep_alive": 0, "stream": False},
                # An unload is instant; it must not inherit the generation read
                # timeout, which exists to cover a multi-minute cold load.
                timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
            )
            resp.raise_for_status()
        except Exception:  # noqa: BLE001 - see the docstring: never the caller's error
            return False
        return True

    async def pull_model(self, name: str) -> AsyncIterator[dict]:
        """Stream pull progress from `/api/pull`.

        Every failure leaves as a seam-level exception rather than an httpx one:
        this feeds an SSE stream, and a raw transport error escaping here would
        surface to a browser as a dead connection with no event explaining it.

        Args:
            name: The exact model tag to download.

        Yields:
            One decoded progress object per NDJSON line, Ollama's own shape.

        Raises:
            ModelNotFoundError: The registry has no such tag (HTTP 404).
            ProviderUnauthorizedError: A proxy in front of the server refused
                the credentials sent (or the lack of them).
            ProviderUnavailableError: The server could not be reached,
                redirected the request, or rejected it.
        """
        payload = {"model": name, "stream": True}
        try:
            # A pull can stall far longer than a generation read; disable the
            # read timeout for this request only.
            async with self._get_client().stream(
                "POST", f"{self._base_url}/api/pull", json=payload, timeout=None
            ) as resp:
                self._raise_for_redirect_or_auth(resp)
                if resp.status_code == 404:
                    raise ModelNotFoundError(
                        f"The model registry has no tag '{name}'. Check the spelling "
                        "against the model library."
                    )
                if resp.status_code >= 400:
                    detail = await self._error_detail(resp)
                    raise ProviderUnavailableError(
                        detail or f"Model server returned HTTP {resp.status_code}."
                    )
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue  # skip a truncated progress line rather than crash
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                "Lost the connection to the model server during the download."
            ) from exc

    async def delete_model(self, name: str) -> None:
        """Remove downloaded weights via `DELETE /api/delete`.

        Args:
            name: The exact model tag to remove.

        Raises:
            ModelNotFoundError: The server does not hold this model (HTTP 404).
            ProviderUnauthorizedError: A proxy in front of the server refused
                the credentials sent (or the lack of them).
            ProviderUnavailableError: The server could not be reached,
                redirected the request, or rejected it.
        """
        try:
            # Ollama takes the tag in a BODY on DELETE, which httpx's
            # convenience `delete()` cannot send - hence the explicit request.
            resp = await self._get_client().request(
                "DELETE", f"{self._base_url}/api/delete", json={"model": name}
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError("The model server is unavailable.") from exc
        self._raise_for_redirect_or_auth(resp)
        if resp.status_code == 404:
            raise ModelNotFoundError(f"Model '{name}' is not installed on the model server.")
        if resp.status_code >= 400:
            detail = self._error_detail_sync(resp)
            raise ProviderUnavailableError(
                detail or f"Model server returned HTTP {resp.status_code}."
            )

    async def capabilities(self, model: str) -> ModelCapabilities:
        """Derive capability flags from `/api/show`.

        This is pure metadata, unlike a generation, so it overrides the long
        generation read timeout with a short one: a capability probe belongs
        inline on the turn that needs it, and a wedged server must not stall
        that turn for the full generation timeout.

        The flags come from the payload's ``capabilities`` array; the native
        window comes from its ``model_info`` map, which names the key after the
        model's own architecture (see :func:`_native_context_length`); the
        descriptive fields come from its ``details`` map; and the reasoning
        plumbing is read off the chat template itself, which is better evidence
        than the ``thinking`` flag beside it (see :func:`_template_thinking`).

        Raises:
            ProviderUnauthorizedError: A proxy in front of the server refused
                the credentials sent (or the lack of them).
            ProviderUnavailableError: The server redirected the request.
        """
        resp = await self._get_client().post(
            f"{self._base_url}/api/show",
            json={"model": model},
            timeout=httpx.Timeout(connect=10.0, read=15.0, write=30.0, pool=10.0),
        )
        self._raise_for_redirect_or_auth(resp)
        resp.raise_for_status()
        payload = resp.json()
        caps = payload.get("capabilities", []) or []
        return ModelCapabilities(
            vision="vision" in caps,
            tools="tools" in caps,
            embedding="embedding" in caps,
            thinking="thinking" in caps,
            completion="completion" in caps,
            context_length=_native_context_length(payload),
            quantization=_detail(payload, "quantization_level"),
            parameter_size=_detail(payload, "parameter_size"),
            family=_detail(payload, "family"),
            template_thinking=_template_thinking(payload),
        )
