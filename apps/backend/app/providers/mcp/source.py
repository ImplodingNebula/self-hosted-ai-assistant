"""The MCP tool source: a remote MCP server adapted into the tool seam.

The backend is the MCP host: :class:`McpToolSource` speaks Streamable HTTP
(the SDK's ``streamable_http_client``) to one configured server, lists that
server's tools, and adapts each into the seam's
:class:`~app.providers.tools.base.Tool`. MCP stays one SOURCE of tools, not
the tool architecture - this source joins the registry alongside the
in-process tools and nothing above the seam can tell them apart.

Two standing rules shape everything here:

- Everything the server sends (names, descriptions, JSON Schemas) is
  UNTRUSTED remote input. It is carried verbatim, never interpolated into a
  prompt or message by this module, and every part of it that rides in the
  prompt is BOUNDED: the description length, the input schema's size, depth
  and string lengths, the number of tools a listing may contribute, and the
  size of a result's text. The schema matters as much as the description
  because a chat template renders the offered tool list into the SYSTEM region
  of the prompt - unfenced prose in a nested ``properties[*].description`` sits
  in more privileged placement than a tool RESULT ever gets.
- Error messages and logs name the server by its configured name, never by
  URL - a URL may embed credentials someday.

The transport is bounded and dumb on purpose: no redirects (an endpoint is
operator-configured, so a 302 is a re-point this deployment never consented to,
including the classic hop to a cloud metadata address) and a hard byte cap per
response (the SDK buffers a whole body, so an unbounded one is an OOM of the
single backend process, taking every concurrent stream with it).
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import timedelta

import httpx
from mcp import ClientSession
from mcp import types as mcp_types
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, Field

from app.providers.tools.base import (
    Tool,
    ToolArtifact,
    ToolContext,
    ToolError,
    ToolRecord,
    ToolResult,
    ToolSource,
    ToolSpec,
)

logger = logging.getLogger(__name__)

# Resolves the per-CALL auth headers for one caller (Phase 5 slice 6b) - the
# acting user's credential, fetched fresh per call. A structural type, so this
# module never imports the concrete resolver (which composes a service): the
# source is handed one and only calls it. Returns headers to MERGE over the
# server's static config headers; ``{}`` means "no per-user credential, use the
# static header".
CredentialResolver = Callable[[ToolContext], Awaitable[dict[str, str]]]

# Descriptions ride in the prompt on every offered turn, so an unbounded one
# is both a context cost and an injection surface; 1024 chars is generous for
# an honest tool and cheap to carry.
_MAX_DESCRIPTION_CHARS = 1024

# The tool NAME is the identity the model calls by, so it cannot be truncated
# (a shortened name is a tool that can never be invoked). A name past this is a
# rejected tool, not a trimmed one. 128 chars fits any honest identifier.
_MAX_NAME_CHARS = 128

# A malicious server could paginate forever; listing is capped, not trusted.
_MAX_LIST_PAGES = 20

# ... and could answer each page with thousands of tools. Every listed tool is
# paid for downstream (selection embeds its description; an offered one rides in
# the prompt), so the total is capped as well as the page count. 150 is far past
# any honest server - the largest published catalogs run to a few dozen - and
# still a bounded embedding batch. It is also half of what the response cap
# below can carry at the per-schema ceiling; the two are chosen together.
_MAX_TOOLS = 150

# The input schema is prompt material exactly like the description, so it takes
# the same treatment. Four bounds, because a schema can be absurd in four ways:
#
#   bytes - the whole thing, measured AFTER prose is clamped. A real tool's
#     schema is a few KB at worst, so 8 KB is generous headroom. The value is
#     not free-standing: see the response cap below, which it has to fit under.
#   depth - nesting, counting a dict/list level each (a schema one object deep
#     costs ~2: "properties" then the property). 12 admits ~5 levels of real
#     nesting; honest schemas rarely pass 3.
#   nodes - total values walked, so a flat dict of a million keys is refused
#     before the walk becomes the attack.
#   string - any single string. Prose keys are TRUNCATED (that is what the
#     top-level description already does); any other string over the cap
#     REJECTS the tool, because silently shortening a "pattern", "enum" or
#     "const" would change what the tool accepts and ship a lie to the model.
_MAX_SCHEMA_BYTES = 8_192
_MAX_SCHEMA_DEPTH = 12
_MAX_SCHEMA_NODES = 2_000
_MAX_SCHEMA_STRING_CHARS = 512
_SCHEMA_PROSE_KEYS = frozenset({"description", "title", "$comment"})

# The SDK buffers a whole response body before this module sees it, so the cap
# has to sit in the transport. Same number as the web page fetcher's, and for
# the same reason: it is far above any honest payload (the result is truncated
# to ~1000 tokens downstream, and a records batch is capped at 50) and far below
# what would hurt a process serving concurrent streams.
#
# THE CAPS ABOVE MUST FIT UNDER THIS ONE. Breaking the transport cap raises,
# and a raised listing yields NO tools at all - so a listing that satisfies
# every per-tool bound but overruns the response would cost a server its whole
# tool set rather than the one bad tool the drop path was built to cost it.
# _MAX_TOOLS * _MAX_SCHEMA_BYTES (150 * 8 KB = 1.2 MB) is the worst case a
# fully-conforming page can reach, and it fits with room for the envelope.
# Raise either cap and this stops holding.
#
# IT ALSO BOUNDS A RESULT'S FILES, and that is the binding constraint on them.
# An artifact travels as base64, which costs 4 bytes per 3, so this cap puts the
# real per-CALL ceiling at ~1.4 MiB of decoded file across every artifact in the
# response - well under the host's own 15 MiB per-file setting, which therefore
# cannot fire through this transport today. The failure mode differs too: over
# this cap the read raises, so the whole CALL fails with an honest tool error
# rather than the file being dropped with a notice. That is the safe direction
# (nothing silently disappears) but it is not the graceful one, and a deployment
# that needs to move larger documents has to raise this cap deliberately - after
# re-doing the arithmetic above, since the SDK buffers the whole body in the one
# process serving every concurrent stream.
_MAX_RESPONSE_BYTES = 2_000_000

# A second, defensive cap on the joined result text, so the join cannot double a
# payload that arrived just under the transport cap. ~16x the per-result token
# ceiling, so it never truncates an honest result - the token budget does that,
# honestly and one layer up. TEXT ONLY: an artifact's bytes never pass through
# it, because they never enter the text path at all.
_MAX_RESULT_CHARS = 64_000

# How many FILES one call may hand back. A per-call structural bound, distinct
# from the per-TURN budget the chat service enforces: this one exists so a
# hostile server cannot make the host decode thousands of blobs before any
# service-layer policy gets a look at them. Deliberately small - a tool that
# produced a document produced one, or a handful.
_MAX_ARTIFACTS = 10

# The uri is a filename HINT and the host re-validates it anyway, so all this
# bound has to do is stop an absurd one from travelling further than the decode.
_MAX_ARTIFACT_NAME_CHARS = 255

# What an artifact is called when the server declared no media type. The generic
# "unknown bytes" type, so nothing downstream mistakes it for something it can
# parse.
_DEFAULT_ARTIFACT_CONTENT_TYPE = "application/octet-stream"


class McpResponseTooLargeError(httpx.HTTPError):
    """A server's response body ran past :data:`_MAX_RESPONSE_BYTES`.

    An ``httpx.HTTPError`` on purpose: it is raised from inside the response
    stream, so anything between here and the caller that already handles a
    transport failure (the SDK, the listing's swallow, the call's ToolError
    wrap) treats an oversized body as exactly that - a failed request - rather
    than as an unexpected crash.
    """

    def __init__(self, max_bytes: int) -> None:
        """Record the cap in a message that names no URL."""
        super().__init__(f"MCP response body exceeded {max_bytes} bytes")
        self.max_bytes = max_bytes


class _CappedByteStream(httpx.AsyncByteStream):
    """Wraps a response stream, failing the read once it passes a byte budget.

    The budget is PER RESPONSE and cumulative over that response's whole life,
    which makes it a bound on a single body only while sessions are short. This
    transport also carries Streamable HTTP's SSE streams, and an SSE response
    stays open: bytes accumulate across every event on it, so a session held
    open long enough would eventually trip a cap meant for one oversized body,
    surfacing as a disconnect nobody could explain.

    It is safe here because a session is opened and closed around each single
    listing or call - which is not a performance choice but the one-client-per-
    call rule that stops one user's ``Authorization`` header being reused for
    another's call. Two unrelated constraints, one mechanism. If MCP client
    POOLING is ever introduced (already gated on a cache key that includes the
    user), this cap has to become per-message or per-body first, and a test
    that dials through the real transport is worth writing at that point.
    """

    def __init__(self, inner: httpx.AsyncByteStream, max_bytes: int) -> None:
        """Bind the wrapped stream and its budget."""
        self._inner = inner
        self._max_bytes = max_bytes

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield the body, raising as soon as it outgrows the budget.

        Raising mid-stream (rather than stopping short) is deliberate: a
        truncated JSON-RPC frame would surface as a confusing protocol error,
        while a raise is an honest failed request the caller already degrades on.
        """
        seen = 0
        async for chunk in self._inner:
            seen += len(chunk)
            if seen > self._max_bytes:
                raise McpResponseTooLargeError(self._max_bytes)
            yield chunk

    async def aclose(self) -> None:
        """Release the wrapped stream, if it holds anything."""
        closer = getattr(self._inner, "aclose", None)
        if closer is not None:
            await closer()


def _env_proxy() -> str | None:
    """The egress proxy from the environment, in httpx's own precedence order.

    httpx resolves these itself ONLY when it builds its own transport, and this
    module supplies one to get the byte cap - so the lookup has to be repeated
    here or a proxied deployment silently loses egress. ``ALL_PROXY`` is the
    fallback for both schemes, matching the convention every other client on
    the box follows.
    """
    return (
        os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("ALL_PROXY")
    )


class _ByteCappedTransport(httpx.AsyncBaseTransport):
    """An httpx transport that refuses to buffer an unbounded response body.

    The MCP SDK reads a whole body into memory (``await response.aread()``), so
    a hostile or broken server could otherwise hand this single-process backend a
    multi-gigabyte listing or tool result and OOM every concurrent chat stream
    with it. Capping at the TRANSPORT is what makes the bound real: every layer
    above it - the token truncation, the records cap - runs only once the payload
    is already resident.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport, max_bytes: int) -> None:
        """Bind the wrapped transport and the per-response byte budget."""
        self._inner = inner
        self._max_bytes = max_bytes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Send the request, returning the response with a capped body stream."""
        response = await self._inner.handle_async_request(request)
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            stream=_CappedByteStream(response.stream, self._max_bytes),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        """Close the wrapped transport."""
        await self._inner.aclose()


class McpServerConfig(BaseModel):
    """Where one MCP server lives and how to authenticate to it.

    Attributes:
        name: Human-readable server name, used in logs and error messages
            in place of the URL (a URL may embed credentials someday).
        url: The full Streamable-HTTP endpoint, e.g.
            ``http://mcp-chassis:8765/mcp``.
        headers: Static auth headers sent on every request. Per-user
            credentials arrive in a later slice; when they do they are
            injected per call, which is why clients are never pooled.
    """

    name: str
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    # Tool names an operator has switched OFF for this server (the per-tool enable
    # map, kept as a deny-list so a newly-advertised tool surfaces by default).
    # Filtered out of a LISTING; empty for a test-connection, which shows the full
    # advertised set so an admin can decide what to toggle.
    disabled_tools: frozenset[str] = Field(default_factory=frozenset)


class McpToolSource(ToolSource):
    """One registered MCP server, presented as a source of seam tools."""

    def __init__(
        self,
        config: McpServerConfig,
        timeout_seconds: float = 30.0,
        list_timeout_seconds: float | None = None,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        credential_resolver: CredentialResolver | None = None,
    ) -> None:
        """Bind the source to one server's config.

        Args:
            config: The server's name, endpoint URL, and static headers.
            timeout_seconds: Budget for a tool CALL's transport and protocol
                waits; a server that hangs must fail the call, not the turn.
            list_timeout_seconds: Budget for a tool LISTING. Deliberately
                separate, and normally much shorter: a listing is discovery
                that runs before the first token of EVERY tool-capable turn,
                so a slow server must degrade to "offers no tools right now"
                quickly. A call is work the user asked for and can wait on.
                None inherits ``timeout_seconds`` (the pre-split behaviour).
            http_client_factory: Builds the httpx client for ONE session.
                None means a real network client carrying the config's
                headers and timeout; tests inject an ASGITransport-backed
                factory to stay hermetic.
            credential_resolver: Resolves the per-CALL auth headers for the
                caller (Phase 5 slice 6b), merged over the config's static
                headers on the client each call builds. None (the default)
                means every call uses the static headers - the pre-6b
                behaviour. Never consulted for a LISTING, which stays on the
                static workspace header so the listing cache never mixes
                callers.
        """
        self._config = config
        self._timeout_seconds = timeout_seconds
        self._list_timeout_seconds = (
            timeout_seconds if list_timeout_seconds is None else list_timeout_seconds
        )
        self._http_client_factory = http_client_factory
        self._credential_resolver = credential_resolver

    @property
    def server_name(self) -> str:
        """The configured server name, safe for logs and error messages."""
        return self._config.name

    def _build_http_client(
        self, timeout_seconds: float, headers_override: dict[str, str] | None = None
    ) -> httpx.AsyncClient:
        """Create a FRESH transport client for exactly one session.

        ``headers_override`` (the caller's per-call credential, slice 6b) is
        merged ON TOP of the client's static headers, so a per-user token
        replaces the shared workspace one for this call only. Applied after the
        client is built so it lands whether the client is the real one or a
        test's injected factory - and the client is still fresh and closed per
        session, so one caller's token never rides another's call.

        The real client is deliberately DUMB about redirects and BOUNDED in what
        it will buffer:

        - ``follow_redirects=False``, matching every other outbound client in
          this codebase (the page fetcher, the OAuth flow, discovery). An MCP
          endpoint is an operator-configured address, never a discovered link,
          so a 302 is the server re-pointing the dial at something the operator
          never approved - a loopback admin port or 169.254.169.254 - and it
          arrives carrying this deployment's Authorization header. Following by
          hand with a per-hop host check (what the page fetcher does, because a
          search result legitimately redirects) would mean reimplementing the
          SDK's POST/session handling for a hop no MCP server needs: the seeded
          chassis and every shipped preset address their endpoint directly.
        - a byte-capped transport, so an unbounded body fails the request
          instead of the process (see :class:`_ByteCappedTransport`).

        Supplying a transport turns OFF httpx's own environment-proxy handling
        (it resolves proxies only when it builds the transport itself), so the
        inner transport reads the proxy variables explicitly. Without that, a
        deployment behind an egress proxy would see every MCP dial fail with a
        generic transport error and nothing pointing at the cause.
        """
        if self._http_client_factory is not None:
            client = self._http_client_factory()
        else:
            client = httpx.AsyncClient(
                headers=dict(self._config.headers),
                timeout=httpx.Timeout(timeout_seconds),
                follow_redirects=False,
                transport=_ByteCappedTransport(
                    httpx.AsyncHTTPTransport(proxy=_env_proxy()), _MAX_RESPONSE_BYTES
                ),
            )
        if headers_override:
            client.headers.update(headers_override)
        return client

    @asynccontextmanager
    async def _session(
        self, timeout_seconds: float, headers_override: dict[str, str] | None = None
    ) -> AsyncIterator[ClientSession]:
        """Open, initialize, and yield a session over a fresh client.

        The session-level read timeout exists because the httpx timeout only
        covers the transport: a server that accepts the request and never
        answers the protocol would otherwise hang the turn. Both budgets are
        the SAME number by design - a listing and a call each get one deadline
        covering transport and protocol together.

        Args:
            timeout_seconds: The deadline for this session. Callers pass the
                listing budget or the call budget; the two differ (see
                ``__init__``).
            headers_override: Per-call auth headers to merge over the static
                ones (a CALL's resolved per-user credential); None for a
                LISTING, which always uses the static workspace header.
        """
        async with (
            self._build_http_client(timeout_seconds, headers_override) as http_client,
            streamable_http_client(self._config.url, http_client=http_client) as (
                read,
                write,
                _get_session_id,
            ),
            ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=timeout_seconds),
            ) as session,
        ):
            await session.initialize()
            yield session

    @asynccontextmanager
    async def call_session(
        self, headers_override: dict[str, str] | None = None
    ) -> AsyncIterator[ClientSession]:
        """Open a session budgeted for one tool CALL.

        Exists so :class:`McpTool` does not have to reach into the source's
        private session helper to pick the right one of two deadlines.
        ``headers_override`` carries the caller's resolved per-call credential.
        """
        async with self._session(self._timeout_seconds, headers_override) as session:
            yield session

    async def resolve_call_headers(self, context: ToolContext) -> dict[str, str]:
        """Resolve the per-call auth headers for ``context``, or ``{}`` for none.

        The per-user credential (slice 6b) is fetched HERE, once per call, so the
        client this call builds carries the acting user's token and no one
        else's. Returns ``{}`` when no resolver is configured (the pre-6b
        behaviour) or the caller has stored no credential - either way the call
        falls back to the static workspace header.
        """
        if self._credential_resolver is None:
            return {}
        return await self._credential_resolver(context)

    async def list_tools_or_raise(self, context: ToolContext) -> list[Tool]:
        """List the server's tools, adapted to the seam - PROPAGATING failure.

        The raising variant behind :meth:`list_tools`. It exists for
        test-connection, which must tell "the server is reachable and offers no
        tools" apart from "unreachable / auth-rejected / protocol mismatch" -
        distinctions :meth:`list_tools` deliberately erases by swallowing every
        failure into an empty list. The chat path uses :meth:`list_tools`; a
        diagnostic caller uses this and classifies the exception.

        Disabled tools (the server's per-tool deny-list, ``disabled_tools`` on
        the config) are dropped here, so a listing offers only what an operator
        left switched on.

        Two bounds on the remote input are enforced here rather than upstream,
        and neither one fails the listing: a tool whose name or input schema
        blows :func:`_adapt_spec`'s limits is DROPPED (the rest of the server's
        tools stay usable), and the total is truncated at :data:`_MAX_TOOLS`.
        Both are logged once per listing, naming the server.

        Raises:
            Exception: Any transport, protocol, or listing failure. Messages from
                the SDK may wrap the cause in an exception group; callers unwrap
                with :func:`_leaf_name`, which also keeps URLs out of the text.
        """
        specs: list[ToolSpec] = []
        dropped: list[str] = []
        truncated = False
        async with self._session(self._list_timeout_seconds) as session:
            cursor: str | None = None
            for _ in range(_MAX_LIST_PAGES):
                listing = await session.list_tools(cursor)
                for remote in listing.tools:
                    if len(specs) >= _MAX_TOOLS:
                        truncated = True
                        break
                    spec = _adapt_spec(remote)
                    if spec is None:
                        dropped.append(remote.name)
                        continue
                    specs.append(spec)
                cursor = listing.nextCursor
                if truncated or cursor is None:
                    break
        if dropped:
            logger.warning(
                "MCP server %r: dropped %d tool(s) whose name or input schema is "
                "past the safe bounds (%s); its other tools are unaffected",
                self._config.name,
                len(dropped),
                # The names are remote input too - preview a few, each clipped.
                ", ".join(repr(name[:60]) for name in sorted(dropped)[:5]),
            )
        if truncated:
            logger.warning(
                "MCP server %r: listing truncated at %d tools; anything past that "
                "is not offered this turn",
                self._config.name,
                _MAX_TOOLS,
            )
        disabled = self._config.disabled_tools
        return [McpTool(spec, self) for spec in specs if spec.name not in disabled]

    async def list_tools(self, context: ToolContext) -> list[Tool]:
        """List the server's tools, adapted to the seam; empty on failure.

        A dead or misbehaving sidecar must not kill the turn, so ANY failure
        here - transport, protocol, a malformed listing - is logged once and
        reported as "no tools": the model is simply not offered this server's
        tools. Never raises.

        Args:
            context: The caller's identity and turn. A LISTING is deliberately
                caller-independent - it uses the static workspace header, which
                is what lets its cache be shared across callers - so this is
                unused here; a per-user credential is resolved per CALL, in
                :meth:`McpTool.run`.

        Returns:
            The adapted tools, or an empty list when the server is unusable.
        """
        try:
            return await self.list_tools_or_raise(context)
        except Exception as exc:
            logger.warning(
                "MCP server %r: tool listing failed (%s); offering no tools this turn",
                self._config.name,
                _leaf_name(exc),
            )
            return []


class McpTool(Tool):
    """One remote MCP tool, adapted to the seam.

    The spec is carried verbatim from the server's listing (untrusted remote
    input - see the module docstring); running it drives one MCP call.
    """

    def __init__(self, spec: ToolSpec, source: McpToolSource) -> None:
        """Bind an adapted spec to the source that can execute it."""
        self._spec = spec
        self._source = source

    @property
    def spec(self) -> ToolSpec:
        """Return the tool's advertised specification."""
        return self._spec

    async def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        """Call the remote tool once, over a dedicated client and session.

        One client per tool call is a LOCKED security rule, not an
        implementation accident: a pooled or shared client would reuse one
        user's Authorization header for another user's call, and no downstream
        guard would catch it - the right token was fetched and simply never
        reached the wire. So the caller's credential is resolved per call and
        merged into THIS call's fresh client, which is closed before this method
        returns.

        Args:
            arguments: The call arguments, matching the spec's JSON Schema.
            context: The caller's identity and turn - the sole source of the
                principal whose credential is injected into this call.

        Returns:
            The joined text content; a server-side error result comes back
            with ``is_error=True`` and its error text as content.

        Raises:
            ToolError: The call could not be made at all (transport failure,
                protocol error, timeout). The message names the server, never
                the URL.
        """
        try:
            headers = await self._source.resolve_call_headers(context)
            async with self._source.call_session(headers) as session:
                raw = await session.call_tool(self._spec.name, arguments)
        except Exception as exc:
            raise ToolError(
                f"tool {self._spec.name!r} failed on MCP server "
                f"{self._source.server_name!r} ({_leaf_name(exc)})"
            ) from exc
        return _to_tool_result(raw, server_name=self._source.server_name)


def _leaf_name(exc: BaseException) -> str:
    """Return the innermost exception type name, for URL-free messages.

    httpx exception TEXT embeds the request URL (which may embed credentials),
    so messages carry only the type name; the SDK transport runs a task group,
    so failures may arrive wrapped in exception groups - unwrap to a leaf.
    """
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return type(exc).__name__


def _truncate(text: str, limit: int) -> str:
    """Clip ``text`` to ``limit`` chars, marking that something was cut."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class _SchemaRejected(Exception):
    """Internal signal: a listed input schema blew one of the structural bounds."""


def _bounded_value(value: object, *, depth: int, counter: list[int]) -> object:
    """Copy one JSON-Schema value, clamping prose and refusing the absurd.

    Recursion is safe by construction: the depth guard fires before descending
    past :data:`_MAX_SCHEMA_DEPTH`, so a schema nested a million deep costs
    twelve frames, not a blown stack.

    Raises:
        _SchemaRejected: The value is past the node, depth, key- or
            string-length bound. Prose keys never raise - they truncate.
    """
    counter[0] += 1
    if counter[0] > _MAX_SCHEMA_NODES:
        raise _SchemaRejected("node count")
    if isinstance(value, dict):
        if depth >= _MAX_SCHEMA_DEPTH:
            raise _SchemaRejected("nesting depth")
        bounded: dict[str, object] = {}
        for key, item in value.items():
            # A key is a property NAME: the identity the model must reproduce in
            # a call, so it truncates as badly as a tool name would.
            if not isinstance(key, str) or len(key) > _MAX_SCHEMA_STRING_CHARS:
                raise _SchemaRejected("key length")
            if key in _SCHEMA_PROSE_KEYS and isinstance(item, str):
                bounded[key] = _truncate(item, _MAX_SCHEMA_STRING_CHARS)
            else:
                bounded[key] = _bounded_value(item, depth=depth + 1, counter=counter)
        return bounded
    if isinstance(value, list):
        if depth >= _MAX_SCHEMA_DEPTH:
            raise _SchemaRejected("nesting depth")
        return [_bounded_value(item, depth=depth + 1, counter=counter) for item in value]
    if isinstance(value, str) and len(value) > _MAX_SCHEMA_STRING_CHARS:
        raise _SchemaRejected("string length")
    return value


def _bounded_schema(schema: dict) -> dict | None:
    """Return a bounded copy of an advertised input schema, or None to reject it.

    The schema is prompt material - a chat template renders it into the SYSTEM
    region, next to the guidance that says only the listed tools exist - so
    nested prose is untrusted text in a MORE privileged position than a tool
    result, and it gets bounded the same way the top-level description is.

    Truncate versus reject, decided per failure: verbose PROSE is clamped (an
    honest server can be wordy, and a clipped sentence still describes the
    field), while a schema that is absurd in shape - too big, too deep, too many
    nodes, a name or a value string past the cap - loses the tool entirely. A
    tool whose schema had to be mangled to fit could not be called correctly
    anyway, so dropping it is the honest outcome and the caller logs it.

    Args:
        schema: The server's ``inputSchema`` for one tool, as parsed JSON.

    Returns:
        The bounded copy, or None when the tool must not be offered.
    """
    try:
        bounded = _bounded_value(schema, depth=0, counter=[0])
        # Measured AFTER clamping, so many small clamped descriptions still add
        # up against one budget - the cap a hostile server would otherwise reach
        # by breadth once it cannot reach it by length.
        serialized = json.dumps(bounded, ensure_ascii=False)
    except (_SchemaRejected, TypeError, ValueError, RecursionError):
        return None
    if len(serialized.encode("utf-8")) > _MAX_SCHEMA_BYTES:
        return None
    return bounded if isinstance(bounded, dict) else None


def _adapt_spec(remote: mcp_types.Tool) -> ToolSpec | None:
    """Map one MCP listing entry onto the seam's ToolSpec, or None to drop it.

    ``readOnlyHint`` is ADVISORY and a missing annotation FAILS CLOSED:
    ``side_effecting`` is False only when the server explicitly declared
    ``readOnlyHint=True``; anything else stays True so the confirmation gate
    applies. The description is length-capped and the input schema bounded
    (see :func:`_bounded_schema`) because both are untrusted remote input that
    ride in the prompt.

    Returns:
        The adapted spec, or None when the tool's name or schema is past the
        bounds and the tool must not be offered at all.
    """
    if len(remote.name) > _MAX_NAME_CHARS:
        return None
    parameters = _bounded_schema(dict(remote.inputSchema or {}))
    if parameters is None:
        return None
    annotations = remote.annotations
    read_only = annotations is not None and annotations.readOnlyHint is True
    return ToolSpec(
        name=remote.name,
        description=_truncate(remote.description or "", _MAX_DESCRIPTION_CHARS),
        parameters=parameters,
        side_effecting=not read_only,
    )


# A server could return a huge structured payload; ingestion is capped, not
# trusted. Records past this are dropped - the tail is least likely to be what
# the model asked about, and the citation budget caps it again downstream.
_MAX_RECORDS = 50

# Each record's TEXT is capped too, and this is the cap that bounds WORK rather
# than memory. A record is the one remote payload that becomes real compute: it
# is chunked, EMBEDDED and written to the session collection on the turn, on the
# same CPU-pinned embedder the chat model is competing with. Fifty records just
# under the transport cap would be tens of megabytes of embedding work from a
# single tool call, so the count cap alone does not bound it. 20 KB is far past
# any honest record (a ticket, a row, a document summary) and still leaves the
# whole batch an ordinary ingest.
_MAX_RECORD_CHARS = 20_000


def _join_capped(parts: list[str]) -> str:
    """Join result text under :data:`_MAX_RESULT_CHARS`, marking a cut honestly.

    The join is where a large-but-admitted payload would DOUBLE in memory, so it
    is budgeted rather than concatenated blindly. The marker is inside the text
    the model reads, because a silently shortened result reads as a complete one.
    """
    joined: list[str] = []
    remaining = _MAX_RESULT_CHARS
    for part in parts:
        if len(part) > remaining:
            if remaining:
                joined.append(part[:remaining])
            joined.append("...[truncated: result too large]")
            break
        joined.append(part)
        remaining -= len(part) + 1  # the newline this part will be joined with
        remaining = max(remaining, 0)
    return "\n".join(joined)


def _artifact_filename_hint(uri: object) -> str:
    """Read a filename HINT out of an embedded resource's uri, defensively.

    The uri is untrusted remote input and is treated as a hint only - the host
    re-validates whatever comes out of here before it names a stored file, and
    substitutes its own name when it cannot. All this does is take the last
    path-ish segment and bound its length, so a megabyte uri cannot travel any
    further than this function.
    """
    text = str(uri or "")
    for separator in ("/", "\\"):
        text = text.rsplit(separator, 1)[-1]
    return text.strip()[:_MAX_ARTIFACT_NAME_CHARS]


def _to_artifact(
    resource: mcp_types.BlobResourceContents, *, server_name: str
) -> ToolArtifact | None:
    """Decode one blob resource into an artifact, or None if it is unusable.

    Never raises. A server that sends malformed base64 loses THAT file and
    nothing else: the call keeps its text, its records and its other artifacts,
    because one bad attachment is not a reason to fail an answer the user is
    waiting for. The log names the server (never the URL, which may embed a
    credential), because a malformed blob is that server's bug to fix.
    """
    try:
        data = base64.b64decode(resource.blob, validate=True)
    except (binascii.Error, ValueError):
        logger.warning(
            "Dropping an artifact from MCP server %r: its blob is not valid base64", server_name
        )
        return None
    if not data:
        return None
    return ToolArtifact(
        filename=_artifact_filename_hint(resource.uri),
        content_type=resource.mimeType or _DEFAULT_ARTIFACT_CONTENT_TYPE,
        data=data,
    )


def _to_tool_result(raw: mcp_types.CallToolResult, *, server_name: str) -> ToolResult:
    """Adapt a raw MCP result into the seam type: text, records, and any files.

    The text content becomes ``content`` (what the model reads when there is
    nothing to cite), capped so the join cannot double an oversized payload.
    ``structuredContent`` matching the records contract becomes ``records``,
    which the generation loop ingests and cites; a payload that does not match is
    ignored here and the result stays text-only, so a server that does not speak
    the contract keeps the old behaviour.

    An ``EmbeddedResource`` carrying a BLOB becomes an artifact - a file the loop
    stores and announces, which the model never reads. The gate is the blob, not
    the uri scheme: the uri is a filename hint, and treating it as a filter would
    silently discard a real file from any server that spells its uris
    differently. Everything else non-text (an image, a text-valued embedded
    resource) keeps the omitted marker it has always had; a blob does NOT trigger
    that marker, because the file was kept rather than dropped.
    """
    parts: list[str] = []
    artifacts: list[ToolArtifact] = []
    omitted = False
    for item in raw.content:
        if isinstance(item, mcp_types.TextContent):
            parts.append(item.text)
        elif isinstance(item, mcp_types.EmbeddedResource) and isinstance(
            item.resource, mcp_types.BlobResourceContents
        ):
            if len(artifacts) >= _MAX_ARTIFACTS:
                omitted = True
                continue
            artifact = _to_artifact(item.resource, server_name=server_name)
            if artifact is None:
                omitted = True
                continue
            artifacts.append(artifact)
        else:
            omitted = True
    if omitted:
        parts.append("[non-text content omitted]")
    return ToolResult(
        content=_join_capped(parts),
        is_error=bool(raw.isError),
        records=_parse_records(raw.structuredContent),
        artifacts=artifacts,
    )


def _parse_records(structured: dict | None) -> list[ToolRecord]:
    """Extract citable records from a structuredContent payload, defensively.

    The contract is ``{"records": [{"id": str, "text": str, "title"?: str,
    "url"?: str}, ...]}``. Everything here is UNTRUSTED remote input, so this
    never raises: a payload that is not this shape yields no records and the
    result falls back to its text content. An item missing a string ``id`` or
    ``text`` is skipped rather than failing the batch, and the list is capped so
    a hostile server cannot flood ingestion.
    """
    if not isinstance(structured, dict):
        return []
    raw_records = structured.get("records")
    if not isinstance(raw_records, list):
        return []
    records: list[ToolRecord] = []
    for item in raw_records[:_MAX_RECORDS]:
        if not isinstance(item, dict):
            continue
        record_id = item.get("id")
        text = item.get("text")
        if not isinstance(record_id, str) or not record_id:
            continue
        if not isinstance(text, str) or not text:
            continue
        title = item.get("title")
        url = item.get("url")
        if len(text) > _MAX_RECORD_CHARS:
            # Truncated, not dropped: the head of an over-long record is still
            # the answer to what was asked far more often than not, and the
            # marker keeps the model from reading a cut record as a whole one.
            text = text[:_MAX_RECORD_CHARS] + "\n...[truncated: record too large]"
        records.append(
            ToolRecord(
                id=record_id,
                text=text,
                title=title if isinstance(title, str) else None,
                url=url if isinstance(url, str) else None,
            )
        )
    return records
