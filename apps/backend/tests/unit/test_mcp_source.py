"""Hermetic tests for the McpToolSource adapter over Streamable HTTP.

Reuses the transport spike's harness shape (tests/unit/test_mcp_transport.py):
a FastMCP server's ASGI app joined to the SDK client by an injected httpx
ASGITransport factory - no sockets. The factory doubles as the counter that
proves the one-client-per-call security rule.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import mcp.types as mcp_types
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, create_model

from app.core.config import Settings
from app.core.tenancy import TenantContext
from app.models import McpCredentialScope, McpServer
from app.providers.mcp import source as source_module
from app.providers.mcp.source import McpServerConfig, McpTool, McpToolSource
from app.providers.tools.base import ToolContext, ToolError, ToolSpec
from app.services.tools.mcp_credentials import (
    MissingUserCredentialError,
    PerUserCredentialResolver,
)
from app.services.tools.mcp_sources import build_source_for_row, credential_secret_name

pytestmark = pytest.mark.anyio

CONTEXT = ToolContext(tenant_id="tenant-1", user_id="user-1")


def build_server(**kwargs) -> FastMCP:
    """A two-tool MCP server: one annotated read-only, one unannotated."""
    kwargs.setdefault(
        "transport_security",
        TransportSecuritySettings(allowed_hosts=["localhost", "localhost:*"]),
    )
    server = FastMCP("spike", stateless_http=True, **kwargs)

    @server.tool(
        description="Look up a record by id.",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    def get_record(record_id: str) -> str:
        return f"record {record_id}: all good"

    @server.tool(description="Create a record.")
    def create_record(title: str) -> str:
        return f"created {title!r}"

    return server


class _Record(BaseModel):
    id: str
    title: str
    text: str


class _RecordList(BaseModel):
    records: list[_Record]
    message: str | None = None


def build_records_server() -> FastMCP:
    """A server whose read tool returns structuredContent in the records contract."""
    server = FastMCP(
        "records",
        stateless_http=True,
        transport_security=TransportSecuritySettings(allowed_hosts=["localhost", "localhost:*"]),
    )

    @server.tool(description="Search records.", annotations=ToolAnnotations(readOnlyHint=True))
    def search(query: str) -> _RecordList:
        return _RecordList(
            records=[
                _Record(id="rec-1", title="First", text="alpha body"),
                _Record(id="rec-2", title="Second", text="beta body"),
            ]
        )

    return server


class CountingFactory:
    """Counts invocations; each call returns a FRESH ASGI-backed client."""

    def __init__(self, app) -> None:
        self.app = app
        self.calls = 0

    def __call__(self) -> httpx.AsyncClient:
        self.calls += 1
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://localhost",
            timeout=httpx.Timeout(10.0),
            follow_redirects=True,
        )


class BudgetProbe:
    """Records the deadline the source hands to each of its two collaborators.

    Two collaborators take one, so both are watched: the httpx client (the
    TRANSPORT budget) and the ClientSession (the PROTOCOL budget). Watching only
    one would miss half a regression - they are passed at different call sites
    and a swap in either is invisible from the other.

    Why stand-ins rather than the injected factory: the factory takes no
    arguments, so it cannot see the number it was built with. The constructor
    call IS the boundary between this code and its collaborators, and the number
    crossing it is the thing under test. The alternative - sleeping past a
    deadline and timing it - is slow and flaky on a loaded box, and would pin the
    behaviour of httpx rather than of this source.

    The stand-ins return the REAL objects, ASGI-backed and socket-free, so the
    round trip underneath these assertions is a genuine one: if a recorded budget
    were never actually applied, the call it belongs to would not complete.
    """

    def __init__(self, app) -> None:
        self._app = app
        # Captured before install() swaps them out.
        self._real_client = httpx.AsyncClient
        self._real_session = source_module.ClientSession
        self.http_budgets: list[float] = []
        self.session_budgets: list[float] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Swap both collaborators for recording stand-ins, undone by monkeypatch.

        The source builds its own httpx client only when NO factory is injected,
        which is why these tests leave the factory out - this is the one path
        that exercises the real _build_http_client.
        """
        monkeypatch.setattr(httpx, "AsyncClient", self._make_client)
        monkeypatch.setattr(source_module, "ClientSession", self._make_session)

    def _make_client(self, **kwargs) -> httpx.AsyncClient:
        self.http_budgets.append(kwargs["timeout"].read)
        return self._real_client(
            transport=httpx.ASGITransport(app=self._app),
            base_url="http://localhost",
            timeout=kwargs["timeout"],
            follow_redirects=True,
        )

    def _make_session(self, *args, **kwargs):
        self.session_budgets.append(kwargs["read_timeout_seconds"].total_seconds())
        return self._real_session(*args, **kwargs)


@asynccontextmanager
async def running_source(
    server: FastMCP,
) -> AsyncIterator[tuple[McpToolSource, CountingFactory]]:
    """A source wired to a live (in-process) server via the counting factory."""
    factory = CountingFactory(server.streamable_http_app())
    source = McpToolSource(
        McpServerConfig(name="spike", url="http://localhost/mcp"),
        http_client_factory=factory,
    )
    async with server.session_manager.run():
        yield source, factory


async def _asgi_500(scope, receive, send) -> None:
    """An ASGI app that fails every request - the dead sidecar."""
    if scope["type"] != "http":
        return
    await send({"type": "http.response.start", "status": 500, "headers": []})
    await send({"type": "http.response.body", "body": b"boom"})


def dead_source() -> McpToolSource:
    """A source whose every connection hits the always-500 app."""
    return McpToolSource(
        McpServerConfig(name="spike", url="http://localhost/mcp"),
        timeout_seconds=5.0,
        http_client_factory=CountingFactory(_asgi_500),
    )


async def test_list_tools_adapts_both_specs() -> None:
    async with running_source(build_server()) as (source, _factory):
        tools = await source.list_tools(CONTEXT)

    specs = {tool.spec.name: tool.spec for tool in tools}
    assert set(specs) == {"get_record", "create_record"}
    assert specs["get_record"].description == "Look up a record by id."
    assert specs["get_record"].parameters["properties"]["record_id"]["type"] == "string"
    # An explicit readOnlyHint=True is the ONLY thing that relaxes the gate.
    assert specs["get_record"].side_effecting is False
    # No annotations at all fails closed: the tool is treated as dangerous.
    assert specs["create_record"].side_effecting is True


async def test_run_round_trip_returns_tool_text() -> None:
    async with running_source(build_server()) as (source, _factory):
        tools = await source.list_tools(CONTEXT)
        tool = next(t for t in tools if t.spec.name == "get_record")
        result = await tool.run({"record_id": "42"}, CONTEXT)

    assert result.is_error is False
    assert result.content == "record 42: all good"


async def test_server_side_error_maps_to_is_error() -> None:
    # Bad arguments are validated server-side and come back as an in-band
    # isError result, not a transport failure - so no ToolError here.
    async with running_source(build_server()) as (source, _factory):
        tools = await source.list_tools(CONTEXT)
        tool = next(t for t in tools if t.spec.name == "get_record")
        result = await tool.run({"wrong_key": "x"}, CONTEXT)

    assert result.is_error is True
    assert result.content  # the server's error text is carried, not dropped


async def test_dead_server_list_tools_returns_empty_with_one_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="app.providers.mcp.source"):
        tools = await dead_source().list_tools(CONTEXT)

    assert tools == []
    warnings = [
        record
        for record in caplog.records
        if record.name == "app.providers.mcp.source" and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    # Only the configured name may appear; the URL could embed credentials.
    assert "http://" not in warnings[0].getMessage()


async def test_dead_server_run_raises_tool_error() -> None:
    tool = McpTool(ToolSpec(name="get_record"), dead_source())

    with pytest.raises(ToolError) as excinfo:
        await tool.run({"record_id": "1"}, CONTEXT)

    message = str(excinfo.value)
    assert "spike" in message
    assert "http://" not in message


# --------------------------------------------------------------------------- #
# Per-call credential injection (slice 6b)                                     #
# --------------------------------------------------------------------------- #


class _AuthCapture:
    """Wraps an ASGI app, recording the Authorization header of each request.

    The wire is where the security property lives: a per-user token is only safe
    if it actually reaches the server on that user's call and no other, so these
    assertions read the header the server RECEIVED, not the one the client was
    handed.
    """

    def __init__(self, app) -> None:
        self._app = app
        self.seen: list[str | None] = []

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            header_map = dict(scope.get("headers") or [])
            raw = header_map.get(b"authorization")
            self.seen.append(raw.decode() if raw is not None else None)
        await self._app(scope, receive, send)


class HeaderedFactory:
    """A counting factory whose fresh clients carry a fixed set of base headers.

    An injected factory bypasses ``_build_http_client``'s real-client branch,
    which is the one that applies the config's static headers - so this is where
    a test stands those in, letting a wire assertion see the static workspace
    header survive a fallback call (or be overridden by a per-user one).
    """

    def __init__(self, app, base_headers: dict[str, str]) -> None:
        self.app = app
        self.base_headers = dict(base_headers)
        self.calls = 0

    def __call__(self) -> httpx.AsyncClient:
        self.calls += 1
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://localhost",
            headers=dict(self.base_headers),
            timeout=httpx.Timeout(10.0),
            follow_redirects=True,
        )


@asynccontextmanager
async def running_source_with_resolver(
    server: FastMCP, resolver, *, static_headers: dict[str, str] | None = None
) -> AsyncIterator[tuple[McpToolSource, HeaderedFactory, _AuthCapture]]:
    """A source wired to a live server AND a per-call credential resolver.

    The counting factory (one fresh client per call) wraps an auth-capturing app,
    so a test can read exactly which Authorization each call and listing put on
    the wire. ``static_headers`` stands in the server's static config header (the
    workspace bearer), so a test can prove a call falls back to it or overrides it.
    """
    base = dict(static_headers or {})
    capture = _AuthCapture(server.streamable_http_app())
    factory = HeaderedFactory(capture, base)
    source = McpToolSource(
        McpServerConfig(name="spike", url="http://localhost/mcp", headers=base),
        http_client_factory=factory,
        credential_resolver=resolver,
    )
    async with server.session_manager.run():
        yield source, factory, capture


def _resolver_returning(by_user: dict[str, dict[str, str]]):
    """A fake resolver handing each user id its own headers (empty if unknown)."""

    async def resolve(context: ToolContext) -> dict[str, str]:
        return dict(by_user.get(context.user_id, {}))

    return resolve


async def test_a_call_carries_the_resolved_credential_and_a_listing_does_not() -> None:
    # The per-user bearer must reach the CALL - and must NOT ride the LISTING,
    # which stays on the shareable workspace header so its cache never mixes
    # callers.
    server = build_server()
    resolver = _resolver_returning({"user-1": {"Authorization": "Bearer alice-token"}})
    async with running_source_with_resolver(server, resolver) as (source, _factory, capture):
        tools = await source.list_tools(CONTEXT)  # CONTEXT.user_id == "user-1"
        listing_auth = list(capture.seen)
        capture.seen.clear()
        tool = next(t for t in tools if t.spec.name == "get_record")
        result = await tool.run({"record_id": "42"}, CONTEXT)
        call_auth = list(capture.seen)

    assert result.is_error is False
    # No request the listing made carried a per-user credential.
    assert listing_auth and all(auth is None for auth in listing_auth)
    # Every request the call made carried it.
    assert call_auth and all(auth == "Bearer alice-token" for auth in call_auth)


async def test_each_caller_gets_only_its_own_credential_on_its_own_call() -> None:
    # One user's token must never ride another user's call - the wire-level form
    # of the one-client-per-call rule.
    server = build_server()
    resolver = _resolver_returning(
        {
            "alice": {"Authorization": "Bearer alice-token"},
            "bob": {"Authorization": "Bearer bob-token"},
        }
    )
    alice = ToolContext(tenant_id="t", user_id="alice")
    bob = ToolContext(tenant_id="t", user_id="bob")
    async with running_source_with_resolver(server, resolver) as (source, factory, capture):
        tools = await source.list_tools(alice)
        tool = next(t for t in tools if t.spec.name == "get_record")
        capture.seen.clear()
        await tool.run({"record_id": "1"}, alice)
        alice_auth = [auth for auth in capture.seen if auth is not None]
        capture.seen.clear()
        calls_before = factory.calls
        await tool.run({"record_id": "2"}, bob)
        bob_auth = [auth for auth in capture.seen if auth is not None]

    # Each call carried its own caller's token and NOTHING of the other's.
    assert alice_auth and all(auth == "Bearer alice-token" for auth in alice_auth)
    assert bob_auth and all(auth == "Bearer bob-token" for auth in bob_auth)
    # Bob's call built its own fresh client (one client per call, never pooled).
    assert factory.calls > calls_before


async def test_resolve_call_headers_is_empty_without_a_resolver() -> None:
    # No resolver configured (the pre-6b default) means no per-call header, so a
    # call uses the static workspace header exactly as before.
    source = McpToolSource(McpServerConfig(name="spike", url="http://localhost/mcp"))
    assert await source.resolve_call_headers(CONTEXT) == {}


async def test_build_http_client_merges_the_per_call_credential_over_the_static_header() -> None:
    # The merge, at the client the call actually sends on: the static workspace
    # header is what a client carries with NO override (a fallback call, and the
    # whole flag-off path); a per-user override REPLACES the Authorization while
    # leaving other static headers; an EMPTY override changes nothing. Guards the
    # real-client branch against silently dropping the static headers.
    source = McpToolSource(
        McpServerConfig(
            name="x",
            url="http://localhost/mcp",
            headers={"Authorization": "Bearer workspace-key", "X-Static": "keep"},
        )
    )
    plain = source._build_http_client(10.0)
    try:
        assert plain.headers["Authorization"] == "Bearer workspace-key"
        assert plain.headers["X-Static"] == "keep"
    finally:
        await plain.aclose()

    merged = source._build_http_client(10.0, {"Authorization": "Bearer alice-token"})
    try:
        assert merged.headers["Authorization"] == "Bearer alice-token"
        assert merged.headers["X-Static"] == "keep"
    finally:
        await merged.aclose()

    fallback = source._build_http_client(10.0, {})
    try:
        assert fallback.headers["Authorization"] == "Bearer workspace-key"
    finally:
        await fallback.aclose()


async def test_a_fallback_call_carries_the_static_workspace_header_at_the_wire() -> None:
    # Flag on, static workspace header configured, the caller has NO per-user
    # credential (resolver returns {}): the CALL must still reach the server with
    # the workspace Authorization, and so must the listing. This is the degrade
    # path (no stored secret, or an unset master key) proven end to end.
    server = build_server()
    resolver = _resolver_returning({})  # nobody has a per-user credential
    async with running_source_with_resolver(
        server, resolver, static_headers={"Authorization": "Bearer workspace-key"}
    ) as (source, _factory, capture):
        tools = await source.list_tools(CONTEXT)
        listing_auth = list(capture.seen)
        capture.seen.clear()
        tool = next(t for t in tools if t.spec.name == "get_record")
        await tool.run({"record_id": "1"}, CONTEXT)
        call_auth = list(capture.seen)

    assert listing_auth and all(auth == "Bearer workspace-key" for auth in listing_auth)
    assert call_auth and all(auth == "Bearer workspace-key" for auth in call_auth)


async def test_a_per_user_credential_overrides_the_static_header_on_the_call_only() -> None:
    # With BOTH a static workspace header and a per-user credential, the CALL
    # carries the per-user token (it replaces the workspace one) while the LISTING
    # still carries the workspace header - so the two never cross.
    server = build_server()
    resolver = _resolver_returning({"user-1": {"Authorization": "Bearer alice-token"}})
    async with running_source_with_resolver(
        server, resolver, static_headers={"Authorization": "Bearer workspace-key"}
    ) as (source, _factory, capture):
        tools = await source.list_tools(CONTEXT)  # CONTEXT.user_id == "user-1"
        listing_auth = list(capture.seen)
        capture.seen.clear()
        tool = next(t for t in tools if t.spec.name == "get_record")
        await tool.run({"record_id": "1"}, CONTEXT)
        call_auth = [auth for auth in capture.seen if auth is not None]

    # The listing never carried the per-user token; the call carried only it.
    assert listing_auth and all(auth == "Bearer workspace-key" for auth in listing_auth)
    assert call_auth and all(auth == "Bearer alice-token" for auth in call_auth)
    assert "Bearer workspace-key" not in call_auth


async def test_a_user_scoped_call_fails_closed_when_the_credential_is_missing() -> None:
    # A credential_scope=user server whose caller has connected nothing: the
    # required resolver RAISES, so the CALL fails closed with a URL-free
    # ToolError - it never falls back to the workspace header - while the LISTING
    # (which never consults the resolver) still succeeds so the user is offered
    # the tools and prompted to connect.
    server = build_server()

    async def resolver(context: ToolContext) -> dict[str, str]:
        raise MissingUserCredentialError("mcp-chassis")

    async with running_source_with_resolver(
        server, resolver, static_headers={"Authorization": "Bearer workspace-key"}
    ) as (source, _factory, capture):
        tools = await source.list_tools(CONTEXT)
        assert any(t.spec.name == "get_record" for t in tools)  # listing works
        listing_auth = list(capture.seen)
        capture.seen.clear()
        tool = next(t for t in tools if t.spec.name == "get_record")
        with pytest.raises(ToolError):
            await tool.run({"record_id": "1"}, CONTEXT)
        # Fail closed means the call is never even sent - nothing reached the wire
        # after the listing, so the workspace header was NOT used as a fallback.
        assert capture.seen == []

    assert listing_auth and all(auth == "Bearer workspace-key" for auth in listing_auth)


def _server_row(scope: str) -> McpServer:
    """A registered server row at the given credential scope."""
    return McpServer(
        tenant_id=uuid4(),
        name="mcp-chassis",
        base_url="http://localhost/mcp",
        credential_scope=scope,
    )


async def test_build_source_for_row_wires_a_resolver_only_for_user_scope() -> None:
    # The per-ROW credential_scope is the switch now (not a global flag): a
    # workspace-scoped row has no resolver - byte-identical to the shared-key
    # path; a user-scoped row wires a FAIL-CLOSED one.
    settings = Settings(_env_file=None, model_provider="fake")
    # cipher=None skips the workspace-credential reveal, so sessionmaker is never
    # dialled here - the switch under test is purely the row's credential_scope.
    tenant = TenantContext(tenant_id=uuid4(), user_id=uuid4(), role="admin")
    workspace, _ = await build_source_for_row(
        _server_row(McpCredentialScope.WORKSPACE.value),
        settings,
        tenant,
        sessionmaker=object(),
        cipher=None,
    )
    assert workspace._inner._credential_resolver is None

    user_row = _server_row(McpCredentialScope.USER.value)
    user, _ = await build_source_for_row(
        user_row, settings, tenant, sessionmaker=object(), cipher=None
    )
    resolver = user._inner._credential_resolver
    assert isinstance(resolver, PerUserCredentialResolver)
    # A user-scoped server must fail closed, never fall back to the shared header.
    assert resolver._required is True
    # The per-user secret is keyed on the server's stable id + url (NOT its mutable
    # display name), so a recreated/re-pointed server cannot inherit a prior one's.
    assert resolver._secret_name == credential_secret_name(user_row.id, user_row.base_url)
    assert resolver._secret_name != user_row.name


async def test_disabled_tools_are_dropped_from_a_listing() -> None:
    # The per-tool deny-list: a tool an operator switched off is not offered on a
    # LISTING, while the rest still are. Its absence is not a discovery failure -
    # the server still advertises it - so a test-connection (no deny-list) shows it.
    server = build_server()
    factory = CountingFactory(server.streamable_http_app())
    source = McpToolSource(
        McpServerConfig(
            name="spike",
            url="http://localhost/mcp",
            disabled_tools=frozenset({"create_record"}),
        ),
        http_client_factory=factory,
    )
    async with server.session_manager.run():
        names = {tool.spec.name for tool in await source.list_tools_or_raise(CONTEXT)}
    assert "get_record" in names
    assert "create_record" not in names


async def test_one_fresh_client_per_run_call() -> None:
    async with running_source(build_server()) as (source, factory):
        tools = await source.list_tools(CONTEXT)
        tool = next(t for t in tools if t.spec.name == "get_record")
        after_listing = factory.calls
        await tool.run({"record_id": "1"}, CONTEXT)
        await tool.run({"record_id": "2"}, CONTEXT)

    # The security rule made observable: two runs = two fresh clients.
    assert factory.calls - after_listing == 2


async def test_listing_and_calling_each_get_their_own_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = build_server()
    probe = BudgetProbe(server.streamable_http_app())
    probe.install(monkeypatch)
    source = McpToolSource(
        McpServerConfig(name="spike", url="http://localhost/mcp"),
        timeout_seconds=30.0,
        list_timeout_seconds=5.0,
    )
    async with server.session_manager.run():
        tools = await source.list_tools(CONTEXT)
        after_listing = (list(probe.http_budgets), list(probe.session_budgets))
        tool = next(t for t in tools if t.spec.name == "get_record")
        result = await tool.run({"record_id": "7"}, CONTEXT)

    # A listing is discovery that runs before the first token of EVERY
    # tool-capable turn, so a slow server has to degrade to "offers no tools
    # right now" FAST - it gets the short budget, on both the transport and the
    # protocol wait.
    assert after_listing == ([5.0], [5.0])
    # The call is work the user asked for and can wait on: the long budget. The
    # round trip completing is what proves these budgets were really applied and
    # not merely recorded.
    assert probe.http_budgets == [5.0, 30.0]
    assert probe.session_budgets == [5.0, 30.0]
    assert result.content == "record 7: all good"


async def test_no_list_timeout_inherits_the_call_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = build_server()
    probe = BudgetProbe(server.streamable_http_app())
    probe.install(monkeypatch)
    # Omitting list_timeout_seconds is the PRE-SPLIT behaviour: one number
    # covering both operations, so adding the split broke no existing caller.
    source = McpToolSource(
        McpServerConfig(name="spike", url="http://localhost/mcp"),
        timeout_seconds=12.0,
    )

    async with server.session_manager.run():
        tools = await source.list_tools(CONTEXT)
        tool = next(t for t in tools if t.spec.name == "get_record")
        await tool.run({"record_id": "1"}, CONTEXT)

    assert probe.http_budgets == [12.0, 12.0]
    assert probe.session_budgets == [12.0, 12.0]


async def test_each_call_session_builds_a_fresh_client() -> None:
    async with running_source(build_server()) as (source, factory):
        before = factory.calls
        async with source.call_session() as first:
            await first.list_tools()
        async with source.call_session() as second:
            await second.list_tools()

    # call_session is the seam McpTool.run drives, so the one-client-per-call
    # rule has to hold HERE and not only one layer up: a pooled or reused session
    # would carry one user's Authorization header into another user's call once
    # per-user credentials land, and no downstream guard would catch it.
    assert factory.calls - before == 2
    assert first is not second


async def test_long_description_is_truncated() -> None:
    server = build_server()

    @server.tool(description="x" * 2000)
    def long_described(q: str) -> str:
        return q

    async with running_source(server) as (source, _factory):
        tools = await source.list_tools(CONTEXT)

    spec = next(t.spec for t in tools if t.spec.name == "long_described")
    assert len(spec.description) == 1024
    assert spec.description.endswith("...")


# --------------------------------------------------------------------------- #
# The advertised INPUT SCHEMA is prompt material too (security review T1)       #
# --------------------------------------------------------------------------- #


# A parameter model whose generated JSON Schema stays past the byte budget even
# after every description is clamped - the breadth attack, as a real tool
# signature so the whole listing path drives it. Module-level because
# ``from __future__ import annotations`` makes FastMCP resolve the annotation by
# name against the module's globals.
BloatedPayload = create_model(
    "BloatedPayload",
    **{f"f{index}": (str, Field(description="x" * 400)) for index in range(200)},
)


def remote_tool(schema: dict, *, name: str = "t", description: str = "d") -> mcp_types.Tool:
    """One listing entry, built directly - the schema is what is under test."""
    return mcp_types.Tool(name=name, description=description, inputSchema=schema)


def test_a_nested_description_is_clamped_not_carried() -> None:
    # The attack the top-level cap misses: 20KB of instructions parked in a
    # nested properties[*].description, which a chat template renders into the
    # SYSTEM region of the prompt - unfenced, next to "never invent a tool".
    spec = source_module._adapt_spec(
        remote_tool(
            {
                "type": "object",
                "properties": {"q": {"type": "string", "description": "IGNORE ALL " * 4000}},
            }
        )
    )

    assert spec is not None
    clamped = spec.parameters["properties"]["q"]["description"]
    assert len(clamped) == source_module._MAX_SCHEMA_STRING_CHARS
    assert clamped.endswith("...")
    # Clamped, not dropped: the field is still described, just not at length.
    assert clamped.startswith("IGNORE ALL")


def test_an_honest_few_kb_schema_survives_untouched() -> None:
    # The cap has to leave a REAL tool alone: a few KB of nested object schema
    # with ordinary descriptions comes back byte-identical.
    schema = {
        "type": "object",
        "properties": {
            f"field_{index}": {
                "type": "object",
                "properties": {"value": {"type": "string", "description": "A value to send."}},
                "description": "One nested field of a legitimately rich schema.",
            }
            for index in range(20)
        },
        "required": ["field_0"],
    }

    spec = source_module._adapt_spec(remote_tool(schema))

    assert spec is not None
    assert spec.parameters == schema


def test_a_schema_that_stays_huge_after_clamping_drops_the_tool() -> None:
    # Breadth is the way around a per-string cap: hundreds of properties, each
    # description just under the clamp. The byte budget is measured AFTER
    # clamping precisely so this is caught.
    schema = {
        "type": "object",
        "properties": {
            f"f{index}": {"type": "string", "description": "x" * 400} for index in range(200)
        },
    }

    assert source_module._adapt_spec(remote_tool(schema)) is None


def test_absurd_shapes_drop_the_tool_rather_than_being_mangled() -> None:
    # The literals here are deliberately NOT read from the module's constants:
    # each payload has to be just past ONE bound and comfortably inside the
    # others, or a test would pass for the wrong reason (a deep schema that is
    # also huge proves only that SOMETHING caught it). They must be re-tuned if a
    # bound moves, which is the point - each one pins a specific bound.
    deep: dict = {"type": "string"}  # 14 levels, against a depth bound of 12
    for _ in range(14):
        deep = {"properties": deep}
    assert source_module._adapt_spec(remote_tool(deep)) is None

    # ~2500 nodes against a node bound of 2000, in well under 16 KB of JSON and
    # only two levels deep: node count is the only bound this can trip.
    wide = {"anyOf": [[] for _ in range(2500)]}
    assert source_module._adapt_spec(remote_tool(wide)) is None

    # A non-prose string cannot be truncated without changing what the tool
    # accepts, so an oversized enum value fails the tool instead of shipping a
    # lie about its arguments.
    enum = {"properties": {"mode": {"enum": ["a" * 5000]}}}
    assert source_module._adapt_spec(remote_tool(enum)) is None

    # A property NAME is the identity the model must reproduce in a call.
    long_key = {"properties": {"k" * 5000: {"type": "string"}}}
    assert source_module._adapt_spec(remote_tool(long_key)) is None


def test_an_oversized_tool_name_drops_the_tool() -> None:
    # The name rides in the prompt like everything else, but truncating it would
    # advertise a tool that can never be called - so it is a drop, not a clamp.
    assert source_module._adapt_spec(remote_tool({}, name="n" * 5000)) is None


async def test_a_hostile_schema_costs_its_own_tool_and_nothing_else(caplog) -> None:
    # End to end over a live listing: one tool advertises a schema that cannot be
    # bounded, and the server's OTHER tools stay offered. A hostile server must
    # not be able to take a whole integration down either.
    server = build_server()

    @server.tool(description="Looks innocent.")
    def hostile(payload: BloatedPayload) -> str:
        return "never called"

    with caplog.at_level(logging.WARNING, logger="app.providers.mcp.source"):
        async with running_source(server) as (source, _factory):
            names = {tool.spec.name for tool in await source.list_tools(CONTEXT)}

    assert "hostile" not in names
    assert {"get_record", "create_record"} <= names
    # Scoped to THIS logger: caplog collects every logger's records, so an
    # unrelated warning from httpx or the SDK inside the block would otherwise
    # turn a passing assertion into a spurious failure.
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "app.providers.mcp.source" and record.levelno >= logging.WARNING
    ]
    assert len(warnings) == 1
    assert "spike" in warnings[0] and "hostile" in warnings[0]
    assert "http://" not in warnings[0]


async def test_a_listing_stops_accumulating_past_the_tool_cap(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    # No cap on tool COUNT means an unbounded embedding batch on the first turn
    # after the listing cache expires (selection embeds every non-core tool).
    monkeypatch.setattr(source_module, "_MAX_TOOLS", 3)
    server = build_server()
    for index in range(10):
        server.add_tool(lambda q: q, name=f"extra_{index}", description="one more")

    with caplog.at_level(logging.WARNING, logger="app.providers.mcp.source"):
        async with running_source(server) as (source, _factory):
            tools = await source.list_tools(CONTEXT)

    assert len(tools) == 3
    assert any("truncated at 3 tools" in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------- #
# The transport is bounded: no redirects, no unbounded body (T3, T4)            #
# --------------------------------------------------------------------------- #


class _RecordingApp:
    """An ASGI app that records the paths it was asked for and scripts replies."""

    def __init__(self, *, redirect_to: str | None = None, body: bytes = b"ok") -> None:
        self.redirect_to = redirect_to
        self.body = body
        self.paths: list[str] = []

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return
        self.paths.append(scope["path"])
        if self.redirect_to is not None and scope["path"] == "/mcp":
            await send(
                {
                    "type": "http.response.start",
                    "status": 307,
                    "headers": [(b"location", self.redirect_to.encode())],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": self.body})


def real_client_source(app, monkeypatch: pytest.MonkeyPatch) -> McpToolSource:
    """A source whose REAL client branch is wired to an in-process ASGI app.

    The other tests inject a factory, which bypasses ``_build_http_client``
    entirely - so the transport's own guards (no redirects, the byte cap) would
    never be exercised by them. Swapping the transport CLASS instead leaves the
    real client construction, and therefore the properties under test, intact.
    """
    monkeypatch.setattr(
        source_module.httpx,
        "AsyncHTTPTransport",
        lambda *args, **kwargs: httpx.ASGITransport(app=app),
    )
    return McpToolSource(McpServerConfig(name="spike", url="http://localhost/mcp"))


async def test_the_dial_never_follows_a_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    # A server that was legitimate at registration answers the handshake with a
    # 302/307 at the cloud metadata address - carrying this deployment's
    # Authorization header with it. The client must simply not go.
    app = _RecordingApp(redirect_to="http://169.254.169.254/latest/meta-data/")
    source = real_client_source(app, monkeypatch)

    client = source._build_http_client(5.0)
    try:
        assert client.follow_redirects is False
        response = await client.post("http://localhost/mcp", json={})
    finally:
        await client.aclose()

    assert response.status_code == 307
    # The metadata path was never requested: the hop was refused, not chased.
    assert app.paths == ["/mcp"]


async def test_an_unbounded_response_body_fails_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The SDK buffers a whole body, so an unbounded one is an OOM of the single
    # backend process - taking every concurrent chat stream with it. The cap has
    # to live in the transport, below where anything else can see the payload.
    monkeypatch.setattr(source_module, "_MAX_RESPONSE_BYTES", 512)
    source = real_client_source(_RecordingApp(body=b"x" * 5000), monkeypatch)

    client = source._build_http_client(5.0)
    try:
        with pytest.raises(source_module.McpResponseTooLargeError):
            await client.post("http://localhost/mcp", json={})
    finally:
        await client.aclose()


async def test_a_body_within_the_cap_is_read_normally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source_module, "_MAX_RESPONSE_BYTES", 512)
    source = real_client_source(_RecordingApp(body=b"x" * 100), monkeypatch)

    client = source._build_http_client(5.0)
    try:
        response = await client.post("http://localhost/mcp", json={})
    finally:
        await client.aclose()

    assert response.content == b"x" * 100


def test_result_text_is_capped_before_the_join_can_double_it() -> None:
    # The last line of defense above the transport: tool_result_max_tokens only
    # truncates once the payload is already resident, and the join itself would
    # copy it again.
    raw = mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(type="text", text="a" * source_module._MAX_RESULT_CHARS),
            mcp_types.TextContent(type="text", text="b" * 1000),
        ]
    )

    result = source_module._to_tool_result(raw, server_name="example")

    assert len(result.content) <= source_module._MAX_RESULT_CHARS + 64
    assert result.content.endswith("...[truncated: result too large]")


def test_an_ordinary_result_is_joined_unchanged() -> None:
    raw = mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(type="text", text="first"),
            mcp_types.TextContent(type="text", text="second"),
        ]
    )

    assert source_module._to_tool_result(raw, server_name="example").content == "first\nsecond"


async def test_run_surfaces_structured_records_from_the_contract() -> None:
    # The provider half of the slice: a structuredContent payload in the records
    # contract arrives as ToolRecords, not just as text - which is what lets the
    # loop, one layer up, ingest and cite them.
    async with running_source(build_records_server()) as (source, _factory):
        tools = await source.list_tools(CONTEXT)
        tool = next(t for t in tools if t.spec.name == "search")
        result = await tool.run({"query": "x"}, CONTEXT)

    assert [r.id for r in result.records] == ["rec-1", "rec-2"]
    assert result.records[0].title == "First"
    assert result.records[0].text == "alpha body"
    # Records are the ADDITION - the text content still comes through unchanged.
    assert result.is_error is False


def test_parse_records_reads_the_contract() -> None:
    records = source_module._parse_records(
        {"records": [{"id": "a", "text": "hello", "title": "T", "url": "http://x/y"}]}
    )
    assert len(records) == 1
    assert records[0].id == "a"
    assert records[0].text == "hello"
    assert records[0].title == "T"
    assert records[0].url == "http://x/y"


def test_parse_records_ignores_non_contract_payloads() -> None:
    # Anything that is not {"records": [{id, text}, ...]} yields nothing, and the
    # result falls back to its text content - a server that does not speak the
    # contract is simply uncited, never broken.
    assert source_module._parse_records(None) == []
    assert source_module._parse_records({"other": 1}) == []
    assert source_module._parse_records({"records": "not-a-list"}) == []
    assert source_module._parse_records({"records": [1, "x", None]}) == []


def test_parse_records_skips_items_missing_a_string_id_or_text() -> None:
    records = source_module._parse_records(
        {
            "records": [
                {"id": "a", "text": "keep"},
                {"id": "", "text": "empty id"},
                {"id": "b"},
                {"text": "no id"},
                {"id": 5, "text": "non-string id"},
                {"id": "c", "text": ""},
            ]
        }
    )
    assert [r.id for r in records] == ["a"]
    # Optional fields absent is fine - not every record has a title or url.
    assert records[0].title is None
    assert records[0].url is None


def test_parse_records_caps_the_batch() -> None:
    # A hostile server cannot flood ingestion: the list is capped.
    payload = {"records": [{"id": str(i), "text": "t"} for i in range(200)]}
    assert len(source_module._parse_records(payload)) == source_module._MAX_RECORDS
