"""The Integrations test-connection service (slice 6b PR3).

Drives ``test_connection`` with an injected fake source - no network. Pins the
two things that matter: the handshake maps a server's advertised tools, and it
NEVER runs one (the fake tools raise if called); and the error taxonomy maps a
failure to a UI code with a URL-free detail.
"""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest

from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext
from app.models import McpAuthType, McpCredentialScope, McpServer
from app.providers.secrets import SecretNotFoundError, SecretScope
from app.providers.tools.base import ToolContext, ToolResult, ToolSpec
from app.schemas.integrations import ConnectionTestRequest, ConnectionTestResult
from app.services import integrations
from app.services.tools.mcp_sources import credential_secret_name

pytestmark = pytest.mark.anyio

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="admin")
SETTINGS = Settings(_env_file=None, model_provider="fake")
REQUEST = ConnectionTestRequest(base_url="http://server/mcp")


class _FakeTool:
    """A tool shaped like an adapted MCP one; running it is a test failure."""

    def __init__(self, name: str, description: str, side_effecting: bool) -> None:
        self.spec = ToolSpec(
            name=name, description=description, parameters={}, side_effecting=side_effecting
        )

    async def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        raise AssertionError("test-connection must never run a tool")


class _FakeSource:
    """A stand-in McpToolSource whose listing is scripted."""

    def __init__(
        self, *, tools: list[_FakeTool] | None = None, error: Exception | None = None
    ) -> None:
        self._tools = tools or []
        self._error = error
        self.listed = False

    async def list_tools_or_raise(self, context: ToolContext) -> list[_FakeTool]:
        self.listed = True
        if self._error is not None:
            raise self._error
        return self._tools


async def _run(source: _FakeSource):
    return await integrations.test_connection(SETTINGS, TENANT, REQUEST, source=source)


async def test_success_maps_the_advertised_tools() -> None:
    source = _FakeSource(
        tools=[
            _FakeTool("search_records", "Search the records.", False),
            _FakeTool("create_record", "Create a record.", True),
        ]
    )
    result = await _run(source)
    assert result.ok
    assert result.error is None
    assert source.listed
    assert [(t.name, t.side_effecting) for t in result.tools] == [
        ("search_records", False),
        ("create_record", True),
    ]


async def test_never_runs_a_tool() -> None:
    # The fake tool's run() raises; a clean success proves the handshake only
    # LISTS - it reads each tool's spec and never invokes it.
    result = await _run(_FakeSource(tools=[_FakeTool("create_record", "Create", True)]))
    assert result.ok


async def test_unreachable_is_classified() -> None:
    result = await _run(_FakeSource(error=httpx.ConnectError("connection refused")))
    assert not result.ok
    assert result.error is not None
    assert result.error.code == "unreachable"
    # The detail is the leaf TYPE name only - never the exception text, which
    # embeds the request URL (which may embed a credential).
    assert result.error.detail == "ConnectError"


async def test_timeout_is_classified() -> None:
    result = await _run(_FakeSource(error=httpx.ReadTimeout("too slow")))
    assert result.error is not None
    assert result.error.code == "timeout"


async def test_auth_rejected_is_classified() -> None:
    request = httpx.Request("POST", "http://server/mcp")
    response = httpx.Response(401, request=request)
    error = httpx.HTTPStatusError("unauthorized", request=request, response=response)
    result = await _run(_FakeSource(error=error))
    assert result.error is not None
    assert result.error.code == "auth_rejected"
    # A status failure surfaces the actionable status code, never the
    # exception's message (which embeds the URL, which may embed a credential).
    assert result.error.detail == "HTTP 401"


async def test_protocol_mismatch_is_classified() -> None:
    result = await _run(_FakeSource(error=ValueError("not valid JSON-RPC")))
    assert result.error is not None
    assert result.error.code == "protocol_mismatch"


async def test_an_unknown_error_is_classified_with_a_next_step() -> None:
    result = await _run(_FakeSource(error=RuntimeError("something else")))
    assert result.error is not None
    assert result.error.code == "unknown"
    # The leaf type name plus a next step: the MCP SDK wraps a mid-handshake
    # death in its own error type, and a bare "McpError" explained nothing to
    # the person who hit it (the server was merely cold and needed a retry).
    # Still no message text - it may embed a URL, which may embed a credential.
    assert result.error.detail.startswith("RuntimeError")
    assert "handshake" in result.error.detail
    assert "something else" not in result.error.detail


async def test_probe_borrows_the_call_budget_for_its_listing(monkeypatch) -> None:
    # The turn's listing budget (5s) fails fast because discovery there is
    # latency nobody asked for; this probe IS the ask - a human clicked Test -
    # so a cold upstream that needs seconds per request must not be failed by
    # the turn's deadline. Hand the constructor mcp_list_timeout_seconds again
    # and this fails.
    captured: dict = {}

    class _SpySource:
        def __init__(self, config, *, timeout_seconds: float, list_timeout_seconds: float) -> None:
            captured["timeout_seconds"] = timeout_seconds
            captured["list_timeout_seconds"] = list_timeout_seconds

        async def list_tools_or_raise(self, context: ToolContext) -> list:
            return []

    monkeypatch.setattr(integrations, "McpToolSource", _SpySource)
    result = await integrations.test_connection(SETTINGS, TENANT, REQUEST)
    assert result.ok
    assert captured["list_timeout_seconds"] == SETTINGS.mcp_timeout_seconds
    assert captured["list_timeout_seconds"] > SETTINGS.mcp_list_timeout_seconds


async def test_classification_unwraps_an_exception_group() -> None:
    # The SDK transport runs a task group, so failures may arrive wrapped.
    group = ExceptionGroup("transport", [httpx.ConnectError("refused")])
    result = await _run(_FakeSource(error=group))
    assert result.error is not None
    assert result.error.code == "unreachable"


# --- the header rule: an env bootstrap key backs only its own sidecar's URL ----


def test_env_key_backs_only_the_chassis_url() -> None:
    settings = Settings(
        _env_file=None,
        model_provider="fake",
        mcp_chassis_url="http://chassis/mcp",
        mcp_chassis_api_key="env-key",
    )
    # The chassis URL: the env bootstrap key backs the probe.
    on = integrations._auth_header(settings, base_url="http://chassis/mcp", credential=None)
    assert on == {"Authorization": "Bearer env-key"}
    # A DIFFERENT URL: no header - the bootstrap key must not reach an unrelated
    # host typed into the test box (the leak PR #97 closed on the chat path).
    off = integrations._auth_header(settings, base_url="http://other/mcp", credential=None)
    assert off == {}


def test_a_one_shot_credential_beats_the_env_key() -> None:
    settings = Settings(
        _env_file=None,
        model_provider="fake",
        mcp_chassis_url="http://chassis/mcp",
        mcp_chassis_api_key="env-key",
    )
    headers = integrations._auth_header(settings, base_url="http://chassis/mcp", credential="typed")
    assert headers == {"Authorization": "Bearer typed"}


def test_every_env_backed_sidecar_gets_its_key_not_just_the_chassis() -> None:
    # A probe that authenticates differently from a real call is the exact
    # divergence this endpoint exists to close, and the doc-gen key was invisible
    # here: the card reported "deployment_key" for the row while Test connection
    # dialled it anonymously and came back auth_rejected. One shared resolver now
    # answers for every configured sidecar.
    settings = Settings(
        _env_file=None,
        model_provider="fake",
        mcp_chassis_url="http://chassis/mcp",
        mcp_chassis_api_key="env-key",
        docgen_mcp_url="http://doc-gen:8765/docgen/mcp",
        docgen_api_key="docgen-key",
    )
    docgen = integrations._auth_header(
        settings, base_url="http://doc-gen:8765/docgen/mcp", credential=None
    )
    assert docgen == {"Authorization": "Bearer docgen-key"}
    # Each key still belongs to ONE url: neither sidecar inherits the other's.
    chassis = integrations._auth_header(settings, base_url="http://chassis/mcp", credential=None)
    assert chassis == {"Authorization": "Bearer env-key"}
    assert integrations._auth_header(settings, base_url="http://other/mcp", credential=None) == {}


# --- test_saved_connection: probe a saved server as the chat path would --------


class _FakeStore:
    """A SecretStore stand-in whose reveal is scripted; records its calls."""

    def __init__(self, *, revealed: str | None = None, error: Exception | None = None) -> None:
        self._revealed = revealed
        self._error = error
        self.reveal_calls: list[tuple] = []

    async def reveal(self, name: str, *, scope: SecretScope) -> str | None:
        self.reveal_calls.append((name, scope))
        if self._error is not None:
            raise self._error
        return self._revealed


def _server(
    *,
    scope: str = McpCredentialScope.WORKSPACE.value,
    base_url: str = "http://crm/mcp",
    auth_type: str = McpAuthType.HEADER.value,
) -> McpServer:
    return McpServer(
        tenant_id=DEFAULT_TENANT_ID,
        name="crm",
        base_url=base_url,
        credential_scope=scope,
        auth_type=auth_type,
    )


def _spy_probe(monkeypatch, captured: dict) -> None:
    """Replace the handshake with a spy recording the URL and headers it would send."""

    async def spy(settings, tenant, *, base_url, headers, source=None):
        captured["base_url"] = base_url
        captured["headers"] = headers
        return ConnectionTestResult(ok=True)

    monkeypatch.setattr(integrations, "_probe", spy)


async def test_saved_connection_none_auth_probes_credential_less(monkeypatch) -> None:
    # The row declares an unauthenticated upstream, so the probe must mirror the
    # chat path and attach NOTHING - not the typed override, not the stored
    # credential (no reveal even happens), not the env bootstrap. Remove the
    # auth_type guard in test_saved_connection and this fails with the override
    # appearing as a bearer header.
    captured: dict = {}
    _spy_probe(monkeypatch, captured)
    store = _FakeStore(revealed="stored-key")
    server = _server(auth_type=McpAuthType.NONE.value)

    result = await integrations.test_saved_connection(
        SETTINGS, TENANT, server, store, override_credential="typed-anyway"
    )

    assert result.ok
    assert captured["headers"] == {}
    assert store.reveal_calls == []


async def test_saved_connection_oauth_probes_with_the_grant_bearer(monkeypatch) -> None:
    # An oauth row tests with the SAME token a chat turn would send - resolved
    # by the caller from the grant (renewed if stale) and passed in. The
    # stored header credential and the typed override stay out of it entirely.
    captured: dict = {}
    _spy_probe(monkeypatch, captured)
    store = _FakeStore(revealed="stored-key")
    server = _server(auth_type=McpAuthType.OAUTH.value)

    result = await integrations.test_saved_connection(
        SETTINGS,
        TENANT,
        server,
        store,
        override_credential="typed-anyway",
        oauth_bearer="at-live",
    )

    assert result.ok
    assert captured["headers"] == {"Authorization": "Bearer at-live"}
    assert store.reveal_calls == []


async def test_saved_connection_oauth_without_a_grant_probes_bare(monkeypatch) -> None:
    # Nothing usable connected: the probe goes out bare, exactly as a chat
    # call would, and fails honestly upstream if the server demands auth.
    captured: dict = {}
    _spy_probe(monkeypatch, captured)
    server = _server(auth_type=McpAuthType.OAUTH.value)

    await integrations.test_saved_connection(
        SETTINGS, TENANT, server, _FakeStore(), override_credential=None
    )

    assert captured["headers"] == {}


async def test_saved_connection_uses_the_stored_credential(monkeypatch) -> None:
    captured: dict = {}
    _spy_probe(monkeypatch, captured)
    store = _FakeStore(revealed="stored-key")
    server = _server(base_url="http://crm/mcp")

    result = await integrations.test_saved_connection(
        SETTINGS, TENANT, server, store, override_credential=None
    )

    assert result.ok
    assert captured["headers"] == {"Authorization": "Bearer stored-key"}
    assert captured["base_url"] == "http://crm/mcp"
    # Revealed under the server's own (workspace) scope, keyed on its id + url.
    assert store.reveal_calls == [
        (credential_secret_name(server.id, server.base_url), SecretScope.WORKSPACE)
    ]


async def test_saved_connection_override_beats_the_stored_credential(monkeypatch) -> None:
    captured: dict = {}
    _spy_probe(monkeypatch, captured)
    store = _FakeStore(revealed="stored-key")

    await integrations.test_saved_connection(
        SETTINGS, TENANT, _server(), store, override_credential="typed-new"
    )

    assert captured["headers"] == {"Authorization": "Bearer typed-new"}
    # The override short-circuits the reveal entirely - no SecretStore hit.
    assert store.reveal_calls == []


async def test_saved_connection_user_scope_reveals_under_user_scope(monkeypatch) -> None:
    captured: dict = {}
    _spy_probe(monkeypatch, captured)
    store = _FakeStore(revealed="user-key")
    server = _server(scope=McpCredentialScope.USER.value)

    await integrations.test_saved_connection(
        SETTINGS, TENANT, server, store, override_credential=None
    )

    assert captured["headers"] == {"Authorization": "Bearer user-key"}
    assert store.reveal_calls == [
        (credential_secret_name(server.id, server.base_url), SecretScope.USER)
    ]


async def test_saved_connection_probes_without_a_credential_when_none_available(
    monkeypatch,
) -> None:
    # A not-connected server (or an unset master key): the reveal fails, so the
    # probe carries no header rather than the test blowing up - and it must NOT
    # fall back to the env chassis key for this non-chassis URL.
    captured: dict = {}
    _spy_probe(monkeypatch, captured)
    store = _FakeStore(error=SecretNotFoundError("nope"))

    await integrations.test_saved_connection(
        SETTINGS, TENANT, _server(), store, override_credential=None
    )

    assert captured["headers"] == {}


async def test_saved_connection_does_not_revalidate_the_stored_url(monkeypatch) -> None:
    # A seeded row can carry a base_url that bypassed schema validation
    # (ensure_seed writes MCP_CHASSIS_URL raw). Probing it must NOT re-validate and
    # raise (a 500 that would break the always-200 contract) - it passes the URL
    # through raw, like the chat path, and degrades to a dial result.
    captured: dict = {}
    _spy_probe(monkeypatch, captured)
    server = _server(base_url="not-a-valid-url")  # would fail _validate_base_url
    store = _FakeStore(error=SecretNotFoundError("nope"))

    # No ValidationError: the call completes and the raw URL reaches the probe.
    await integrations.test_saved_connection(
        SETTINGS, TENANT, server, store, override_credential=None
    )

    assert captured["base_url"] == "not-a-valid-url"


async def test_a_redirect_is_named_as_a_redirect_not_a_dead_server() -> None:
    # The dial refuses to follow redirects on purpose - a 302 is a server
    # re-pointing a credential-bearing call somewhere nobody approved. That
    # makes the DIAGNOSTIC the whole mitigation: an admin whose vendor endpoint
    # moved needs to be told to register the new URL, not left reading the same
    # "unreachable" an offline host produces. The hint carries no URL of its
    # own; the Location header is the remote side's text.
    request = httpx.Request("POST", "http://server/mcp")
    response = httpx.Response(307, headers={"location": "http://elsewhere/mcp"}, request=request)
    error = httpx.HTTPStatusError("redirect", request=request, response=response)
    result = await _run(_FakeSource(error=error))
    assert result.error is not None
    assert result.error.code == "protocol_mismatch"
    assert "redirected" in result.error.detail
    assert "elsewhere" not in result.error.detail
