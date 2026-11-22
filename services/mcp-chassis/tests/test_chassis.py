"""Hermetic chassis tests: the real MCP protocol, no sockets.

Mirrors the backend's transport-spike harness
(apps/backend/tests/unit/test_mcp_transport.py): the chassis ASGI app on one
side, the SDK's streamable-HTTP client on the other, joined by httpx's
ASGITransport. The parent app's lifespan is entered explicitly because
ASGITransport does not run lifespans - and the lifespan is exactly what
starts each wrapper's session manager, so these tests exercise that wiring.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent
from starlette.applications import Starlette

from app import example_wrapper
from app.main import build_app

pytestmark = pytest.mark.anyio

BASE_URL = "http://localhost"
API_KEY = "test-chassis-key"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _fresh_records() -> Iterator[None]:
    # create_record mutates the module-level mock store; snapshot and restore
    # so tests stay order-independent.
    snapshot = {record_id: dict(record) for record_id, record in example_wrapper.RECORDS.items()}
    yield
    example_wrapper.RECORDS.clear()
    example_wrapper.RECORDS.update(snapshot)


@asynccontextmanager
async def chassis_http_client(
    app: Starlette, headers: dict[str, str] | None = None, base_url: str = BASE_URL
) -> AsyncIterator[httpx.AsyncClient]:
    """An httpx client over the chassis app, with its lifespan running."""
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(10.0),
            follow_redirects=True,
        ) as http_client,
    ):
        yield http_client


@asynccontextmanager
async def mcp_session(
    app: Starlette, headers: dict[str, str] | None = None, base_url: str = BASE_URL
) -> AsyncIterator[ClientSession]:
    """A real client session against the example wrapper's mount."""
    async with (
        chassis_http_client(app, headers, base_url) as http_client,
        streamable_http_client(f"{base_url}/example/mcp", http_client=http_client) as (
            read,
            write,
            _get_session_id,
        ),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


async def call_text(session: ClientSession, tool: str, args: dict[str, str]) -> str:
    """Calls a tool and returns its single text content, asserting success."""
    result = await session.call_tool(tool, args)
    assert result.isError is False
    assert isinstance(result.content[0], TextContent)
    return result.content[0].text


async def test_healthz_is_open() -> None:
    async with chassis_http_client(build_app()) as http_client:
        response = await http_client.get("/healthz")

    assert response.status_code == 200
    assert response.text == "ok"


async def test_tools_list_on_the_example_mount() -> None:
    async with mcp_session(build_app()) as session:
        listed = await session.list_tools()

    by_name = {tool.name: tool for tool in listed.tools}
    assert set(by_name) == {"search_records", "get_record", "create_record"}
    # Read-only tools carry the advisory hint; the side-effecting one stays
    # unannotated so the platform's fail-closed default treats it as such.
    assert by_name["search_records"].annotations.readOnlyHint is True
    assert by_name["get_record"].annotations.readOnlyHint is True
    assert by_name["create_record"].annotations is None


async def test_search_get_create_round_trip() -> None:
    async with mcp_session(build_app()) as session:
        found = await call_text(session, "search_records", {"query": "printer"})
        assert "rec-001" in found
        assert "Printer offline" in found

        record = await call_text(session, "get_record", {"record_id": "rec-001"})
        assert "status: open" in record
        assert "Rolling back to 2.0" in record

        created = await call_text(
            session,
            "create_record",
            {"title": "Login page 500s", "body": "Every login attempt returns HTTP 500."},
        )
        assert "rec-004" in created

        readback = await call_text(session, "get_record", {"record_id": "rec-004"})
        assert "Login page 500s" in readback


async def test_read_tools_return_structured_records() -> None:
    # The platform's citable-records contract: a read tool's structuredContent
    # is a `records` list of {id, title, text}. This is what lets the backend
    # ingest and cite records instead of parsing a formatted string.
    async with mcp_session(build_app()) as session:
        found = await session.call_tool("search_records", {"query": "printer"})
        assert found.isError is False
        assert found.structuredContent is not None
        records = found.structuredContent["records"]
        # Full body, not a snippet - the whole reason to return structured.
        assert any(r["id"] == "rec-001" and "Rolling back to 2.0" in r["text"] for r in records)
        assert all({"id", "title", "text"} <= r.keys() for r in records)

        one = await session.call_tool("get_record", {"record_id": "rec-002"})
        assert one.structuredContent is not None
        assert one.structuredContent["records"][0]["id"] == "rec-002"

        # Nothing to cite, but the model still learns what happened via `message`,
        # which the backend ignores for citation.
        missing = await session.call_tool("get_record", {"record_id": "rec-999"})
        assert missing.structuredContent["records"] == []
        assert "No record with id" in (missing.structuredContent.get("message") or "")


async def test_unknown_record_id_is_a_message_not_an_error() -> None:
    async with mcp_session(build_app()) as session:
        text = await call_text(session, "get_record", {"record_id": "rec-999"})

    assert "No record with id" in text


async def test_no_match_search_says_so() -> None:
    async with mcp_session(build_app()) as session:
        text = await call_text(session, "search_records", {"query": "zzz-nothing"})

    assert "No records match" in text


@pytest.mark.parametrize(
    "headers",
    [None, {"Authorization": "Bearer wrong-key"}],
    ids=["missing-header", "wrong-key"],
)
async def test_auth_rejects_calls_without_the_bearer_key(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str] | None
) -> None:
    monkeypatch.setenv("MCP_CHASSIS_API_KEY", API_KEY)
    app = build_app()

    # The 401 lands on the initialize POST and the SDK surfaces it as an
    # httpx.HTTPStatusError wrapped in the client task group's exception
    # group(s) - never as a usable session.
    with pytest.raises(BaseExceptionGroup) as excinfo:
        async with mcp_session(app, headers=headers):
            pytest.fail("an unauthenticated session must not initialize")
    assert excinfo.group_contains(httpx.HTTPStatusError, match="401 Unauthorized")


async def test_auth_accepts_the_bearer_key_and_healthz_stays_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_CHASSIS_API_KEY", API_KEY)
    app = build_app()

    async with mcp_session(app, headers={"Authorization": f"Bearer {API_KEY}"}) as session:
        listed = await session.list_tools()
    assert {tool.name for tool in listed.tools} == {
        "search_records",
        "get_record",
        "create_record",
    }

    # The liveness probe must not require the key. A fresh app is built
    # because a FastMCP session manager's run() is once-per-instance, so an
    # app's lifespan cannot be entered twice.
    async with chassis_http_client(build_app()) as http_client:
        response = await http_client.get("/healthz")
    assert response.status_code == 200


async def test_a_blank_api_key_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # A blank value is what a failed compose variable substitution produces;
    # the chassis boots open (with a warning) rather than locking on "Bearer ".
    monkeypatch.setenv("MCP_CHASSIS_API_KEY", "")

    async with mcp_session(build_app()) as session:
        listed = await session.list_tools()

    assert len(listed.tools) == 3


async def test_a_disallowed_host_is_rejected() -> None:
    # The SDK's DNS-rebinding protection rejects unlisted Host headers with a
    # 421 before the protocol layer sees the request.
    with pytest.raises(BaseExceptionGroup) as excinfo:
        async with mcp_session(build_app(), base_url="http://evil.example"):
            pytest.fail("a disallowed Host must not initialize")
    assert excinfo.group_contains(httpx.HTTPStatusError, match="421")


async def test_the_allowed_hosts_env_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    # The env var is the whole reason transport security is configurable: a
    # deployment reaching the chassis under another hostname lists it here.
    monkeypatch.setenv("MCP_CHASSIS_ALLOWED_HOSTS", "chassis.internal,chassis.internal:*")

    async with mcp_session(build_app(), base_url="http://chassis.internal") as session:
        listed = await session.list_tools()

    assert len(listed.tools) == 3


async def test_blank_allowed_hosts_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An empty allow list would be the SDK's reject-everything state.
    monkeypatch.setenv("MCP_CHASSIS_ALLOWED_HOSTS", "  ,  ")

    async with mcp_session(build_app()) as session:
        listed = await session.list_tools()

    assert len(listed.tools) == 3


def test_example_api_base_url_warns_that_main_ships_only_the_mock(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("EXAMPLE_API_BASE_URL", "http://upstream.example")

    with caplog.at_level("WARNING", logger="app.example_wrapper"):
        build_app()

    assert "not implemented on main" in caplog.text
