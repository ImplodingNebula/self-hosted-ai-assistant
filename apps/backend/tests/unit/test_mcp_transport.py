"""The Streamable-HTTP transport spike: the one genuine unknown in Phase 5.

Proves, hermetically, that the backend can drive a real MCP server over the
real Streamable HTTP protocol: a FastMCP server's ASGI app on one side, the
SDK's ``streamable_http_client`` on the other, joined by an in-process httpx
``ASGITransport`` instead of a network socket (the client takes the httpx
client outright, which is also what makes per-call credential injection
straightforward later). Every McpToolSource test reuses this harness shape.

Facts these tests pin because the seam and the chassis depend on them:

- ``tools/list`` carries JSON Schemas and (optional) annotations; a
  ``readOnlyHint`` is ADVISORY - the adapter maps only an explicit true to
  ``side_effecting=False`` and everything else fails closed.
- The server's session manager must be RUNNING for the ASGI app to accept
  requests (its lifespan does this under uvicorn; tests run it explicitly).
- The server validates the Host header (DNS-rebinding protection) and
  rejects unlisted hosts with 421 - including bare ``localhost`` - so the
  chassis MUST configure ``allowed_hosts`` explicitly for its
  compose-network name; there is no permissive default to lean on.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import TextContent, ToolAnnotations

pytestmark = pytest.mark.anyio


def build_spike_server(**kwargs) -> FastMCP:
    """A minimal MCP server: one annotated read-only tool, one unannotated.

    ``allowed_hosts`` is explicit because the SDK's DNS-rebinding protection
    ships with an EMPTY allow list: without this, every request - localhost
    included - is rejected 421 before the protocol layer sees it.
    """
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


@asynccontextmanager
async def mcp_session(
    server: FastMCP, base_url: str = "http://localhost"
) -> AsyncIterator[ClientSession]:
    """A real client session over Streamable HTTP, no sockets involved."""
    app = server.streamable_http_app()
    async with (
        server.session_manager.run(),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=base_url,
            timeout=httpx.Timeout(10.0),
            follow_redirects=True,
        ) as http_client,
        streamable_http_client(f"{base_url}/mcp", http_client=http_client) as (
            read,
            write,
            _get_session_id,
        ),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


async def test_initialize_and_list_tools_over_streamable_http() -> None:
    async with mcp_session(build_spike_server()) as session:
        listed = await session.list_tools()

    by_name = {tool.name: tool for tool in listed.tools}
    assert set(by_name) == {"get_record", "create_record"}
    # The JSON Schema the model will see rides in the listing.
    assert by_name["get_record"].inputSchema["properties"]["record_id"]["type"] == "string"
    # Annotations are optional and advisory: present on the annotated tool,
    # absent on the other - the adapter must fail closed on absence.
    assert by_name["get_record"].annotations.readOnlyHint is True
    assert by_name["create_record"].annotations is None


async def test_call_tool_round_trip() -> None:
    async with mcp_session(build_spike_server()) as session:
        result = await session.call_tool("get_record", {"record_id": "42"})

    assert result.isError is False
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "record 42: all good"


async def test_call_with_bad_arguments_reports_a_tool_error() -> None:
    # Argument validation happens server-side and comes back as an in-band
    # error result, not a transport failure - the adapter must surface it as
    # a ToolError-style result, never a crash.
    async with mcp_session(build_spike_server()) as session:
        result = await session.call_tool("get_record", {"wrong_key": "x"})

    assert result.isError is True


async def test_non_localhost_host_needs_an_allowed_hosts_entry() -> None:
    # The SDK's DNS-rebinding protection rejects unlisted Host headers with a
    # 421 BEFORE the protocol layer sees the request. In compose the backend
    # reaches the chassis as http://mcp-chassis:8765, so the chassis must list
    # its own service name - localhost-only testing would never catch this.
    security = TransportSecuritySettings(allowed_hosts=["mcp-chassis", "mcp-chassis:*"])
    server = build_spike_server(transport_security=security)
    async with mcp_session(server, base_url="http://mcp-chassis") as session:
        result = await session.call_tool("get_record", {"record_id": "7"})

    assert result.isError is False
