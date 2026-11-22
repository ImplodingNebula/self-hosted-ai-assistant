"""Shared harness: the real MCP protocol over an in-process ASGI transport.

Mirrors the chassis's test harness - the doc-gen ASGI app on one side, the SDK's
streamable-HTTP client on the other, joined by httpx's ASGITransport. The parent
app's lifespan is entered explicitly because ASGITransport does not run
lifespans, and that lifespan is exactly what starts the session manager.

Every document test goes through this rather than calling the renderers
directly, so what is asserted is what a CLIENT receives.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import BlobResourceContents, ContentBlock, EmbeddedResource, TextContent
from starlette.applications import Starlette

BASE_URL = "http://localhost"


@asynccontextmanager
async def docgen_http_client(
    app: Starlette, headers: dict[str, str] | None = None, base_url: str = BASE_URL
) -> AsyncIterator[httpx.AsyncClient]:
    """An httpx client over the doc-gen app, with its lifespan running."""
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        ) as http_client,
    ):
        yield http_client


@asynccontextmanager
async def mcp_session(
    app: Starlette, headers: dict[str, str] | None = None, base_url: str = BASE_URL
) -> AsyncIterator[ClientSession]:
    """A real client session against the /docgen mount."""
    async with (
        docgen_http_client(app, headers, base_url) as http_client,
        streamable_http_client(f"{base_url}/docgen/mcp", http_client=http_client) as (
            read,
            write,
            _get_session_id,
        ),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


async def call_content(session: ClientSession, tool: str, args: dict) -> list[ContentBlock]:
    """Call a tool and return its content blocks, asserting the call succeeded."""
    result = await session.call_tool(tool, args)
    assert result.isError is False, result.content
    return list(result.content)


def artifact_of(content: list[ContentBlock]) -> tuple[str, BlobResourceContents]:
    """Unpack the two-block artifact contract, asserting its ORDER.

    The order is the seam the platform host reads positionally: the summary
    first (the only part the model sees), the file second.
    """
    assert len(content) == 2, content
    summary, embedded = content
    assert isinstance(summary, TextContent)
    assert isinstance(embedded, EmbeddedResource)
    assert isinstance(embedded.resource, BlobResourceContents)
    return summary.text, embedded.resource


def artifact_bytes(resource: BlobResourceContents) -> bytes:
    """The artifact's raw bytes, decoded from its base64 blob."""
    return base64.b64decode(resource.blob)


async def make_document(session: ClientSession, fmt: str, spec: dict) -> tuple[str, bytes, str]:
    """Call ``create_document`` and return (summary, file bytes, filename)."""
    summary, resource = artifact_of(
        await call_content(
            session,
            "create_document",
            {
                "format": fmt,
                "spec": spec,
            },
        )
    )
    return summary, artifact_bytes(resource), str(resource.uri).removeprefix("artifact://")


def text_only(content: list[ContentBlock]) -> str:
    """A refusal: one text block, no artifact. Returns the message."""
    assert len(content) == 1, content
    assert isinstance(content[0], TextContent)
    return content[0].text
