"""Unit tests for the tool seam value types and the composing registry."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.providers.tools.base import (
    Tool,
    ToolContext,
    ToolNotFoundError,
    ToolResult,
    ToolSource,
    ToolSpec,
)
from app.services.tools.registry import ToolRegistry

CTX = ToolContext(tenant_id="t1", user_id="u1", conversation_id="c1", session_id="s1")


class _EchoTool(Tool):
    """A trivial tool that echoes its arguments, tagged with a source label."""

    def __init__(self, name: str, label: str) -> None:
        self._name = name
        self._label = label

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self._name, description=f"echo from {self._label}")

    async def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        return ToolResult(content=f"{self._label}:{arguments.get('text', '')}:{context.user_id}")


class _StaticSource(ToolSource):
    """A source that offers a fixed list of tools, ignoring the context."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = tools

    async def list_tools(self, context: ToolContext) -> list[Tool]:
        return list(self._tools)


# --- the seam value types ---


def test_tool_context_is_frozen() -> None:
    # A tool must not be able to mutate the identity it was handed.
    with pytest.raises(ValidationError):
        CTX.tenant_id = "other"  # type: ignore[misc]


def test_tool_spec_defaults_to_side_effecting() -> None:
    # A missing annotation must fail closed: an un-annotated tool is dangerous.
    assert ToolSpec(name="x").side_effecting is True


# --- the registry ---


async def test_a_registry_with_no_sources_offers_no_tools() -> None:
    registry = ToolRegistry([])
    assert await registry.list_tools(CTX) == []


async def test_lists_and_dispatches_tools() -> None:
    registry = ToolRegistry([_StaticSource([_EchoTool("echo", "A")])])
    tools = await registry.list_tools(CTX)
    assert [t.spec.name for t in tools] == ["echo"]

    result = await registry.run("echo", {"text": "hi"}, CTX)
    assert result.content == "A:hi:u1"
    assert result.is_error is False


async def test_unknown_tool_raises() -> None:
    registry = ToolRegistry([])
    with pytest.raises(ToolNotFoundError) as excinfo:
        await registry.run("nope", {}, CTX)
    assert excinfo.value.name == "nope"


async def test_earlier_source_wins_a_name_collision() -> None:
    # A later source must not be able to shadow a built-in tool's name.
    registry = ToolRegistry(
        [
            _StaticSource([_EchoTool("shared", "first")]),
            _StaticSource([_EchoTool("shared", "second"), _EchoTool("only-second", "second")]),
        ]
    )
    names = sorted(t.spec.name for t in await registry.list_tools(CTX))
    assert names == ["only-second", "shared"]
    # The surviving "shared" is the first source's.
    result = await registry.run("shared", {"text": "x"}, CTX)
    assert result.content.startswith("first:")


async def test_get_tool_returns_none_when_absent() -> None:
    registry = ToolRegistry([_StaticSource([_EchoTool("echo", "A")])])
    assert await registry.get_tool("echo", CTX) is not None
    assert await registry.get_tool("ghost", CTX) is None
