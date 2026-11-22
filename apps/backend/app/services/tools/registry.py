"""The tool registry: one catalog composed from several tool sources.

The registry is the single place the generation loop talks to about tools. It
fans a request out across its :class:`~app.providers.tools.base.ToolSource` s -
the in-process local source, and later the per-tenant MCP sources - and presents
their union as one catalog, so the loop never needs to know which source a tool
came from. A tool name is unique across the catalog: the first source to offer a
name wins, and a later collision is dropped with a warning (a remote server must
not be able to shadow a built-in tool).

The registry is a small, stateless composer, and it is built PER TURN rather than
once per process. That is deliberate, not an oversight: the in-process local
tools carry the request's own collaborators (its embedder, its search provider),
so a single process-wide registry could not hold them. What IS process-wide is
the set of MCP sources - see
:class:`~app.services.tools.mcp_sources.McpSourceProvider` - because each caches
its remote listing, and rebuilding them per request would discard that cache and
put a round trip to every configured server in front of every turn.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from app.providers.tools.base import (
    Tool,
    ToolContext,
    ToolNotFoundError,
    ToolResult,
    ToolSource,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Composes several tool sources into one name-unique catalog."""

    def __init__(self, sources: Iterable[ToolSource]) -> None:
        """Build a registry over an ordered collection of sources.

        Args:
            sources: The tool sources to compose, in precedence order: an
                earlier source's tool wins a name collision with a later one.
        """
        self._sources = list(sources)

    async def list_tools(self, context: ToolContext) -> list[Tool]:
        """Return every tool offered to ``context``, deduplicated by name.

        Sources are consulted in construction order; the first to offer a given
        name keeps it, and a later duplicate is dropped with a warning so a
        remote source cannot shadow a built-in tool.

        Args:
            context: The caller's identity and turn, passed to each source.

        Returns:
            The composed catalog, at most one tool per name.
        """
        by_name: dict[str, Tool] = {}
        for source in self._sources:
            for tool in await source.list_tools(context):
                name = tool.spec.name
                if name in by_name:
                    logger.warning(
                        "Duplicate tool name %r; keeping the earlier source's and dropping the later",
                        name,
                    )
                    continue
                by_name[name] = tool
        return list(by_name.values())

    async def get_tool(self, name: str, context: ToolContext) -> Tool | None:
        """Return the tool offered to ``context`` under ``name``, or None."""
        for tool in await self.list_tools(context):
            if tool.spec.name == name:
                return tool
        return None

    async def run(self, name: str, arguments: dict, context: ToolContext) -> ToolResult:
        """Run the named tool for ``context``.

        Args:
            name: The tool name the model asked for.
            arguments: The call arguments.
            context: The caller's identity and turn - the sole source of the
                tenant/user the call acts for.

        Returns:
            The tool's result.

        Raises:
            ToolNotFoundError: When no source offers a tool called ``name`` -
                the model invented or misspelled a tool. The caller turns this
                into an honest notice rather than a silent no-op.

        Note:
            This re-lists the sources to resolve ``name``, so it is NOT what the
            generation loop uses: now that a listing can cross the network, the
            loop lists once per turn and dispatches against that snapshot rather
            than paying a fan-out per call. This remains the convenient entry
            point for a caller that holds only a name.
        """
        tool = await self.get_tool(name, context)
        if tool is None:
            raise ToolNotFoundError(name)
        return await tool.run(arguments, context)
