"""The tool seam: transport-agnostic value types and ABCs for tools.

Local in-process tools and remote MCP tools both satisfy the same
:class:`~app.providers.tools.base.Tool` / :class:`~app.providers.tools.base.ToolSource`
interface here; the concrete registry that composes sources into one catalog
lives in ``app.services.tools``. MCP is one *source* of tools, not the tool
architecture. See docs/ARCHITECTURE.md section 7.
"""
