"""The doc-gen MCP server: two tools, each answering with one real file.

Both tools are READ-ONLY in the MCP sense - they touch no upstream system and
change nothing anywhere; they turn the arguments they were given into bytes.
The ``readOnlyHint`` annotation is therefore correct AND load-bearing: the
platform fails closed on a missing hint and would route every document through
a confirmation prompt.

Both return a ``CallToolResult`` directly. That return annotation is what tells
FastMCP to pass the result through untouched - a pydantic-model return would be
serialized into ``structuredContent``, which is the platform's citable-RECORDS
contract, not the artifact one.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, ToolAnnotations

from app import artifacts, charts, documents
from app.artifacts import DocGenError

logger = logging.getLogger(__name__)

_CREATE_DOCUMENT_DESCRIPTION = (
    "Create a real downloadable spreadsheet, document, PDF, CSV or Markdown file "
    "from structured data, and return the file itself. Use it whenever the user "
    "asks to make, write, export, save or download a report, spreadsheet, "
    "workbook, table or document. Choose format xlsx for a spreadsheet (spec: "
    "sheets, each with rows and an optional header_row), docx or pdf for a "
    "formatted document (spec: title plus blocks of type heading, paragraph, "
    "bullets, numbered or table), csv for a plain table (spec: rows), or md for "
    "Markdown (spec: text). Set spec.filename to name the file. The answer is a "
    "one-line summary plus the generated file."
)

# This description must TEACH the `spec` envelope in prose, because the schema
# alone demonstrably does not: the first live probe (2026-08-09, qwen3.5:9b)
# flattened every field to the top level - `{series, labels, title, format}` with
# no `spec` - twice in one turn, even though the schema marks `spec` required.
# create_document is the proof of the cure: its description names `spec` three
# times and the same model nests it correctly. Prose the schema cannot carry is
# also kept: `series` has a default, so nothing in the schema marks it as the one
# field a chart cannot be drawn without.
#
# The words a user would reach for ("label the pie wedges", "axis labels") are
# retrieval, not decoration: the host's top-K tool selection embeds this
# description whole and cosine-ranks it - the schema is invisible to selection,
# so the description has to carry that vocabulary or the tool stops being
# offered on exactly the turns that want it.
_RENDER_CHART_DESCRIPTION = (
    "Render a chart image - line, bar, pie or scatter - from numeric data and "
    "return it as a downloadable png or svg file. Use it whenever the user asks "
    "to plot, graph, visualize or chart numbers, or wants a picture of a trend or "
    "a comparison. Send everything inside the single spec object: spec.series is "
    "required (at least one entry with its values), and spec.type, a title, axis "
    "labels, wedge labels, format png or svg and a filename are optional. The "
    "answer is a one-line summary plus the image file."
)

_UNEXPECTED_FAILURE = (
    "Generating the file failed unexpectedly. Try again with a smaller or simpler "
    "spec; if it keeps failing, tell the user the file could not be generated."
)


def build_docgen_server(allowed_hosts: list[str]) -> FastMCP:
    """Build the ``doc-gen`` FastMCP server.

    Args:
        allowed_hosts: Host header values the transport accepts. Explicit
            because the SDK's DNS-rebinding protection ships with an EMPTY allow
            list - without this every request, localhost included, is rejected
            with 421 before the protocol layer sees it.

    Returns:
        A stateless Streamable-HTTP FastMCP server exposing the document and
        chart tools.
    """
    server = FastMCP(
        "doc-gen",
        stateless_http=True,
        transport_security=TransportSecuritySettings(allowed_hosts=allowed_hosts),
    )

    @server.tool(
        description=_CREATE_DOCUMENT_DESCRIPTION,
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    def create_document(
        format: documents.DocumentFormat, spec: documents.DocumentSpec
    ) -> CallToolResult:
        """Render one document and return it as an artifact."""
        try:
            rendered = documents.render(format, spec)
            filename = artifacts.sanitize_filename(
                spec.filename or spec.title,
                fallback=documents.FALLBACK_STEMS[format],
                extension=format,
            )
            return artifacts.artifact_result(
                summary=(
                    f"Created {filename} ({rendered.detail}, "
                    f"{artifacts.human_bytes(len(rendered.data))})."
                ),
                filename=filename,
                data=rendered.data,
                media_type=artifacts.MEDIA_TYPES[format],
            )
        except DocGenError as exc:
            # A spec the caller can fix: hand back the reason as ordinary text
            # so the model can correct it and try again.
            return artifacts.message_result(str(exc))
        except Exception:
            # Never let a library traceback reach the model as tool output.
            logger.exception("rendering a %s document failed", format)
            return artifacts.message_result(_UNEXPECTED_FAILURE)

    @server.tool(
        description=_RENDER_CHART_DESCRIPTION,
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    def render_chart(spec: charts.ChartSpec) -> CallToolResult:
        """Render one chart image and return it as an artifact."""
        try:
            rendered = charts.render_chart(spec)
            filename = artifacts.sanitize_filename(
                spec.filename or spec.title,
                fallback=charts.FALLBACK_STEM,
                extension=spec.format,
            )
            return artifacts.artifact_result(
                summary=(
                    f"Created {filename}: a {rendered.chart_type} chart "
                    f"({rendered.detail}, {artifacts.human_bytes(len(rendered.data))})."
                ),
                filename=filename,
                data=rendered.data,
                media_type=artifacts.MEDIA_TYPES[spec.format],
            )
        except DocGenError as exc:
            return artifacts.message_result(str(exc))
        except Exception:
            logger.exception("rendering a %s chart failed", spec.type)
            return artifacts.message_result(_UNEXPECTED_FAILURE)

    return server
