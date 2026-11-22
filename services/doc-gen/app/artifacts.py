"""The artifact result shape, the filename rules, and the caps every tool shares.

A file-producing tool answers with two content blocks in a FIXED order: one
``TextContent`` summary line - the only part the model reads - followed by
exactly one ``EmbeddedResource`` carrying the bytes, which the host turns into a
download. Both halves are built here rather than at each tool, so the order and
the ``artifact://`` URI shape cannot drift apart between tools.

Every limit lives here or beside the renderer it bounds, as a module constant:
a cap that is a literal buried in a branch is a cap nobody can find, and these
exist to keep one malformed spec from turning into a gigabyte of base64.
"""

from __future__ import annotations

import base64
import re
from typing import Any

from mcp.types import BlobResourceContents, CallToolResult, EmbeddedResource, TextContent
from pydantic import BaseModel, ConfigDict

# Ceiling on the RAW bytes of one artifact, checked before base64 (which inflates
# by a third): the encoded string, the JSON envelope and the SSE frame are all
# copies of it, so the moment to refuse is while there is still only one.
#
# The VALUE is set by the host, not by what a document might want to be. The
# platform's MCP client reads a whole response body under a hard
# _MAX_RESPONSE_BYTES = 2_000_000 cap, and breaking it raises: the ENTIRE tool
# call fails, with no path that keeps the summary and drops the file. So the
# arithmetic has to close with room to spare - 1_200_000 raw bytes * 4/3 =
# 1_600_000 base64 characters, leaving 400,000 for the summary text and the JSON
# envelope around them. An artifact this sidecar is willing to EMIT must never be
# one the host transport kills on arrival: refusing here costs the caller one
# correctable message, and refusing there costs them the whole turn's tool call.
MAX_OUTPUT_BYTES = 1_200_000

# Ceiling on the total TEXT in one request, summed across every string anywhere
# in the spec. Every other cap bounds a QUANTITY - blocks, rows, cells, points -
# and none of them bounds the SIZE of one field, so a handful of bullet items
# can satisfy every count and still hand a renderer megabytes of text. That
# matters because the work is paid BEFORE the output check can refuse it:
# reportlab lays the text out to discover how big the result is, and layout cost
# is not obviously linear (the 10,000-wedge pie is the same shape of surprise).
# One total, checked once, up front - not a length limit sprinkled per field,
# which is the version that goes stale the day a field is added.
#
# This is the SECOND line, not the first: the MCP server refuses a POST body
# over DEFAULT_MAX_REQUEST_BODY_SIZE (4 MiB) with a 413 before parsing it, so
# the truly enormous request never arrives. What it does not cover is the band
# underneath - a spec that is accepted, is nowhere near any per-item count, and
# is still megabytes of prose headed for a layout engine. That band is this cap.
#
# Kept at or under MAX_OUTPUT_BYTES so an at-cap input cannot be one the output
# check was always going to refuse. Necessary, not sufficient: a renderer can
# still inflate its input, and the output check stays the backstop for that.
MAX_INPUT_CHARS = 1_000_000

# Filenames are handed to a browser download, so they are sanitized HERE and
# never trusted: one segment matching [A-Za-z0-9][A-Za-z0-9._-]* plus the
# extension the format dictates.
MAX_FILENAME_STEM = 80

_DISALLOWED = re.compile(r"[^A-Za-z0-9._-]+")
_LEADING_JUNK = re.compile(r"^[^A-Za-z0-9]+")

# What each format's bytes are, on the wire. A wrong media type is the
# difference between a file that opens and a file the browser renders as text.
MEDIA_TYPES: dict[str, str] = {
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
    "csv": "text/csv",
    "md": "text/markdown",
    "png": "image/png",
    "svg": "image/svg+xml",
}


class DocGenError(Exception):
    """Base for the failures a CALLER can fix by sending a different spec."""


class SpecError(DocGenError):
    """The spec cannot be rendered as asked (nothing to write, wrong shape)."""


class CapExceeded(DocGenError):
    """The spec is well-formed but larger than a hard limit allows."""


def _drop_inert_annotations(schema: dict[str, Any]) -> None:
    """Strip the JSON Schema keywords that describe a field without bounding it.

    ``title`` and a null ``default`` are annotations, not constraints: no
    validator reads either, so removing them cannot change which arguments this
    server accepts. Pydantic emits both anyway, and both are empty - a generated
    ``title`` is the property name title-cased ("Header Row" for ``header_row``),
    and ``"default": null`` only restates what leaving a field out of
    ``required`` already says.

    Empty everywhere except in cost. Every offered tool's input schema rides in
    the chat prompt on EVERY agentic turn, so the platform pays for these bytes
    once per turn, forever, to tell the model nothing it cannot read off the
    property name beside them.

    Args:
        schema: One model's generated schema, mutated in place - the contract
            of pydantic's ``json_schema_extra`` hook.
    """
    schema.pop("title", None)
    for field_schema in schema.get("properties", {}).values():
        field_schema.pop("title", None)
        # Membership first, and `is None` rather than falsiness: `false` and `0`
        # are defaults that genuinely inform, and a missing key is not a null one.
        if "default" in field_schema and field_schema["default"] is None:
            del field_schema["default"]


class SpecModel(BaseModel):
    """Base for the request models published INTO a tool's input schema.

    Declares the trim once instead of per model, which is a smaller claim than
    it looks: a spec model added later still has to INHERIT this to get it.
    That is a real drift risk, not a solved one, so the trim is covered by a
    test that reads the published schema rather than by this base alone.

    Out of reach here: the root arguments model, which the MCP SDK builds per
    tool from the handler's signature. It is not a ``SpecModel``, so its own
    title survives. Left alone deliberately - the only way to reach it is the
    SDK's private tool registry, and a boot-time break on an SDK minor bump
    costs far more than the bytes it would save.
    """

    model_config = ConfigDict(json_schema_extra=_drop_inert_annotations)


def total_input_chars(value: object) -> int:
    """Sum the length of every string anywhere inside a spec.

    Walks the value structurally - models, lists, dicts, strings - rather than
    naming fields, so a field added later is counted without anyone remembering
    to add it here. That completeness is the point: a size check that has to be
    maintained field by field is one that silently stops covering the newest
    field, which is exactly where an unbounded string would arrive.

    Args:
        value: Any part of a request - a spec model, a list, or a leaf.

    Returns:
        The total number of characters in every string reachable from it.
    """
    if isinstance(value, str):
        return len(value)
    if isinstance(value, BaseModel):
        return sum(total_input_chars(getattr(value, name)) for name in type(value).model_fields)
    if isinstance(value, list | tuple):
        return sum(total_input_chars(item) for item in value)
    if isinstance(value, dict):
        return sum(total_input_chars(key) + total_input_chars(item) for key, item in value.items())
    return 0


def check_input_size(spec: BaseModel) -> None:
    """Refuse an oversized request before any renderer touches it.

    Args:
        spec: The request model.

    Raises:
        CapExceeded: The spec carries more than :data:`MAX_INPUT_CHARS`.
    """
    total = total_input_chars(spec)
    if total > MAX_INPUT_CHARS:
        raise CapExceeded(
            f"The request carries {total} characters of text, over the limit of "
            f"{MAX_INPUT_CHARS}. Send less text, or split it across several files."
        )


def sanitize_filename(candidate: str | None, *, fallback: str, extension: str) -> str:
    """Build a safe single-segment filename with the format's extension forced on.

    Args:
        candidate: The caller's preferred name - a request field, or a title.
            Fully untrusted: path separators, spaces and anything outside the
            allowed set are removed rather than rejected, because a bad name is
            not a reason to fail a document the caller did want.
        fallback: The stem used when nothing usable survives sanitizing.
        extension: The extension the format dictates; always applied, so the
            name can never disagree with the bytes.

    Returns:
        A filename matching ``[A-Za-z0-9][A-Za-z0-9._-]*`` and ending in
        ``.<extension>``.
    """
    raw = (candidate or "").strip()
    # Both separators, because a model writes whichever it has seen; taking the
    # last segment is what stops "../../etc/passwd" from ever being a filename.
    raw = raw.replace("\\", "/").rsplit("/", 1)[-1]
    suffix = f".{extension}"
    stem = raw[: -len(suffix)] if raw.lower().endswith(suffix) else raw
    stem = _DISALLOWED.sub("-", stem)
    stem = _LEADING_JUNK.sub("", stem)
    stem = stem[:MAX_FILENAME_STEM].rstrip("._-")
    return f"{stem or fallback}{suffix}"


def human_bytes(size: int) -> str:
    """Render a byte count for the summary line the model reads.

    Binary units (KiB/MiB), because the arithmetic is binary - labelling a
    division by 1024 as "MB" is what makes a stated limit disagree with itself.
    """
    if size < 1024:
        return f"{size} B"
    kibibytes = size / 1024
    if kibibytes < 1024:
        return f"{kibibytes:.1f} KiB"
    return f"{kibibytes / 1024:.1f} MiB"


def message_result(text: str) -> CallToolResult:
    """A text-only answer: no file, no error flag, an instruction to the model.

    Deliberately NOT ``isError``: a refused cap or an unusable spec is something
    the caller can correct on the next attempt, and the platform treats a failed
    tool as one to stop offering for the rest of the turn - which would take the
    retry away along with the mistake.
    """
    return CallToolResult(content=[TextContent(type="text", text=text)])


def artifact_result(*, summary: str, filename: str, data: bytes, media_type: str) -> CallToolResult:
    """Pack rendered bytes into the two-block artifact result.

    Args:
        summary: The one-line description the model reads (the file is opaque
            to it).
        filename: An already-sanitized single-segment name.
        data: The rendered file's raw bytes.
        media_type: The bytes' media type.

    Returns:
        A result whose content is ``[TextContent, EmbeddedResource]``, in that
        order.

    Raises:
        CapExceeded: If the raw bytes exceed :data:`MAX_OUTPUT_BYTES`. Checked
            here, the one place every artifact passes through, so no renderer
            can forget it.
    """
    if len(data) > MAX_OUTPUT_BYTES:
        # The remedy is picked off the media type because the useful advice
        # differs sharply: an SVG holds one element per data point and so grows
        # without bound, where the SAME chart as a png is a fixed-size raster.
        # "Send less data" would be misleading advice for the one case where
        # changing a single field fixes it.
        remedy = (
            'Request format "png" instead of "svg" - it stays small however '
            "many points it draws - or plot fewer points."
            if media_type == MEDIA_TYPES["svg"]
            else "Send less data - fewer rows, fewer blocks, or split it across several files."
        )
        raise CapExceeded(
            f"The generated file is {human_bytes(len(data))}, over the "
            f"{human_bytes(MAX_OUTPUT_BYTES)} limit. {remedy}"
        )
    return CallToolResult(
        content=[
            TextContent(type="text", text=summary),
            EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(
                    uri=f"artifact://{filename}",
                    blob=base64.b64encode(data).decode("ascii"),
                    mimeType=media_type,
                ),
            ),
        ]
    )
