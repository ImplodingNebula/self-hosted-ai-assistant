"""Document renderers: one structured spec in, one real file out.

The model fills a schema and pinned library code writes the file - nothing the
model sends is ever executed. The spec models are deliberately FLAT and
forgiving (one optional field per format rather than a discriminated union, an
unknown block rendered as a paragraph rather than a refusal), because a small
model fills them and a near-miss should still produce the document that was
asked for.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Literal
from xml.sax.saxutils import escape

from docx import Document
from openpyxl import Workbook
from openpyxl.styles import Font
from pydantic import Field
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.artifacts import CapExceeded, SpecError, SpecModel, check_input_size

DocumentFormat = Literal["xlsx", "docx", "pdf", "csv", "md"]

# Cells across the whole workbook. openpyxl holds every cell in memory and the
# writer walks them all, so the ceiling is on the TOTAL, not per sheet.
MAX_WORKBOOK_CELLS = 50_000
# Blocks in one docx/pdf. A document is a report, not a data dump; past this the
# caller wants a spreadsheet.
MAX_DOCUMENT_BLOCKS = 200
# Rows in one table block. Bounded separately from the block count because a
# single table is the one block that can be arbitrarily tall on its own.
MAX_TABLE_ROWS = 500
# Rows in one CSV. The format has no structural limit of its own, so without
# this the only ceiling is the output byte check - which refuses only AFTER the
# whole file has been built in memory. This refuses while the request is still
# just a request. There is no sibling character cap for Markdown: the
# total-input cap in artifacts.py counts exactly the same characters and more of
# them (block text as well as raw text), so a second one would only be a way for
# the two to disagree.
MAX_CSV_ROWS = 50_000

# Excel rejects these in a sheet name, and openpyxl raises ValueError rather
# than repairing it - so an unrepaired model-chosen name loses the whole file.
_EXCEL_TITLE_BAD = re.compile(r"[\[\]:*?/\\]")
# Excel's sheet-name length limit. openpyxl only WARNS past it and writes the
# long name anyway, producing a file some applications cannot open - a failure
# that surfaces at the recipient rather than here, which is why it is truncated.
_EXCEL_TITLE_MAX = 31
# Control characters openpyxl refuses to write. Stripped rather than raised on:
# a stray \x07 in a model-written string is not worth losing the workbook over.
_CONTROL_CHARS = re.compile(r"[\000-\010\013\014\016-\037]")

# The page geometry the PDF renderer shares between its margins and its tables:
# a table has to be told how wide it may be, and the answer is "the page, less
# its margins".
_PDF_MARGIN = 20 * mm
_PDF_USABLE_WIDTH = A4[0] - 2 * _PDF_MARGIN

# The stem a filename falls back to when the caller named neither a file nor a
# title - descriptive of what the file IS, since that is what a download shows.
FALLBACK_STEMS: dict[str, str] = {
    "xlsx": "spreadsheet",
    "docx": "document",
    "pdf": "document",
    "csv": "table",
    "md": "notes",
}

CellValue = str | int | float | bool | None


class Sheet(SpecModel):
    """One worksheet of a spreadsheet."""

    name: str | None = None
    # No "the first row may be a header" here: `header_row` sits directly below
    # and says it operatively, which is the copy that decides the rendering.
    rows: list[list[CellValue]] = Field(default_factory=list, description="Row-major cell values.")
    header_row: bool = Field(
        default=False, description="True when the first row holds column headings (rendered bold)."
    )


# Every docstring below is published INTO the tool's input schema, so each one
# is written for the model that fills the form, and each is paid for on every
# agentic turn. A fact therefore gets stated ONCE, on the field that acts on it:
# the block kinds are listed on ``type`` (whose values they literally are) and
# not restated in prose that would have to paraphrase them - the old class
# docstring said "bullet list, numbered list" for values spelled "bullets" and
# "numbered", which cost tokens to introduce an ambiguity. The reasoning behind
# the shapes stays in comments like this one, which the schema never sees.
#
# ``type`` is a plain string rather than an enum on purpose: an unrecognized
# value renders as a paragraph, so a typo cannot cost the whole document.
class Block(SpecModel):
    """One piece of a document."""

    type: str = Field(
        default="paragraph", description="heading | paragraph | bullets | numbered | table"
    )
    text: str | None = Field(default=None, description="Text for a heading or paragraph.")
    items: list[str] | None = Field(default=None, description="List entries for bullets/numbered.")
    rows: list[list[str]] | None = Field(default=None, description="Row-major cells for a table.")
    level: int | None = Field(default=None, description="Heading level, 1-3.")


# One flat model rather than one per format: a small model picks FIELDS far more
# reliably than it picks a variant. Each renderer reads only what it needs, and
# falls back across the near-miss fields (bare `rows` sent for an xlsx becomes a
# single sheet; `blocks` sent for md are rendered down to Markdown).
class DocumentSpec(SpecModel):
    """The document's content. Fill only the fields the chosen format uses."""

    filename: str | None = Field(default=None, description="Preferred file name, without a path.")
    title: str | None = Field(default=None, description="Document title (docx, pdf, md).")
    sheets: list[Sheet] | None = Field(default=None, description="Worksheets (xlsx).")
    blocks: list[Block] | None = Field(default=None, description="Body blocks (docx, pdf).")
    rows: list[list[CellValue]] | None = Field(default=None, description="Row-major cells (csv).")
    text: str | None = Field(default=None, description="Raw body text (md).")


@dataclass(frozen=True)
class Rendered:
    """Rendered bytes plus the shape note that goes in the summary line."""

    data: bytes
    detail: str


def render(fmt: DocumentFormat, spec: DocumentSpec) -> Rendered:
    """Render ``spec`` in ``fmt``.

    Args:
        fmt: One of xlsx, docx, pdf, csv, md.
        spec: The caller's structured content.

    Returns:
        The file's bytes and a short human description of its shape.

    Raises:
        SpecError: The spec holds nothing to write.
        CapExceeded: The spec is larger than a hard limit allows.
    """
    # First, and for every format: the total-text check is the only one that
    # bounds how much WORK a renderer is about to be handed, and it has to
    # happen before any of them starts.
    check_input_size(spec)
    renderers = {
        "xlsx": _render_xlsx,
        "docx": _render_docx,
        "pdf": _render_pdf,
        "csv": _render_csv,
        "md": _render_md,
    }
    return renderers[fmt](spec)


def _plural(count: int, noun: str) -> str:
    """Count plus noun, pluralized the naive way (every noun here is regular)."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _clean_text(value: str) -> str:
    """Strip the control characters the Office writers reject."""
    return _CONTROL_CHARS.sub("", value)


def _sheets_of(spec: DocumentSpec) -> list[Sheet]:
    """The workbook's sheets, accepting a bare ``rows`` as a one-sheet workbook."""
    if spec.sheets:
        return spec.sheets
    if spec.rows:
        return [Sheet(rows=spec.rows)]
    raise SpecError("A spreadsheet needs `sheets` (each with `rows`), or a top-level `rows`.")


def _blocks_of(spec: DocumentSpec) -> list[Block]:
    """The document's blocks, accepting a bare ``text`` as one paragraph."""
    if spec.blocks:
        return spec.blocks
    if spec.text:
        return [Block(type="paragraph", text=spec.text)]
    if spec.title:
        return []
    raise SpecError("A document needs `blocks` (or at least a `title` or `text`).")


def _check_document_caps(blocks: list[Block]) -> None:
    """Reject an oversized document before any of it is written."""
    if len(blocks) > MAX_DOCUMENT_BLOCKS:
        raise CapExceeded(
            f"{len(blocks)} blocks exceeds the limit of {MAX_DOCUMENT_BLOCKS} per document. "
            "Split the content across several documents."
        )
    for block in blocks:
        if block.rows is not None and len(block.rows) > MAX_TABLE_ROWS:
            raise CapExceeded(
                f"A table block has {len(block.rows)} rows, over the limit of "
                f'{MAX_TABLE_ROWS} per table. Use a spreadsheet (format "xlsx") for '
                "data that large."
            )


def _block_text(block: Block) -> str:
    """The text an unrecognized or text-shaped block contributes."""
    if block.text:
        return block.text
    if block.items:
        return " ".join(block.items)
    return ""


def _heading_level(block: Block) -> int:
    """Heading level clamped to 1-3; anything else is a typo, not a reason to fail."""
    return min(3, max(1, block.level or 1))


def _table_grid(rows: list[list[str]]) -> tuple[list[list[str]], int]:
    """Rectangularize a ragged table: every row padded to the widest one."""
    width = max((len(row) for row in rows), default=0)
    return [
        [_clean_text(str(cell)) for cell in row] + [""] * (width - len(row)) for row in rows
    ], width


def _render_xlsx(spec: DocumentSpec) -> Rendered:
    """Write an .xlsx workbook with openpyxl, one worksheet per sheet spec."""
    sheets = _sheets_of(spec)
    total_cells = sum(len(row) for sheet in sheets for row in sheet.rows)
    if total_cells > MAX_WORKBOOK_CELLS:
        raise CapExceeded(
            f"{total_cells} cells exceeds the limit of {MAX_WORKBOOK_CELLS} per workbook. "
            "Send fewer rows or split the data across several files."
        )

    workbook = Workbook()
    workbook.remove(workbook.active)
    used_titles: set[str] = set()
    for index, sheet in enumerate(sheets, start=1):
        worksheet = workbook.create_sheet(_sheet_title(sheet.name, index, used_titles))
        for row_index, row in enumerate(sheet.rows, start=1):
            for column_index, value in enumerate(row, start=1):
                if isinstance(value, str):
                    value = _clean_text(value)
                cell = worksheet.cell(row=row_index, column=column_index, value=value)
                if isinstance(value, str):
                    # Force TEXT. openpyxl turns a leading "=" into a live
                    # formula, so a string arriving from a model (which may be
                    # echoing an attacker-controlled web page) would otherwise
                    # execute the moment the user opens the download. The cell
                    # reads back byte-identical, so honest data loses nothing.
                    cell.data_type = "s"
        if sheet.header_row and sheet.rows:
            for cell in worksheet[1]:
                cell.font = Font(bold=True)

    buffer = io.BytesIO()
    workbook.save(buffer)
    return Rendered(buffer.getvalue(), _plural(len(sheets), "sheet"))


def _sheet_title(name: str | None, index: int, used: set[str]) -> str:
    """A worksheet title Excel accepts: legal characters, <=31 chars, unique."""
    title = _EXCEL_TITLE_BAD.sub("-", _clean_text(name or "")).strip()[:_EXCEL_TITLE_MAX]
    title = title or f"Sheet{index}"
    if title in used:
        # openpyxl de-duplicates on its own, but by APPENDING - which can push a
        # 31-character name over the limit above and reintroduce exactly the
        # problem it was truncated to avoid. Suffixing within the budget here
        # keeps every sheet and keeps every name legal.
        suffix = f"-{index}"
        title = f"{title[: _EXCEL_TITLE_MAX - len(suffix)]}{suffix}"
    used.add(title)
    return title


def _render_docx(spec: DocumentSpec) -> Rendered:
    """Write a .docx with python-docx, mapping each block to a Word style."""
    blocks = _blocks_of(spec)
    _check_document_caps(blocks)

    document = Document()
    if spec.title:
        document.add_heading(_clean_text(spec.title), level=0)
    for block in blocks:
        kind = (block.type or "").strip().lower()
        if kind == "heading":
            document.add_heading(_clean_text(_block_text(block)), level=_heading_level(block))
        elif kind in {"bullets", "numbered"}:
            style = "List Bullet" if kind == "bullets" else "List Number"
            for item in block.items or []:
                document.add_paragraph(_clean_text(item), style=style)
        elif kind == "table" and any(block.rows or []):
            # `any`, not truthiness: rows=[[]] is a non-empty list of empty rows,
            # which yields a zero-column table that both writers refuse.
            grid, width = _table_grid(block.rows or [])
            table = document.add_table(rows=len(grid), cols=width)
            table.style = "Table Grid"
            for row_index, row in enumerate(grid):
                for column_index, value in enumerate(row):
                    table.cell(row_index, column_index).text = value
        else:
            # Unknown type (and a table block with no rows): render whatever text
            # it carries rather than failing the whole document over one block.
            document.add_paragraph(_clean_text(_block_text(block)))

    buffer = io.BytesIO()
    document.save(buffer)
    return Rendered(buffer.getvalue(), _plural(len(blocks), "block"))


def _render_pdf(spec: DocumentSpec) -> Rendered:
    """Write a PDF with reportlab's platypus flowables (same block model as docx)."""
    blocks = _blocks_of(spec)
    _check_document_caps(blocks)

    styles = getSampleStyleSheet()
    story: list[object] = []
    if spec.title:
        story.append(Paragraph(_pdf_text(spec.title), styles["Title"]))
    for block in blocks:
        kind = (block.type or "").strip().lower()
        if kind == "heading":
            story.append(
                Paragraph(_pdf_text(_block_text(block)), styles[f"Heading{_heading_level(block)}"])
            )
        elif kind in {"bullets", "numbered"}:
            items = [
                ListItem(Paragraph(_pdf_text(item), styles["BodyText"]))
                for item in block.items or []
            ]
            if items:
                story.append(ListFlowable(items, bulletType="bullet" if kind == "bullets" else "1"))
        elif kind == "table" and any(block.rows or []):
            grid, width = _table_grid(block.rows or [])
            # colWidths is NOT optional here. Left to size itself, reportlab
            # measures the natural width of the content and RAISES when it
            # overflows the frame - so one long cell (a description column) kills
            # the whole document. Dividing the usable page evenly makes every
            # table fit by construction; wide ones wrap hard, which beats
            # failing.
            table = Table(
                [[Paragraph(_pdf_text(cell), styles["BodyText"]) for cell in row] for row in grid],
                colWidths=[_PDF_USABLE_WIDTH / width] * width,
            )
            table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(table)
        else:
            story.append(Paragraph(_pdf_text(_block_text(block)), styles["BodyText"]))
        story.append(Spacer(1, 4 * mm))

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=spec.title or "",
        leftMargin=_PDF_MARGIN,
        rightMargin=_PDF_MARGIN,
        topMargin=_PDF_MARGIN,
        bottomMargin=_PDF_MARGIN,
    )
    document.build(story)
    return Rendered(buffer.getvalue(), _plural(len(blocks), "block"))


def _pdf_text(value: str) -> str:
    """Escape text for a platypus Paragraph, which parses a small markup dialect.

    Unescaped, a stray ``&`` or ``<`` in ordinary prose aborts the whole render.
    """
    return escape(_clean_text(value))


def _render_csv(spec: DocumentSpec) -> Rendered:
    """Write UTF-8 CSV with the stdlib writer (RFC 4180 quoting, CRLF endings)."""
    rows = spec.rows
    if not rows and spec.sheets:
        rows = spec.sheets[0].rows
    if not rows:
        raise SpecError("A CSV needs `rows` - a list of rows, each a list of cell values.")
    if len(rows) > MAX_CSV_ROWS:
        raise CapExceeded(
            f"{len(rows)} rows exceeds the limit of {MAX_CSV_ROWS} per CSV. "
            "Send fewer rows or split the data across several files."
        )

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    for row in rows:
        writer.writerow(["" if cell is None else cell for cell in row])
    return Rendered(buffer.getvalue().encode("utf-8"), _plural(len(rows), "row"))


def _render_md(spec: DocumentSpec) -> Rendered:
    """Write Markdown: the caller's text verbatim, or their blocks rendered down."""
    if spec.text:
        body = spec.text
    elif spec.blocks:
        _check_document_caps(spec.blocks)
        body = "\n\n".join(_markdown_block(block) for block in spec.blocks)
    elif spec.title:
        body = ""
    else:
        raise SpecError("A Markdown file needs `text` (or `blocks`, or at least a `title`).")

    if spec.title:
        body = f"# {spec.title}\n\n{body}" if body else f"# {spec.title}\n"
    if not body.endswith("\n"):
        body += "\n"
    return Rendered(body.encode("utf-8"), _plural(len(body.splitlines()), "line"))


def _markdown_block(block: Block) -> str:
    """One block as Markdown; an unknown type degrades to a plain paragraph."""
    kind = (block.type or "").strip().lower()
    if kind == "heading":
        return f"{'#' * (_heading_level(block) + 1)} {_block_text(block)}"
    if kind == "bullets":
        return "\n".join(f"- {item}" for item in block.items or [])
    if kind == "numbered":
        return "\n".join(f"{n}. {item}" for n, item in enumerate(block.items or [], start=1))
    if kind == "table" and any(block.rows or []):
        grid, width = _table_grid(block.rows or [])
        # A pipe inside a cell is the column separator, so an unescaped one
        # silently splits the row into the wrong number of columns.
        head, *body = [[cell.replace("|", r"\|") for cell in row] for row in grid]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * width]
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        return "\n".join(lines)
    return _block_text(block)
