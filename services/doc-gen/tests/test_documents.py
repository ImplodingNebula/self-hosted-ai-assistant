"""Document round trips: every format is REOPENED with a library that reads it.

"Bytes came back" proves nothing - a truncated workbook and a real one are both
bytes. Each test here writes through the real tool call and then parses the
result with openpyxl / python-docx / pypdf / csv, asserting the content survived
the trip.
"""

from __future__ import annotations

import csv
import io

import pytest
from docx import Document
from openpyxl import load_workbook
from pypdf import PdfReader

from app import artifacts, documents
from app.main import build_app
from tests.helpers import call_content, make_document, mcp_session, text_only

pytestmark = pytest.mark.anyio


async def test_xlsx_round_trip_reopens_with_openpyxl() -> None:
    spec = {
        "filename": "quarterly report",
        "sheets": [
            {
                "name": "Revenue",
                "header_row": True,
                "rows": [["Region", "Q1", "Q2"], ["North", 120.5, 130], ["South", 90, 95]],
            },
            {"name": "Notes", "rows": [["Prepared for the board", True, None]]},
        ],
    }
    async with mcp_session(build_app()) as session:
        summary, data, filename = await make_document(session, "xlsx", spec)

    assert filename == "quarterly-report.xlsx"
    assert "2 sheets" in summary

    workbook = load_workbook(io.BytesIO(data))
    assert workbook.sheetnames == ["Revenue", "Notes"]
    revenue = workbook["Revenue"]
    assert [cell.value for cell in revenue[1]] == ["Region", "Q1", "Q2"]
    assert revenue.cell(row=2, column=2).value == 120.5
    assert revenue.cell(row=2, column=3).value == 130
    # header_row is what makes the first row read as headings, not data.
    assert revenue.cell(row=1, column=1).font.bold is True
    assert revenue.cell(row=2, column=1).font.bold is not True
    notes = workbook["Notes"]
    assert notes.cell(row=1, column=2).value is True
    assert notes.cell(row=1, column=3).value is None


async def test_xlsx_string_cells_never_become_live_formulas() -> None:
    # A model can echo attacker-controlled text, and openpyxl turns a leading
    # "=" into a formula that runs when the user opens the download. The value
    # must survive verbatim AS TEXT.
    spec = {"sheets": [{"rows": [["=SUM(A2:A9)", "=cmd|'/c calc'!A0", "+1+1", "plain"]]}]}
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(session, "xlsx", spec)

    sheet = load_workbook(io.BytesIO(data)).active
    for column, expected in enumerate(
        ["=SUM(A2:A9)", "=cmd|'/c calc'!A0", "+1+1", "plain"], start=1
    ):
        cell = sheet.cell(row=1, column=column)
        assert cell.value == expected
        assert cell.data_type == "s", f"column {column} was written as {cell.data_type}"


async def test_xlsx_accepts_bare_rows_as_one_sheet() -> None:
    # Forgiveness: a caller who sends the csv shape for a spreadsheet still gets
    # a spreadsheet rather than a refusal.
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(session, "xlsx", {"rows": [["a", 1]]})

    sheet = load_workbook(io.BytesIO(data)).active
    assert sheet.cell(row=1, column=1).value == "a"


async def test_xlsx_sheet_names_are_made_legal_and_unique() -> None:
    # openpyxl RAISES on an illegal character and only WARNS past 31 characters
    # (writing a file some applications cannot open), so a model-chosen name has
    # to be repaired here or it costs the file - loudly or at the recipient.
    # The two identical LONG names are the case the de-duplication exists for:
    # openpyxl de-duplicates by APPENDING, which would push a name truncated to
    # exactly 31 characters back over the limit.
    spec = {
        "sheets": [
            {"name": "Q1/Q2 [draft]", "rows": [["x"]]},
            {"name": "n" * 60, "rows": [["y"]]},
            {"name": "n" * 60, "rows": [["z"]]},
        ]
    }
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(session, "xlsx", spec)

    names = load_workbook(io.BytesIO(data)).sheetnames
    assert len(names) == 3
    assert len(set(names)) == 3
    assert all(len(name) <= 31 for name in names)
    assert not any(set(name) & set("[]:*?/\\") for name in names)


async def test_docx_round_trip_reopens_with_python_docx() -> None:
    spec = {
        "title": "Incident review",
        "blocks": [
            {"type": "heading", "text": "Summary", "level": 2},
            {"type": "paragraph", "text": "The service recovered after a restart."},
            {"type": "bullets", "items": ["Detected at 09:12", "Resolved at 09:31"]},
            {"type": "numbered", "items": ["Roll back", "Re-deploy"]},
            {"type": "table", "rows": [["Stage", "Owner"], ["Detect", "On-call"]]},
        ],
    }
    async with mcp_session(build_app()) as session:
        summary, data, filename = await make_document(session, "docx", spec)

    assert filename == "Incident-review.docx"
    assert "5 blocks" in summary

    document = Document(io.BytesIO(data))
    texts = [paragraph.text for paragraph in document.paragraphs]
    assert "Incident review" in texts
    assert "Summary" in texts
    assert "The service recovered after a restart." in texts
    styles = {paragraph.text: paragraph.style.name for paragraph in document.paragraphs}
    assert styles["Detected at 09:12"] == "List Bullet"
    assert styles["Roll back"] == "List Number"
    assert len(document.tables) == 1
    assert document.tables[0].cell(0, 1).text == "Owner"
    assert document.tables[0].cell(1, 0).text == "Detect"


async def test_docx_renders_an_unknown_block_as_a_paragraph() -> None:
    # Forgiving by design: a small model inventing a block type must not cost
    # the whole document.
    spec = {"blocks": [{"type": "callout", "text": "Read this first."}]}
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(session, "docx", spec)

    document = Document(io.BytesIO(data))
    assert "Read this first." in [paragraph.text for paragraph in document.paragraphs]


async def test_docx_pads_a_ragged_table() -> None:
    spec = {"blocks": [{"type": "table", "rows": [["a", "b", "c"], ["d"]]}]}
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(session, "docx", spec)

    table = Document(io.BytesIO(data)).tables[0]
    assert len(table.columns) == 3
    assert table.cell(1, 2).text == ""


async def test_pdf_round_trip_reopens_with_pypdf() -> None:
    # pypdf, not reportlab: a reader that did not write the file is what makes
    # this a round trip rather than reportlab agreeing with itself.
    spec = {
        "title": "Quarterly report",
        "blocks": [
            {"type": "heading", "text": "Highlights", "level": 1},
            {"type": "paragraph", "text": "Revenue grew across every region."},
            {"type": "table", "rows": [["Region", "Growth"], ["North", "8%"]]},
        ],
    }
    async with mcp_session(build_app()) as session:
        summary, data, filename = await make_document(session, "pdf", spec)

    assert filename == "Quarterly-report.pdf"
    assert "3 blocks" in summary
    assert data.startswith(b"%PDF-")

    reader = PdfReader(io.BytesIO(data))
    assert len(reader.pages) >= 1
    text = "".join(page.extract_text() for page in reader.pages)
    assert "Quarterly report" in text
    assert "Revenue grew across every region." in text
    assert "North" in text


# reportlab's Paragraph parses a small markup dialect, so an unescaped "&" or
# "<" aborts the render. Escaping is applied at FIVE call sites, and a test that
# only drives prose leaves the other four free to lose it - which is not
# hypothetical: dropping it from table cells left the whole suite green. One
# test per path, each asserting through the path it names.
_MARKUP = "Tom & Jerry <best> \"quoted\" 'apostrophe'"


def _pdf_text_of(data: bytes) -> str:
    """Every page's extracted text, joined."""
    return "".join(page.extract_text() for page in PdfReader(io.BytesIO(data)).pages)


async def test_pdf_survives_markup_characters_in_prose() -> None:
    spec = {"blocks": [{"type": "paragraph", "text": _MARKUP}]}
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(session, "pdf", spec)

    assert "Tom & Jerry <best>" in _pdf_text_of(data)


async def test_pdf_survives_markup_characters_in_the_title() -> None:
    # The title is its own call site - and the likeliest place for an ampersand
    # ("Sales & Marketing"), which is exactly what aborts an unescaped render.
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(
            session, "pdf", {"title": _MARKUP, "blocks": [{"type": "paragraph", "text": "body"}]}
        )

    assert "Tom & Jerry <best>" in _pdf_text_of(data)


async def test_pdf_survives_markup_characters_in_a_table_cell() -> None:
    # Table cells are their own call site too, and had no coverage at all: the
    # escaping was removed from this path and all 85 other tests stayed green.
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(
            session, "pdf", {"blocks": [{"type": "table", "rows": [["header"], [_MARKUP]]}]}
        )

    assert "Tom & Jerry <best>" in _pdf_text_of(data)


async def test_pdf_tables_survive_content_wider_than_the_page() -> None:
    # Regression: left to size its own columns, reportlab measures the natural
    # width of the content and RAISES when it overflows the frame - so a single
    # long description cell failed the whole document. Columns now divide the
    # usable page, which makes every table fit by construction.
    spec = {
        "blocks": [
            {"type": "table", "rows": [["x" * 400, "a short one"]]},
            {"type": "table", "rows": [[f"column heading {n}" for n in range(15)]]},
        ]
    }
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(session, "pdf", spec)

    assert data.startswith(b"%PDF-")
    assert len(PdfReader(io.BytesIO(data)).pages) >= 1


@pytest.mark.parametrize("fmt", ["docx", "pdf", "md"])
async def test_a_table_block_with_no_columns_does_not_fail_the_document(fmt: str) -> None:
    # rows=[[]] is a non-empty list of EMPTY rows, which asks for a zero-column
    # table. reportlab REFUSES one outright ("must have at least a row and
    # column"); python-docx accepts it and writes an empty table nobody wanted.
    # The block degrades to a paragraph instead, and the rest of the document
    # still renders.
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(
            session,
            fmt,
            {"blocks": [{"type": "table", "rows": [[]]}, {"type": "paragraph", "text": "after"}]},
        )

    if fmt == "docx":
        document = Document(io.BytesIO(data))
        assert document.tables == []
        assert "after" in [paragraph.text for paragraph in document.paragraphs]
    elif fmt == "pdf":
        assert "after" in "".join(page.extract_text() for page in PdfReader(io.BytesIO(data)).pages)
    else:
        assert "after" in data.decode("utf-8")


@pytest.mark.parametrize("fmt", ["xlsx", "docx"])
async def test_control_characters_are_stripped_rather_than_failing_the_file(fmt: str) -> None:
    # Both Office writers REFUSE a control character outright (openpyxl raises
    # IllegalCharacterError, python-docx raises "All strings must be XML
    # compatible"), so one stray byte in model-written text would cost the whole
    # file. It is stripped, and the rest of the text survives.
    dirty = "a\x07b\x0bc"
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(
            session,
            fmt,
            {"rows": [[dirty]], "blocks": [{"type": "paragraph", "text": dirty}]},
        )

    if fmt == "xlsx":
        assert load_workbook(io.BytesIO(data)).active.cell(row=1, column=1).value == "abc"
    else:
        assert "abc" in [paragraph.text for paragraph in Document(io.BytesIO(data)).paragraphs]


async def test_markdown_table_cells_escape_the_column_separator() -> None:
    # An unescaped pipe inside a cell IS the column separator, so it silently
    # splits the row into the wrong number of columns.
    spec = {"blocks": [{"type": "table", "rows": [["a|b", "c"], ["d", "e"]]}]}
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(session, "md", spec)

    assert r"| a\|b | c |" in data.decode("utf-8")


async def test_pdf_accepts_a_title_only_document() -> None:
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(session, "pdf", {"title": "Just a title"})

    assert len(PdfReader(io.BytesIO(data)).pages) == 1


async def test_csv_round_trip_reads_with_the_csv_module() -> None:
    spec = {"filename": "export.csv", "rows": [["name", "count"], ["Zoe", 3], ["Ana", None]]}
    async with mcp_session(build_app()) as session:
        summary, data, filename = await make_document(session, "csv", spec)

    assert filename == "export.csv"
    assert "3 rows" in summary
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
    assert rows == [["name", "count"], ["Zoe", "3"], ["Ana", ""]]


async def test_csv_quotes_separators_and_keeps_utf8() -> None:
    spec = {"rows": [["a,b", 'say "hi"'], ["Zoë", "naïve"]]}
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(session, "csv", spec)

    rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
    assert rows == [["a,b", 'say "hi"'], ["Zoë", "naïve"]]


async def test_md_passes_text_through_under_its_title() -> None:
    spec = {"title": "Notes", "text": "- one\n- two\n"}
    async with mcp_session(build_app()) as session:
        _, data, filename = await make_document(session, "md", spec)

    assert filename == "Notes.md"
    assert data.decode("utf-8") == "# Notes\n\n- one\n- two\n"


async def test_md_renders_blocks_when_no_text_was_sent() -> None:
    spec = {
        "blocks": [
            {"type": "heading", "text": "Steps", "level": 1},
            {"type": "numbered", "items": ["first", "second"]},
            {"type": "table", "rows": [["k", "v"], ["a", "b"]]},
        ]
    }
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(session, "md", spec)

    body = data.decode("utf-8")
    assert "## Steps" in body
    assert "1. first" in body
    assert "| k | v |" in body


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("xlsx", "sheets"),
        ("docx", "blocks"),
        ("pdf", "blocks"),
        ("csv", "rows"),
        ("md", "text"),
    ],
)
async def test_an_empty_spec_is_told_what_is_missing(fmt: str, expected: str) -> None:
    # A refusal the caller can act on, as ordinary text: no artifact, no error
    # flag (an errored tool gets dropped for the rest of the turn, taking the
    # retry with it).
    async with mcp_session(build_app()) as session:
        result = await session.call_tool("create_document", {"format": fmt, "spec": {}})

    assert result.isError is False
    assert expected in text_only(list(result.content))


# The cap tests below spell their boundaries as LITERALS on purpose. Deriving
# the sizes from the module constants makes the test move WITH the constant:
# raise the cap and a derived test happily re-proves the new one, which is a
# test that can never fail. Written as literals, each test fails if the cap is
# raised, lowered, or stopped being checked at all.


async def test_the_workbook_cell_cap_is_enforced_at_the_boundary() -> None:
    # 5,000 rows x 10 columns = the 50,000-cell limit exactly.
    row = ["x"] * 10
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(session, "xlsx", {"rows": [row] * 5_000})
        assert load_workbook(io.BytesIO(data)).active.max_row == 5_000

        over = await call_content(
            session,
            "create_document",
            {"format": "xlsx", "spec": {"rows": [row] * 5_000 + [["x"]]}},
        )

    message = text_only(over)
    assert "50001 cells" in message
    assert str(documents.MAX_WORKBOOK_CELLS) in message


# Parametrized over every format that shares the guard, not just the first one:
# three formats call _check_document_caps, and a test that only drives docx
# leaves the pdf and md paths free to lose their call entirely.
@pytest.mark.parametrize("fmt", ["docx", "pdf", "md"])
async def test_the_document_block_cap_is_enforced_at_the_boundary(fmt: str) -> None:
    block = {"type": "paragraph", "text": "line"}
    at_cap = [block] * 200
    async with mcp_session(build_app()) as session:
        await make_document(session, fmt, {"blocks": at_cap})

        over = await call_content(
            session,
            "create_document",
            {"format": fmt, "spec": {"blocks": at_cap + [block]}},
        )

    message = text_only(over)
    assert "201 blocks" in message
    assert str(documents.MAX_DOCUMENT_BLOCKS) in message


@pytest.mark.parametrize("fmt", ["docx", "pdf", "md"])
async def test_the_table_row_cap_is_enforced_at_the_boundary(fmt: str) -> None:
    rows = [["a", "b"]] * 500
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(
            session, fmt, {"blocks": [{"type": "table", "rows": rows}]}
        )
        if fmt == "docx":
            assert len(Document(io.BytesIO(data)).tables[0].rows) == 500

        over = await call_content(
            session,
            "create_document",
            {
                "format": fmt,
                "spec": {"blocks": [{"type": "table", "rows": rows + [["c", "d"]]}]},
            },
        )

    message = text_only(over)
    assert "501 rows" in message
    assert str(documents.MAX_TABLE_ROWS) in message


async def test_the_csv_row_cap_is_enforced_at_the_boundary() -> None:
    # CSV has no structural limit of its own, so without this the only ceiling
    # is the output-byte check - which refuses only AFTER building the file.
    at_cap = [["a", "b"]] * 50_000
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(session, "csv", {"rows": at_cap})
        assert len(data.splitlines()) == 50_000

        over = await call_content(
            session, "create_document", {"format": "csv", "spec": {"rows": at_cap + [["c", "d"]]}}
        )

    message = text_only(over)
    assert "50001 rows" in message
    assert str(documents.MAX_CSV_ROWS) in message


async def test_the_total_input_cap_is_enforced_at_the_boundary() -> None:
    at_cap = "x" * 1_000_000
    async with mcp_session(build_app()) as session:
        _, data, _ = await make_document(session, "md", {"text": at_cap})
        assert len(data) == 1_000_001  # the trailing newline the writer adds

        over = await call_content(
            session, "create_document", {"format": "md", "spec": {"text": at_cap + "x"}}
        )

    message = text_only(over)
    assert "1000001 characters" in message
    assert str(artifacts.MAX_INPUT_CHARS) in message


async def test_the_input_cap_counts_every_string_in_the_spec_together() -> None:
    # The failure this exists for: every OTHER cap bounds a quantity, so fifty
    # bullet items of 40k characters each passes all of them and still hands
    # reportlab two megabytes to lay out - work paid in full before the output
    # check gets a chance to refuse the result. No single field is over any
    # limit here, and the request is nowhere near the 4 MiB body size the MCP
    # server itself refuses; the TOTAL is what is over.
    items = ["x" * 40_000] * 50
    async with mcp_session(build_app()) as session:
        over = await call_content(
            session,
            "create_document",
            {"format": "pdf", "spec": {"blocks": [{"type": "bullets", "items": items}]}},
        )

    message = text_only(over)
    # 2,000,000 of bullet text plus the 7 characters of the block's own "bullets"
    # type - the walk counts every string it reaches, including the ones nobody
    # thinks of as content.
    assert "2000007 characters" in message
    assert str(artifacts.MAX_INPUT_CHARS) in message


async def test_the_input_cap_covers_charts_as_well_as_documents() -> None:
    # Labels are caller-controlled strings that matplotlib lays out, so the
    # chart tool needs the same bound - wiring it into only one of the two tools
    # would leave the other with no size check at all.
    async with mcp_session(build_app()) as session:
        over = await call_content(
            session,
            "render_chart",
            {"spec": {"labels": ["x" * 600_000, "y" * 600_000], "series": [{"values": [1, 2]}]}},
        )

    assert str(artifacts.MAX_INPUT_CHARS) in text_only(over)


def test_the_input_cap_fits_under_the_output_byte_cap() -> None:
    # An at-cap input must not be one the output check was always going to
    # refuse anyway, or the earlier limit is decoration. Necessary, not
    # sufficient - a renderer can still inflate its input, which is what leaves
    # the output check as the backstop rather than as the only guard.
    assert artifacts.MAX_INPUT_CHARS <= artifacts.MAX_OUTPUT_BYTES


def test_the_input_walk_reaches_nested_strings_not_just_top_level_ones() -> None:
    # Counting only the fields it knows by name is how a size check quietly
    # stops covering the newest one; the walk is structural for that reason.
    spec = documents.DocumentSpec.model_validate(
        {
            "title": "abcde",
            "blocks": [{"type": "table", "rows": [["ab", "c"]], "items": ["de"]}],
            "sheets": [{"name": "fg", "rows": [["hij", 42, True, None]]}],
        }
    )
    # title 5 + type 5 + cells 3 + items 2 + name 2 + cell 3 = 20; the int, the
    # bool and the None contribute nothing.
    assert artifacts.total_input_chars(spec) == 20


def test_the_output_byte_cap_rejects_before_base64() -> None:
    # The single choke point every artifact passes through, so no renderer can
    # forget it. At 1,200,000 bytes it packs; one byte over it refuses - and
    # refuses BEFORE base64, which would have made a third copy of an oversized
    # payload.
    at_cap = artifacts.artifact_result(
        summary="ok",
        filename="big.pdf",
        data=b"\0" * 1_200_000,
        media_type="application/pdf",
    )
    assert len(at_cap.content) == 2

    with pytest.raises(artifacts.CapExceeded) as excinfo:
        artifacts.artifact_result(
            summary="ok",
            filename="big.pdf",
            data=b"\0" * 1_200_001,
            media_type="application/pdf",
        )
    assert "1.1 MiB limit" in str(excinfo.value)


def test_the_output_byte_cap_fits_under_the_host_transport_cap() -> None:
    # The reason the cap is 1,200,000 and not a rounder, larger number: the
    # platform's MCP client reads a whole response body under a hard 2,000,000
    # byte ceiling, and breaking it raises - failing the ENTIRE tool call rather
    # than dropping the file and keeping the summary. Base64 costs 4/3, so this
    # is the arithmetic that has to keep closing if the cap is ever raised.
    host_response_cap = 2_000_000
    base64_length = 4 * ((artifacts.MAX_OUTPUT_BYTES + 2) // 3)
    assert base64_length == 1_600_000
    # The remaining headroom carries the summary text and the JSON envelope.
    assert host_response_cap - base64_length >= 400_000


def test_an_oversized_svg_is_told_to_ask_for_a_png() -> None:
    # An SVG holds one element per data point, so it is the one output that
    # grows without bound - and the same chart as a png is a fixed-size raster.
    # "Send less data" would be misleading where changing one field fixes it.
    with pytest.raises(artifacts.CapExceeded) as excinfo:
        artifacts.artifact_result(
            summary="ok",
            filename="big.svg",
            data=b"\0" * (artifacts.MAX_OUTPUT_BYTES + 1),
            media_type="image/svg+xml",
        )
    assert "png" in str(excinfo.value)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("../../etc/passwd", "passwd.csv"),
        (r"C:\Users\me\report.csv", "report.csv"),
        ("Q1 2026 review", "Q1-2026-review.csv"),
        ("  ...leading junk", "leading-junk.csv"),
        ("résumé", "r-sum.csv"),
        ("", "table.csv"),
        (None, "table.csv"),
        ("report.CSV", "report.csv"),
        ("a" * 200, "a" * 80 + ".csv"),
    ],
)
def test_filenames_are_sanitized_to_one_safe_segment(candidate: str | None, expected: str) -> None:
    assert artifacts.sanitize_filename(candidate, fallback="table", extension="csv") == expected
