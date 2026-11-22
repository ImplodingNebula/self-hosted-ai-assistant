"""Office formats: what is read, what is refused, and what must never be dropped."""

import zipfile
from io import BytesIO

import pytest

from app.parsing.archive import DOCX_MARKER, XLSX_MARKER
from app.parsing.base import ParseFailedError, UnsupportedFileError
from app.parsing.office import DocxParser, PptxParser, XlsxParser
from app.parsing.registry import default_registry

# Fixtures are BUILT, never committed: a binary blob in the repo is a thing
# nobody can review, and the same libraries that read these formats write them.


def make_docx(paragraphs: list[str] | None = None, table: list[list[str]] | None = None) -> bytes:
    """Build a .docx with the given paragraphs and (optionally) one table."""
    import docx

    document = docx.Document()
    for text in paragraphs or []:
        document.add_paragraph(text)
    if table:
        built = document.add_table(rows=len(table), cols=len(table[0]))
        for row_index, row in enumerate(table):
            for cell_index, value in enumerate(row):
                built.cell(row_index, cell_index).text = value
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_xlsx(sheets: dict[str, list[list[object]]]) -> bytes:
    """Build an .xlsx from {sheet name: rows}."""
    import openpyxl

    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for name, rows in sheets.items():
        worksheet = workbook.create_sheet(title=name)
        for row in rows:
            worksheet.append(row)
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def make_pptx(slides: list[tuple[str, str]], notes: str | None = None) -> bytes:
    """Build a .pptx of (title, body) slides, with optional notes on the first."""
    from pptx import Presentation

    presentation = Presentation()
    layout = presentation.slide_layouts[1]  # title and content
    for title, body in slides:
        slide = presentation.slides.add_slide(layout)
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    if notes and presentation.slides:
        presentation.slides[0].notes_slide.notes_text_frame.text = notes
    buffer = BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


async def test_a_word_document_reads_as_one_unit() -> None:
    data = make_docx(["The rollout begins on Tuesday.", "Rollback takes ten minutes."])

    documents = await DocxParser().parse(data, filename="plan.docx")

    assert len(documents) == 1
    assert "rollout begins on Tuesday" in documents[0].text
    assert "Rollback takes ten minutes" in documents[0].text
    # One unit, so the filename already names it - a title would just repeat.
    assert documents[0].title is None


async def test_a_word_documents_tables_are_not_dropped() -> None:
    # The bug this pins: `document.paragraphs` returns ONLY paragraphs, so a file
    # whose content lives in a table (a form, a spec sheet, a price list) would
    # ingest as EMPTY while reporting success - the worst kind of wrong, because
    # the file looks readable and every answer from it is silently ungrounded.
    data = make_docx(
        paragraphs=["Rates for 2026:"],
        table=[["Region", "Rate"], ["EMEA", "4.5%"], ["APAC", "6.1%"]],
    )

    documents = await DocxParser().parse(data, filename="rates.docx")

    text = documents[0].text
    assert "Rates for 2026" in text
    assert "EMEA" in text
    assert "4.5%" in text
    assert "APAC" in text
    # The caption still leads its table: the body is walked in reading order.
    assert text.index("Rates for 2026") < text.index("EMEA")


async def test_an_empty_word_document_yields_nothing() -> None:
    assert await DocxParser().parse(make_docx([]), filename="empty.docx") == []


async def test_a_spreadsheet_is_one_unit_per_sheet_named_by_its_tab() -> None:
    # The sheet name is the whole point: a citation that says "Q3 Forecast" tells
    # the reader where the number came from; "book.xlsx" does not.
    data = make_xlsx(
        {
            "Q3 Forecast": [["Region", "Revenue"], ["EMEA", 41207]],
            "Assumptions": [["Churn", "2%"]],
        }
    )

    documents = await XlsxParser().parse(data, filename="book.xlsx")

    assert [d.title for d in documents] == ["Q3 Forecast", "Assumptions"]
    assert "41207" in documents[0].text
    assert "Churn" in documents[1].text


async def test_an_empty_sheet_is_skipped_but_the_next_keeps_its_position() -> None:
    data = make_xlsx({"Blank": [], "Data": [["value", 7]]})

    documents = await XlsxParser().parse(data, filename="book.xlsx")

    assert [d.title for d in documents] == ["Data"]
    # Ordinal is the sheet's REAL position, not its position among non-empty ones,
    # so a merged file stays in workbook order.
    assert documents[0].ordinal == 1


async def test_a_presentation_is_one_unit_per_slide_and_keeps_its_notes() -> None:
    # Notes are often where the actual argument lives - the slide is three words
    # and a chart - so dropping them would throw away the answerable text.
    data = make_pptx(
        [("Margins", "Up 4 points"), ("Risks", "Supply")],
        notes="The gain is one-off: a supplier rebate, not pricing.",
    )

    documents = await PptxParser().parse(data, filename="deck.pptx")

    assert [d.title for d in documents] == ["slide 1", "slide 2"]
    assert "Up 4 points" in documents[0].text
    assert "supplier rebate" in documents[0].text
    assert "Supply" in documents[1].text


async def test_the_registry_claims_each_office_format_by_extension() -> None:
    registry = default_registry()

    assert isinstance(registry.resolve(filename="a.docx"), DocxParser)
    assert isinstance(registry.resolve(filename="a.xlsx"), XlsxParser)
    assert isinstance(registry.resolve(filename="a.pptx"), PptxParser)
    # Macro-enabled files are the same container; nothing here executes a macro.
    assert isinstance(registry.resolve(filename="a.docm"), DocxParser)


async def test_a_mislabelled_office_file_is_claimed_by_what_is_inside_it() -> None:
    # Every office format is a zip, so shape alone claims nothing - the part
    # inside decides. Without this, a .docx sent as octet-stream with no
    # extension would fall through to the text parser and decode to garbage.
    registry = default_registry()

    parser = registry.resolve(
        filename="attachment", content_type="application/octet-stream", data=make_docx(["hi"])
    )

    assert isinstance(parser, DocxParser)


async def test_a_binary_format_declared_as_plain_text_is_still_claimed_by_its_signature() -> None:
    """A declared text/plain must not beat a real signature.

    text/plain is what a browser sends when it does not recognise a file, and what a
    hand-rolled upload sends when nobody set a type - so it is a guess, not a fact, and
    it arrives on genuinely binary files all the time. Honouring it outright would hand
    a .docx to the reader that reads anything, badly: the zip decodes to mojibake, the
    row goes READY, and the garbage is embedded and retrievable.

    The text parser's claim is held until the signatures have had their say.
    """
    registry = default_registry()

    parser = registry.resolve(
        filename="attachment", content_type="text/plain", data=make_docx(["real prose"])
    )

    assert isinstance(parser, DocxParser)


async def test_a_genuine_text_file_declared_as_text_is_still_text() -> None:
    # The other half: holding the text claim must not lose it. Nothing signs plain
    # prose, so the held claim is honoured once the signatures come up empty.
    from app.parsing.text import PlainTextParser

    registry = default_registry()

    parser = registry.resolve(
        filename="notes.txt", content_type="text/plain", data=b"just some notes"
    )

    assert isinstance(parser, PlainTextParser)


async def test_a_plain_zip_is_refused_rather_than_read_as_text() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", "just a zip")

    with pytest.raises(UnsupportedFileError):
        default_registry().resolve(filename="bundle.zip", data=buffer.getvalue())


async def test_a_legacy_office_file_is_refused_honestly() -> None:
    # .doc/.xls/.ppt are OLE2 compound files, not zips. Nothing here reads them,
    # and they must be refused at upload rather than half-decoded into nonsense.
    ole2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512

    with pytest.raises(UnsupportedFileError):
        default_registry().resolve(filename="report.doc", data=ole2)


def _repack(data: bytes, part: str, content: bytes) -> bytes:
    """Rebuild an OOXML archive with one part's bytes replaced.

    DEFLATED, not stored: a zipfile writes uncompressed by default, and a bomb
    that is not actually compressed is not a bomb - it would sail past a ratio
    check while proving nothing.
    """
    source = BytesIO(data)
    out = BytesIO()
    with (
        zipfile.ZipFile(source) as original,
        zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as rebuilt,
    ):
        for entry in original.infolist():
            payload = content if entry.filename == part else original.read(entry.filename)
            rebuilt.writestr(entry.filename, payload)
    return out.getvalue()


async def test_a_word_documents_content_controls_are_not_dropped() -> None:
    # The same silent-loss class as the tables bug: a form's fields live inside
    # content controls (w:sdt), which are neither paragraphs nor tables, so a body
    # walk that only knew those two would ingest the form as empty.
    import docx
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls

    document = docx.Document()
    document.add_paragraph("Applicant details:")
    sdt = parse_xml(
        f"<w:sdt {nsdecls('w')}><w:sdtContent>"
        "<w:p><w:r><w:t>Jane Marisol Okonkwo</w:t></w:r></w:p>"
        "</w:sdtContent></w:sdt>"
    )
    document.element.body.append(sdt)
    buffer = BytesIO()
    document.save(buffer)

    documents = await DocxParser().parse(buffer.getvalue(), filename="form.docx")

    assert "Jane Marisol Okonkwo" in documents[0].text


async def test_an_office_file_declaring_an_xml_entity_is_refused() -> None:
    """The XXE guard, pinned.

    A real office file never declares a DOCTYPE. One that does is either broken
    or trying to make the XML parser read the server's disk (or expand a few
    nested entities into gigabytes). The container is checked directly rather
    than trusting each library's parser settings - openpyxl in particular
    switches to lxml whenever lxml is importable and does NOT disable entities,
    and lxml became importable the moment python-docx was added.
    """
    hostile = (
        b'<?xml version="1.0"?>\n'
        b'<!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>\n'
        b"<workbook>&xxe;</workbook>"
    )
    data = _repack(make_xlsx({"S": [["a"]]}), XLSX_MARKER, hostile)

    with pytest.raises(ParseFailedError, match="document type or entity"):
        await XlsxParser().parse(data, filename="book.xlsx")


async def test_an_entity_hidden_behind_a_long_comment_is_still_caught() -> None:
    # A DOCTYPE is only legal in the prolog, but a part may carry a long leading
    # comment, so a fixed head-window scan could be jumped: pad a comment past the
    # window, then declare the entity. The whole part is scanned, so it is caught.
    hidden = (
        b'<?xml version="1.0"?>\n'
        b"<!-- " + b"A" * 20000 + b" -->\n"
        b'<!DOCTYPE x [<!ENTITY e "boom">]>\n'
        b"<workbook>&e;</workbook>"
    )
    data = _repack(make_xlsx({"S": [["a"]]}), XLSX_MARKER, hidden)

    with pytest.raises(ParseFailedError, match="document type or entity"):
        await XlsxParser().parse(data, filename="book.xlsx")


async def test_an_archive_with_absurdly_many_parts_is_refused() -> None:
    # Each container check is per-part, and so is the parsing library after it, so
    # a file with hundreds of thousands of tiny parts is a way to make that work
    # expensive. It is refused on the count, before any of it is done.
    from app.parsing.archive import _MAX_ENTRIES

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(DOCX_MARKER, b"<w:document/>")
        for index in range(_MAX_ENTRIES + 5):
            archive.writestr(f"junk/{index}.xml", b"<a/>")

    with pytest.raises(ParseFailedError, match="too many parts"):
        await DocxParser().parse(buffer.getvalue(), filename="many.docx")


async def test_a_compression_bomb_is_refused_without_being_decompressed() -> None:
    # A zip declares each entry's uncompressed size in its own directory, so an
    # archive that expands absurdly is recognised for what it is BEFORE anything
    # is inflated. The upload cap bounds what ARRIVES; it says nothing about what
    # that becomes - 60MB of padding here rides in as a few dozen KB.
    bomb = _repack(make_docx(["x"]), DOCX_MARKER, b"<w:document>" + b"A" * (60 * 1024 * 1024))
    assert len(bomb) < 1024 * 1024  # it really is small on the wire

    with pytest.raises(ParseFailedError, match="compression bomb"):
        await DocxParser().parse(bomb, filename="bomb.docx")


async def test_an_entity_hidden_in_a_duplicated_part_name_is_still_caught() -> None:
    """The scan reads the parts, not the names, and this is why.

    A zip may hold two entries under ONE name - nothing in the format forbids it -
    and a lookup by name resolves to whichever the central directory saw last. So a
    scan that walks the names and reads each one back by name examines only the
    second copy, and a hostile first copy is never looked at. The check would pass,
    and a parsing library reading the archive its own way might well take the copy
    that was never scanned.

    Iterating the entries themselves closes it: every copy of a duplicated name is
    an entry, so every copy is read.
    """
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            DOCX_MARKER,
            b'<?xml version="1.0"?>\n'
            b'<!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>\n'
            b"<w:document>&xxe;</w:document>",
        )
        # Same name, second slot: the innocent one, and the only one a name-keyed
        # read would ever see.
        archive.writestr(DOCX_MARKER, b"<w:document/>")

    with pytest.raises(ParseFailedError, match="document type or entity"):
        await DocxParser().parse(buffer.getvalue(), filename="twins.docx")


async def test_a_corrupt_office_file_fails_honestly() -> None:
    with pytest.raises(ParseFailedError, match="corrupt"):
        await DocxParser().parse(b"PK\x03\x04 not really a zip", filename="broken.docx")


async def test_an_empty_file_yields_nothing_rather_than_failing() -> None:
    assert await XlsxParser().parse(b"", filename="empty.xlsx") == []
