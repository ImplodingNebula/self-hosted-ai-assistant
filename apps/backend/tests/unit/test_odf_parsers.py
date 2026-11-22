"""OpenDocument: what is read, what is skipped, and the sheet that lies about its size."""

import zipfile
from io import BytesIO

import pytest

from app.parsing.base import ParseFailedError
from app.parsing.odf import OdpParser, OdsParser, OdtParser
from app.parsing.registry import default_registry

# Fixtures are BUILT, never committed. Unlike the office formats there is no library
# here that WRITES OpenDocument, so these assemble the zip and the XML by hand - which
# is no loss: it puts the exact hostile shapes each test needs in plain sight.

_NAMESPACES = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
    'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"'
)

ODT_TYPE = "application/vnd.oasis.opendocument.text"
ODS_TYPE = "application/vnd.oasis.opendocument.spreadsheet"
ODP_TYPE = "application/vnd.oasis.opendocument.presentation"


def make_package(
    media_type: str, parts: dict[str, bytes | str], *, compress: bool = False
) -> bytes:
    """Build an OpenDocument package: the mimetype entry, then the given parts.

    The mimetype entry is written first and STORED rather than deflated, which is
    what the format requires and what makes a package identifiable from its first
    few dozen bytes.
    """
    buffer = BytesIO()
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, "w", compression=method) as archive:
        archive.writestr(zipfile.ZipInfo("mimetype"), media_type, compress_type=zipfile.ZIP_STORED)
        for name, content in parts.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def make_odf(media_type: str, body: str, *, compress: bool = False) -> bytes:
    """Build a package whose content.xml wraps ``body`` in a document shell."""
    content = (
        f'<?xml version="1.0"?><office:document-content {_NAMESPACES}>'
        f"<office:body>{body}</office:body></office:document-content>"
    )
    return make_package(
        media_type,
        {"content.xml": content, "META-INF/manifest.xml": "<manifest/>"},
        compress=compress,
    )


def make_odt(body: str) -> bytes:
    """Build an .odt whose text body is ``body``."""
    return make_odf(ODT_TYPE, f"<office:text>{body}</office:text>")


def make_ods(tables: str) -> bytes:
    """Build an .ods whose spreadsheet body is the given table elements."""
    return make_odf(ODS_TYPE, f"<office:spreadsheet>{tables}</office:spreadsheet>")


def make_odp(pages: str) -> bytes:
    """Build an .odp whose presentation body is the given draw:page elements."""
    return make_odf(ODP_TYPE, f"<office:presentation>{pages}</office:presentation>")


def cells(*values: str) -> str:
    """Render one table row of text cells."""
    body = "".join(
        f"<table:table-cell><text:p>{value}</text:p></table:table-cell>" for value in values
    )
    return f"<table:table-row>{body}</table:table-row>"


async def test_a_text_document_reads_as_one_unit() -> None:
    data = make_odt("<text:h>Turbine report</text:h><text:p>The bearing failed.</text:p>")

    documents = await OdtParser().parse(data, filename="report.odt")

    assert len(documents) == 1
    assert "Turbine report" in documents[0].text
    assert "The bearing failed" in documents[0].text
    # One unit, so the filename already names it - a title would just repeat.
    assert documents[0].title is None


async def test_a_text_documents_tables_are_not_dropped() -> None:
    # The same silent-loss class the Word reader pins: a document whose content is a
    # table (a form, a price list) must not ingest as empty while reporting success.
    data = make_odt(
        "<text:p>Rates for 2026:</text:p>"
        f"<table:table>{cells('Region', 'Rate')}{cells('EMEA', '4.5%')}</table:table>"
    )

    text = (await OdtParser().parse(data, filename="rates.odt"))[0].text

    assert "Rates for 2026" in text
    assert "EMEA" in text
    assert "4.5%" in text
    # The caption still leads its table: the body is walked in reading order.
    assert text.index("Rates for 2026") < text.index("EMEA")


async def test_deleted_text_is_not_read_back_out_of_a_tracked_change() -> None:
    """Text that was DELETED must not be answered from.

    A document with tracked changes keeps the superseded text in the file, inside
    text:tracked-changes. Walking the tree naively pulls it back out - so a contract
    whose price was revised would ground an answer on the price that was struck,
    stated as fact and cited. The retraction is invisible; the wrong number is not.
    """
    data = make_odt(
        "<text:tracked-changes><text:changed-region><text:deletion>"
        "<text:p>The fee is 12,000 EUR.</text:p>"
        "</text:deletion></text:changed-region></text:tracked-changes>"
        "<text:p>The fee is 9,000 EUR.</text:p>"
    )

    text = (await OdtParser().parse(data, filename="contract.odt"))[0].text

    assert "9,000" in text
    assert "12,000" not in text


async def test_a_spreadsheet_is_one_unit_per_sheet_named_by_its_tab() -> None:
    # The sheet name is the whole point: a citation that says "Q3 Forecast" tells the
    # reader where the number came from; "book.ods" does not.
    data = make_ods(
        f'<table:table table:name="Q3 Forecast">{cells("Region", "Revenue")}'
        f"{cells('EMEA', '41207')}</table:table>"
        f'<table:table table:name="Assumptions">{cells("Churn", "2%")}</table:table>'
    )

    documents = await OdsParser().parse(data, filename="book.ods")

    assert [document.title for document in documents] == ["Q3 Forecast", "Assumptions"]
    assert "41207" in documents[0].text
    assert "Churn" in documents[1].text


async def test_a_blank_sheet_claiming_a_billion_cells_is_read_in_an_instant() -> None:
    """The repeat bomb - which is also just what LibreOffice saves.

    ODF run-length-encodes repetition, and a genuinely empty sheet declares itself as
    one row repeated 1,048,576 times holding one cell repeated 16,384 times. Expanding
    those counts literally means materialising seventeen billion cells from a file of
    a few hundred bytes: the worker dies, and it dies on a FILE THAT IS NOT HOSTILE.

    The rule that makes it safe is that a repeat of nothing is nothing. Empty repeats
    are never materialised at any count - so the real rows come back, and the padding
    costs nothing.
    """
    data = make_ods(
        f'<table:table table:name="Q3 Forecast">{cells("Region", "Revenue")}'
        f"{cells('EMEA', '41207')}"
        '<table:table-row table:number-rows-repeated="1048576">'
        '<table:table-cell table:number-columns-repeated="16384"/>'
        "</table:table-row></table:table>"
        # A number no integer type would like, in the position a reader must read.
        '<table:table table:name="Padding">'
        '<table:table-row table:number-rows-repeated="99999999999999999999">'
        '<table:table-cell table:number-columns-repeated="99999999999999999999"/>'
        "</table:table-row></table:table>"
    )

    documents = await OdsParser().parse(data, filename="book.ods")

    # The real sheet reads normally; the sheet that is entirely padding holds no text
    # and is skipped, exactly as an empty sheet is.
    assert [document.title for document in documents] == ["Q3 Forecast"]
    assert "41207" in documents[0].text
    assert "\n\n" not in documents[0].text.strip()


async def test_a_document_of_many_amplifying_tables_is_bounded_not_summed() -> None:
    """The per-sheet cap bounds each table; this proves the document total is bounded too.

    Run-length encoding amplifies PER TABLE, so a file well under the content.xml cap
    can still declare hundreds of sheets that each expand to the per-sheet ceiling -
    and hundreds times 8MB is gigabytes the worker does not have. The document budget
    is what stops that: the read halts at the total, and a small file cannot turn into
    an out-of-memory kill.
    """
    from app.parsing.odf import _MAX_DOCUMENT_CHARS

    tables = "".join(
        f'<table:table table:name="S{index}">'
        '<table:table-row table:number-rows-repeated="256">'
        '<table:table-cell table:number-columns-repeated="256"><text:p>x</text:p></table:table-cell>'
        "</table:table-row></table:table>"
        for index in range(200)
    )
    data = make_ods(tables)
    assert len(data) < 128 * 1024  # tiny on the wire

    documents = await OdsParser().parse(data, filename="huge.ods")

    total = sum(len(document.text) for document in documents)
    # Bounded near the document ceiling - not the 200 x 8MB an unbounded read would build.
    assert total < _MAX_DOCUMENT_CHARS + 64 * 1024


async def test_a_paragraph_of_nothing_but_space_runs_cannot_grow_unbounded() -> None:
    # A paragraph is assembled from elements, so many text:s runs (up to 1000 spaces
    # each) would build a huge string in one paragraph, below every table/document cap.
    # The per-paragraph ceiling stops it.
    from app.parsing.odf import _MAX_INLINE_CHARS

    spaces = '<text:s text:c="1000"/>' * 5000  # 5,000,000 spaces if taken literally
    data = make_odt(f"<text:p>Start{spaces}End</text:p>")

    text = (await OdtParser().parse(data, filename="spaces.odt"))[0].text

    assert "Start" in text
    assert len(text) <= _MAX_INLINE_CHARS


async def test_a_list_inside_a_table_cell_is_not_read_as_empty() -> None:
    # A cell's content is not always a bare paragraph: a bulleted cell wraps its
    # paragraphs in a text:list. Reading only the cell's direct children would drop it
    # entirely - silent data loss, the worst kind, because the file looks readable.
    data = make_odt(
        "<table:table><table:table-row><table:table-cell>"
        "<text:list><text:list-item><text:p>First bullet</text:p></text:list-item>"
        "<text:list-item><text:p>Second bullet</text:p></text:list-item></text:list>"
        "</table:table-cell></table:table-row></table:table>"
    )

    text = (await OdtParser().parse(data, filename="checklist.odt"))[0].text

    assert "First bullet" in text
    assert "Second bullet" in text


async def test_a_repeated_cell_that_actually_holds_text_is_repeated() -> None:
    # The other half of the rule: a repeat of SOMETHING is something. A merged run of
    # three "Yes" cells is real content and must survive - the bound applies to how
    # many, not to whether.
    data = make_ods(
        '<table:table table:name="Audit"><table:table-row>'
        '<table:table-cell table:number-columns-repeated="3">'
        "<text:p>Yes</text:p></table:table-cell>"
        "</table:table-row></table:table>"
    )

    text = (await OdsParser().parse(data, filename="audit.ods"))[0].text

    assert text.count("Yes") == 3


async def test_a_presentation_is_one_unit_per_slide_and_keeps_its_notes() -> None:
    # Notes are often where the actual argument lives - the slide is three words and a
    # chart - so dropping them would throw away the answerable text.
    data = make_odp(
        '<draw:page draw:name="page1">'
        "<draw:frame><draw:text-box><text:p>Magnet supply</text:p></draw:text-box></draw:frame>"
        "<presentation:notes><draw:frame><draw:text-box>"
        "<text:p>The gain is one-off: a supplier rebate.</text:p>"
        "</draw:text-box></draw:frame></presentation:notes>"
        "</draw:page>"
        '<draw:page draw:name="page2">'
        "<draw:frame><draw:text-box><text:p>Risks</text:p></draw:text-box></draw:frame>"
        "</draw:page>"
    )

    documents = await OdpParser().parse(data, filename="deck.odp")

    assert [document.title for document in documents] == ["slide 1", "slide 2"]
    assert "Magnet supply" in documents[0].text
    assert "supplier rebate" in documents[0].text
    assert "Risks" in documents[1].text


async def test_the_registry_claims_each_opendocument_format_by_extension() -> None:
    registry = default_registry()

    assert isinstance(registry.resolve(filename="a.odt"), OdtParser)
    assert isinstance(registry.resolve(filename="a.ods"), OdsParser)
    assert isinstance(registry.resolve(filename="a.odp"), OdpParser)
    # A template is the same content.xml under a different name.
    assert isinstance(registry.resolve(filename="a.ott"), OdtParser)


async def test_a_mislabelled_opendocument_file_is_claimed_by_the_type_it_declares() -> None:
    """Every OpenDocument format has the SAME parts, so only the mimetype entry sorts them.

    An .ods sent as application/octet-stream with no extension must still reach the
    spreadsheet reader - not the text reader (which would flatten every sheet into one
    unit and lose the tab names that citations are made of), and certainly not the
    plain-text fallback, which would decode the zip to garbage.
    """
    registry = default_registry()
    data = make_ods(f'<table:table table:name="S">{cells("value", "7")}</table:table>')

    parser = registry.resolve(
        filename="attachment", content_type="application/octet-stream", data=data
    )

    assert isinstance(parser, OdsParser)


async def test_an_opendocument_file_declaring_an_xml_entity_is_refused() -> None:
    # The XXE guard, on the ODF suffix list. content.xml is handed to an XML parser,
    # so a DOCTYPE in it is either broken or an attempt to read the server's disk.
    hostile = (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>\n'
        "<office:document-content>&xxe;</office:document-content>"
    )
    data = make_package(ODT_TYPE, {"content.xml": hostile, "META-INF/manifest.xml": "<manifest/>"})

    with pytest.raises(ParseFailedError, match="document type or entity"):
        await OdtParser().parse(data, filename="report.odt")


async def test_a_password_protected_opendocument_file_is_refused_honestly() -> None:
    # An encrypted ODF is a valid zip whose parts are ciphertext, so it does not fail
    # as corrupt - it parses to nothing. Saying so beats ingesting an empty document
    # and letting every answer from it be silently ungrounded.
    manifest = (
        '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">'
        '<manifest:file-entry manifest:full-path="content.xml">'
        "<manifest:encryption-data/>"
        "</manifest:file-entry></manifest:manifest>"
    )
    data = make_package(
        ODT_TYPE, {"content.xml": "<encrypted/>", "META-INF/manifest.xml": manifest}
    )

    with pytest.raises(ParseFailedError, match="password"):
        await OdtParser().parse(data, filename="secret.odt")


async def test_an_opendocument_compression_bomb_is_refused_without_being_decompressed() -> None:
    # A zip declares each entry's uncompressed size in its own directory, so an archive
    # that expands absurdly is recognised BEFORE anything is inflated. The upload cap
    # bounds what arrives; it says nothing about what that becomes.
    padding = "A" * (60 * 1024 * 1024)
    data = make_package(
        ODT_TYPE,
        {"content.xml": f"<office:document-content>{padding}", "META-INF/manifest.xml": "<m/>"},
        compress=True,
    )
    assert len(data) < 1024 * 1024  # it really is small on the wire

    with pytest.raises(ParseFailedError, match="compression bomb"):
        await OdtParser().parse(data, filename="bomb.odt")


async def test_a_corrupt_opendocument_file_fails_honestly() -> None:
    with pytest.raises(ParseFailedError, match="corrupt"):
        await OdtParser().parse(b"PK\x03\x04 not really a zip", filename="broken.odt")


async def test_an_empty_file_yields_nothing_rather_than_failing() -> None:
    assert await OdsParser().parse(b"", filename="empty.ods") == []
