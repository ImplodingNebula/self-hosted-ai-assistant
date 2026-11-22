"""EPUB: read in the order the book prescribes, from where the book says to read it."""

import zipfile
from io import BytesIO

import pytest

from app.parsing.base import ParseFailedError
from app.parsing.epub import EpubParser
from app.parsing.registry import default_registry

# Fixtures are BUILT, never committed. A book is a zip, a package file and some XHTML,
# and writing all three by hand is what lets each test state the exact structure it is
# about - a spine that disagrees with the archive, a chapter one directory over.

CHAPTER = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    "<!DOCTYPE html>\n"
    '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>{title}</title></head>'
    "<body>{body}</body></html>"
)

CONTAINER = (
    '<?xml version="1.0"?>'
    '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
    '<rootfile full-path="{opf}" media-type="application/oebps-package+xml"/>'
    "</rootfiles></container>"
)


def make_package(items: list[tuple[str, str, str]], spine: list[str]) -> str:
    """Render an OPF from (id, href, media type) manifest items and a spine of ids."""
    manifest = "".join(
        f'<item id="{item_id}" href="{href}" media-type="{media_type}"/>'
        for item_id, href, media_type in items
    )
    reading_order = "".join(f'<itemref idref="{item_id}"/>' for item_id in spine)
    return (
        '<?xml version="1.0"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
        f"<manifest>{manifest}</manifest><spine>{reading_order}</spine></package>"
    )


def make_epub(
    parts: dict[str, bytes | str], *, opf: str = "OEBPS/content.opf", compress: bool = False
) -> bytes:
    """Build a book: the mimetype entry, the container, then the given parts."""
    buffer = BytesIO()
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, "w", compression=method) as archive:
        archive.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        archive.writestr("META-INF/container.xml", CONTAINER.format(opf=opf))
        for name, content in parts.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def make_book(chapters: list[tuple[str, str, str]], spine: list[str] | None = None) -> bytes:
    """Build a book from (id, heading, prose) chapters; ``spine`` overrides the order."""
    items = [
        (item_id, f"text/{item_id}.xhtml", "application/xhtml+xml") for item_id, _, _ in chapters
    ]
    parts: dict[str, bytes | str] = {
        "OEBPS/content.opf": make_package(items, spine or [item_id for item_id, _, _ in chapters])
    }
    for item_id, heading, prose in chapters:
        parts[f"OEBPS/text/{item_id}.xhtml"] = CHAPTER.format(
            title="Untitled", body=f"<h1>{heading}</h1><p>{prose}</p>"
        )
    return make_epub(parts)


async def test_a_book_is_one_unit_per_chapter_labelled_by_its_heading() -> None:
    # A citation that says "The Return" tells the reader where the sentence came from;
    # "book.epub" does not, and "section 7" barely does.
    data = make_book(
        [
            ("c1", "The Return", "He came back to Ostrava."),
            ("c2", "The Departure", "She left on a Tuesday."),
        ]
    )

    documents = await EpubParser().parse(data, filename="book.epub")

    assert [document.title for document in documents] == ["The Return", "The Departure"]
    assert "Ostrava" in documents[0].text
    assert "Tuesday" in documents[1].text
    assert [document.ordinal for document in documents] == [0, 1]


async def test_the_spine_decides_the_reading_order_not_the_archive() -> None:
    """The one thing an EPUB reader must not get wrong.

    Zip entry order is arbitrary and the manifest is an unordered set; only the spine
    says what order the book is READ in. A reader that walks the archive returns a book
    in an order nobody wrote - so its chapter numbers are wrong, and every citation from
    it points at the wrong chapter while looking perfectly authoritative.

    Here the spine reverses the archive, so the two orders cannot be confused.
    """
    data = make_book(
        [
            ("c1", "The Return", "He came back to Ostrava."),
            ("c2", "The Departure", "She left on a Tuesday."),
        ],
        spine=["c2", "c1"],
    )

    documents = await EpubParser().parse(data, filename="book.epub")

    assert [document.title for document in documents] == ["The Departure", "The Return"]


async def test_a_chapter_path_is_resolved_against_the_package_not_the_archive_root() -> None:
    """The classic EPUB bug, and it fails SILENTLY.

    The package sits in a subdirectory and lists a chapter as "text/ch1.xhtml", which is
    the archive entry "OEBPS/text/ch1.xhtml". Resolving that against the root instead
    finds nothing - every lookup misses, no chapter is read, and the book ingests as
    empty while reporting success. Nothing raises; the file just quietly grounds nothing.
    """
    data = make_epub(
        {
            "OEBPS/content.opf": make_package(
                [("c1", "text/ch1.xhtml", "application/xhtml+xml")], ["c1"]
            ),
            "OEBPS/text/ch1.xhtml": CHAPTER.format(
                title="Untitled", body="<h1>Chapter One</h1><p>He came back to Ostrava.</p>"
            ),
        }
    )

    documents = await EpubParser().parse(data, filename="book.epub")

    assert len(documents) == 1
    assert "Ostrava" in documents[0].text


async def test_a_chapter_falls_back_to_its_title_then_to_its_position() -> None:
    data = make_epub(
        {
            "OEBPS/content.opf": make_package(
                [
                    ("c1", "titled.xhtml", "application/xhtml+xml"),
                    ("c2", "bare.xhtml", "application/xhtml+xml"),
                ],
                ["c1", "c2"],
            ),
            "OEBPS/titled.xhtml": CHAPTER.format(title="Front Matter", body="<p>A note.</p>"),
            "OEBPS/bare.xhtml": CHAPTER.format(title="", body="<p>Unlabelled prose.</p>"),
        }
    )

    documents = await EpubParser().parse(data, filename="book.epub")

    assert [document.title for document in documents] == ["Front Matter", "section 2"]


async def test_a_chapter_naming_a_remote_url_is_skipped_rather_than_fetched() -> None:
    """A document reader that makes outbound requests can be used as one.

    A package may point a chapter at http://attacker/ - and a reader that dutifully
    resolves it turns every upload into a request from inside the network, to a host the
    uploader chose. The href names nothing in the archive, so it is passed over.
    """
    data = make_epub(
        {
            "OEBPS/content.opf": make_package(
                [
                    ("c1", "http://169.254.169.254/latest/meta-data/", "application/xhtml+xml"),
                    ("c2", "ch2.xhtml", "application/xhtml+xml"),
                ],
                ["c1", "c2"],
            ),
            "OEBPS/ch2.xhtml": CHAPTER.format(
                title="Untitled", body="<h1>Real</h1><p>Real prose.</p>"
            ),
        }
    )

    documents = await EpubParser().parse(data, filename="book.epub")

    assert [document.title for document in documents] == ["Real"]
    assert "Real prose" in documents[0].text


async def test_a_chapter_escaping_the_archive_is_skipped() -> None:
    # Nothing is extracted to disk, so this is not a traversal in the usual sense - the
    # name simply addresses no entry. It is refused anyway: a resolver that can return
    # a path outside the archive stops being safe the day something does touch the disk.
    data = make_epub(
        {
            "OEBPS/content.opf": make_package(
                [
                    ("c1", "../../../etc/passwd", "application/xhtml+xml"),
                    ("c2", "ch2.xhtml", "application/xhtml+xml"),
                ],
                ["c1", "c2"],
            ),
            "OEBPS/ch2.xhtml": CHAPTER.format(
                title="Untitled", body="<h1>Real</h1><p>Real prose.</p>"
            ),
        }
    )

    documents = await EpubParser().parse(data, filename="book.epub")

    assert [document.title for document in documents] == ["Real"]


async def test_a_percent_encoded_chapter_name_is_decoded_before_it_is_looked_up() -> None:
    # Hrefs are percent-encoded; the archive entry is not. A chapter called "part 1"
    # is stored under that name, and looking it up as "part%201.xhtml" misses.
    data = make_epub(
        {
            "OEBPS/content.opf": make_package(
                [("c1", "part%201.xhtml", "application/xhtml+xml")], ["c1"]
            ),
            "OEBPS/part 1.xhtml": CHAPTER.format(
                title="Untitled", body="<h1>Part One</h1><p>Prose.</p>"
            ),
        }
    )

    documents = await EpubParser().parse(data, filename="book.epub")

    assert [document.title for document in documents] == ["Part One"]


async def test_a_broken_chapter_reference_costs_that_chapter_not_the_book() -> None:
    # Real books carry dangling links. Losing an entire uploaded book over one of them
    # would be a poor trade - the same reasoning as the PDF reader's "one bad page is
    # not a bad file".
    data = make_epub(
        {
            "OEBPS/content.opf": make_package(
                [
                    ("c1", "missing.xhtml", "application/xhtml+xml"),
                    ("c2", "ch2.xhtml", "application/xhtml+xml"),
                ],
                ["c1", "ghost", "c2"],  # "ghost" is in no manifest at all
            ),
            "OEBPS/ch2.xhtml": CHAPTER.format(
                title="Untitled", body="<h1>Survivor</h1><p>Still here.</p>"
            ),
        }
    )

    documents = await EpubParser().parse(data, filename="book.epub")

    assert [document.title for document in documents] == ["Survivor"]


async def test_a_chapter_listed_a_thousand_times_is_read_once() -> None:
    # A package may legitimately list one file twice; a hostile one lists a big chapter
    # a thousand times, and the archive stays small either way.
    data = make_epub(
        {
            "OEBPS/content.opf": make_package(
                [("c1", "ch1.xhtml", "application/xhtml+xml")], ["c1"] * 1000
            ),
            "OEBPS/ch1.xhtml": CHAPTER.format(
                title="Untitled", body="<h1>Once</h1><p>Read one time.</p>"
            ),
        }
    )

    documents = await EpubParser().parse(data, filename="book.epub")

    assert len(documents) == 1


async def test_a_chapters_doctype_does_not_refuse_the_book() -> None:
    """Every real EPUB chapter opens with <!DOCTYPE html>, and it must stay readable.

    The container's entity scan deliberately does not cover .xhtml parts: refusing a
    bare doctype would refuse the entire format. They are safe by a different route -
    they go to the lenient HTML parser, which resolves no entity declarations at all.
    The parts that DO reach an XML parser (the container, the package) are the ones the
    scan covers, which is why the fixture above carries a doctype and this passes.
    """
    documents = await EpubParser().parse(
        make_book([("c1", "Doctyped", "It reads.")]), filename="book.epub"
    )

    assert documents[0].text.strip().startswith("Doctyped")


async def test_a_package_declaring_an_xml_entity_is_refused() -> None:
    # The package IS handed to an XML parser, so a DOCTYPE in it is the XXE case.
    hostile = (
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>\n'
        '<package xmlns="http://www.idpf.org/2007/opf"><manifest/><spine>&xxe;</spine></package>'
    )

    with pytest.raises(ParseFailedError, match="document type or entity"):
        await EpubParser().parse(make_epub({"OEBPS/content.opf": hostile}), filename="x.epub")


def _encryption_xml(algorithm: str, uri: str) -> str:
    """Render a META-INF/encryption.xml declaring one encrypted resource."""
    return (
        '<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
        'xmlns:enc="http://www.w3.org/2001/04/xmlenc#"><enc:EncryptedData>'
        f'<enc:EncryptionMethod Algorithm="{algorithm}"/>'
        f'<enc:CipherData><enc:CipherReference URI="{uri}"/></enc:CipherData>'
        "</enc:EncryptedData></encryption>"
    )


async def test_a_drm_protected_book_is_refused_honestly() -> None:
    # An encrypted book parses to nothing rather than failing, so without this it would
    # ingest as an empty document and quietly ground nothing. A real cipher on a content
    # document (here AES on a chapter) is DRM.
    data = make_epub(
        {
            "OEBPS/content.opf": make_package(
                [("c1", "ch1.xhtml", "application/xhtml+xml")], ["c1"]
            ),
            "OEBPS/ch1.xhtml": CHAPTER.format(title="", body="<p>ciphertext</p>"),
            "META-INF/encryption.xml": _encryption_xml(
                "http://www.w3.org/2001/04/xmlenc#aes256-cbc", "OEBPS/ch1.xhtml"
            ),
        }
    )

    with pytest.raises(ParseFailedError, match="digital rights management"):
        await EpubParser().parse(data, filename="bought.epub")


async def test_a_book_that_only_obfuscates_its_fonts_is_still_read() -> None:
    """Font obfuscation is not DRM, and a book that uses it must not be refused.

    A large slice of ordinary commercial EPUBs scramble their embedded fonts - to
    satisfy a foundry's licence - with the IDPF or Adobe algorithm, while leaving every
    word of prose in the clear. Both write a META-INF/encryption.xml, so refusing on the
    mere presence of that part throws away readable books. This reader never touches the
    font, so it reads the text and lets the scrambled typeface be.
    """
    data = make_epub(
        {
            "OEBPS/content.opf": make_package(
                [
                    ("c1", "ch1.xhtml", "application/xhtml+xml"),
                    ("f1", "fonts/body.otf", "application/vnd.ms-opentype"),
                ],
                ["c1"],
            ),
            "OEBPS/ch1.xhtml": CHAPTER.format(
                title="", body="<h1>Readable</h1><p>The text is in the clear.</p>"
            ),
            "OEBPS/fonts/body.otf": b"scrambled-font-bytes",
            "META-INF/encryption.xml": _encryption_xml(
                "http://www.idpf.org/2008/embedding", "OEBPS/fonts/body.otf"
            ),
        }
    )

    documents = await EpubParser().parse(data, filename="typeset.epub")

    assert [document.title for document in documents] == ["Readable"]
    assert "the clear" in documents[0].text


async def test_an_ncx_with_its_standard_doctype_does_not_refuse_the_book() -> None:
    """Virtually every EPUB 2 carries a toc.ncx with a NISO DOCTYPE, and it must read.

    The NCX is a navigation aid the reader never parses - reading order comes from the
    OPF spine - so it is deliberately outside the container's entity scan. Were it inside,
    that standard DOCTYPE would be read as hostile and a shelf of perfectly good books
    would be refused: the exact failure the .xhtml exclusion exists to prevent, one part
    over.
    """
    ncx = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE ncx PUBLIC "-//NISO//DTD ncx 2005-1//EN" '
        '"http://www.daisy.org/z3986/2005/ncx-2005-1.dtd">\n'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><navMap/></ncx>'
    )
    data = make_epub(
        {
            "OEBPS/content.opf": make_package(
                [
                    ("nav", "toc.ncx", "application/x-dtbncx+xml"),
                    ("c1", "ch1.xhtml", "application/xhtml+xml"),
                ],
                ["c1"],
            ),
            "OEBPS/toc.ncx": ncx,
            "OEBPS/ch1.xhtml": CHAPTER.format(title="", body="<h1>Real</h1><p>Prose.</p>"),
        }
    )

    documents = await EpubParser().parse(data, filename="classic.epub")

    assert [document.title for document in documents] == ["Real"]


async def test_a_utf16_chapter_is_decoded_by_its_own_encoding_not_mangled() -> None:
    # Almost every book is UTF-8, but UTF-16 is permitted - and a hard UTF-8 decode
    # would turn a UTF-16 chapter into mojibake (every NUL a replacement char) and index
    # and cite it in that state. The BOM and the XML declaration are honoured instead.
    xhtml = (
        '<?xml version="1.0" encoding="utf-16"?><!DOCTYPE html>'
        "<html><body><h1>Chapitre</h1><p>Le cafe coute cinq euros.</p></body></html>"
    )
    data = make_epub(
        {
            "OEBPS/content.opf": make_package(
                [("c1", "ch1.xhtml", "application/xhtml+xml")], ["c1"]
            ),
            "OEBPS/ch1.xhtml": xhtml.encode("utf-16"),
        }
    )

    documents = await EpubParser().parse(data, filename="french.epub")

    assert "Chapitre" in documents[0].text
    assert "cinq euros" in documents[0].text
    assert "�" not in documents[0].text  # no replacement characters


async def test_an_oversized_package_file_is_refused_before_it_is_parsed() -> None:
    # The opf is read whole and turned into a tree, so its size has to be gated on its
    # own: a package a little over the structural cap is refused there, before it is
    # parsed into a tree several times its size. (A package that ALSO compresses absurdly
    # is caught earlier by the archive bomb check; this pins the size gate itself, so the
    # padding is high-entropy and rides in close to its real size.)
    import os

    from app.parsing.epub import _MAX_PACKAGE_BYTES

    filler = os.urandom((_MAX_PACKAGE_BYTES + 64 * 1024) // 2).hex()  # incompressible
    opf = (
        '<?xml version="1.0"?>'
        '<package xmlns="http://www.idpf.org/2007/opf">'
        f"<!-- {filler} --><manifest/><spine/></package>"
    )
    data = make_epub({"OEBPS/content.opf": opf})

    with pytest.raises(ParseFailedError, match="oversized"):
        await EpubParser().parse(data, filename="fat.epub")


async def test_a_book_with_no_container_fails_honestly() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip")
        archive.writestr("OEBPS/ch1.xhtml", "<html><body><p>Orphaned.</p></body></html>")

    with pytest.raises(ParseFailedError, match="container"):
        await EpubParser().parse(buffer.getvalue(), filename="broken.epub")


async def test_the_registry_claims_a_book_by_extension_and_by_what_it_declares() -> None:
    registry = default_registry()
    data = make_book([("c1", "One", "Prose.")])

    assert isinstance(registry.resolve(filename="a.epub"), EpubParser)
    # Mislabelled: a book is a zip like every office format, so only the mimetype entry
    # it carries tells it apart from one.
    assert isinstance(
        registry.resolve(filename="attachment", content_type="application/octet-stream", data=data),
        EpubParser,
    )


async def test_a_corrupt_book_fails_honestly() -> None:
    with pytest.raises(ParseFailedError, match="corrupt"):
        await EpubParser().parse(b"PK\x03\x04 not really a zip", filename="broken.epub")


async def test_an_empty_file_yields_nothing_rather_than_failing() -> None:
    assert await EpubParser().parse(b"", filename="empty.epub") == []
