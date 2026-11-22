"""Shared defences for the formats that are really a zip full of XML.

Three families this app reads are the same thing underneath: a zip archive whose
entries are XML parts. An OOXML office file (``.docx``, ``.xlsx``, ``.pptx``), an
OpenDocument file (``.odt``, ``.ods``, ``.odp``) and an EPUB all share that
container - and so they share its hazards, which is why the defences live here
rather than in seven parsers. All of them run BEFORE a single byte reaches a
parsing library.

**Zip bombs.** A few hundred kilobytes of highly compressible XML can inflate to
gigabytes. The upload size cap bounds what arrives; it says nothing about what
that expands to. So the archive's own directory is read first - it declares each
entry's uncompressed size, without decompressing anything - and a file whose
declared total is absurd, or whose compression ratio could only come from padding
designed to inflate, is refused before it is opened.

**XML entities.** An XML parser that resolves entities can be talked into reading
files off the server's disk (an external entity), or into expanding a few nested
internal entities into gigabytes of memory (the "billion laughs"). Where each
library stands is a moving target: python-docx and python-pptx build their parser
with ``resolve_entities=False``; openpyxl's element parser blocks external
entities but still EXPANDS internal ones, and it silently switches to lxml
whenever lxml is importable - which lxml became the moment python-docx was added.
The standard library's own parser expands internal entities too.

Rather than track which library has which parser configured which way, the
container is checked directly: a part that declares a DOCTYPE or an entity is
refused before any parser sees it. The whole of each part is scanned, not just its
head - a DOCTYPE is only legal in the prolog, but a part can carry a long leading
comment, and scanning a fixed window would let a declaration hide past it.

**Which parts get scanned is a per-format decision, and it is the sharp edge.**
The scan is keyed on a suffix list, so a format whose XML lives in a part that
list does not name is not protected at all - and that is not hypothetical: an
EPUB's package document is ``.opf`` and its EPUB-2 navigation is ``.ncx``, both of
which are XML, both of which a reader must parse, and neither of which ends in
``.xml``. Each caller therefore passes the suffixes ITS reader will hand to an XML
parser, and the rule to hold a new format to is exactly that: scan what you parse.

That list is one forgotten word away from a hole, so it is not the only wall. The
parsers read XML through ``defusedxml``, which refuses an entity declaration
outright. This check is the cheap gate that runs first, on the whole container;
that one is the gate that does not depend on anyone remembering a suffix.
"""

from __future__ import annotations

import zipfile
from io import BytesIO

from app.parsing.base import ParseFailedError

# The zip local-file-header signature. Every one of these formats starts with it,
# and so does every other zip, which is why claiming a format takes more than this.
ZIP_SIGNATURE = b"PK\x03\x04"

# The part whose presence identifies each OOXML format inside the archive.
DOCX_MARKER = "word/document.xml"
XLSX_MARKER = "xl/workbook.xml"
PPTX_MARKER = "ppt/presentation.xml"

# The parts each family's reader actually hands to an XML parser, and therefore
# exactly what each must scan. OOXML is the default, so its callers are unchanged.
OOXML_XML_SUFFIXES = (".xml", ".rels")
# ODF reads content.xml and META-INF/manifest.xml; .rdf is included because the
# package may carry one and it is XML by any other name.
ODF_XML_SUFFIXES = (".xml", ".rdf")
# EPUB: container.xml (.xml) and the package document (.opf) - the only two parts the
# reader hands to an XML parser. Two whole families of part are deliberately ABSENT,
# for the SAME reason: scanning a part the reader never parses can only refuse a good
# book, never protect anything. Chapters (.xhtml/.html) carry <!DOCTYPE html> and go to
# the lenient HTML parser. And the EPUB-2 table of contents (.ncx) carries the standard
# NISO DOCTYPE that virtually every such book declares - and the reader drives its order
# from the OPF spine, never touching the NCX - so scanning it would refuse a shelf of
# perfectly good books to guard a part that is never read. Getting this list wrong is
# how a container defence starts rejecting the very files it exists to let through.
EPUB_XML_SUFFIXES = (".xml", ".opf")

# The ODF and EPUB packages both put their media type in one entry whose bytes ARE
# that type. Every OpenDocument format carries the same parts, so unlike OOXML the
# format cannot be told by which part is inside - only by what this one says.
MIMETYPE_ENTRY = "mimetype"
EPUB_MIMETYPE = "application/epub+zip"

# A media type is tens of bytes. An entry claiming to be one and declaring far more
# is not a mimetype, and reading it would be a place to hide work.
_MAX_MIMETYPE_BYTES = 128

# Where an ODF package records that its parts are encrypted, and the marker it
# writes there. An EPUB says the same thing by simply carrying an encryption part.
_ODF_MANIFEST = "META-INF/manifest.xml"
_ODF_ENCRYPTION_MARKER = b"encryption-data"
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
EPUB_ENCRYPTION_PART = "META-INF/encryption.xml"

# Ceilings for what an archive may expand to. The uploaded file is already capped
# (attachment_max_bytes, 25MB by default); these bound the INFLATED size, which
# that cap says nothing about. A real document of any plausible size - including a
# heavily illustrated book, whose bulk is already-compressed images - is orders of
# magnitude inside all three.
_MAX_UNCOMPRESSED_BYTES = 400 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 200
# Below this, ratio is not evidence of anything: a small, highly repetitive XML
# part legitimately compresses enormously well.
_RATIO_FLOOR_BYTES = 1 * 1024 * 1024

# The most parts a real file has. A deck with a thousand slides and their media, or
# a 1,200-page illustrated book, is still well under this; hundreds of thousands of
# empty parts is not a document, it is an attempt to make the checks below (each of
# which is per-part) expensive. Refusing on the count bounds that work before it is
# done.
_MAX_ENTRIES = 10_000

# Entity markers, matched case-insensitively (XML keywords are case-sensitive and
# these are the only legal spellings, but a lie need not be well-formed).
_DOCTYPE_MARKERS = (b"<!doctype", b"<!entity")
# How much of a part to hold while scanning it for those markers. The part is read
# in chunks and discarded, so memory stays flat however large the part; a small
# overlap between chunks catches a marker that straddles a chunk boundary.
_SCAN_CHUNK_BYTES = 64 * 1024
_MARKER_OVERLAP = max(len(marker) for marker in _DOCTYPE_MARKERS) - 1


def is_zip(data: bytes) -> bool:
    """Return whether ``data`` begins with the zip signature."""
    return data.startswith(ZIP_SIGNATURE)


def contains_part(data: bytes, marker: str) -> bool:
    """Return whether ``data`` is a zip holding ``marker``, without extracting it.

    Used to claim a mislabelled file by its content: every OOXML format looks
    identical from the outside (a zip), so the format is decided by which part is
    inside. Reads only the archive's directory. Never raises: anything that is not
    a readable zip, or that carries an absurd number of parts, simply is not a
    file this claims.
    """
    if not is_zip(data):
        return False
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            names = archive.namelist()
            return len(names) <= _MAX_ENTRIES and marker in names
    except (zipfile.BadZipFile, OSError, ValueError):
        return False


def package_mimetype(data: bytes) -> str | None:
    """Return the media type an ODF or EPUB package declares, or None.

    These two containers put their media type in a single entry named ``mimetype``,
    whose bytes are that type and nothing else - and reading it is the only way to
    tell them apart. :func:`contains_part` cannot: an ``.odt``, an ``.ods`` and an
    ``.odp`` all carry exactly the same parts, so "which part is inside" - the
    question that decides an OOXML format - has the same answer for all three. What
    differs is what this one entry SAYS.

    Reads the archive's directory and that entry alone, and checks the entry's
    DECLARED size before reading a byte of it. Never raises: anything that is not a
    readable zip carrying a plausible mimetype entry simply is not claimed.
    """
    if not is_zip(data):
        return None
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            if len(archive.namelist()) > _MAX_ENTRIES:
                return None
            info = archive.getinfo(MIMETYPE_ENTRY)
            if info.file_size > _MAX_MIMETYPE_BYTES:
                return None
            with archive.open(info) as part:
                raw = part.read(_MAX_MIMETYPE_BYTES)
        return raw.decode("ascii").strip().lower() or None
    except (
        zipfile.BadZipFile,
        KeyError,
        OSError,
        ValueError,
        RuntimeError,
        UnicodeDecodeError,
    ):
        # KeyError: no such entry. RuntimeError: zipfile's way of saying an entry
        # needs a password. The rest are the many ways bytes fail to be an archive.
        return None


def is_encrypted_package(data: bytes) -> bool:
    """Return whether the package says its own contents are encrypted.

    A password-protected OpenDocument file records it in the package manifest; an
    EPUB with DRM carries an encryption part. Both are perfectly valid zips whose
    parts happen to be ciphertext, so without this they would parse "successfully"
    into gibberish, and that gibberish would be indexed and cited as though it were
    the document. Saying plainly that the file is locked is the honest answer.

    Never raises: bytes that are not a readable archive are not an encrypted one.
    """
    if not is_zip(data):
        return False
    manifest: bytes | None = None
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            if EPUB_ENCRYPTION_PART in archive.namelist():
                return True
            try:
                info = archive.getinfo(_ODF_MANIFEST)
            except KeyError:
                return False
            if info.file_size > _MAX_MANIFEST_BYTES:
                return False
            with archive.open(info) as part:
                manifest = part.read(_MAX_MANIFEST_BYTES)
    except (zipfile.BadZipFile, OSError, ValueError, RuntimeError):
        return False
    # A substring search, not an XML parse: this runs before any parser is allowed
    # near the file, which is the entire point of it.
    return _ODF_ENCRYPTION_MARKER in manifest.lower()


def parse_xml(data: bytes, *, what: str) -> object:
    """Parse one XML part of an uploaded package, refusing entity declarations.

    The second wall behind :func:`check_archive`, and the one that does not depend
    on a suffix list being complete. ``defusedxml`` refuses an entity declaration
    outright - internal, external, or the nested kind that expands a few lines into
    gigabytes - while still accepting the plain ``<!DOCTYPE html>`` that every real
    EPUB chapter and many an XHTML fragment legitimately carry. Both properties
    matter: without the first, the standard library's parser would happily expand
    the entity; without the second, ordinary books would be rejected.

    Imported here rather than at module scope: these modules are re-imported inside
    the spawned child that does the reading, so their import cost is paid per file.

    Raises:
        ParseFailedError: The part is not well-formed XML, or it declares an entity.
    """
    from defusedxml.common import DefusedXmlException
    from defusedxml.ElementTree import ParseError, fromstring

    try:
        return fromstring(data)
    except DefusedXmlException as exc:
        raise ParseFailedError(
            f"The {what} declares an XML entity, which a real one never does, so it was not read."
        ) from exc
    except ParseError as exc:
        raise ParseFailedError(f"The {what} is corrupt and could not be read.") from exc


def _check_entry_count(archive: zipfile.ZipFile, what: str) -> None:
    """Refuse an archive with far more parts than any real document has.

    First, because the two checks below are per-part: a file with hundreds of
    thousands of tiny entries is not a document but a way to make those checks
    (and the parsing libraries after them) grind. Bounding the count bounds the
    work.

    Raises:
        ParseFailedError: The archive declares more than ``_MAX_ENTRIES`` parts.
    """
    if len(archive.infolist()) > _MAX_ENTRIES:
        raise ParseFailedError(f"The {what} has far too many parts to be a real file.")


def _check_bomb(archive: zipfile.ZipFile, what: str) -> None:
    """Refuse an archive that declares an implausible expansion.

    Reads the zip directory only - nothing is decompressed to find this out.

    Raises:
        ParseFailedError: The archive expands past the ceiling, or a single entry
            shows a compression ratio only padding could produce.
    """
    total = 0
    for entry in archive.infolist():
        total += entry.file_size
        if total > _MAX_UNCOMPRESSED_BYTES:
            raise ParseFailedError(
                f"The {what} expands to more than "
                f"{_MAX_UNCOMPRESSED_BYTES // (1024 * 1024)}MB and was not read."
            )
        if (
            entry.file_size > _RATIO_FLOOR_BYTES
            and entry.compress_size > 0
            and entry.file_size // entry.compress_size > _MAX_COMPRESSION_RATIO
        ):
            raise ParseFailedError(f"The {what} looks like a compression bomb and was not read.")


def _part_declares_entity(part: object) -> bool:
    """Return whether an XML part contains a DOCTYPE or entity declaration.

    Scans the WHOLE part, in bounded chunks, rather than a fixed head window: a
    DOCTYPE is only legal in the prolog, but a part may carry a long leading
    comment, and a window scan would let a declaration hide just past it. Total
    bytes read here are bounded by the uncompressed ceiling that ``_check_bomb``
    has already enforced.
    """
    tail = b""
    while True:
        chunk = part.read(_SCAN_CHUNK_BYTES)  # type: ignore[attr-defined]
        if not chunk:
            return False
        window = (tail + chunk).lower()
        if any(marker in window for marker in _DOCTYPE_MARKERS):
            return True
        # Keep the last few bytes so a marker split across two chunks is still seen.
        tail = window[-_MARKER_OVERLAP:]


def _check_entities(archive: zipfile.ZipFile, what: str, xml_suffixes: tuple[str, ...]) -> None:
    """Refuse an archive whose XML declares a DOCTYPE or an entity.

    A legitimate part of these formats never carries one, so this costs a real file
    only the read it would do anyway, and it closes external-entity and
    entity-expansion attacks regardless of how a parsing library happens to be
    configured (see the module docstring - the picture there is not uniform, and it
    moves).

    Every PHYSICAL entry is scanned, addressed by its directory record rather than
    by its name, and that distinction is load-bearing. A zip may carry two entries
    with the SAME name, and looking one up by name gets the LAST of them - so a
    scan driven by names would read the innocent twin, twice, and never see the
    hostile one hiding in front of it. The archive would then pass a check that had
    not, in fact, looked at it.

    Raises:
        ParseFailedError: A part declares a DOCTYPE or an entity.
    """
    for info in archive.infolist():
        if not info.filename.lower().endswith(xml_suffixes):
            continue
        try:
            with archive.open(info) as part:
                declares = _part_declares_entity(part)
        except (zipfile.BadZipFile, OSError, ValueError, RuntimeError) as exc:
            raise ParseFailedError(
                f"The {what} could not be read ({info.filename} is unreadable)."
            ) from exc
        if declares:
            raise ParseFailedError(
                f"The {what} declares an XML document type or entity, which a real "
                "one never does, so it was not read."
            )


def check_archive(
    data: bytes, *, what: str, xml_suffixes: tuple[str, ...] = OOXML_XML_SUFFIXES
) -> None:
    """Refuse a hostile zip-and-XML container before any library opens it.

    Args:
        data: The uploaded file's bytes.
        what: What is being read ("Word document"), for the message the person
            who uploaded it eventually sees.
        xml_suffixes: Which part extensions to scan for entity declarations. Pass
            the ones THIS format's reader will hand to an XML parser, and only
            those. Both mistakes are possible and they fail in opposite directions:
            omit a suffix the reader parses (an EPUB's ``.opf``) and that part is
            never checked at all; add one the reader does NOT parse as XML (an
            EPUB's ``.xhtml``, which goes to the lenient HTML parser) and every real
            file of that format is refused, because an ordinary chapter
            legitimately opens with ``<!DOCTYPE html>``. Defaults to the OOXML set.

    Raises:
        ParseFailedError: The file is not a readable zip, expands implausibly, or
            declares an XML entity.
    """
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            _check_entry_count(archive, what)
            _check_bomb(archive, what)
            _check_entities(archive, what, xml_suffixes)
    except zipfile.BadZipFile as exc:
        raise ParseFailedError(f"The {what} is corrupt and could not be read.") from exc
