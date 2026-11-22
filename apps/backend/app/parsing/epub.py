"""EPUB: a book, read chapter by chapter, in the order it is meant to be read.

An EPUB is a zip of XHTML documents plus a package file that says which of them are
chapters and what order they come in. So it is a zip-of-XML container like the
office formats (and inherits their defences from :mod:`app.parsing.archive`), and
its chapters are HTML like a fetched web page (and are turned into prose by the same
extractor, :mod:`app.parsing.html`). It is one unit per chapter, because that is
what a reader would cite.

Three things about the format decide the shape of this module.

**Reading order lives in the spine, and nowhere else.** The obvious instinct is to
reach for the table of contents; it is wrong. The TOC is a navigation aid - routinely
incomplete, sometimes out of order, often pointing at fragments of a file rather than
whole ones - while the ``<spine>`` is the normative reading order in every version of
the format. Walking the archive's own entry list is worse still: zip order is
arbitrary, so a book read that way comes out in an order nobody wrote, and every
citation from it points at the wrong chapter.

**Chapter paths are relative to the package file, not to the archive.** The package
usually sits in a subdirectory (``OEBPS/content.opf``), and a chapter it lists as
``text/ch1.xhtml`` is the archive entry ``OEBPS/text/ch1.xhtml``. Resolving those
against the archive root instead is the classic EPUB bug, and it fails silently: every
lookup misses, no chapter is found, and the book ingests as empty while reporting
success.

**A chapter's markup must NOT go through an XML parser.** A real EPUB chapter opens
with ``<!DOCTYPE html>``, so the container's entity check deliberately does not scan
``.xhtml`` parts - refusing them would refuse every genuine book. They are safe by a
different mechanism: they are read with the lenient HTML parser, which does not
resolve entity declarations at all. The parts that ARE handed to an XML parser - the
container file, the package - are the ones the entity scan covers, and they are the
ones that must never be moved out of it.

Nothing here touches the filesystem or the network. A chapter path is a key into the
archive's own directory, not a path on disk, so a name like ``../../etc/passwd``
addresses nothing but the attacker's own bytes; and a package that lists a chapter at
``http://example.com/ch1.xhtml`` is skipped rather than fetched, because a document
reader that makes outbound requests is a document reader that can be used as one.
"""

from __future__ import annotations

import asyncio
import logging
import posixpath
from typing import Any, ClassVar
from urllib.parse import unquote

from app.parsing.archive import (
    EPUB_ENCRYPTION_PART,
    EPUB_MIMETYPE,
    EPUB_XML_SUFFIXES,
    check_archive,
    package_mimetype,
)
from app.parsing.base import DocumentParser, ParsedDocument, ParseFailedError
from app.parsing.isolation import run_isolated

logger = logging.getLogger(__name__)

# Generous for a long book, bounded so a pathological one cannot run forever.
_DEFAULT_PARSE_TIMEOUT_SECONDS = 60.0

# What this format is called in a message the person who uploaded it reads.
_WHAT = "EPUB"

# The one entry whose location the format fixes; everything else is found from it.
_CONTAINER_PART = "META-INF/container.xml"

# Bounds. A 1,200-page illustrated book runs to a couple of thousand archive entries
# and a few million characters, so these are far outside anything real - they exist
# because a hostile package can list one 5MB chapter ten thousand times over, and the
# archive would still be 5MB.
_MAX_SPINE_ITEMS = 2_000
_MAX_CHAPTER_BYTES = 32 * 1024 * 1024
_MAX_TOTAL_CHARS = 16 * 1024 * 1024

# The container, the package and the encryption declaration are structural: a real one
# is a few KB, the largest imaginable a few hundred. They are read whole and handed to
# an XML parser, so an oversized one is a way to build a multi-GB tree from a part the
# container's expansion check (which permits 400MB total) would wave through. A real opf
# is nothing like this large; a hostile one is refused before it is decompressed.
_MAX_PACKAGE_BYTES = 8 * 1024 * 1024

# What a chapter is. Anything else in the reading order (an SVG plate, a font) simply
# holds no prose, and is passed over rather than treated as a failure.
_CHAPTER_TYPES = frozenset({"application/xhtml+xml", "text/html"})
_CHAPTER_SUFFIXES = (".xhtml", ".html", ".htm")

# The two algorithms a book uses to OBFUSCATE an embedded font - the IDPF scheme and
# Adobe's - to satisfy a foundry's licence while leaving every word of text in the
# clear. A book that "encrypts" only with these is readable; treating it as DRM would
# refuse a large slice of ordinary commercial books.
_FONT_OBFUSCATION_ALGORITHMS = frozenset(
    {
        "http://www.idpf.org/2008/embedding",
        "http://ns.adobe.com/pdf/enc#RC",
    }
)


def _local(tag: object) -> str:
    """Return an element's local name, ignoring whatever namespace it declared.

    Real books misspell, omit and invent namespaces constantly, and a reader that
    insists on the correct URI rejects a shelf full of perfectly readable files for a
    typo in their metadata. The tag names themselves are unambiguous.
    """
    name = str(tag)
    return name.rsplit("}", 1)[-1].lower()


def _resolve(base: str, href: str) -> str | None:
    """Turn a package href into the archive entry name it points at, or None.

    A zip entry name is a dictionary key, not a filesystem path, which is why this
    uses ``posixpath`` and string operations only: ``os.path`` would produce
    backslashes on Windows and the lookup would never match. It is also why traversal
    is not a vulnerability here - nothing is ever extracted to disk, so a name that
    climbs out of the archive simply names no entry. It is refused anyway, because a
    resolver that can return nonsense is one that stops being safe the day somebody
    does touch the filesystem.

    Returns None for an href that names something outside the book entirely: a remote
    URL (which must never be fetched), or a path that escapes the archive.
    """
    href = href.split("#", 1)[0].split("?", 1)[0].strip()
    if not href or "://" in href:
        return None
    # Hrefs are percent-encoded, and a chapter called "part 1.xhtml" is stored under
    # that name, not under "part%201.xhtml". Decode first, then validate what was
    # decoded - never the other way round.
    name = unquote(href).replace("\\", "/")
    if name.startswith("/"):
        name = name.lstrip("/")
    elif base:
        name = posixpath.join(base, name)
    name = posixpath.normpath(name)
    if name in (".", "..") or name.startswith(("../", "/")):
        return None
    return name


def _read_structural(archive: Any, name: str) -> bytes:
    """Read a structural part (container, package), refusing an oversized one.

    These are the parts handed to an XML parser, so their size has to be checked
    before they are read: the archive-wide expansion check permits far more than any
    real one of these could be, and a 300MB package inflates into a tree that does not
    fit the worker.

    Raises:
        KeyError: The archive has no such entry (the caller says what that means).
        ParseFailedError: The entry is larger than a structural part can plausibly be.
    """
    info = archive.getinfo(name)
    if info.file_size > _MAX_PACKAGE_BYTES:
        raise ParseFailedError(
            f"The {_WHAT} has an oversized {name.rsplit('/', 1)[-1]} and could not be read."
        )
    with archive.open(info) as part:
        return part.read(_MAX_PACKAGE_BYTES)


def _decode_chapter(raw: bytes) -> str:
    """Decode a chapter's bytes honouring its own declared encoding.

    Almost every EPUB is UTF-8, but the format also permits UTF-16, and a book that
    used it would, under a hard-coded UTF-8 decode, arrive as mojibake - every NUL
    byte a replacement character - and be indexed and cited in that state. So the byte
    order mark and the XML declaration are consulted first; UTF-8 is the fallback, with
    ``errors="replace"`` so a genuinely broken byte can never take the chapter down.
    """
    for bom, encoding in (
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    ):
        if raw.startswith(bom):
            return raw.decode(encoding, errors="replace")
    head = raw[:200].lower()
    if b"encoding=" in head and (b"utf-16" in head or b"utf16" in head):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _is_drm_encrypted(archive: Any) -> bool:
    """Return whether the book's TEXT is encrypted, not merely its fonts.

    ``META-INF/encryption.xml`` is present for two unrelated reasons: real DRM, which
    encrypts the chapters, and ordinary font obfuscation, which scrambles an embedded
    typeface to satisfy its licence and leaves the prose untouched. Refusing a book on
    the mere presence of that part throws away a large class of perfectly readable
    commercial books whose only "encryption" is a mangled font this reader never looks
    at anyway.

    So the part is read. If every declared algorithm is font obfuscation the book is
    readable and this returns False; anything else - a real cipher, or a declaration
    that cannot be made sense of - is treated as DRM, because an honest refusal beats a
    silently empty or garbled ingest. The container check has already refused this part
    if it declared an entity, so parsing it here is safe.
    """
    from app.parsing.archive import parse_xml

    try:
        raw = _read_structural(archive, EPUB_ENCRYPTION_PART)
    except KeyError:
        return False
    except ParseFailedError:
        return True
    try:
        root = parse_xml(raw, what=_WHAT)
    except ParseFailedError:
        return True
    algorithms = [
        (element.get("Algorithm") or "").strip()
        for element in root.iter()
        if _local(element.tag) == "encryptionmethod"
    ]
    if not algorithms:
        # An encryption declaration whose method cannot be read is not one to trust.
        return True
    return any(algorithm not in _FONT_OBFUSCATION_ALGORITHMS for algorithm in algorithms)


def _drm_check(data: bytes) -> bool:
    """Open the package and report whether its text is under DRM; runs in the parent.

    A thin wrapper so the check can run in a worker thread with a zip it opens itself,
    keeping the decision - and its clean error message - out of the child process.
    Corruption is not diagnosed here; a later step reports it with a better sentence.
    """
    import zipfile
    from io import BytesIO

    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            return _is_drm_encrypted(archive)
    except zipfile.BadZipFile:
        return False


def _read_package(archive: Any, opf_name: str) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Return the package's manifest (id -> (name, media type)) and its spine."""
    from app.parsing.archive import parse_xml

    package = parse_xml(_read_structural(archive, opf_name), what=_WHAT)
    base = posixpath.dirname(opf_name)
    manifest: dict[str, tuple[str, str]] = {}
    spine: list[str] = []
    for element in package.iter():
        tag = _local(element.tag)
        if tag == "item":
            item_id = element.get("id")
            href = element.get("href")
            if item_id and href:
                name = _resolve(base, href)
                if name is not None:
                    manifest[item_id] = (name, (element.get("media-type") or "").lower())
        elif tag == "itemref":
            idref = element.get("idref")
            if idref and len(spine) < _MAX_SPINE_ITEMS:
                spine.append(idref)
    return manifest, spine


def _opf_name(archive: Any) -> str:
    """Find the package file, which is the only thing the container file is for.

    Raises:
        ParseFailedError: The book has no readable container, or it names no package.
    """
    from app.parsing.archive import parse_xml

    try:
        container = parse_xml(_read_structural(archive, _CONTAINER_PART), what=_WHAT)
    except KeyError as exc:
        raise ParseFailedError(
            f"The {_WHAT} is missing its container file and could not be read."
        ) from exc

    fallback: str | None = None
    for element in container.iter():
        if _local(element.tag) != "rootfile":
            continue
        name = _resolve("", element.get("full-path") or "")
        if name is None:
            continue
        if (element.get("media-type") or "").lower() == "application/oebps-package+xml":
            return name
        fallback = fallback or name
    if fallback is not None:
        # A book that names its package without declaring its type is still a book.
        return fallback
    raise ParseFailedError(f"The {_WHAT} names no package file and could not be read.")


def read_epub(data: bytes) -> list[tuple[str, str]]:
    """Extract each chapter as (label, text), in reading order; runs in the child.

    Top-level so the spawn bootstrap can import it, with everything heavy imported
    inside because the child re-imports this module.

    A chapter that cannot be read is SKIPPED, not fatal. Real books carry broken
    references - a spine entry naming a manifest item that is not there, a manifest
    item naming a file that is not in the archive - and losing an entire uploaded book
    over one dangling link would be a poor trade. The reasoning matches the PDF
    reader's: one bad page is not a bad file.
    """
    import zipfile
    from io import BytesIO

    from app.parsing.html import EPUB_SKIP_TAGS, extract_document

    with zipfile.ZipFile(BytesIO(data)) as archive:
        names = set(archive.namelist())
        opf_name = _opf_name(archive)
        if opf_name not in names:
            raise ParseFailedError(
                f"The {_WHAT} points at a package file it does not contain, so it "
                "could not be read."
            )
        manifest, spine = _read_package(archive, opf_name)

        chapters: list[tuple[str, str]] = []
        seen: set[str] = set()
        total = 0
        for position, idref in enumerate(spine, start=1):
            if total >= _MAX_TOTAL_CHARS:
                logger.warning("Stopped reading an EPUB at its total size limit")
                break
            item = manifest.get(idref)
            if item is None:
                logger.debug("EPUB spine names %r, which its manifest does not list", idref)
                continue
            name, media_type = item
            if name in seen:
                # A package may legitimately list one file twice; a hostile one lists
                # a big chapter a thousand times. Either way, read it once.
                continue
            if name not in names:
                logger.debug("EPUB manifest names %r, which the book does not contain", name)
                continue
            if media_type and media_type not in _CHAPTER_TYPES:
                # An illustration or a font in the reading order: no prose to take.
                continue
            if not media_type and not name.lower().endswith(_CHAPTER_SUFFIXES):
                continue
            info = archive.getinfo(name)
            if info.file_size > _MAX_CHAPTER_BYTES:
                logger.warning("Skipped an EPUB chapter larger than the per-chapter limit")
                continue
            seen.add(name)
            markup = _decode_chapter(archive.read(name))
            text, label = extract_document(
                markup, max_chars=_MAX_TOTAL_CHARS - total, skip_tags=EPUB_SKIP_TAGS
            )
            if not text.strip():
                # A cover, or the table of contents (whose <nav> the extractor drops
                # by design - a contents page indexed as prose grounds nothing and
                # cites worse).
                continue
            total += len(text)
            chapters.append((label or f"section {position}", text))
    return chapters


class EpubParser(DocumentParser):
    """Reads a book as one unit per chapter, in the order the book prescribes."""

    media_types: ClassVar[frozenset[str]] = frozenset({"application/epub+zip"})
    extensions: ClassVar[frozenset[str]] = frozenset({".epub"})

    def __init__(self, *, timeout_seconds: float = _DEFAULT_PARSE_TIMEOUT_SECONDS) -> None:
        """Create a parser that gives one book ``timeout_seconds`` to be read."""
        self._timeout = timeout_seconds

    def sniff(self, data: bytes) -> bool:
        """Claim a book by the media type its package declares.

        Like the OpenDocument formats, an EPUB puts its own media type in a
        ``mimetype`` entry - so a book sent as ``application/octet-stream``, or with
        no extension at all, is still recognisably a book.
        """
        return package_mimetype(data) == EPUB_MIMETYPE

    async def parse(
        self, data: bytes, *, filename: str, media_type: str | None = None
    ) -> list[ParsedDocument]:
        """Extract each chapter's text, in spine order; an empty book yields nothing.

        Each chapter's label is its own first heading, falling back to its title tag
        and then to its position, so a citation says "The Return" where the book
        offers it and "section 4" where it does not.

        Raises:
            ParseFailedError: The book is corrupt, hostile, locked, or too slow.
        """
        if not data:
            return []
        # Off the event loop: inspecting the container walks every part, and on a
        # hostile file that is real work, which would otherwise stall every other
        # ingest job on the worker.
        await asyncio.to_thread(check_archive, data, what=_WHAT, xml_suffixes=EPUB_XML_SUFFIXES)
        if await asyncio.to_thread(_drm_check, data):
            raise ParseFailedError(
                f"The {_WHAT} is protected by digital rights management, so its text "
                "could not be read."
            )
        chapters = await run_isolated(read_epub, data, timeout_seconds=self._timeout, what=_WHAT)
        documents: list[ParsedDocument] = []
        for index, (label, text) in enumerate(chapters):
            documents.append(ParsedDocument(ordinal=index, text=text, title=label))
        return documents
