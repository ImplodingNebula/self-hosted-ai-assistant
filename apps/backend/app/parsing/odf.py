"""OpenDocument formats: the text, spreadsheet and presentation LibreOffice writes.

The open twin of :mod:`app.parsing.office`. An ``.odt``, ``.ods`` or ``.odp`` is a
zip of XML parts, like an OOXML file, so it inherits the same container defences
from :mod:`app.parsing.archive` - and the same split into the units a person would
actually cite: a spreadsheet is one unit per SHEET, a presentation one per SLIDE,
and a text document stays one unit, because it has no natural seam.

Two things differ from the office formats, and both shape the code.

**The format is not in the parts, it is in one entry.** Every OpenDocument file
carries the same parts - ``content.xml``, ``styles.xml``, a manifest - so "which
part is inside", the question that decides an OOXML format, cannot tell an ``.odt``
from an ``.ods``. What tells them apart is the package's ``mimetype`` entry, whose
bytes ARE the media type. Hence the sniff on
:func:`~app.parsing.archive.package_mimetype` rather than on a marker part.

**An empty spreadsheet claims to be enormous, and it is telling the truth.** ODF
run-length-encodes repetition: a blank sheet does not write a million empty cells,
it writes one cell saying "and 1,048,575 more like me". A reader that expands those
counts literally allocates a billion cells from a 900-byte file - and this is not an
attack, it is what LibreOffice saves for an empty spreadsheet. One rule makes it
safe, and it is applied everywhere below: **a repeat of nothing is nothing.** Only
cells that actually hold text are ever materialised, and even then within a bounded
budget. The xlsx reader already relies on the same insight when it drops trailing
empty cells; ODF simply forces the issue.

Headers and footers need no rule at all: in ODF they live in ``styles.xml``, which
is never read. They are excluded by construction.

Reading runs in a killable child process (see :mod:`app.parsing.isolation`): the
XML parse underneath is native code, and the repeat expansion is a loop over counts
that someone else chose.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any, ClassVar

from app.parsing.archive import (
    ODF_XML_SUFFIXES,
    check_archive,
    is_encrypted_package,
    package_mimetype,
)
from app.parsing.base import DocumentParser, ParsedDocument, ParseFailedError
from app.parsing.isolation import run_isolated

logger = logging.getLogger(__name__)

# Generous for a large document, bounded so a pathological one cannot run forever.
# The same value the office formats use: consistency is worth more than a number
# tuned per format, and the container check already bounds what can arrive.
_DEFAULT_PARSE_TIMEOUT_SECONDS = 60.0

# How a spreadsheet row and a table row are laid out as text. Matches the office
# parsers, so a cited row reads the same whichever suite wrote the file.
_CELL_SEPARATOR = " | "

# The one part that is read. styles.xml, meta.xml and settings.xml are not: they
# hold formatting, metadata and view state - and the first is where headers and
# footers live, which is why excluding those costs no code.
_CONTENT_PART = "content.xml"

# The largest content.xml this will parse, checked against the DECLARED size in the
# archive's directory before a byte of it is decompressed. The container check
# permits 400MB of total expansion, which is right for a book full of images but far
# too much to turn into an XML tree: a parsed tree costs several times its source,
# and the dev tier has 4GB in total. A real document's content.xml is a few MB.
_MAX_CONTENT_BYTES = 32 * 1024 * 1024

# Bounds on what one sheet may expand to. They exist because the repeat counts are
# attacker-controlled - and, per the module docstring, absurd even in honest files.
# A sheet that hits them is TRUNCATED rather than refused: a spreadsheet with a
# million filled cells is not a document anyone is asking questions about, but the
# part that was read is still worth answering from.
_MAX_COLUMNS_PER_ROW = 1024
_MAX_ROWS_PER_SHEET = 100_000
_MAX_SHEET_CHARS = 8 * 1024 * 1024
_MAX_SHEETS = 1_000

# The most text one WHOLE document may yield, across every sheet, slide and table in
# it. The per-sheet cap above bounds each PIECE; this bounds their sum, because the
# run-length encoding amplifies per-piece - a file well under the 32MB content.xml
# cap can still declare a thousand sheets that each expand to the per-sheet ceiling,
# and a thousand times 8MB is 8GB the 4GB tier does not have. Matches EPUB's total.
_MAX_DOCUMENT_CHARS = 16 * 1024 * 1024

# The most text a SINGLE paragraph or table cell may build. A real one is a sentence;
# this exists because a paragraph is made of elements (text:s space runs of up to
# _MAX_SPACES each, and any number of them), so without a ceiling one paragraph of
# nothing but space runs assembles a multi-GB string before any higher cap is even
# consulted. The join happens here, so the bound has to be here.
_MAX_INLINE_CHARS = 64 * 1024
# The most text one cell may gather across all the paragraphs (and list items) inside
# it - the same idea one level up, so a cell full of list items cannot sum past it.
_MAX_CELL_CHARS = 256 * 1024
# The most times a NON-EMPTY cell or row is repeated. Real merged runs are small. An
# empty repeat costs nothing and is never counted against this.
_MAX_REPEAT = 256
# text:s carries a count of consecutive spaces, and it is as attacker-controlled as
# any other number in the file.
_MAX_SPACES = 1_000

# The OpenDocument namespaces, in Clark notation ({uri}local) - how both the stdlib
# parser and lxml spell a qualified tag.
_OFFICE = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_TEXT = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_TABLE = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_DRAW = "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
_PRESENTATION = "urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"

# Media types - which are also exactly what a package's mimetype entry says. Each
# format's template twin (.ott/.ots/.otp) has an identical content.xml, so claiming
# it costs nothing and refusing it would be arbitrary.
ODT_TYPES = frozenset(
    {
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.text-template",
    }
)
ODS_TYPES = frozenset(
    {
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.spreadsheet-template",
    }
)
ODP_TYPES = frozenset(
    {
        "application/vnd.oasis.opendocument.presentation",
        "application/vnd.oasis.opendocument.presentation-template",
    }
)


def _q(namespace: str, local: str) -> str:
    """Return a tag in Clark notation, the spelling an XML parser matches on."""
    return f"{{{namespace}}}{local}"


_P = _q(_TEXT, "p")
_H = _q(_TEXT, "h")
_TABLE_TAG = _q(_TABLE, "table")
_TABLE_ROW = _q(_TABLE, "table-row")
_TABLE_CELL = _q(_TABLE, "table-cell")
_COVERED_CELL = _q(_TABLE, "covered-table-cell")
_ROWS_REPEATED = _q(_TABLE, "number-rows-repeated")
_COLUMNS_REPEATED = _q(_TABLE, "number-columns-repeated")
_ROW_CONTAINERS = (_q(_TABLE, "table-header-rows"), _q(_TABLE, "table-row-group"))
_DRAW_PAGE = _q(_DRAW, "page")
_NOTES = _q(_PRESENTATION, "notes")

# Subtrees that are never the document's own text, and are never descended into.
#
# text:tracked-changes is the sharp one: it holds the SUPERSEDED text of every
# accepted deletion, so walking it would resurrect sentences the author deleted and
# let an answer be grounded on them as though they were current. The rest are
# generated or marginal - an index duplicates headings that are already indexed, a
# footnote repeats and interleaves badly mid-paragraph, a comment is a remark ABOUT
# the document rather than the document. All three match the calls office.py made.
_SKIPPED_TAGS = frozenset(
    {
        _q(_TEXT, "tracked-changes"),
        _q(_TEXT, "note"),
        _q(_TEXT, "table-of-content"),
        _q(_TEXT, "illustration-index"),
        _q(_TEXT, "alphabetical-index"),
        _q(_TEXT, "object-index"),
        _q(_TEXT, "table-index"),
        _q(_TEXT, "user-index"),
        _q(_TEXT, "bibliography"),
        _q(_TEXT, "sequence-decls"),
        _q(_TEXT, "variable-decls"),
        _q(_TEXT, "user-field-decls"),
        _q(_OFFICE, "annotation"),
        _q(_OFFICE, "annotation-end"),
    }
)


def _repeat(element: Any, attribute: str) -> int:
    """Read a repeat count, trusting nothing about it.

    The attribute is a string in a file somebody uploaded, so it may be absent,
    negative, non-numeric, or ``99999999999999999999`` - which is a perfectly valid
    Python integer and an effectively infinite loop. It is parsed defensively and
    clamped HERE, before it can reach a ``range()``, because the one thing that must
    never happen is an allocation sized by it.
    """
    raw = element.get(attribute)
    if raw is None:
        return 1
    try:
        count = int(raw)
    except (TypeError, ValueError):
        return 1
    return max(1, min(count, _MAX_REPEAT))


class _Budget:
    """A shared, decrementing character allowance for one document's text.

    Passed down the read rather than checked at the top, because the amplification it
    guards against happens DEEP in the walk - a paragraph of repeated spaces, a cell
    repeated across a row - and a cap that only looked at the finished total would
    already have allocated the gigabytes it was meant to prevent. Every place that
    commits text to the output takes from it first; when it reaches zero the readers
    stop, and the file is truncated rather than refused (the part that was read is
    still worth answering from).
    """

    __slots__ = ("remaining",)

    def __init__(self, total: int) -> None:
        """Start with ``total`` characters to spend."""
        self.remaining = total

    def take(self, text: str) -> str:
        """Return as much of ``text`` as the allowance still permits, and spend it."""
        if self.remaining <= 0:
            return ""
        if len(text) <= self.remaining:
            self.remaining -= len(text)
            return text
        clipped = text[: self.remaining]
        self.remaining = 0
        return clipped


def _inline_text(element: Any) -> str:
    """Collect one paragraph's text, including the whitespace ODF hides.

    ODF does not store runs of spaces, tabs or line breaks as characters - it stores
    them as ELEMENTS (``text:s``, ``text:tab``, ``text:line-break``). A walk that
    only gathered character data would glue words together into run-on tokens that
    neither a person nor a retriever can use. This puts the whitespace back.

    The build stops at ``_MAX_INLINE_CHARS``. That ceiling is not about real
    paragraphs, which never approach it - it is because a paragraph is assembled from
    elements, and a hostile one made of nothing but ``text:s`` runs would otherwise
    grow an unbounded string here, below every higher cap, before the join.
    """
    parts: list[str] = []
    length = 0

    def emit(piece: str) -> bool:
        """Append what still fits; return False once the paragraph ceiling is hit."""
        nonlocal length
        if length >= _MAX_INLINE_CHARS:
            return False
        room = _MAX_INLINE_CHARS - length
        if len(piece) > room:
            piece = piece[:room]
        parts.append(piece)
        length += len(piece)
        return length < _MAX_INLINE_CHARS

    def walk(node: Any) -> bool:
        for child in node:
            tag = child.tag
            if tag in _SKIPPED_TAGS:
                # Even for a skipped child, its TAIL is the paragraph's own text
                # continuing after it, and dropping that would eat a sentence.
                if child.tail and not emit(child.tail):
                    return False
                continue
            if tag == _q(_TEXT, "s"):
                raw = child.get(_q(_TEXT, "c"))
                try:
                    spaces = max(1, min(int(raw), _MAX_SPACES)) if raw else 1
                except (TypeError, ValueError):
                    spaces = 1
                if not emit(" " * spaces):
                    return False
            elif tag == _q(_TEXT, "tab"):
                if not emit("\t"):
                    return False
            elif tag == _q(_TEXT, "line-break"):
                if not emit("\n"):
                    return False
            else:
                if child.text and not emit(child.text):
                    return False
                if not walk(child):
                    return False
            if child.tail and not emit(child.tail):
                return False
        return True

    if element.text:
        emit(element.text)
    walk(element)
    return "".join(parts)


def _cell_text(cell: Any) -> str:
    """Return one cell's displayed text, falling back to its typed value.

    ODF writes both: the value a formula computed AND the string a person sees. The
    displayed text wins for the reason the xlsx reader asks for cached values -
    ``=SUM(B2:B9)`` tells a reader nothing and ``41,207`` is the answer they came
    for. Unlike xlsx there is no "generated but never opened, so blank" caveat: ODF
    always writes the display text. The typed value is the fallback for a cell that
    somehow carries no paragraph at all.

    Paragraphs are gathered by descending through the cell, not by reading its direct
    children only, because a cell's text is not always a bare paragraph: a bulleted or
    numbered cell wraps its paragraphs in a ``text:list``, and a walk that stopped at
    the top would read that cell as empty and silently drop content a person can see.
    """
    paragraphs: list[str] = []
    length = 0

    def collect(node: Any) -> None:
        nonlocal length
        for child in node:
            if length >= _MAX_CELL_CHARS:
                return
            tag = child.tag
            if tag in _SKIPPED_TAGS:
                continue
            if tag in (_P, _H):
                part = _inline_text(child).strip()
                if part:
                    paragraphs.append(part)
                    length += len(part)
            else:
                # A list, or a sub-list inside it: the paragraphs are further down.
                collect(child)

    collect(cell)
    text = " ".join(paragraphs)
    if text:
        return text
    for attribute in ("string-value", "value", "date-value", "time-value", "boolean-value"):
        value = cell.get(_q(_OFFICE, attribute))
        if value:
            return str(value).strip()
    return ""


def _rows_of(node: Any) -> list[Any]:
    """Return a table's rows, reaching through the header and group wrappers."""
    found: list[Any] = []
    for child in node:
        if child.tag == _TABLE_ROW:
            found.append(child)
        elif child.tag in _ROW_CONTAINERS:
            found.extend(_rows_of(child))
    return found


def _read_table(table: Any, budget: _Budget) -> list[str]:
    """Turn one sheet or table into lines of separated cells.

    Where the repeat encoding is defused, and every rule below is the same rule: a
    repeat of nothing is nothing. An empty row emits no line however many times it
    claims to repeat - which is how a blank sheet's 1,048,576 rows collapse to zero
    work - and an empty cell is never materialised at all. It only advances the
    column position of whatever non-empty cell comes after it, and if none does, it
    cost a single integer.

    Two ceilings bound the cost. ``chars`` / ``_MAX_SHEET_CHARS`` bounds this one
    table; the shared ``budget`` bounds the whole document, so a file of many tables
    that each stay under the per-table cap still cannot sum past the document total.
    """
    lines: list[str] = []
    chars = 0

    for row in _rows_of(table):
        if len(lines) >= _MAX_ROWS_PER_SHEET or chars >= _MAX_SHEET_CHARS or budget.remaining <= 0:
            logger.warning("Stopped reading an OpenDocument sheet at its size limit")
            break
        cells: list[str] = []
        pending_empty = 0
        line_chars = 0
        for cell in row:
            if cell.tag not in (_TABLE_CELL, _COVERED_CELL):
                continue
            if len(cells) >= _MAX_COLUMNS_PER_ROW or line_chars >= _MAX_SHEET_CHARS:
                # Bound the BUILD of one row, not just the table: a single row of many
                # repeated non-empty cells would otherwise materialise its whole join
                # before any per-table or document cap could apply.
                break
            # A covered cell is the shadow of a merge and holds no text of its own.
            text = "" if cell.tag == _COVERED_CELL else _cell_text(cell)
            repeat = _repeat(cell, _COLUMNS_REPEATED)
            if not text:
                pending_empty += repeat
                continue
            room = max(0, _MAX_COLUMNS_PER_ROW - len(cells))
            if pending_empty:
                # Only now does an empty run cost anything, and only as much as is
                # needed to put THIS cell in its right column.
                cells.extend([""] * min(pending_empty, room))
                pending_empty = 0
                room = max(0, _MAX_COLUMNS_PER_ROW - len(cells))
            count = min(repeat, room)
            cells.extend([text] * count)
            line_chars += len(text) * count
        if not any(cells):
            continue
        line = _CELL_SEPARATOR.join(cells)
        for _ in range(_repeat(row, _ROWS_REPEATED)):
            if (
                len(lines) >= _MAX_ROWS_PER_SHEET
                or chars >= _MAX_SHEET_CHARS
                or budget.remaining <= 0
            ):
                break
            taken = budget.take(line)
            lines.append(taken)
            chars += len(taken)
            if len(taken) < len(line):
                # The document allowance ran out mid-line; nothing more will fit.
                break
    return lines


def _blocks_of(nodes: Iterable[Any], budget: _Budget) -> list[str]:
    """Walk a body in document order, turning it into blocks of text.

    Shared by all three formats, because their bodies differ less than they look: a
    paragraph is a paragraph whether it sits in a document, a slide's text box or a
    table cell. Tables are read IN PLACE, so a caption still precedes the table it
    introduces - the same reasoning as the Word reader, and the same bug avoided,
    because a form, rate card or spec sheet whose content lives entirely inside a
    table would otherwise ingest as empty while reporting success.

    The shared ``budget`` is spent as blocks are committed; once it is gone the walk
    stops, so the total text a document yields is bounded however it is structured.
    """
    blocks: list[str] = []

    def walk(items: Iterable[Any]) -> None:
        for child in items:
            if budget.remaining <= 0:
                return
            tag = child.tag
            if tag in _SKIPPED_TAGS:
                continue
            if tag in (_P, _H):
                text = _inline_text(child).strip()
                if text:
                    blocks.append(budget.take(text))
            elif tag == _TABLE_TAG:
                blocks.extend(_read_table(child, budget))
            else:
                # Lists, sections, frames, text boxes, groups, shapes: containers
                # whose text lives in the paragraphs beneath them. Descending
                # generically rather than naming each one is what lets a deck built
                # from grouped boxes - or a document using an element nobody thought
                # of - still read.
                walk(child)

    walk(nodes)
    return blocks


def _body(root: Any, kind: str) -> Any | None:
    """Return ``office:text`` / ``office:spreadsheet`` / ``office:presentation``."""
    for body in root.iter(_q(_OFFICE, "body")):
        for child in body:
            if child.tag == _q(_OFFICE, kind):
                return child
    return None


def read_text_document(content: bytes, what: str) -> str:
    """Extract a text document's text in reading order; runs in the child process.

    Top-level and given raw XML rather than the archive, because the spawn bootstrap
    re-imports this module in the child and the container work has already been done
    (and already refused a hostile file) on the way in.
    """
    from app.parsing.archive import parse_xml

    body = _body(parse_xml(content, what=what), "text")
    return "\n".join(_blocks_of(body, _Budget(_MAX_DOCUMENT_CHARS))) if body is not None else ""


def read_spreadsheet(content: bytes, what: str) -> list[tuple[str, str]]:
    """Extract each sheet as (name, text); runs in the child process."""
    from app.parsing.archive import parse_xml

    body = _body(parse_xml(content, what=what), "spreadsheet")
    if body is None:
        return []
    budget = _Budget(_MAX_DOCUMENT_CHARS)
    sheets: list[tuple[str, str]] = []
    for table in body:
        if table.tag != _TABLE_TAG:
            continue
        if len(sheets) >= _MAX_SHEETS or budget.remaining <= 0:
            logger.warning("Stopped reading an OpenDocument spreadsheet at its size limit")
            break
        sheets.append((table.get(_q(_TABLE, "name")) or "", "\n".join(_read_table(table, budget))))
    return sheets


def read_presentation(content: bytes, what: str) -> list[str]:
    """Extract each slide's text, in slide order; runs in the child process.

    Speaker notes are included, and appended at the END of their slide rather than
    left where they sit. In ODF the notes are a CHILD of the slide, so a walk that
    did not know that would fold them into the middle of the slide's own words,
    unlabelled. They are kept for the reason the pptx reader keeps them: the slide is
    often three words and a chart, and the argument is in the notes.
    """
    from app.parsing.archive import parse_xml

    body = _body(parse_xml(content, what=what), "presentation")
    if body is None:
        return []
    budget = _Budget(_MAX_DOCUMENT_CHARS)
    slides: list[str] = []
    for page in body:
        if budget.remaining <= 0:
            logger.warning("Stopped reading an OpenDocument presentation at its size limit")
            break
        if page.tag != _DRAW_PAGE:
            # presentation:header-decl and its friends: the deck's repeating chrome.
            continue
        blocks: list[str] = []
        notes: list[str] = []
        for child in page:
            if child.tag == _NOTES:
                notes.extend(_blocks_of([child], budget))
            else:
                blocks.extend(_blocks_of([child], budget))
        if notes:
            blocks.append("Notes: " + "\n".join(notes))
        slides.append("\n".join(blocks))
    return slides


def _extract_content(data: bytes, what: str) -> bytes:
    """Pull content.xml out of the package, refusing one too large to parse.

    Runs in the PARENT (off the event loop), not the child, so that a missing or
    oversized part is reported as the plain sentence it deserves rather than arriving
    wrapped in the child's exception type.

    Raises:
        ParseFailedError: The package is corrupt, has no content, or holds more text
            than can be turned into a tree here.
    """
    import zipfile
    from io import BytesIO

    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            try:
                info = archive.getinfo(_CONTENT_PART)
            except KeyError as exc:
                raise ParseFailedError(f"The {what} has no content and could not be read.") from exc
            if info.file_size > _MAX_CONTENT_BYTES:
                raise ParseFailedError(
                    f"The {what} holds more text than can be read here "
                    f"({_MAX_CONTENT_BYTES // (1024 * 1024)}MB is the limit)."
                )
            with archive.open(info) as part:
                return part.read(_MAX_CONTENT_BYTES)
    except zipfile.BadZipFile as exc:
        raise ParseFailedError(f"The {what} is corrupt and could not be read.") from exc


class _OdfParser(DocumentParser):
    """Shared plumbing: identify the package, check it, then read it in a child."""

    #: The media types this format's ``mimetype`` entry may declare.
    package_types: ClassVar[frozenset[str]] = frozenset()
    #: What this format is called in a message the person who uploaded it reads.
    what: ClassVar[str] = "file"

    def __init__(self, *, timeout_seconds: float = _DEFAULT_PARSE_TIMEOUT_SECONDS) -> None:
        """Create a parser that gives one file ``timeout_seconds`` to be read."""
        self._timeout = timeout_seconds

    def sniff(self, data: bytes) -> bool:
        """Claim a file that IS this format, whatever it was called.

        Not by a marker part, the way the office formats do it: every OpenDocument
        file holds the same parts, so the presence of ``content.xml`` claims all
        three at once and distinguishes them not at all. The package's own
        ``mimetype`` entry is the only thing that can.
        """
        mimetype = package_mimetype(data)
        return mimetype is not None and mimetype in self.package_types

    async def _content(self, data: bytes) -> bytes:
        """Check the container, refuse a locked file, and return its content.xml.

        Raises:
            ParseFailedError: The file is corrupt, hostile, or password-protected.
        """
        # Off the event loop: inspecting the container walks every part, and on a
        # hostile file (many parts, large XML) that is real work, which would
        # otherwise stall every other ingest job on the worker.
        await asyncio.to_thread(check_archive, data, what=self.what, xml_suffixes=ODF_XML_SUFFIXES)
        if await asyncio.to_thread(is_encrypted_package, data):
            raise ParseFailedError(
                f"The {self.what} is password-protected, so its text could not be read."
            )
        return await asyncio.to_thread(_extract_content, data, self.what)


class OdtParser(_OdfParser):
    """Reads an OpenDocument text document as one unit of text, tables included."""

    media_types: ClassVar[frozenset[str]] = ODT_TYPES
    extensions: ClassVar[frozenset[str]] = frozenset({".odt", ".ott"})
    package_types: ClassVar[frozenset[str]] = ODT_TYPES
    what: ClassVar[str] = "OpenDocument text document"

    async def parse(
        self, data: bytes, *, filename: str, media_type: str | None = None
    ) -> list[ParsedDocument]:
        """Extract the document's text; an empty document yields nothing.

        Raises:
            ParseFailedError: The file is corrupt, hostile, locked, or too slow.
        """
        if not data:
            return []
        content = await self._content(data)
        text = await run_isolated(
            read_text_document, content, self.what, timeout_seconds=self._timeout, what=self.what
        )
        if not text.strip():
            return []
        # One unit, so the filename is already its name - no title to add.
        return [ParsedDocument(ordinal=0, text=text)]


class OdsParser(_OdfParser):
    """Reads an OpenDocument spreadsheet as one unit per sheet, named by its tab."""

    media_types: ClassVar[frozenset[str]] = ODS_TYPES
    extensions: ClassVar[frozenset[str]] = frozenset({".ods", ".ots"})
    package_types: ClassVar[frozenset[str]] = ODS_TYPES
    what: ClassVar[str] = "OpenDocument spreadsheet"

    async def parse(
        self, data: bytes, *, filename: str, media_type: str | None = None
    ) -> list[ParsedDocument]:
        """Extract each sheet's cells; an empty sheet is skipped, its position kept.

        Raises:
            ParseFailedError: The file is corrupt, hostile, locked, or too slow.
        """
        if not data:
            return []
        content = await self._content(data)
        sheets = await run_isolated(
            read_spreadsheet, content, self.what, timeout_seconds=self._timeout, what=self.what
        )
        documents: list[ParsedDocument] = []
        for index, (name, text) in enumerate(sheets):
            if not text.strip():
                continue
            documents.append(ParsedDocument(ordinal=index, text=text, title=name or None))
        return documents


class OdpParser(_OdfParser):
    """Reads an OpenDocument presentation as one unit per slide, notes included."""

    media_types: ClassVar[frozenset[str]] = ODP_TYPES
    extensions: ClassVar[frozenset[str]] = frozenset({".odp", ".otp"})
    package_types: ClassVar[frozenset[str]] = ODP_TYPES
    what: ClassVar[str] = "OpenDocument presentation"

    async def parse(
        self, data: bytes, *, filename: str, media_type: str | None = None
    ) -> list[ParsedDocument]:
        """Extract each slide's text; a slide with none is skipped, its number kept.

        A picture-only slide holds no text and is left out rather than indexed as
        blank, but the slides that remain keep their true numbers, so a citation
        still reads "slide 9".

        Raises:
            ParseFailedError: The file is corrupt, hostile, locked, or too slow.
        """
        if not data:
            return []
        content = await self._content(data)
        slides = await run_isolated(
            read_presentation, content, self.what, timeout_seconds=self._timeout, what=self.what
        )
        documents: list[ParsedDocument] = []
        for index, text in enumerate(slides):
            if not text.strip():
                continue
            documents.append(ParsedDocument(ordinal=index, text=text, title=f"slide {index + 1}"))
        return documents
