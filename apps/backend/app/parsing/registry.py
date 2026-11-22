"""Parser registry: chooses the parser for an uploaded file.

The registry is the extension point of the whole attachments pipeline. A new
file type is a new :class:`~app.parsing.base.DocumentParser` plus one line here;
nothing upstream or downstream changes.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from app.parsing.base import DocumentParser, UnsupportedFileError
from app.parsing.epub import EpubParser
from app.parsing.image import ImageParser
from app.parsing.ocr import OcrConfig
from app.parsing.odf import OdpParser, OdsParser, OdtParser
from app.parsing.office import DocxParser, PptxParser, XlsxParser
from app.parsing.pdf import PdfParser
from app.parsing.rtf import RtfParser
from app.parsing.text import PlainTextParser, looks_like_text

if TYPE_CHECKING:
    from app.core.config import Settings


def normalize_media_type(content_type: str | None) -> str | None:
    """Reduce a Content-Type header to a bare lowercase media type.

    ``"Text/Plain; charset=utf-8"`` becomes ``"text/plain"``. Returns None for a
    missing or empty header, which is common: browsers often send nothing at all
    for a source file.
    """
    if not content_type:
        return None
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type or None


def extension_of(filename: str) -> str:
    """Return the lowercase, dot-prefixed extension of ``filename``, or "".

    The name is treated as a label, never a path: only the final component is
    considered, so a client sending ``"../../etc/passwd"`` yields no extension
    and decides nothing. Where bytes are written is the storage seam's business,
    and it accepts only keys the app itself mints.
    """
    name = PurePosixPath(filename.replace("\\", "/")).name.strip()
    suffix = PurePosixPath(name).suffix.lower()
    # A dotfile such as ".env" has no suffix by pathlib's reckoning, but its
    # name is exactly the extension that identifies it.
    if not suffix and name.startswith("."):
        return name.lower()
    return suffix


class ParserRegistry:
    """Dispatches an uploaded file to the parser that handles it."""

    def __init__(self, parsers: list[DocumentParser] | None = None) -> None:
        """Create a registry over ``parsers``, tried in registration order."""
        self._parsers: list[DocumentParser] = list(parsers or [])

    def register(self, parser: DocumentParser) -> None:
        """Add ``parser``, giving it lower precedence than those already present."""
        self._parsers.append(parser)

    @property
    def parsers(self) -> tuple[DocumentParser, ...]:
        """The registered parsers, in precedence order."""
        return tuple(self._parsers)

    def resolve(
        self, *, filename: str, content_type: str | None = None, data: bytes | None = None
    ) -> DocumentParser:
        r"""Return the parser for this file.

        Resolution runs in three passes: declared handlers (media type or
        extension) first; then each parser's content signature (``sniff``); then
        the plain-text fallback. The signature pass is what makes a mislabelled
        binary format safe - a PDF sent as ``application/octet-stream`` with no
        extension is claimed by its ``%PDF-`` bytes rather than reaching the text
        parser and decoding to garbage. The text fallback runs last so a real
        format whose container looks textual - an SVG, say - is still claimed by
        its own parser, and so "source code in any language" holds: a `.zig` or
        `.nut` file nobody listed parses because its bytes are text.

        **A declared claim by the text parser does not end the search.** It is the
        one parser whose declared surface is a guess rather than a fact: it claims
        ``text/plain``, which is what a browser says when it does not know, and what
        anybody uploading by hand says when they cannot be bothered. Letting that win
        outright would put a real format into the reader that can read anything
        badly - an RTF declared ``text/plain`` is ASCII all the way down, so it would
        sail through and be indexed, retrieved and CITED as its own font tables. So a
        text claim is HELD, the signatures are tried, and it is honoured only if no
        format recognises the bytes. Every other parser's declared claim is final:
        those name formats they actually own.

        Neither the filename nor the declared type is trusted for anything but
        dispatch. A client can lie about both; the signature pass narrows what a
        lie can achieve, and any residual mislabel is contained - it produces junk
        text at worst, never a write outside the storage root (that is the storage
        seam's business, and it accepts only keys the app itself mints).

        Args:
            filename: The client's filename, used for its extension only.
            content_type: The client's declared Content-Type, if any.
            data: The file's bytes. None means "not supplied", which disables both
                the signature pass and the text sniff; an empty ``b""`` is a real,
                and textual, file.

        Returns:
            The parser that claims this file.

        Raises:
            UnsupportedFileError: No parser handles it, and it neither carries a
                known signature nor looks like text (or no bytes were supplied).
        """
        media_type = normalize_media_type(content_type)
        extension = extension_of(filename)
        declared_text: DocumentParser | None = None
        for parser in self._parsers:
            if not parser.handles(media_type, extension):
                continue
            if isinstance(parser, PlainTextParser):
                # Held, not returned: a signature below may know better. With no
                # bytes to check there is nothing that could, so it stands.
                if data is None:
                    return parser
                declared_text = parser
                break
            return parser
        if data is not None:
            for parser in self._parsers:
                if parser.sniff(data):
                    return parser
            if declared_text is not None:
                return declared_text
            if looks_like_text(data):
                for parser in self._parsers:
                    if isinstance(parser, PlainTextParser):
                        return parser
        if declared_text is not None:
            return declared_text
        raise UnsupportedFileError(
            f"No parser handles {filename!r}"
            + (f" (declared as {media_type})" if media_type else "")
        )


def default_registry(settings: Settings | None = None) -> ParserRegistry:
    """Build the registry with every parser this build supports.

    Order is precedence. The plain-text parser is registered last on purpose: it
    claims the broadest surface, so every specific format gets first refusal.

    That ordering is not a nicety for RTF - it is the whole reason its parser
    works. An RTF is ASCII from end to end, so it satisfies every test for "this
    is text" and the plain-text parser would happily claim it and index its font
    tables. Registering RtfParser ahead of PlainTextParser, and giving it a
    signature to sniff on, is what keeps a document out of the fallback that can
    read it wrongly.

    The zip formats - OOXML, OpenDocument and EPUB - cannot be claimed by shape
    alone, since from the outside each is just a zip. An office file recognises
    itself by the part inside the archive; an OpenDocument or an EPUB by the media
    type its package declares, because all three OpenDocument formats carry
    identical parts and only that entry tells them apart. So registering all seven
    costs nothing in ambiguity.

    The LEGACY office formats (``.doc``, ``.xls``, ``.ppt``) are not zips at all and
    no parser here claims them: they are refused at upload with an honest 415 rather
    than being decoded into garbage by the text fallback.

    Args:
        settings: When given and OCR is enabled, the PDF parser is built to read
            scanned pages by OCR (and with the larger timeout that slower work
            needs). Omitted - as on the request path, which only ever dispatches
            and never parses - the PDF parser stays text-layer only.
    """
    if settings is not None and settings.ocr_enabled:
        pdf = PdfParser(
            timeout_seconds=settings.ocr_timeout_seconds,
            ocr=OcrConfig(
                language=settings.ocr_language,
                dpi=settings.ocr_dpi,
                min_chars=settings.ocr_min_chars,
                max_pages=settings.ocr_max_pages,
            ),
        )
        image = ImageParser(
            timeout_seconds=settings.ocr_timeout_seconds, language=settings.ocr_language
        )
    else:
        pdf = PdfParser()
        image = ImageParser()
    return ParserRegistry(
        [
            pdf,
            image,
            DocxParser(),
            XlsxParser(),
            PptxParser(),
            OdtParser(),
            OdsParser(),
            OdpParser(),
            EpubParser(),
            RtfParser(),
            PlainTextParser(),
        ]
    )
