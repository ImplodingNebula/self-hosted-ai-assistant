"""PDF parser: extract each page's text as one addressable unit.

A born-digital PDF carries its text, and pypdfium2 reads it a page at a time -
fast, and with no system dependency (the wheel bundles the PDFium engine). A
SCANNED page carries a picture of text and no text layer, so direct extraction
finds nothing there. When OCR is configured (see :mod:`app.parsing.ocr`), such a
page is rendered to an image and read by Tesseract, so a scanned document is
still answerable; where OCR is unavailable the page simply comes back empty, as
it always did.

The extraction runs in a separate, killable process (see
:mod:`app.parsing.isolation`) because a malformed or hostile PDF can send a
native PDFium call into an unbounded loop that no timeout in this process could
ever interrupt. OCR runs there too - it is the same per-page work, only slower.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from app.parsing.base import DocumentParser, ParsedDocument
from app.parsing.isolation import run_isolated
from app.parsing.ocr import OcrConfig, image_to_text, ocr_available

logger = logging.getLogger(__name__)

# How long one PDF gets to extract before its worker is killed. Generous for a
# large born-digital file, bounded so a hostile one cannot run forever.
_DEFAULT_PARSE_TIMEOUT_SECONDS = 60.0

# What this format is called in a message the person who uploaded it reads.
_WHAT = "PDF"

# The PDF signature, and how far into the file it may start. A real PDF carries
# "%PDF-" at the very beginning, tolerating at most a short BOM or stray prefix.
# Scanning deeper (the spec allows up to 1 KB of leading junk) would let a text
# file that merely MENTIONS "%PDF-" be misclaimed as a broken PDF.
_SIGNATURE = b"%PDF-"
_MAX_SIGNATURE_OFFSET = 8

# Refuse a PDF with an absurd page count rather than walking it page by page: a
# bomb can declare millions of pages. A real document is orders of magnitude
# under this. It bounds page-count blowups only - a decompression bomb that
# inflates a few pages to gigabytes is bounded by the worker's own memory, which
# belongs at the container level (a cgroup limit + restart), not here.
_DEFAULT_MAX_PAGES = 10_000


def _ocr_page(page: object, ocr: OcrConfig) -> str:
    """Render one page to an image and read its text with OCR.

    Runs in the child process; imports the image libraries there, where the spawn
    bootstrap can re-import a light module. ``scale`` converts the requested DPI
    to PDFium's 72-DPI base unit.
    """
    bitmap = page.render(scale=ocr.dpi / 72.0)  # type: ignore[attr-defined]
    image = bitmap.to_pil()
    try:
        return image_to_text(image, language=ocr.language)
    finally:
        image.close()


def _read_pages(data: bytes, max_pages: int, ocr: OcrConfig | None) -> list[str]:
    """Extract every page's text, reading scanned pages by OCR when configured.

    Top-level so the spawn bootstrap can import it. Raising here is safe: the
    isolation layer turns whatever this raises into a seam error in the parent,
    because a traceback cannot cross a process boundary on its own.

    A page whose text layer is sparse - a scan - is rendered and OCR'd when
    ``ocr`` is set, up to its page cap. An OCR that fails on one page is logged
    and left as the sparse text it was: one unreadable page must not fail the
    whole document, and downstream that page is simply skipped like any empty one.
    """
    import pypdfium2 as pdfium  # imported in the child, never on the request path

    pdf = pdfium.PdfDocument(data)
    try:
        page_count = len(pdf)
        if page_count > max_pages:
            raise ValueError(f"has {page_count} pages, over the {max_pages}-page limit")
        pages: list[str] = []
        ocred = 0
        for index in range(page_count):
            page = pdf[index]
            try:
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_range()
                finally:
                    textpage.close()
                if ocr is not None and len(text.strip()) < ocr.min_chars and ocred < ocr.max_pages:
                    ocred += 1
                    try:
                        text = _ocr_page(page, ocr) or text
                    except Exception as exc:  # noqa: BLE001 - one bad page is not a bad file
                        logger.warning("OCR failed on page %d: %s", index + 1, exc)
            finally:
                page.close()
            pages.append(text)
    finally:
        pdf.close()
    return pages


class PdfParser(DocumentParser):
    """Reads a PDF's text layer as one :class:`ParsedDocument` per page."""

    media_types: ClassVar[frozenset[str]] = frozenset({"application/pdf"})
    extensions: ClassVar[frozenset[str]] = frozenset({".pdf"})

    def __init__(
        self,
        *,
        timeout_seconds: float = _DEFAULT_PARSE_TIMEOUT_SECONDS,
        max_pages: int = _DEFAULT_MAX_PAGES,
        ocr: OcrConfig | None = None,
    ) -> None:
        """Create a PDF parser capped at ``timeout_seconds`` and ``max_pages``.

        Args:
            timeout_seconds: How long one PDF gets before its child is killed.
            max_pages: The most pages a PDF may declare before it is refused.
            ocr: When set, scanned pages are read by OCR. Silently downgraded to
                None if the Tesseract binary is absent, so a deployment without
                it still reads born-digital PDFs rather than failing to start -
                the capable and graceful paths are the same code with one gate.
        """
        self._timeout = timeout_seconds
        self._max_pages = max_pages
        if ocr is not None and not ocr_available():
            logger.warning(
                "OCR is enabled but the Tesseract binary was not found on PATH; "
                "scanned PDF pages will not be read until it is installed."
            )
            ocr = None
        self._ocr = ocr

    def sniff(self, data: bytes) -> bool:
        """Claim a file that carries the PDF signature at its start, however named.

        Closes the hole the plain-text sniff leaves: a PDF sent with the wrong
        type or extension would otherwise reach the text parser and decode to
        garbage, or be refused outright. Anchored near offset 0 so a text file
        that merely mentions the signature further in is not misclaimed.
        """
        return data.find(_SIGNATURE, 0, _MAX_SIGNATURE_OFFSET + len(_SIGNATURE)) != -1

    async def parse(
        self, data: bytes, *, filename: str, media_type: str | None = None
    ) -> list[ParsedDocument]:
        """Extract each page's text; a page with none is skipped, its number kept.

        A page that holds no text - blank, or a scan that OCR could not read - is
        left out rather than embedded as whitespace, but the pages that remain
        keep their true page numbers, so a citation still reads "page 7".

        Raises:
            ParseFailedError: The PDF is corrupt, encrypted, or too slow to read.
        """
        if not data:
            return []
        pages = await run_isolated(
            _read_pages, data, self._max_pages, self._ocr, timeout_seconds=self._timeout, what=_WHAT
        )
        documents: list[ParsedDocument] = []
        for index, text in enumerate(pages):
            if not text.strip():
                continue
            documents.append(ParsedDocument(ordinal=index, text=text, title=f"page {index + 1}"))
        return documents
