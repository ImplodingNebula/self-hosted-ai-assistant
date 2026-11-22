"""OCR backfill for scanned PDF pages: the heuristic, the cap, and degradation.

A scanned page is modelled as an image saved as a PDF - Pillow writes exactly
that, a page whose only content is a picture and whose text layer is empty. The
tests that need to actually read one require the Tesseract binary and are marked
``ocr`` (and skipped where it is absent); the rest drive the decision logic with
the OCR step stubbed, so they run everywhere and pin the behaviour that matters:
which pages get OCR'd, how many, and what happens when OCR is unavailable.
"""

from __future__ import annotations

from io import BytesIO

import pytest

from app.parsing import pdf as pdf_module
from app.parsing.ocr import OcrConfig, ocr_available
from app.parsing.pdf import PdfParser, _read_pages
from tests.unit.test_pdf_parser import _make_pdf

needs_tesseract = pytest.mark.skipif(
    not ocr_available(), reason="the Tesseract binary is not installed"
)


def _scanned_pdf(text: str, *, size: tuple[int, int] = (1200, 300)) -> bytes:
    """Build a one-page PDF whose text is a picture, not a text layer.

    Pillow saving an image as a PDF produces exactly a scan: the page renders the
    image and carries no extractable text, so direct extraction finds nothing and
    only OCR can read it.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    # The default bitmap font is small; scale it up so OCR has clear glyphs.
    draw.text((20, 120), text, fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PDF", resolution=150.0)
    return buffer.getvalue()


def test_a_scan_is_left_empty_when_ocr_is_off() -> None:
    # No OCR configured: an image-only page has no text layer, so it yields
    # nothing - exactly the behaviour before OCR existed.
    pages = _read_pages(_scanned_pdf("Invoice 4471"), 10_000, None)

    assert pages == [""] or pages[0].strip() == ""


def test_the_heuristic_ocrs_only_the_sparse_pages(monkeypatch) -> None:
    # A born-digital page carries its text and must NOT be OCR'd (OCR is slow and
    # would only re-derive what is already there); a scan, being sparse, must be.
    # The OCR step is stubbed so this runs without the binary and asserts exactly
    # which pages reached it.
    ocred: list[int] = []

    def fake_ocr_page(page: object, ocr: OcrConfig) -> str:
        ocred.append(1)
        return "OCR RECOVERED TEXT"

    monkeypatch.setattr(pdf_module, "_ocr_page", fake_ocr_page)

    born_digital = _make_pdf(["This page has a real, extractable text layer."])
    text_pages = _read_pages(born_digital, 10_000, OcrConfig())
    assert len(ocred) == 0  # nothing sparse, so OCR never ran
    assert "extractable text layer" in text_pages[0]

    scan = _scanned_pdf("Scanned only")
    scan_pages = _read_pages(scan, 10_000, OcrConfig())
    assert len(ocred) == 1  # the one sparse page was sent to OCR
    assert scan_pages[0] == "OCR RECOVERED TEXT"


def test_ocr_respects_the_per_document_page_cap(monkeypatch) -> None:
    # OCR is seconds per page, so a long scan is bounded: past the cap, a sparse
    # page keeps whatever little it had rather than being OCR'd.
    calls: list[int] = []
    monkeypatch.setattr(pdf_module, "_ocr_page", lambda page, ocr: calls.append(1) or "ocr text")

    # Three image-only pages, cap of two.
    from PIL import Image

    images = [Image.new("RGB", (800, 200), "white") for _ in range(3)]
    buffer = BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:])

    _read_pages(buffer.getvalue(), 10_000, OcrConfig(max_pages=2))

    assert len(calls) == 2


def test_a_single_pages_ocr_failure_does_not_fail_the_document(monkeypatch) -> None:
    # One unreadable page must not sink the whole file: the page is left as its
    # sparse self (skipped downstream) and the rest are read.
    def boom(page: object, ocr: OcrConfig) -> str:
        raise RuntimeError("tesseract fell over")

    monkeypatch.setattr(pdf_module, "_ocr_page", boom)

    pages = _read_pages(_scanned_pdf("unreadable"), 10_000, OcrConfig())

    assert pages == [""] or pages[0].strip() == ""


def test_ocr_downgrades_gracefully_when_tesseract_is_absent(monkeypatch) -> None:
    # Configured for OCR on a host without the binary: the parser must build,
    # warn, and fall back to text-layer-only rather than fail. A scan then yields
    # nothing, exactly as with OCR off.
    monkeypatch.setattr(pdf_module, "ocr_available", lambda: False)

    parser = PdfParser(ocr=OcrConfig())

    assert parser._ocr is None


@needs_tesseract
@pytest.mark.ocr
async def test_ocr_reads_a_real_scanned_page() -> None:
    # The end-to-end path with the real engine: an image-only PDF whose text only
    # OCR can recover. Only runs where Tesseract is installed.
    scan = _scanned_pdf("HELLO OCR WORLD", size=(1600, 400))

    documents = await PdfParser(ocr=OcrConfig(dpi=200)).parse(scan, filename="scan.pdf")

    assert documents, "OCR recovered no text from the scanned page"
    recovered = documents[0].text.upper()
    # OCR is not pixel-perfect; assert the distinctive tokens, not an exact match.
    assert "OCR" in recovered
    assert "HELLO" in recovered or "WORLD" in recovered
