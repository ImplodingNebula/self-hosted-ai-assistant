"""Optical character recognition, for the pages a PDF only pictures.

A born-digital PDF carries its text and the PDF parser reads it directly. A
SCANNED one carries an image of text and no text layer at all, so direct
extraction yields nothing. OCR is what reads those: it recognises the characters
in the picture, so a text-only model can still answer from a document that only
ever existed on paper.

OCR is done by Tesseract, which is a SYSTEM binary, not a Python package - the
one dependency in this whole pipeline that a deployment can genuinely lack (an
image built without it, a bare local checkout). So it is treated as optional
everywhere: :func:`ocr_available` reports whether the binary is installed, and a
caller that finds it absent simply does not OCR - a scanned page comes back empty,
exactly as it did before this existed, rather than failing the upload. The
capable path and the graceful one are the same code with one gate.

This module is import-light on purpose: it is reached inside the PDF parser's
child process (see :mod:`app.parsing.isolation`), which re-imports its module on
spawn, and the heavy libraries (Pillow, pytesseract) load only when a page
actually needs reading.
"""

from __future__ import annotations

import functools
import logging
import shutil
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# The Tesseract executable's name, as it is invoked on PATH.
_TESSERACT_BINARY = "tesseract"


@dataclass(frozen=True)
class OcrConfig:
    """How to OCR a scanned page, and when a page counts as one.

    A frozen dataclass so it crosses the process boundary into the parser's child
    by value, carrying every knob the child needs without reaching back for
    settings it cannot see.

    Attributes:
        language: Tesseract language(s) to recognise, e.g. ``"eng"`` or
            ``"eng+deu"``. The matching language data must be installed.
        dpi: Resolution to render a page at before OCR. Higher reads small type
            better but is slower and heavier; 200-300 is the usual range.
        min_chars: A page whose text layer holds fewer than this many
            non-whitespace characters is treated as a scan and sent to OCR. Zero
            for a true scan; a small floor also catches a page carrying only a
            stray watermark or page number.
        max_pages: The most pages of one document to OCR. OCR is seconds per page,
            so a very long scan is bounded rather than allowed to run for minutes;
            pages past the cap keep whatever sparse text they had.
    """

    language: str = "eng"
    dpi: int = 200
    min_chars: int = 16
    max_pages: int = 20


@functools.cache
def ocr_available() -> bool:
    """Return whether the Tesseract binary is on PATH. Checked once per process."""
    return shutil.which(_TESSERACT_BINARY) is not None


def image_to_text(image: Any, *, language: str) -> str:
    """Recognise the text in a rendered page image.

    Args:
        image: A PIL image of the page (the PDF parser renders it).
        language: Tesseract language(s) to recognise with.

    Returns:
        The recognised text, stripped. Empty when the page holds no readable text.

    Raises:
        RuntimeError: OCR failed for a reason other than an absent binary (a
            missing language pack, a Tesseract crash). The caller treats a page
            that will not OCR the same as one with no text - it is skipped, not
            fatal - so this surfaces as a log line, not a failed upload.
    """
    import pytesseract

    try:
        return pytesseract.image_to_string(image, lang=language).strip()
    except pytesseract.TesseractError as exc:
        # A recognised-but-unreadable page (e.g. the language pack is not
        # installed). Named distinctly so the caller can log it and move on.
        raise RuntimeError(f"OCR failed: {exc}") from exc
