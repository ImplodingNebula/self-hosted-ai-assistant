"""Image parser: read the text in an uploaded image with OCR.

An image is very often a picture OF text - a screenshot of an error, a photo of
a page, a chart with labels - and this reads that text with the same OCR engine
the PDF parser uses for scans, so an image is answerable by an ordinary text
model. It is the graceful half of image support: it works on every tier, needing
no vision-capable model.

The other half - a model actually SEEING a photograph that carries no text - needs
a vision-capable chat model, which no tier runs today (vision lives in a separate
on-demand model). That routing is a later slice. What matters here is that an
image with no readable text is still ACCEPTED, not failed: it is a valid file a
vision model could look at, so the ingest pipeline stores it and marks it ready
rather than rejecting it for having no text (see attachment_ingest).

OCR runs in the same killable subprocess the other native parsers use, because
Tesseract is native and cannot be interrupted in-thread. Where Tesseract is not
installed the parser still CLAIMS the image (so it is stored and accepted) and
simply returns no text, exactly as it would for a picture that holds none.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from app.parsing.base import DocumentParser, ParsedDocument
from app.parsing.isolation import run_isolated
from app.parsing.ocr import ocr_available

logger = logging.getLogger(__name__)

# How long one image gets to OCR before its child is killed. Generous for a
# large scan, bounded so a pathological file cannot run forever.
_DEFAULT_PARSE_TIMEOUT_SECONDS = 120.0

# What this format is called in a message the person who uploaded it reads.
_WHAT = "image"

# The largest image, in pixels, this will decode. 80 megapixels is far more than
# any page scan or screenshot (an A4 page at 600 DPI is about 35), and far less
# than what it takes to exhaust a small container: a decoded pixel costs roughly
# three bytes, so this bounds one image at a few hundred MB rather than the
# gigabytes Pillow's own default would allow. See _read_image_text.
MAX_IMAGE_PIXELS = 80_000_000

# Magic-byte signatures for the raster formats claimed by content sniff. Only
# strong, distinctive prefixes are listed: BMP's two-byte "BM" is too weak to
# claim a file on its own (it collides with ordinary text), so BMP is dispatched
# by media type and extension only. WEBP and the two TIFF byte orders are matched
# in :meth:`sniff` because their signatures are not a simple prefix.
_PREFIX_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n",  # png
    b"\xff\xd8\xff",  # jpeg
    b"GIF87a",  # gif
    b"GIF89a",
    b"II*\x00",  # tiff, little-endian
    b"MM\x00*",  # tiff, big-endian
)


def _read_image_text(data: bytes, language: str) -> str:
    """OCR an image's bytes; runs in the child process.

    Top-level so the spawn bootstrap can import it. Pillow opens the bytes and
    the shared OCR helper reads them; both are imported here, in the child, to
    keep the module cheap for the re-import spawn does.
    """
    from io import BytesIO

    from PIL import Image, ImageOps

    from app.parsing.ocr import image_to_text

    # A raster image's file SIZE says nothing about how much memory it becomes. A
    # 12000x12000 PNG of flat colour compresses to a couple of hundred KB - well
    # inside the upload limit on the way in - and decodes to some 430MB on the way
    # out, which is how a worker on a small box gets killed by the kernel. Pillow
    # has a guard for this but pitches it far too high to protect a 4GB machine,
    # and below twice its ceiling it only warns.
    #
    # So the header's declared size is checked before a single pixel is decoded,
    # and Pillow's own ceiling is lowered behind it, for the formats that can lie
    # about their size until they are unpacked.
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

    image = Image.open(BytesIO(data))
    try:
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"it is {width}x{height} pixels, past the "
                f"{MAX_IMAGE_PIXELS // 1_000_000} megapixel limit"
            )
        image.load()
        try:
            # A phone photograph of a page is almost always stored landscape with
            # an EXIF tag saying which way is up. Tesseract does not read that tag,
            # so an unrotated photo OCRs sideways and comes back as nothing at all -
            # the single most likely way a perfectly good photo of a document reads
            # as empty. This applies the rotation the camera recorded.
            #
            # In place, because the copy it makes otherwise is a second full decoded
            # image - at the ceiling above, a few hundred more MB - and it makes that
            # copy even for an image with no orientation tag at all, which is most of
            # them. In place it is a true no-op on those.
            ImageOps.exif_transpose(image, in_place=True)
            return image_to_text(image, language=language)
        except Exception as exc:  # noqa: BLE001 - a failed read is not an unusable image
            # OCR present but unable to read this image (a missing language pack, an
            # unusual colour mode, a malformed EXIF block) is the picture-with-no-text
            # case, not a bad file: degrade to no text so the image is still accepted,
            # matching the PDF parser's per-page handling. Only a failure to DECODE
            # the image (above) means the file itself is unreadable.
            logger.warning("Could not read the text in an image: %s", exc)
            return ""
    finally:
        image.close()


class ImageParser(DocumentParser):
    """Reads an uploaded image as one text unit, via OCR when it is available."""

    media_types: ClassVar[frozenset[str]] = frozenset(
        {
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
            "image/bmp",
            "image/tiff",
        }
    )
    extensions: ClassVar[frozenset[str]] = frozenset(
        {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
    )

    def __init__(
        self, *, timeout_seconds: float = _DEFAULT_PARSE_TIMEOUT_SECONDS, language: str = "eng"
    ) -> None:
        """Create an image parser that OCRs with ``language`` under a time cap."""
        self._timeout = timeout_seconds
        self._language = language

    def sniff(self, data: bytes) -> bool:
        """Claim a file that carries a known image signature, however it was named.

        Lets a mislabelled image (sent as octet-stream, or with no extension) be
        recognised by its bytes rather than falling through to the text parser
        and decoding to garbage.
        """
        if any(data.startswith(sig) for sig in _PREFIX_SIGNATURES):
            return True
        # WEBP is "RIFF....WEBP"; the four size bytes between the two markers mean
        # it is not a simple prefix.
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"

    async def parse(
        self, data: bytes, *, filename: str, media_type: str | None = None
    ) -> list[ParsedDocument]:
        """Extract the image's text by OCR; an image with none yields nothing.

        No text is not a failure for an image - unlike a document, its value is
        not only its text - so an empty result here becomes an accepted, ready
        attachment upstream rather than a rejected one.

        Raises:
            ParseFailedError: The image is unreadable, or OCR ran past the cap.
        """
        if not data or not ocr_available():
            # No bytes, or no OCR engine: the image is still a valid attachment,
            # it just contributes no text here.
            return []
        text = await run_isolated(
            _read_image_text, data, self._language, timeout_seconds=self._timeout, what=_WHAT
        )
        if not text.strip():
            return []
        return [ParsedDocument(ordinal=0, text=text)]
