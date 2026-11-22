"""Image attachments: dispatch, OCR text, and acceptance of a text-less image."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterator
from io import BytesIO

import pytest

from app.parsing.image import MAX_IMAGE_PIXELS, ImageParser, _read_image_text
from app.parsing.ocr import ocr_available
from app.parsing.registry import default_registry

needs_tesseract = pytest.mark.skipif(
    not ocr_available(), reason="the Tesseract binary is not installed"
)


@pytest.fixture(autouse=True)
def _keep_pillow_s_ceiling() -> Iterator[None]:
    """Undo the parser's edit to Pillow's global pixel ceiling.

    In production ``_read_image_text`` only ever runs in a spawned child, so the
    global it lowers dies with that process. The tests below call it in-process to
    reach the code a monkeypatch cannot follow into a child - which leaves the
    ceiling lowered for every test that runs afterwards.
    """
    from PIL import Image

    ceiling = Image.MAX_IMAGE_PIXELS
    yield
    Image.MAX_IMAGE_PIXELS = ceiling


def _png(text: str | None = None, *, size: tuple[int, int] = (640, 200)) -> bytes:
    """Build a PNG, optionally with text drawn on it for OCR."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, "white")
    if text:
        ImageDraw.Draw(image).text((20, 80), text, fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _png_header_claiming(width: int, height: int) -> bytes:
    """A valid PNG header declaring ``width`` x ``height``, with no pixels behind it.

    A decompression bomb IS this: a header that promises far more image than the
    file could possibly hold. Building one by hand rather than by rendering a real
    100-megapixel image keeps the test from allocating the very 300MB it is about.
    """

    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    # width, height, 8-bit, truecolour, default compression/filter/interlace.
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b""))
        + chunk(b"IEND", b"")
    )


def test_the_registry_claims_images_by_type_and_extension() -> None:
    registry = default_registry()

    assert isinstance(registry.resolve(filename="a.png"), ImageParser)
    assert isinstance(registry.resolve(filename="a.jpg"), ImageParser)
    assert isinstance(registry.resolve(filename="a.webp"), ImageParser)
    assert isinstance(registry.resolve(filename="x", content_type="image/png"), ImageParser)


def test_a_mislabelled_image_is_claimed_by_its_bytes() -> None:
    # Sent as octet-stream with no extension: only the PNG signature can claim it,
    # otherwise it would fall through to the text parser and decode to garbage.
    registry = default_registry()

    parser = registry.resolve(
        filename="clipboard", content_type="application/octet-stream", data=_png()
    )

    assert isinstance(parser, ImageParser)


def test_an_ocr_error_degrades_to_no_text_rather_than_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OCR present but unable to read this image - the classic case is a Tesseract
    # binary installed without the configured language pack - must not fail the
    # upload: the image is still an accepted, text-less attachment. Tested on the
    # in-process helper because the real parse OCRs in a spawned child that
    # re-imports fresh, out of a parent monkeypatch's reach.
    from app.parsing import ocr as ocr_module

    def _boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("tesseract: failed loading language 'eng'")

    monkeypatch.setattr(ocr_module, "image_to_text", _boom)

    # The image opens and loads fine; only the OCR step fails.
    assert _read_image_text(_png("unreadable"), "eng") == ""


async def test_an_image_without_ocr_yields_no_text_but_does_not_fail() -> None:
    # No OCR configured for the parser (or unavailable): a valid image still
    # parses to an empty result rather than raising - it is a real file a vision
    # model could see, just with no text here.
    documents = await ImageParser().parse(_png(), filename="photo.png")

    assert documents == []


@needs_tesseract
@pytest.mark.ocr
async def test_ocr_reads_the_text_in_an_image() -> None:
    documents = await ImageParser(language="eng").parse(
        _png("HELLO IMAGE", size=(900, 240)), filename="shot.png"
    )

    assert documents, "OCR recovered no text from the image"
    recovered = documents[0].text.upper()
    assert "HELLO" in recovered or "IMAGE" in recovered


def test_a_decompression_bomb_is_refused_before_it_is_decoded() -> None:
    """A file small enough to upload can still be big enough to kill the worker.

    An image's file SIZE says nothing about the memory it becomes. A 12000x12000
    PNG of flat colour compresses to a few hundred KB - well under the upload
    limit - and decodes to some 430MB, which on a small box means the kernel kills
    the process doing the decoding. So the size DECLARED IN THE HEADER is checked
    before a single pixel is unpacked.

    The bomb here is a header and nothing else: 60 bytes claiming 144 megapixels,
    with no image data behind them at all. That is the whole point - if the check
    ever regressed to inspecting the decoded image, this file could not be decoded
    and the test would fail rather than quietly passing on a real one.
    """
    bomb = _png_header_claiming(12_000, 12_000)

    # 144 megapixels: past the ceiling, but inside the band where Pillow's own
    # guard merely WARNS. That band is exactly what this check exists to cover, so
    # it is the band worth pinning.
    assert len(bomb) < 100

    with pytest.raises(ValueError, match="megapixel"):
        _read_image_text(bomb, "eng")


def test_an_ordinary_screenshot_is_nowhere_near_the_ceiling() -> None:
    # The cap has to be generous enough that no real page scan trips it: an A4
    # page at 600 DPI is about 35 megapixels, comfortably inside 80.
    assert MAX_IMAGE_PIXELS > 40_000_000


@pytest.mark.ocr
@pytest.mark.skipif(not ocr_available(), reason="Tesseract is not installed")
def test_a_photo_taken_sideways_is_turned_upright_before_it_is_read() -> None:
    """A phone photo of a page is the case this exists for.

    Cameras store the sensor's raw landscape frame and record the rotation as an
    EXIF tag rather than rotating the pixels. Tesseract does not read that tag, so
    an unrotated photo is OCR'd sideways and comes back as nothing at all - which
    is the single most likely way a perfectly good photo of a document reads as
    empty. Applying the recorded rotation is what makes it readable.
    """
    from PIL import Image

    upright = Image.new("RGB", (900, 240), "white")
    from PIL import ImageDraw

    ImageDraw.Draw(upright).text((20, 90), "ROTATED TEXT", fill="black")

    # Store it the way a camera does: rotated on disk, with the tag that says so.
    buffer = BytesIO()
    # PIL's ROTATE_90 turns counter-clockwise, so undoing it means a quarter turn
    # CLOCKWISE - which is orientation 6. (8 is the other way, and would leave the
    # text upside down: exactly as unreadable, and a much more confusing failure.)
    sideways = upright.transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[274] = 6
    sideways.save(buffer, format="JPEG", exif=exif, quality=95)

    text = _read_image_text(buffer.getvalue(), "eng")

    # Read sideways, this comes back as noise or as nothing at all. Read upright,
    # the words are there. (OCR still mis-reads the odd letter, so the assertion
    # is on a word being found, not on a perfect transcription.)
    assert "TEXT" in text.upper()
