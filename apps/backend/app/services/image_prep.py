"""Size an image for a vision model, and say what it will cost in tokens.

A vision model does not read pixels, it reads a grid of fixed-size cells, and
the serving runtime clamps that grid into a band before the model ever sees it.
Both ends of that band were MEASURED against this project's runtime rather than
taken from a model card, because the card describes the weights and the clamp
belongs to the server:

    qwen3-vl:2b on Ollama 0.32.0, cost read from ``prompt_eval_count``

        28x28 through 1024x1024 ... 1026 tokens (identical - a FLOOR)
        1920x1080 ................. 2043
        2560x1440 ................. 3603
        4000x3000 ................. 4018 (a CEILING near 4015)

Two consequences drive everything here. Shrinking an image below the floor buys
NOTHING - a 768x432 image and a 1024x576 image both cost about 1059 tokens, and
the smaller one simply reads less text. And the budget is spent on CELLS, not on
the length of the longest side, so fitting a 16:9 image inside a square box
wastes almost half of it: 1024x576 is 576 cells where 1338x753 is 1008 cells for
the same price.

Measured on a transcription task (five codes at font sizes 44 down to 12):

    source       as-is         square box    this module
    2560x1440    5/5 @ 3627    4/5 @ 1059    5/5 @ 1102
    1920x1080    5/5 @ 2067    5/5 @ 1059    5/5 @ 1102
    3000x2000    5/5 @ 4083    3/5 @ 1107    4/5 @ 1107

So normalizing is free up to about 1440p and costs the finest print above it.
That residual is a real trade rather than a bug - a caller who knows the fine
print is the question can raise ``budget_cells``, which is why the budget is an
argument and not a constant.
"""

from __future__ import annotations

import io
import logging
import math

from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError

from app.parsing.image import MAX_IMAGE_PIXELS

logger = logging.getLogger(__name__)

# One visual token covers a 32x32 pixel cell: Qwen3-VL ships patch_size 16 with
# a 2x2 spatial merge. NOT the 14/28 of Qwen2-VL and Qwen2.5-VL - Qwen3-VL still
# names its processor Qwen2VLImageProcessorFast, whose CLASS default is 14, so
# reading the class instead of the checkpoint config is an easy way to land 23%
# under the true cost. The measurement above independently pins 32: a 1024x1024
# image costs exactly 1024 visual tokens.
CELL_PIXELS = 32

# The runtime never charges less than this many cells however small the image,
# so it is both the floor of the cost and the target worth normalizing to.
FLOOR_CELLS = 1024

# Sentinels wrapping the grid, plus headroom. The sentinels alone are 2 to 3,
# but the MEASURED cost of a normalized image ran up to about 80 tokens above the
# cell arithmetic, so the estimate carries a margin on top. Deliberately generous:
# over-reserving costs some retrieved context and shows up in the answer, while
# under-reserving silently overfills the window and the runtime then truncates
# from the HEAD - dropping the system prompt, and the tool guidance and injection
# defense inside it, with no error anywhere.
_ESTIMATE_MARGIN_TOKENS = 96

# Formats forwarded untouched when the image already fits. Everything else is
# re-encoded to PNG even at a size needing no resize, so an exotic container
# (BMP, TIFF, a multi-frame GIF) cannot reach a model that may not decode it
# purely because it happened to be small.
_PASS_THROUGH_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})


class ImagePrepError(Exception):
    """An image could not be decoded or re-encoded."""


def cells_for(width: int, height: int) -> int:
    """Return the grid size an image of these dimensions occupies.

    The runtime rounds each side to the NEAREST whole cell rather than rounding
    up, which the measurements pin exactly: 1512x1512 costs 2211 tokens, and
    ``round(1512/32) == 47`` gives 47*47 + sentinels, where rounding up would
    predict 48*48 and overshoot by 95.
    """
    return max(1, round(width / CELL_PIXELS)) * max(1, round(height / CELL_PIXELS))


def estimate_tokens(width: int, height: int, *, ceiling_cells: int | None = None) -> int:
    """Estimate the prompt tokens an image of these dimensions will cost.

    Clamped to the runtime's floor, because an image smaller than the floor is
    charged at the floor anyway. ``ceiling_cells`` applies the server's upper
    clamp when the caller knows it; left unset the estimate keeps growing, which
    errs toward reserving too much.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        ceiling_cells: Upper clamp the runtime applies, if known.

    Returns:
        Estimated prompt tokens, sentinels included.
    """
    grid = max(FLOOR_CELLS, cells_for(width, height))
    if ceiling_cells is not None:
        grid = min(grid, ceiling_cells)
    return grid + _ESTIMATE_MARGIN_TOKENS


def target_size(width: int, height: int, *, budget_cells: int = FLOOR_CELLS) -> tuple[int, int]:
    """Return the largest size within ``budget_cells`` that keeps the aspect ratio.

    Returns the input unchanged when it already fits - this NEVER upscales. A
    small image is already charged at the floor, so enlarging it would add
    interpolated detail the model would be charged for twice over: once in
    tokens it does not need and once in the false impression of resolution.

    Args:
        width: Source width in pixels.
        height: Source height in pixels.
        budget_cells: Grid size to fit within.

    Returns:
        The (width, height) to resize to.
    """
    if width <= 0 or height <= 0:
        raise ImagePrepError(f"image has no area: {width}x{height}")
    if cells_for(width, height) <= budget_cells:
        return width, height
    # cells ~ (w*s / CELL) * (h*s / CELL) = budget, so s = sqrt(budget * CELL^2 / area).
    scale = math.sqrt(budget_cells * CELL_PIXELS * CELL_PIXELS / (width * height))
    new_w, new_h = _scaled(width, height, scale)
    # Rounding each side to the nearest cell can push the product one row or
    # column over the budget; step down until it fits rather than trusting the
    # closed form, which is off by one often enough to matter.
    while cells_for(new_w, new_h) > budget_cells and scale > 0.05:
        scale *= 0.98
        new_w, new_h = _scaled(width, height, scale)
    return new_w, new_h


def _scaled(width: int, height: int, scale: float) -> tuple[int, int]:
    """Apply a scale factor, never dropping a side below one whole cell."""
    return (
        max(CELL_PIXELS, round(width * scale)),
        max(CELL_PIXELS, round(height * scale)),
    )


def normalize(data: bytes, *, budget_cells: int = FLOOR_CELLS) -> tuple[bytes, str, int]:
    """Resize an image down to ``budget_cells`` and report its token cost.

    Returns the ORIGINAL bytes untouched when the image already fits, so an
    image that needs nothing done to it is not silently re-encoded (which would
    cost quality for no benefit and change bytes the caller may be comparing).

    Args:
        data: The encoded image as stored.
        budget_cells: Grid size to fit within; larger keeps more detail and
            costs proportionally more tokens.

    Returns:
        A tuple of (encoded bytes, media type, estimated prompt tokens).

    Raises:
        ImagePrepError: The bytes are not a decodable image, or re-encoding
            failed. Callers treat this as "this image cannot be delivered" and
            must keep saying so to the user.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            # Refuse on the HEADER, before a single pixel is decoded. Pillow's own
            # guard is not enough alone: its ceiling is pitched for large servers,
            # below twice that ceiling it only WARNS, and file size does not bound
            # decoded size at all - a couple of hundred KB of PNG can unpack to
            # hundreds of MB. Same limit as the parser that read this file on
            # upload, imported rather than restated so the two cannot drift.
            if width * height > MAX_IMAGE_PIXELS:
                raise ImagePrepError(f"image is too large to decode safely: {width}x{height}")
            new_w, new_h = target_size(width, height, budget_cells=budget_cells)
            if (new_w, new_h) == (width, height) and (img.format or "") in _PASS_THROUGH_FORMATS:
                return data, _media_type(img.format), estimate_tokens(width, height)
            # Flatten onto white through the alpha channel: a transparent PNG
            # otherwise composites onto black once the model rasterizes it, and
            # dark-on-transparent text (a cropped screenshot) turns unreadable.
            resized = img.convert("RGBA") if img.mode in ("P", "LA", "RGBA") else img
            resized = resized.resize((new_w, new_h), Image.LANCZOS)
            if resized.mode == "RGBA":
                flat = Image.new("RGB", resized.size, (255, 255, 255))
                flat.paste(resized, mask=resized.split()[-1])
                resized = flat
            elif resized.mode != "RGB":
                resized = resized.convert("RGB")
            buf = io.BytesIO()
            # PNG, not JPEG: the point of sending pixels is usually text or a
            # chart, and JPEG ringing lands hardest on exactly those edges.
            resized.save(buf, format="PNG", optimize=True)
    except (
        UnidentifiedImageError,
        # NOT an OSError and NOT a ValueError - it descends straight from
        # Exception, so leaving it out of this tuple let it escape the whole
        # turn instead of skipping one image, taking the honesty notice with it.
        DecompressionBombError,
        MemoryError,
        NotImplementedError,
        OSError,
        ValueError,
    ) as exc:
        raise ImagePrepError(f"could not prepare image: {exc}") from exc
    logger.debug("normalized image %dx%d -> %dx%d", width, height, new_w, new_h)
    return buf.getvalue(), "image/png", estimate_tokens(new_w, new_h)


def _media_type(pillow_format: str | None) -> str:
    """Map Pillow's format name to a media type, defaulting to PNG."""
    if not pillow_format:
        return "image/png"
    fmt = pillow_format.lower()
    return "image/jpeg" if fmt in ("jpeg", "jpg") else f"image/{fmt}"
