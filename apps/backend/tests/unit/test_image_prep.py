"""Tests for sizing an image against a vision model's cell budget.

The expected numbers come from a live measurement against qwen3-vl:2b on Ollama
0.32.0 (cost read from prompt_eval_count), not from a model card - see the
module docstring of app.services.image_prep.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.services.image_prep import (
    CELL_PIXELS,
    FLOOR_CELLS,
    ImagePrepError,
    cells_for,
    estimate_tokens,
    normalize,
    target_size,
)


def png(width: int, height: int, *, mode: str = "RGB", color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def dims(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as img:
        return img.size


# --- the grid ---


def test_cells_round_to_nearest_not_up() -> None:
    # Pinned by measurement: 1512x1512 cost 2211 tokens, which is 47*47 plus
    # sentinels. Rounding UP would predict 48*48 and overshoot by 95.
    assert cells_for(1512, 1512) == 47 * 47
    assert cells_for(1024, 1024) == 32 * 32
    assert cells_for(1920, 1080) == 60 * 34
    assert cells_for(2560, 1440) == 80 * 45


def test_cells_never_zero_for_a_tiny_image() -> None:
    # round(10/32) is 0; a sub-cell image still occupies one cell.
    assert cells_for(10, 10) == 1
    assert cells_for(1, 1) == 1


# --- the cost estimate ---


def test_estimate_is_flat_below_the_floor() -> None:
    # MEASURED: every size from 28x28 to 1024x1024 cost the same 1026 tokens.
    floor = estimate_tokens(1024, 1024)
    for w, h in [(28, 28), (112, 112), (448, 448), (512, 512), (784, 784)]:
        assert estimate_tokens(w, h) == floor
    assert floor >= FLOOR_CELLS


# Every image cost measured on the runtime, as (width, height, prompt tokens).
# The floor rows matter as much as the large ones: an image well under the floor
# still cost the floor.
MEASURED_COSTS = [
    (28, 28, 1026),
    (512, 512, 1026),
    (1024, 1024, 1026),
    (1338, 753, 1081),
    (1512, 1512, 2211),
    (1920, 1080, 2043),
    (2560, 1440, 3603),
]


@pytest.mark.parametrize(("width", "height", "measured"), MEASURED_COSTS)
def test_estimate_never_undercounts_a_measured_cost(width, height, measured) -> None:
    """The estimate must sit AT or ABOVE what the runtime actually charged.

    The asymmetry is the whole point and it is not a style preference. Reserving
    too much costs some retrieved context, which is visible in the answer.
    Reserving too little overfills the window, and the runtime then truncates the
    prompt from the HEAD with no error at all - dropping the system prompt, and
    the tool guidance and injection defense inside it. So this asserts a floor,
    never equality.
    """
    assert estimate_tokens(width, height) >= measured


@pytest.mark.parametrize(("width", "height", "measured"), MEASURED_COSTS)
def test_estimate_is_not_wastefully_above_the_measured_cost(width, height, measured) -> None:
    # The other side of the same coin: a reserve far above the truth would
    # squeeze out grounding for no reason. Generous, not arbitrary.
    assert estimate_tokens(width, height) <= measured + 200


def test_estimate_applies_a_known_ceiling() -> None:
    # The runtime clamps around 4015 cells; a caller that knows it must not
    # reserve for the uncapped grid of a huge image.
    uncapped = estimate_tokens(4000, 3000)
    capped = estimate_tokens(4000, 3000, ceiling_cells=4015)
    assert uncapped > capped
    # Still at or above the 4018 actually measured for that image.
    assert capped >= 4018


# --- choosing a target size ---


def test_target_fits_the_budget_and_keeps_aspect() -> None:
    w, h = target_size(2560, 1440)
    assert cells_for(w, h) <= FLOOR_CELLS
    assert abs((w / h) - (2560 / 1440)) < 0.02


def test_target_beats_a_square_box_on_a_wide_image() -> None:
    # The reason this module exists. Fitting 16:9 into a 1024 square gives
    # 1024x576 = 576 cells, wasting 44% of a budget that is charged whole -
    # and measurably losing text the area fit keeps.
    boxed = cells_for(1024, 576)
    area_w, area_h = target_size(2560, 1440)
    assert cells_for(area_w, area_h) > boxed * 1.5
    assert area_w > 1024


def test_target_never_upscales() -> None:
    # An image below the floor is already charged at the floor; enlarging it
    # invents detail and buys nothing.
    assert target_size(320, 240) == (320, 240)
    assert target_size(64, 64) == (64, 64)


def test_target_honours_a_bigger_budget() -> None:
    small = target_size(4000, 3000, budget_cells=FLOOR_CELLS)
    large = target_size(4000, 3000, budget_cells=4 * FLOOR_CELLS)
    assert large[0] > small[0] and large[1] > small[1]
    assert cells_for(*large) <= 4 * FLOOR_CELLS


def test_target_rejects_an_empty_image() -> None:
    with pytest.raises(ImagePrepError):
        target_size(0, 100)


# --- normalize ---


def test_normalize_shrinks_a_large_image_and_reports_its_cost() -> None:
    data, media_type, tokens = normalize(png(2560, 1440))
    w, h = dims(data)
    assert cells_for(w, h) <= FLOOR_CELLS
    assert media_type == "image/png"
    # Measured cost of the normalized 2560x1440 screenshot was 1102.
    assert abs(tokens - 1102) <= 80


def test_normalize_returns_original_bytes_when_it_already_fits() -> None:
    # Re-encoding an image that needs nothing costs quality for no benefit.
    original = png(800, 600)
    data, _, tokens = normalize(original)
    assert data is original
    assert tokens >= FLOOR_CELLS


def test_normalize_preserves_the_media_type_of_an_untouched_jpeg() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (10, 20, 30)).save(buf, format="JPEG")
    data, media_type, _ = normalize(buf.getvalue())
    assert data is not None
    assert media_type == "image/jpeg"


def test_normalize_flattens_transparency_onto_white() -> None:
    # A cropped screenshot is often dark text on transparency. Compositing onto
    # the default black would make exactly that unreadable.
    buf = io.BytesIO()
    Image.new("RGBA", (2000, 2000), (0, 0, 0, 0)).save(buf, format="PNG")
    data, _, _ = normalize(buf.getvalue())
    with Image.open(io.BytesIO(data)) as img:
        assert img.mode == "RGB"
        assert img.getpixel((5, 5)) == (255, 255, 255)


def test_normalize_raises_on_bytes_that_are_not_an_image() -> None:
    with pytest.raises(ImagePrepError):
        normalize(b"this is not a picture")


def test_normalize_raises_on_truncated_image_bytes() -> None:
    with pytest.raises(ImagePrepError):
        normalize(png(2000, 2000)[:120])


def test_normalize_respects_a_larger_budget() -> None:
    _, _, cheap = normalize(png(3000, 2000))
    _, _, detailed = normalize(png(3000, 2000), budget_cells=4 * FLOOR_CELLS)
    assert detailed > cheap * 3


def test_normalized_sides_land_on_whole_cells() -> None:
    # Not required for correctness, but a side that straddles a cell boundary
    # pays for a row the image only partly fills.
    data, _, _ = normalize(png(3000, 2000))
    w, h = dims(data)
    assert cells_for(w, h) <= FLOOR_CELLS
    assert w >= CELL_PIXELS and h >= CELL_PIXELS


def test_normalize_refuses_a_decompression_bomb_as_ImagePrepError() -> None:
    """A pixel bomb must be skippable, not fatal.

    PIL raises DecompressionBombError, which descends straight from Exception -
    NOT from OSError or ValueError. Leaving it out of the caught set let it
    escape the whole chat turn instead of skipping one image, and it took the
    honesty notice with it: the turn died with no answer and no explanation.
    A 210 KB PNG can carry 192 million pixels, so file size does not bound this.
    """
    bomb = png(16000, 12000, mode="1", color=0)
    assert len(bomb) < 1_000_000, "the point is that a bomb is SMALL on disk"
    with pytest.raises(ImagePrepError):
        normalize(bomb)


def test_normalize_refuses_on_the_header_before_decoding() -> None:
    # The guard reads declared dimensions and refuses, rather than relying on
    # Pillow - whose own ceiling only WARNS below twice its limit.
    from app.parsing.image import MAX_IMAGE_PIXELS

    side = int(MAX_IMAGE_PIXELS**0.5) + 2000
    over = png(side, side, mode="1", color=0)
    with pytest.raises(ImagePrepError):
        normalize(over)


def test_a_small_exotic_format_is_converted_rather_than_forwarded() -> None:
    # The fast path forwards bytes untouched, but only for formats a vision
    # model can be relied on to decode. A small BMP must not slip through
    # unconverted just because it needed no resize.
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (10, 20, 30)).save(buf, format="BMP")
    data, media_type, _ = normalize(buf.getvalue())
    assert media_type == "image/png"
    assert data != buf.getvalue()
    with Image.open(io.BytesIO(data)) as img:
        assert img.format == "PNG"
        assert img.size == (320, 240), "converting must not also resize it"


def test_a_small_png_still_takes_the_untouched_fast_path() -> None:
    original = png(320, 240)
    data, media_type, _ = normalize(original)
    assert data is original
    assert media_type == "image/png"
