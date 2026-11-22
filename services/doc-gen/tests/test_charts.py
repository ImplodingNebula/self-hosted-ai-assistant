"""Chart rendering: real image bytes, the caps, and the forgiving type mapping.

Charts are checked structurally (PNG signature and IHDR dimensions, SVG root
element) rather than pixel by pixel: matplotlib's exact pixels are its business,
but "this is a PNG of the size we asked for" is ours.
"""

from __future__ import annotations

import struct

import pytest
from pydantic import ValidationError

from app import charts
from app.main import build_app
from tests.helpers import artifact_bytes, artifact_of, call_content, mcp_session, text_only

pytestmark = pytest.mark.anyio

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_dimensions(data: bytes) -> tuple[int, int]:
    """Width and height from a PNG's IHDR chunk (the first chunk, always)."""
    assert data.startswith(_PNG_SIGNATURE), data[:16]
    assert data[12:16] == b"IHDR"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


async def render(session, spec: dict) -> tuple[str, bytes, str]:
    """Call ``render_chart`` and return (summary, image bytes, filename)."""
    summary, resource = artifact_of(await call_content(session, "render_chart", {"spec": spec}))
    return summary, artifact_bytes(resource), str(resource.uri).removeprefix("artifact://")


async def test_png_chart_is_a_real_png() -> None:
    spec = {
        "type": "line",
        "title": "Signups",
        "x_label": "Week",
        "y_label": "Count",
        "labels": ["w1", "w2", "w3"],
        "series": [{"name": "2026", "values": [10, 14, 19]}],
        "format": "png",
    }
    async with mcp_session(build_app()) as session:
        summary, data, filename = await render(session, spec)

    assert filename == "Signups.png"
    assert "line chart" in summary
    assert "3 points" in summary
    width, height = png_dimensions(data)
    assert (width, height) == (990, 550)


async def test_svg_chart_is_a_real_svg() -> None:
    spec = {"type": "bar", "series": [{"values": [1, 2]}], "format": "svg", "filename": "bars"}
    async with mcp_session(build_app()) as session:
        _, data, filename = await render(session, spec)

    assert filename == "bars.svg"
    body = data.decode("utf-8")
    assert "<svg" in body
    assert body.rstrip().endswith("</svg>")


async def test_the_same_spec_renders_to_the_same_dimensions_twice() -> None:
    # Not a byte-for-byte determinism claim (matplotlib stamps metadata), but a
    # guard on the figure geometry being fixed rather than data-dependent.
    spec = {"type": "scatter", "series": [{"values": [3, 1, 4, 1, 5]}], "format": "png"}
    async with mcp_session(build_app()) as session:
        _, first, _ = await render(session, spec)
        _, second, _ = await render(session, spec)

    assert png_dimensions(first) == png_dimensions(second)


async def test_multiple_series_render_with_a_legend() -> None:
    spec = {
        "type": "bar",
        "labels": ["a", "b"],
        "series": [{"name": "one", "values": [1, 2]}, {"name": "two", "values": [3, 4]}],
        "format": "svg",
    }
    async with mcp_session(build_app()) as session:
        summary, data, _ = await render(session, spec)

    assert "2 series" in summary
    body = data.decode("utf-8")
    assert "one" in body and "two" in body


async def test_an_unknown_chart_type_falls_back_to_bar_and_says_so() -> None:
    # Forgiving: a small model naming a type this module does not draw still
    # gets a picture, and the summary reports what was actually drawn.
    async with mcp_session(build_app()) as session:
        summary, _, _ = await render(session, {"type": "waterfall", "series": [{"values": [1, 2]}]})

    assert "bar chart" in summary


async def test_a_type_alias_is_honored() -> None:
    async with mcp_session(build_app()) as session:
        summary, _, _ = await render(session, {"type": "donut", "series": [{"values": [1, 2]}]})

    assert "pie chart" in summary


async def test_a_pie_of_signed_data_is_refused_with_a_reason() -> None:
    async with mcp_session(build_app()) as session:
        content = await call_content(
            session, "render_chart", {"spec": {"type": "pie", "series": [{"values": [1, -2]}]}}
        )

    assert "non-negative" in text_only(content)


async def test_the_pie_wedge_cap_is_enforced_at_the_boundary() -> None:
    # The measured reason this cap is tighter than the point cap: pie cost is
    # not linear, and a 10,000-wedge pie took ~53s - the whole tool deadline.
    at_cap = [1.0] * 50
    async with mcp_session(build_app()) as session:
        summary, data, _ = await render(session, {"type": "pie", "series": [{"values": at_cap}]})
        assert "pie chart" in summary
        assert data.startswith(_PNG_SIGNATURE)

        over = await call_content(
            session,
            "render_chart",
            {"spec": {"type": "pie", "series": [{"values": at_cap + [1.0]}]}},
        )

    message = text_only(over)
    assert "51 wedges" in message
    assert str(charts.MAX_PIE_WEDGES) in message


async def test_category_ticks_are_dropped_once_they_stop_being_readable() -> None:
    # MAX_CATEGORY_TICKS is a performance guard with a measured reason: one tick
    # label per data point was the single slowest part of a large chart (the
    # suite ran 20x slower before it). At the cap the caller's labels appear; one
    # past it the axis falls back to matplotlib's own numeric ticks.
    at_cap = [f"cat{n}" for n in range(30)]
    async with mcp_session(build_app()) as session:
        _, labelled, _ = await render(
            session,
            {"labels": at_cap, "series": [{"values": [1.0] * 30}], "format": "svg"},
        )
        _, over, _ = await render(
            session,
            {"labels": at_cap + ["cat30"], "series": [{"values": [1.0] * 31}], "format": "svg"},
        )

    assert "cat29" in labelled.decode("utf-8")
    assert "cat29" not in over.decode("utf-8")


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_a_series_of_nan_or_infinity_is_refused_not_silently_blank(bad: str) -> None:
    # Both are JSON literals Python's parser accepts and pydantic will happily
    # bind to a float, and matplotlib draws NOTHING for either - so unguarded
    # they produce a BLANK picture whose summary still boasts of its points.
    #
    # Asserted at the model, not over the protocol: the client's JSON serializer
    # mangles a non-finite float on the way out, so a protocol-level call errors
    # either way and would pass with this constraint deleted.
    with pytest.raises(ValidationError):
        charts.ChartSpec.model_validate({"series": [{"values": [1.0, float(bad)]}]})

    # The finite case still validates - the constraint sits on the ELEMENT type,
    # and putting it on the list field instead rejects every series there is.
    assert charts.ChartSpec.model_validate({"series": [{"values": [1.0, 2.0]}]}).series[0].values


async def test_a_chart_with_no_numbers_is_refused_with_a_reason() -> None:
    async with mcp_session(build_app()) as session:
        content = await call_content(session, "render_chart", {"spec": {"series": []}})

    assert "series" in text_only(content)


async def test_the_chart_point_cap_is_enforced_at_the_boundary() -> None:
    # Literal boundaries, not `MAX_CHART_POINTS +/- 1`: a test that derives its
    # sizes from the constant simply re-proves whatever the constant says, and
    # can never fail when the cap is relaxed.
    at_cap = list(range(10_000))
    async with mcp_session(build_app()) as session:
        summary, data, _ = await render(session, {"type": "line", "series": [{"values": at_cap}]})
        assert "10000 points" in summary
        assert data.startswith(_PNG_SIGNATURE)

        over = await call_content(
            session,
            "render_chart",
            {"spec": {"type": "line", "series": [{"values": at_cap + [1]}]}},
        )

    message = text_only(over)
    assert "10001 points" in message
    assert str(charts.MAX_CHART_POINTS) in message


async def test_points_are_counted_across_every_series() -> None:
    # The cap is on the TOTAL: two half-sized series must not slip past it.
    half = [1.0] * 5_001
    async with mcp_session(build_app()) as session:
        over = await call_content(
            session,
            "render_chart",
            {"spec": {"series": [{"values": half}, {"values": half}]}},
        )

    assert "10002 points" in text_only(over)
