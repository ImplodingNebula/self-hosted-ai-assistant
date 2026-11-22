"""Chart rendering: numeric series in, one png or svg image out.

Two matplotlib rules shape this module and both are load-bearing on a server:

- The Agg backend is selected before anything can import pyplot, so no import
  order can leave the process reaching for a display that is not there.
- The figure is built through the object-oriented API (a ``Figure`` with its own
  canvas), never pyplot. pyplot keeps a global registry of figures and a notion
  of the "current" one, and this server renders requests on worker threads -
  through pyplot, two turns would be reaching for the same current figure. The
  OO API removes that shared state; it is not a claim that matplotlib as a whole
  is thread-safe, which upstream does not promise.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Annotated, Literal

import matplotlib

matplotlib.use("Agg")

from matplotlib.axes import Axes  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from pydantic import Field  # noqa: E402

from app.artifacts import CapExceeded, SpecError, SpecModel, check_input_size  # noqa: E402

ChartImageFormat = Literal["png", "svg"]

# Points across every series in one chart. Past this a chart is unreadable long
# before it is slow, and the render cost is linear in points.
MAX_CHART_POINTS = 10_000

# Wedges in one pie. Tighter than the point cap because a pie is the one chart
# whose cost is not linear: a MEASURED 10,000-wedge pie took ~53s to draw, which
# would spend the platform's whole tool deadline on a picture nobody can read.
MAX_PIE_WEDGES = 50

# Above this many x positions, category labels stop being readable and start
# being expensive, so the axis falls back to matplotlib's numeric ticks.
MAX_CATEGORY_TICKS = 30

# The stem a chart filename falls back to when nothing else names it.
FALLBACK_STEM = "chart"

_FIGURE_SIZE = (9.0, 5.0)
_FIGURE_DPI = 110
# Aliases a model reaches for. Mapping them beats refusing them: the caller
# wanted a picture of their numbers, and "column" is a bar chart.
_TYPE_ALIASES = {
    "column": "bar",
    "columns": "bar",
    "histogram": "bar",
    "hist": "bar",
    "bars": "bar",
    "area": "line",
    "lines": "line",
    "doughnut": "pie",
    "donut": "pie",
    "points": "scatter",
}
_KNOWN_TYPES = {"line", "bar", "pie", "scatter"}


# NaN and Infinity are both JSON literals that pydantic accepts for a float, and
# matplotlib draws nothing for either - so an unguarded series yields a BLANK
# picture whose summary line still boasts of its points. Rejecting them turns a
# silent blank into a correctable error. The constraint has to sit on the
# ELEMENT type: on the list field it would be read as a constraint on the list
# and every series would fail validation.
PlottableNumber = Annotated[float, Field(allow_inf_nan=False)]


class Series(SpecModel):
    """One named run of numbers."""

    name: str | None = Field(default=None, description="Legend label for this series.")
    values: list[PlottableNumber] = Field(default_factory=list, description="The numbers to plot.")


# These docstrings are published INTO the tool's input schema, so they address
# the model filling the form; the reasoning stays in comments. Only `series` is
# required - everything else has a working default, because a chart nobody
# labelled still beats a refusal.
class ChartSpec(SpecModel):
    """The chart to draw."""

    type: str = Field(default="bar", description="line | bar | pie | scatter")
    title: str | None = Field(default=None, description="Chart title.")
    x_label: str | None = Field(default=None, description="X axis label.")
    y_label: str | None = Field(default=None, description="Y axis label.")
    labels: list[str] | None = Field(
        default=None, description="Category labels for the x axis, or the pie wedges."
    )
    series: list[Series] = Field(default_factory=list, description="One entry per plotted run.")
    format: ChartImageFormat = Field(default="png", description="Image format: png or svg.")
    filename: str | None = Field(default=None, description="Preferred file name, without a path.")


@dataclass(frozen=True)
class RenderedChart:
    """Rendered image bytes plus the shape note that goes in the summary line."""

    data: bytes
    detail: str
    chart_type: str


def render_chart(spec: ChartSpec) -> RenderedChart:
    """Render ``spec`` as a png or svg image.

    Args:
        spec: The chart request.

    Returns:
        The image's bytes, a short description of its shape, and the chart type
        actually drawn (which differs from the request when an unknown type was
        normalized).

    Raises:
        SpecError: There are no numbers to plot, or the data cannot be drawn as
            the chart asked for (signed values in a pie).
        CapExceeded: The data exceeds the point or pie-wedge limit.
    """
    # Labels and series names are strings a caller controls, and matplotlib lays
    # every one of them out; the same total-text bound applies here as to a
    # document, and for the same reason - before any of that work starts.
    check_input_size(spec)
    series = [entry for entry in spec.series if entry.values]
    if not series:
        raise SpecError("A chart needs `series` - at least one entry with a non-empty `values`.")
    total_points = sum(len(entry.values) for entry in series)
    if total_points > MAX_CHART_POINTS:
        raise CapExceeded(
            f"{total_points} points exceeds the limit of {MAX_CHART_POINTS} per chart. "
            "Aggregate the data or split it across several charts."
        )

    chart_type = _normalize_type(spec.type)
    figure = Figure(figsize=_FIGURE_SIZE, dpi=_FIGURE_DPI, layout="constrained")
    # Attaching the canvas explicitly is what makes savefig work without pyplot.
    FigureCanvasAgg(figure)
    axes = figure.subplots()

    if chart_type == "pie":
        _draw_pie(axes, spec, series[0])
    else:
        _draw_xy(axes, spec, series, chart_type)

    if spec.title:
        axes.set_title(spec.title)

    buffer = io.BytesIO()
    figure.savefig(buffer, format=spec.format)
    detail = f"{_plural(len(series), 'series')}, {_plural(total_points, 'point')}"
    return RenderedChart(buffer.getvalue(), detail, chart_type)


def _plural(count: int, noun: str) -> str:
    """Count plus noun; "series" is already its own plural."""
    if noun == "series":
        return f"{count} series"
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _normalize_type(requested: str) -> str:
    """Map a requested chart type onto one this module draws.

    An unknown type becomes a bar chart rather than an error: bars read
    correctly for labelled categories, which is what an unrecognized request
    almost always is, and the summary line names what was actually drawn.
    """
    kind = (requested or "").strip().lower()
    kind = _TYPE_ALIASES.get(kind, kind)
    return kind if kind in _KNOWN_TYPES else "bar"


def _categories(spec: ChartSpec, length: int) -> list[str]:
    """X axis category labels, padded or trimmed to the data's length."""
    labels = [str(label) for label in (spec.labels or [])]
    if len(labels) >= length:
        return labels[:length]
    return labels + [str(index + 1) for index in range(len(labels), length)]


def _draw_pie(axes: Axes, spec: ChartSpec, series: Series) -> None:
    """Draw the first series as pie wedges; negatives are refused, not silently absolute."""
    if len(series.values) > MAX_PIE_WEDGES:
        raise CapExceeded(
            f"{len(series.values)} wedges exceeds the limit of {MAX_PIE_WEDGES} per pie chart. "
            "Aggregate the small categories, or use a bar chart."
        )
    if any(value < 0 for value in series.values):
        raise SpecError("A pie chart needs non-negative values. Use a bar chart for signed data.")
    if not any(value > 0 for value in series.values):
        raise SpecError("A pie chart needs at least one value above zero.")
    axes.pie(series.values, labels=_categories(spec, len(series.values)), autopct="%1.1f%%")
    axes.set_aspect("equal")


def _draw_xy(axes: Axes, spec: ChartSpec, series: list[Series], chart_type: str) -> None:
    """Draw line, bar or scatter series against shared x positions."""
    longest = max(len(entry.values) for entry in series)
    positions = range(longest)

    if chart_type == "bar":
        # Grouped bars: each series gets a slice of the slot, so a multi-series
        # bar chart does not overpaint itself.
        width = 0.8 / len(series)
        for index, entry in enumerate(series):
            offsets = [
                position + (index - (len(series) - 1) / 2) * width
                for position in positions[: len(entry.values)]
            ]
            axes.bar(offsets, entry.values, width=width, label=entry.name)
    else:
        draw = axes.plot if chart_type == "line" else axes.scatter
        for entry in series:
            draw(list(positions[: len(entry.values)]), entry.values, label=entry.name)

    if spec.labels and longest <= MAX_CATEGORY_TICKS:
        # Category ticks only while they can still be read. Past the ceiling
        # matplotlib's own numeric ticks are both legible and far cheaper - one
        # tick per data point is a wall of ink and the slowest part of the draw.
        categories = _categories(spec, longest)
        rotated = _labels_are_long(categories)
        axes.set_xticks(list(positions))
        axes.set_xticklabels(
            categories, rotation=45 if rotated else 0, ha="right" if rotated else "center"
        )
    if spec.x_label:
        axes.set_xlabel(spec.x_label)
    if spec.y_label:
        axes.set_ylabel(spec.y_label)
    if any(entry.name for entry in series):
        axes.legend()


def _labels_are_long(categories: list[str]) -> bool:
    """Whether the x labels need rotating to stay readable."""
    return sum(len(label) for label in categories) > 60
