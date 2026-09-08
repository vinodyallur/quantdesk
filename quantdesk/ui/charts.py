"""Charts drawn in text.

No plotting dependency. A candle is a vertical run of block characters and a price axis is
right-aligned numbers, which is all a trading chart needs in a terminal and avoids pulling
a rendering library into a process that also has to keep trading.

Two properties worth stating because they are easy to get wrong and hard to notice:

**A candle body must never disappear.** A doji has an open equal to its close, and naive
scaling rounds that to zero rows and draws nothing, so the bar silently vanishes from the
chart. Every body is given at least one row.

**The axis is computed from what is drawn, not from the whole series.** Scaling to the
full history and then showing the last sixty bars produces a chart where price appears
pinned to one edge, which reads as a market that has stopped moving.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text

from quantdesk.core.types import Bar

UP = "#33dd77"
DOWN = "#ff5566"
WICK_UP = "#1f8b4c"
WICK_DOWN = "#a63340"
AXIS = "#667788"
GRID = "#1d2733"
MARK = "bold #ffb000"

#: Body, upper wick, lower wick, and the sliver used when a candle spans one row.
_BODY = "\u2588"
_WICK = "\u2502"
_DOJI = "\u2500"


@dataclass(slots=True)
class Scale:
    """Linear price-to-row mapping for one chart."""

    low: float
    high: float
    rows: int

    @property
    def span(self) -> float:
        return max(1e-12, self.high - self.low)

    def row_of(self, price: float) -> int:
        """Row index, 0 at the top. Clamped so an outlier cannot draw off-canvas."""
        fraction = (price - self.low) / self.span
        row = int(round((1.0 - fraction) * (self.rows - 1)))
        return max(0, min(self.rows - 1, row))


def candlestick(
    bars: list[Bar],
    width: int = 100,
    height: int = 18,
    forming: Bar | None = None,
) -> Text:
    """Render the most recent bars as candles, newest at the right.

    ``forming`` is the in-progress bar. It is drawn in amber so it reads as provisional,
    because a chart that shows it identically to a closed bar invites acting on it.
    """
    axis_width = 11
    plot_width = max(10, width - axis_width)
    series = list(bars[-plot_width:])
    if forming is not None:
        series = (series + [forming])[-plot_width:]
    if not series:
        return Text("no bars yet", style=AXIS)

    scale = Scale(
        low=min(b.low for b in series),
        high=max(b.high for b in series),
        rows=max(6, height),
    )

    # One column per bar, each a list of (char, style) by row.
    columns: list[list[tuple[str, str] | None]] = []
    for i, bar in enumerate(series):
        provisional = forming is not None and i == len(series) - 1
        rising = bar.close >= bar.open
        body_style = MARK if provisional else (UP if rising else DOWN)
        wick_style = MARK if provisional else (WICK_UP if rising else WICK_DOWN)

        top_row = scale.row_of(bar.high)
        bottom_row = scale.row_of(bar.low)
        body_top = scale.row_of(max(bar.open, bar.close))
        body_bottom = scale.row_of(min(bar.open, bar.close))
        # A doji collapses to a single row; give it one so the bar stays visible.
        if body_top == body_bottom:
            char_for_body = _DOJI
        else:
            char_for_body = _BODY

        column: list[tuple[str, str] | None] = [None] * scale.rows
        for row in range(top_row, bottom_row + 1):
            column[row] = (_WICK, wick_style)
        for row in range(body_top, body_bottom + 1):
            column[row] = (char_for_body, body_style)
        columns.append(column)

    out = Text()
    for row in range(scale.rows):
        price = scale.high - (row / max(1, scale.rows - 1)) * scale.span
        out.append(f"{price:>10,.2f} ", style=AXIS)
        for column in columns:
            cell = column[row]
            if cell is None:
                out.append(" ")
            else:
                out.append(cell[0], style=cell[1])
        out.append("\n")

    first, last = series[0], series[-1]
    move = (last.close / first.close - 1.0) if first.close else 0.0
    out.append(f"{'':>10} ", style=AXIS)
    out.append(f"{first.ts:%m-%d %H:%M}", style=AXIS)
    out.append(" -> ", style=AXIS)
    out.append(f"{last.ts:%m-%d %H:%M}", style=AXIS)
    out.append(f"   {len(series)} bars   ", style=AXIS)
    out.append(f"{move:+.2%}", style=UP if move >= 0 else DOWN)
    if forming is not None:
        out.append("   last candle is still forming", style=MARK)
    return out


def line_chart(
    values: list[float],
    width: int = 100,
    height: int = 8,
    label: str = "",
    baseline: float | None = None,
) -> Text:
    """Render a series as a filled area, newest at the right.

    Used for the equity curve. ``baseline`` draws a reference row - starting capital - so
    the chart answers "am I up or down" without reading the axis.
    """
    axis_width = 11
    plot_width = max(10, width - axis_width)
    series = _resample(values, plot_width)
    if not series:
        return Text("no data yet", style=AXIS)

    low, high = min(series), max(series)
    if baseline is not None:
        low, high = min(low, baseline), max(high, baseline)
    scale = Scale(low=low, high=high, rows=max(4, height))
    base_row = scale.row_of(baseline) if baseline is not None else None

    out = Text()
    for row in range(scale.rows):
        price = scale.high - (row / max(1, scale.rows - 1)) * scale.span
        out.append(f"{price:>10,.2f} ", style=AXIS)
        for value in series:
            value_row = scale.row_of(value)
            above = baseline is None or value >= baseline
            if base_row is None:
                out.append(_BODY if value_row == row else " ", style=UP)
                continue
            # Fill between the value and the baseline, not down to the canvas floor.
            # Filling to the floor paints an entire winning session in the losing colour,
            # because every row below the line then counts as "below".
            top, bottom = min(value_row, base_row), max(value_row, base_row)
            if row == base_row and value_row == base_row:
                out.append(_DOJI, style=GRID)
            elif top <= row <= bottom:
                out.append(_BODY, style=UP if above else DOWN)
            elif row == base_row:
                out.append(_DOJI, style=GRID)
            else:
                out.append(" ")
        out.append("\n")

    if label:
        out.append(f"{'':>10} ", style=AXIS)
        out.append(label + "\n", style=AXIS)
    return out


def _resample(values: list[float], width: int) -> list[float]:
    """Fit a series to a column count by averaging buckets.

    Averaging rather than taking every nth point: sampling drops spikes entirely, and on
    an equity curve the spike is usually the thing worth seeing.
    """
    if not values:
        return []
    if len(values) <= width:
        return list(values)
    out: list[float] = []
    step = len(values) / width
    for i in range(width):
        start = int(i * step)
        end = max(start + 1, int((i + 1) * step))
        bucket = values[start:end]
        out.append(sum(bucket) / len(bucket))
    return out
