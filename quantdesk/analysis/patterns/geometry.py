"""Line geometry for the boundary-based patterns.

Triangles, wedges, flags, channels and necklines are all *pairs of lines fitted
through pivots*, so the fitting, convergence and quality maths lives here once
instead of five times.

Murphy draws these by eye and by definition through two touches per line. That is
kept - :func:`line_through` is the two-point construction he specifies - but two
extra things are needed to do it mechanically:

* **A fit quality measure.** A human rejects a "triangle" whose pivots do not
  really sit on the lines. :func:`fit_error` gives the mean deviation in ATR
  units so a detector can apply the same judgement.
* **Convergence in scale-free terms.** Whether two lines converge "meaningfully"
  cannot be a raw slope comparison, because slope depends on the instrument's
  price and volatility. Everything here is expressed per-ATR-per-bar.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantdesk.analysis.swings import SwingPoint


@dataclass(slots=True)
class Line:
    """A straight line in (bar index, price) space."""

    x1: int
    y1: float
    x2: int
    y2: float

    @property
    def slope(self) -> float:
        dx = self.x2 - self.x1
        if dx == 0:
            return 0.0
        return (self.y2 - self.y1) / dx

    def value_at(self, index: int) -> float:
        return self.y1 + self.slope * (index - self.x1)

    def distance(self, index: int, price: float) -> float:
        """Signed gap from line to price. Positive means price is above."""
        return price - self.value_at(index)

    def slope_atr(self, atr: float) -> float:
        """Slope in ATR per bar, so it means the same across instruments."""
        if atr <= 0:
            return 0.0
        return self.slope / atr

    def is_flat(self, atr: float, tolerance: float = 0.05) -> bool:
        """Flat enough to count as horizontal.

        Needed for the ascending triangle's flat top and the descending
        triangle's flat bottom, which are what distinguish them from the
        symmetrical case.
        """
        return abs(self.slope_atr(atr)) <= tolerance


def line_through(a: SwingPoint, b: SwingPoint) -> Line:
    """Murphy's two-point construction."""
    return Line(a.index, a.price, b.index, b.price)


def best_fit_line(points: list[SwingPoint]) -> Line | None:
    """Least-squares line through three or more pivots.

    Used only when a boundary has more touches than the two needed to draw it.
    With exactly two points this is identical to :func:`line_through`.
    """
    n = len(points)
    if n < 2:
        return None
    if n == 2:
        return line_through(points[0], points[1])

    mean_x = sum(p.index for p in points) / n
    mean_y = sum(p.price for p in points) / n
    num = sum((p.index - mean_x) * (p.price - mean_y) for p in points)
    den = sum((p.index - mean_x) ** 2 for p in points)
    if den <= 0:
        return None
    slope = num / den
    intercept = mean_y - slope * mean_x
    x1 = points[0].index
    x2 = points[-1].index
    return Line(x1, intercept + slope * x1, x2, intercept + slope * x2)


def fit_error(line: Line, points: list[SwingPoint], atr: float) -> float:
    """Mean absolute deviation of pivots from a line, in ATR units.

    A detector uses this to refuse shapes that only loosely resemble the
    formation. Returns ``inf`` when ATR is unusable, which fails closed.
    """
    if not points or atr <= 0:
        return float("inf")
    total = sum(abs(line.distance(p.index, p.price)) for p in points)
    return total / len(points) / atr


def intersection(a: Line, b: Line) -> tuple[float, float] | None:
    """Where two lines cross, as (bar index, price).

    This is the triangle's apex, and Murphy uses it as the pattern's deadline:
    the breakout is expected between two-thirds and three-quarters of the way
    from base to apex. Returns None for parallel lines.
    """
    ds = a.slope - b.slope
    if abs(ds) < 1e-15:
        return None
    # a.y1 + a.slope*(x - a.x1) == b.y1 + b.slope*(x - b.x1)
    x = (b.y1 - b.slope * b.x1 - a.y1 + a.slope * a.x1) / ds
    return x, a.value_at(int(round(x)))


def converging(upper: Line, lower: Line, atr: float, min_rate: float = 0.01) -> bool:
    """Do the boundaries close on each other fast enough to matter?

    Rate is the shrink in gap per bar, in ATR. A pair of near-parallel lines
    technically meets somewhere, but a rectangle is not a triangle, so a minimum
    convergence rate is required.
    """
    if atr <= 0:
        return False
    rate = lower.slope_atr(atr) - upper.slope_atr(atr)
    return rate >= min_rate


def diverging(upper: Line, lower: Line, atr: float, min_rate: float = 0.01) -> bool:
    """The broadening formation: boundaries pulling apart."""
    if atr <= 0:
        return False
    rate = upper.slope_atr(atr) - lower.slope_atr(atr)
    return rate >= min_rate


def gap_at(upper: Line, lower: Line, index: int) -> float:
    """Vertical distance between the boundaries at a bar."""
    return upper.value_at(index) - lower.value_at(index)


def apex_progress(
    upper: Line, lower: Line, base_index: int, current_index: int
) -> float:
    """How far through the base-to-apex span the current bar sits.

    Murphy's timing rule reads directly off this: expect the breakout between
    0.67 and 0.75, and past 0.75 the triangle is losing its potency. Returns
    ``inf`` when the lines do not converge, so a non-triangle never looks timely.
    """
    cross = intersection(upper, lower)
    if cross is None:
        return float("inf")
    apex_x = cross[0]
    span = apex_x - base_index
    if span <= 0:
        return float("inf")
    return (current_index - base_index) / span


#: Murphy: "prices should break out ... somewhere between two-thirds to
#: three-quarters of the horizontal width of the triangle."
APEX_WINDOW_START = 2.0 / 3.0
APEX_WINDOW_END = 0.75


def slant(line_a: Line, line_b: Line, atr: float) -> float:
    """Average slope of two boundaries, in ATR per bar.

    This is what separates a wedge from a symmetrical triangle. Both converge,
    but Murphy notes the wedge has "a noticeable slant" - and crucially it slants
    *against* the prevailing trend, which is why a falling wedge is bullish.
    """
    if atr <= 0:
        return 0.0
    return (line_a.slope_atr(atr) + line_b.slope_atr(atr)) / 2.0
