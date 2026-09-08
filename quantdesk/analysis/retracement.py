"""Percentage retracements, Fibonacci levels and speed resistance lines.

Murphy's exact numbers, which is the whole point of this module:

* **50%** is the best-known retracement.
* **One-third (33%) is the minimum** and **two-thirds (66%) the maximum** for a
  correction within an intact trend. These come from Dow Theory.
* Elliott/Fibonacci practitioners use **38%** and **62%**.
* Murphy states his own preference explicitly: combine both, giving a minimum
  retracement *zone* of **33-38%** and a maximum zone of **62-66%**. Some round
  further to 40-60%.
* Crucially: **beyond the two-thirds point the odds favour a full reversal**
  rather than a retracement, and price then usually retraces the entire prior
  move. That turns the 66% level into a decision boundary, not just another line,
  and it is what :func:`classify_retracement` encodes.
* For trade timing he cites the **40-60%** pullback zone as the entry area.
* Gann divided the trend into eighths, attaching special weight to 3/8, 4/8 and
  5/8, plus the thirds.

Speed resistance lines (Edson Gould) measure the *rate* of a trend rather than
its retracement depth: drop a vertical from the extreme to the trend's origin,
divide into thirds, and draw lines from the origin through those points.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quantdesk.analysis.swings import SwingLeg, SwingPoint

# Murphy's retracement ratios. Dow thirds plus the Fibonacci pair.
DOW_RATIOS: tuple[float, ...] = (1 / 3, 0.50, 2 / 3)
FIBONACCI_RATIOS: tuple[float, ...] = (0.382, 0.50, 0.618)
GANN_EIGHTHS: tuple[float, ...] = tuple(i / 8 for i in range(1, 8))

#: Murphy's stated preference: blended minimum and maximum retracement zones.
MIN_ZONE: tuple[float, float] = (1 / 3, 0.382)
MAX_ZONE: tuple[float, float] = (0.618, 2 / 3)
#: His trade-timing pullback zone.
ENTRY_ZONE: tuple[float, float] = (0.40, 0.60)
#: Beyond this, a reversal is more likely than a continuation.
REVERSAL_THRESHOLD: float = 2 / 3


class RetracementZone(str, Enum):
    SHALLOW = "shallow"
    """Under the 33% minimum. Correction may not be finished."""
    MINIMUM = "minimum"
    """In the 33-38% blended minimum zone."""
    ENTRY = "entry"
    """In the 40-60% zone Murphy names for timing entries."""
    MAXIMUM = "maximum"
    """In the 62-66% zone. Last defence of the trend."""
    BROKEN = "broken"
    """Beyond 66%: odds favour full reversal, not continuation."""


@dataclass(slots=True)
class RetracementLevel:
    ratio: float
    price: float
    label: str

    def distance_pct(self, price: float) -> float:
        if price <= 0:
            return 0.0
        return (self.price - price) / price


@dataclass(slots=True)
class RetracementMap:
    """Retracement levels for one completed swing leg."""

    start_price: float
    end_price: float
    start_index: int
    end_index: int
    levels: list[RetracementLevel]

    @property
    def is_up_leg(self) -> bool:
        return self.end_price > self.start_price

    @property
    def height(self) -> float:
        return abs(self.end_price - self.start_price)

    def retracement_of(self, price: float) -> float:
        """Fraction of the leg given back at ``price``, unsigned.

        Kept for callers that only care about magnitude. Prefer
        :meth:`signed_retracement_of` when the answer will drive a decision: the
        absolute value here cannot tell a 20% pullback from a 20% *extension* past
        the leg's end, and those argue in opposite directions.
        """
        if self.height <= 1e-12:
            return 0.0
        return abs(self.end_price - price) / self.height

    def signed_retracement_of(self, price: float) -> float:
        """Position relative to the leg, signed so the regimes are distinguishable.

        * below 0 - price has *extended* beyond the leg's end; the move is still
          running and there is nothing to retrace yet
        * 0 to 1 - a genuine retracement, and Murphy's 40-60% entry zone lives here
        * above 1 - the whole leg has been given back, so this is a reversal rather
          than a correction
        """
        if self.height <= 1e-12:
            return 0.0
        if self.is_up_leg:
            return (self.end_price - price) / self.height
        return (price - self.end_price) / self.height

    def level(self, ratio: float) -> float:
        """Price at a given retracement ratio."""
        if self.is_up_leg:
            return self.end_price - self.height * ratio
        return self.end_price + self.height * ratio

    def zone_bounds(self, zone: tuple[float, float]) -> tuple[float, float]:
        a, b = self.level(zone[0]), self.level(zone[1])
        return (min(a, b), max(a, b))

    def nearest_level(self, price: float) -> RetracementLevel | None:
        if not self.levels:
            return None
        return min(self.levels, key=lambda lv: abs(lv.price - price))


def build_retracements(
    start: SwingPoint | float,
    end: SwingPoint | float,
    include_gann: bool = False,
) -> RetracementMap:
    """Build the retracement map for a leg running from ``start`` to ``end``."""
    s_px = start.price if isinstance(start, SwingPoint) else float(start)
    e_px = end.price if isinstance(end, SwingPoint) else float(end)
    s_ix = start.index if isinstance(start, SwingPoint) else 0
    e_ix = end.index if isinstance(end, SwingPoint) else 0

    height = abs(e_px - s_px)
    up = e_px > s_px

    ratios: list[tuple[float, str]] = [
        (1 / 3, "33% (Dow min)"),
        (0.382, "38.2% (Fib)"),
        (0.50, "50%"),
        (0.618, "61.8% (Fib)"),
        (2 / 3, "66% (Dow max)"),
    ]
    if include_gann:
        existing = {round(r, 4) for r, _ in ratios}
        for r in GANN_EIGHTHS:
            if round(r, 4) not in existing:
                ratios.append((r, f"{r * 8:.0f}/8 (Gann)"))
        ratios.sort(key=lambda t: t[0])

    levels = [
        RetracementLevel(
            ratio=r,
            price=(e_px - height * r) if up else (e_px + height * r),
            label=label,
        )
        for r, label in ratios
    ]
    return RetracementMap(
        start_price=s_px,
        end_price=e_px,
        start_index=s_ix,
        end_index=e_ix,
        levels=levels,
    )


def classify_retracement(fraction: float) -> RetracementZone:
    """Map a retracement fraction onto Murphy's zones.

    The BROKEN boundary at two-thirds is the meaningful one: past it he says the
    odds shift from "correction within a trend" to "trend reversal", so a
    pullback-buying agent must stand down rather than treat a deeper discount as
    a better entry.
    """
    if fraction > REVERSAL_THRESHOLD:
        return RetracementZone.BROKEN
    if fraction >= MAX_ZONE[0]:
        return RetracementZone.MAXIMUM
    if fraction >= ENTRY_ZONE[0]:
        return RetracementZone.ENTRY
    if fraction >= MIN_ZONE[0]:
        return RetracementZone.MINIMUM
    return RetracementZone.SHALLOW


def in_entry_zone(fraction: float) -> bool:
    """True inside the 40-60% pullback window Murphy names for entries."""
    return ENTRY_ZONE[0] <= fraction <= ENTRY_ZONE[1]


# ------------------------------------------------------- speed resistance lines


@dataclass(slots=True)
class SpeedLines:
    """Edson Gould's speed resistance lines.

    Constructed from the trend's origin and its extreme: the vertical span is cut
    into thirds and lines are drawn from the origin through the 1/3 and 2/3
    points. Unlike other trendlines these deliberately cut through price action.

    Murphy's cascade: in a correcting uptrend the decline should hold at the 2/3
    line; failing that, at the 1/3 line; and if that breaks too, price likely
    returns to the trend's origin. Like all trendlines, they reverse roles once
    broken.

    A new extreme invalidates the lines and they must be redrawn, which
    :meth:`needs_redraw` reports.
    """

    origin_index: int
    origin_price: float
    extreme_index: int
    extreme_price: float

    @property
    def is_up(self) -> bool:
        return self.extreme_price > self.origin_price

    @property
    def height(self) -> float:
        return abs(self.extreme_price - self.origin_price)

    def _line_price(self, fraction: float, index: int) -> float:
        """Price of the speedline at ``fraction`` of the span, at ``index``."""
        dx = self.extreme_index - self.origin_index
        if dx <= 0:
            return self.origin_price
        # The line passes through the point at `fraction` of the vertical span,
        # measured from the origin, at the extreme's bar index.
        if self.is_up:
            target = self.origin_price + self.height * fraction
        else:
            target = self.origin_price - self.height * fraction
        slope = (target - self.origin_price) / dx
        return self.origin_price + slope * (index - self.origin_index)

    def upper(self, index: int) -> float:
        """The 2/3 speedline: first support in a correcting uptrend."""
        return self._line_price(2 / 3, index)

    def lower(self, index: int) -> float:
        """The 1/3 speedline: second line of defence."""
        return self._line_price(1 / 3, index)

    def zone(self, index: int, price: float) -> str:
        """Which speedline band ``price`` currently occupies."""
        up, lo = self.upper(index), self.lower(index)
        if self.is_up:
            if price >= up:
                return "above_2/3"
            if price >= lo:
                return "between"
            return "below_1/3"
        if price <= up:
            return "below_2/3"
        if price <= lo:
            return "between"
        return "above_1/3"

    def needs_redraw(self, price: float) -> bool:
        """A new extreme means the lines must be rebuilt."""
        return (
            price > self.extreme_price if self.is_up else price < self.extreme_price
        )


def build_speedlines(origin: SwingPoint, extreme: SwingPoint) -> SpeedLines:
    return SpeedLines(
        origin_index=origin.index,
        origin_price=origin.price,
        extreme_index=extreme.index,
        extreme_price=extreme.price,
    )


def fibonacci_projection(leg_height: float, base_price: float, up: bool) -> dict[str, float]:
    """Fibonacci extension targets beyond a completed leg.

    Murphy cites 1.618 in the Elliott chapter as the standard extension ratio,
    alongside equality (1.0) between impulse legs.
    """
    ratios = {"1.000": 1.0, "1.382": 1.382, "1.618": 1.618, "2.618": 2.618}
    sign = 1.0 if up else -1.0
    return {k: base_price + sign * leg_height * v for k, v in ratios.items()}
