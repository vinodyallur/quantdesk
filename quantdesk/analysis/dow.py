"""Dow Theory trend classification.

Murphy's definition of trend, which the whole book builds on:

* **Uptrend** - successively higher peaks *and* higher troughs.
* **Downtrend** - successively lower peaks *and* lower troughs.
* **Sideways** - horizontal peaks and troughs, which he stresses is a third
  distinct state occupying at least a third of the time, not a rounding error
  between up and down. Trend-following tools fail here, and he is explicit that
  standing aside is usually the correct response.

Both conditions must hold. That is the part naive implementations drop, and it
matters: higher peaks with flat troughs is a weakening advance, not an uptrend,
and treating the two identically is how a trend filter ends up confirming a top.

This module also encodes Murphy's early-warning structure, which is where the
real value is. He describes specific pre-reversal tells:

* Failure to exceed the previous peak in an uptrend is the first warning.
* A corrective dip reaching all the way back to the previous low suggests the
  uptrend is at least flattening.
* Violation of that support level makes a reversal likely.

Those states are reported explicitly rather than being collapsed into a single
direction, so an agent can size down on a warning instead of waiting for the
reversal to be confirmed and complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quantdesk.analysis.swings import SwingKind, SwingPoint


class TrendDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"

    @property
    def sign(self) -> int:
        if self is TrendDirection.UP:
            return 1
        if self is TrendDirection.DOWN:
            return -1
        return 0


class TrendDegree(str, Enum):
    """Murphy's three classifications of trend.

    He gives bar counts in calendar terms (major over a year for stocks, over six
    months for futures; intermediate three weeks to months; near term under two
    to three weeks). Since this desk may run on any bar size, the degree is
    derived from how many bars the structure spans, calibrated by the caller.
    """

    MAJOR = "major"
    INTERMEDIATE = "intermediate"
    NEAR_TERM = "near_term"


class TrendHealth(str, Enum):
    """Where the current structure sits between intact and reversed."""

    INTACT = "intact"
    """Peaks and troughs both still advancing in the trend direction."""
    PEAK_FAILURE = "peak_failure"
    """Failed to exceed the prior extreme: Murphy's first warning."""
    SUPPORT_TESTED = "support_tested"
    """Correction reached back to the prior trough/peak: trend flattening."""
    SUPPORT_BROKEN = "support_broken"
    """Prior trough/peak violated: reversal likely."""
    REVERSED = "reversed"
    """Structure now satisfies the opposite trend definition."""


@dataclass(slots=True)
class TrendState:
    """Classified trend structure at a point in time."""

    direction: TrendDirection
    health: TrendHealth
    peaks_rising: bool | None
    troughs_rising: bool | None
    #: Bars spanned by the pivots used for this classification.
    span_bars: int = 0
    #: Number of confirmed pivots the classification is based on.
    pivots_used: int = 0
    last_peak: float = 0.0
    last_trough: float = 0.0
    rationale: str = ""

    @property
    def is_trending(self) -> bool:
        return self.direction in (TrendDirection.UP, TrendDirection.DOWN)

    @property
    def confidence(self) -> float:
        """How much to trust this classification, in [0, 1].

        Degrades for warning states rather than flipping to the opposite trend,
        which is the behaviour Murphy's early-warning sequence implies: a peak
        failure means take the uptrend less seriously, not go short.
        """
        if self.direction is TrendDirection.UNKNOWN:
            return 0.0
        base = {
            TrendHealth.INTACT: 1.0,
            TrendHealth.PEAK_FAILURE: 0.6,
            TrendHealth.SUPPORT_TESTED: 0.45,
            TrendHealth.SUPPORT_BROKEN: 0.2,
            TrendHealth.REVERSED: 0.1,
        }[self.health]
        if self.direction is TrendDirection.SIDEWAYS:
            base *= 0.5
        # More confirmed pivots means a better established structure.
        pivot_factor = min(1.0, self.pivots_used / 4.0)
        return base * (0.5 + 0.5 * pivot_factor)


def classify_trend(
    points: list[SwingPoint],
    last_price: float | None = None,
    flat_tolerance: float = 0.0025,
) -> TrendState:
    """Classify trend from confirmed swing pivots.

    Parameters
    ----------
    points:
        Confirmed pivots, oldest first. Only the most recent two peaks and two
        troughs are needed, which is the minimum Murphy's definition requires.
    last_price:
        Current price, used to detect the early-warning states against the most
        recent structure. Optional; without it only the pivot sequence is judged.
    flat_tolerance:
        Fractional band within which two pivots count as level rather than
        rising or falling. Murphy says "horizontal" without quantifying it, so
        this is a chosen parameter.
    """
    peaks = [p for p in points if p.kind is SwingKind.PEAK]
    troughs = [p for p in points if p.kind is SwingKind.TROUGH]

    if len(peaks) < 2 or len(troughs) < 2:
        return TrendState(
            direction=TrendDirection.UNKNOWN,
            health=TrendHealth.INTACT,
            peaks_rising=None,
            troughs_rising=None,
            pivots_used=len(points),
            rationale="need at least two peaks and two troughs",
        )

    p_prev, p_last = peaks[-2], peaks[-1]
    t_prev, t_last = troughs[-2], troughs[-1]

    peaks_rising = _compare(p_prev.price, p_last.price, flat_tolerance)
    troughs_rising = _compare(t_prev.price, t_last.price, flat_tolerance)

    # Murphy's definition: BOTH sequences must agree for a trend to exist.
    if peaks_rising is True and troughs_rising is True:
        direction = TrendDirection.UP
    elif peaks_rising is False and troughs_rising is False:
        direction = TrendDirection.DOWN
    else:
        direction = TrendDirection.SIDEWAYS

    health, note = _assess_health(
        direction, p_last, t_last, peaks_rising, troughs_rising, last_price
    )

    span = max(p_last.index, t_last.index) - min(p_prev.index, t_prev.index)
    desc = {
        TrendDirection.UP: "higher peaks and higher troughs",
        TrendDirection.DOWN: "lower peaks and lower troughs",
        TrendDirection.SIDEWAYS: "peaks and troughs disagree",
        TrendDirection.UNKNOWN: "insufficient structure",
    }[direction]

    return TrendState(
        direction=direction,
        health=health,
        peaks_rising=peaks_rising,
        troughs_rising=troughs_rising,
        span_bars=span,
        pivots_used=len(points),
        last_peak=p_last.price,
        last_trough=t_last.price,
        rationale=f"{desc}{('; ' + note) if note else ''}",
    )


def _compare(prev: float, latest: float, tol: float) -> bool | None:
    """True if rising, False if falling, None if level within tolerance."""
    if prev <= 0:
        return None
    change = (latest - prev) / prev
    if abs(change) <= tol:
        return None
    return change > 0


def _assess_health(
    direction: TrendDirection,
    last_peak: SwingPoint,
    last_trough: SwingPoint,
    peaks_rising: bool | None,
    troughs_rising: bool | None,
    last_price: float | None,
) -> tuple[TrendHealth, str]:
    """Locate the structure on Murphy's warning sequence."""
    if direction is TrendDirection.UP:
        if troughs_rising is False:
            return TrendHealth.REVERSED, "troughs now falling"
        if last_price is not None and last_price < last_trough.price:
            return (
                TrendHealth.SUPPORT_BROKEN,
                f"price {last_price:.4g} below prior trough {last_trough.price:.4g}",
            )
        if peaks_rising is None:
            return TrendHealth.PEAK_FAILURE, "failed to exceed prior peak"
        return TrendHealth.INTACT, ""

    if direction is TrendDirection.DOWN:
        if troughs_rising is True:
            return TrendHealth.REVERSED, "troughs now rising"
        if last_price is not None and last_price > last_peak.price:
            return (
                TrendHealth.SUPPORT_BROKEN,
                f"price {last_price:.4g} above prior peak {last_peak.price:.4g}",
            )
        if peaks_rising is None:
            return TrendHealth.PEAK_FAILURE, "failed to undercut prior peak"
        return TrendHealth.INTACT, ""

    # Sideways: flag which side of the range price is pressing against.
    if last_price is not None:
        if last_price > last_peak.price:
            return TrendHealth.SUPPORT_BROKEN, "breaking above range"
        if last_price < last_trough.price:
            return TrendHealth.SUPPORT_BROKEN, "breaking below range"
    return TrendHealth.INTACT, "range bound"


def classify_degree(
    span_bars: int,
    bars_per_day: float,
    major_days: float = 180.0,
    intermediate_days: float = 21.0,
) -> TrendDegree:
    """Map a structure's bar span onto Murphy's three trend degrees.

    Defaults follow his futures figures: major beyond roughly six months,
    intermediate from about three weeks, near term below that.
    """
    if bars_per_day <= 0:
        return TrendDegree.NEAR_TERM
    days = span_bars / bars_per_day
    if days >= major_days:
        return TrendDegree.MAJOR
    if days >= intermediate_days:
        return TrendDegree.INTERMEDIATE
    return TrendDegree.NEAR_TERM
