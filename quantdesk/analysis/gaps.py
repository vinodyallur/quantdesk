"""Price gaps and single-bar reversal signals.

Murphy's three gap types, and what distinguishes them is *where in the move they
appear*, not how they look:

* **Breakaway** - at the completion of a price pattern, on heavy volume, starting
  a significant move. Usually *not* filled.
* **Runaway / measuring** - roughly midway through a move, on moderate volume.
  Its measuring use is the valuable part: double the distance travelled so far to
  estimate the remainder.
* **Exhaustion** - near the end of a move. Price closing back through it is the
  tell that the move is over.

He explicitly debunks "gaps are always filled": some should be, some should not.
An unfilled up gap acts as support, and a close back below an up gap is a sign of
weakness regardless of type.

**Island reversal** - an exhaustion gap one way followed by a breakaway gap the
other, leaving a few bars stranded. Signals a reversal of some magnitude.

Reversal days are also here since they are single-bar structures:

* **Top reversal** - a new high for the move, then a close below the prior close.
* **Bottom reversal** - a new low, then a close above the prior close.
* The wider the range and the heavier the volume, the more it means. An **outside
  bar** (engulfing the prior range) is not required but adds significance, and a
  high-volume bottom reversal is a **selling climax**.

Weekly and monthly reversals carry more weight than daily; this module works on
whatever bar size it is fed, so the caller decides the timeframe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from quantdesk.core.types import Bar


class GapType(str, Enum):
    COMMON = "common"
    BREAKAWAY = "breakaway"
    RUNAWAY = "runaway"
    EXHAUSTION = "exhaustion"


@dataclass(slots=True)
class Gap:
    """An unfilled area between one bar's range and the next."""

    index: int
    ts: datetime
    up: bool
    #: Edges of the empty area.
    low: float
    high: float
    gap_type: GapType = GapType.COMMON
    volume_ratio: float = 1.0
    """Bar volume relative to its recent average. Heavy volume implies breakaway."""
    filled: bool = False
    filled_at: int | None = None
    closed_through: bool = False
    """True once price CLOSED beyond the gap against its direction."""

    @property
    def size(self) -> float:
        return self.high - self.low

    def size_atr(self, atr: float) -> float:
        return self.size / atr if atr > 0 else 0.0

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0

    def acts_as(self) -> str:
        """Unfilled up gaps support the market; down gaps cap it."""
        return "support" if self.up else "resistance"

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high


class GapTracker:
    """Detects, classifies and follows the fate of price gaps.

    Parameters
    ----------
    min_atr:
        Minimum gap size in ATR to be worth recording. Murphy gives no threshold,
        so this is chosen; without it every rounding tick becomes a "gap".
    heavy_volume:
        Volume ratio above which a gap counts as heavy-volume, pushing
        classification toward breakaway. Also a chosen parameter.
    """

    def __init__(
        self,
        min_atr: float = 0.25,
        heavy_volume: float = 1.5,
        max_gaps: int = 30,
    ) -> None:
        self.min_atr = min_atr
        self.heavy_volume = heavy_volume
        self.max_gaps = max_gaps
        self._gaps: list[Gap] = []
        self._prev: Bar | None = None
        self._index = -1

    def update(
        self,
        bar: Bar,
        atr: float,
        avg_volume: float = 0.0,
        move_bars: int = 0,
        pattern_breakout: bool = False,
    ) -> Gap | None:
        """Feed a bar. Returns a newly detected gap, if any.

        ``move_bars`` is how long the current move has been running and
        ``pattern_breakout`` whether a chart pattern just completed. Both feed
        classification, because position within the move is precisely what
        separates Murphy's three gap types - the shape alone cannot.
        """
        self._index += 1
        prev = self._prev
        self._prev = bar

        # Update the status of existing gaps first.
        self._check_fills(bar)

        if prev is None or atr <= 0:
            return None

        if bar.low > prev.high:
            up, lo, hi = True, prev.high, bar.low
        elif bar.high < prev.low:
            up, lo, hi = False, bar.high, prev.low
        else:
            return None

        if (hi - lo) < self.min_atr * atr:
            return None

        vol_ratio = (bar.volume / avg_volume) if avg_volume > 0 else 1.0
        gap = Gap(
            index=self._index,
            ts=bar.ts,
            up=up,
            low=lo,
            high=hi,
            volume_ratio=vol_ratio,
            gap_type=self._classify(vol_ratio, move_bars, pattern_breakout),
        )
        self._gaps.append(gap)
        if len(self._gaps) > self.max_gaps:
            del self._gaps[: len(self._gaps) - self.max_gaps]
        return gap

    def _classify(
        self, vol_ratio: float, move_bars: int, pattern_breakout: bool
    ) -> GapType:
        """Classify by position in the move and volume, per Murphy."""
        # A pattern completing plus heavy volume is the breakaway signature.
        if pattern_breakout or (vol_ratio >= self.heavy_volume and move_bars <= 3):
            return GapType.BREAKAWAY
        if move_bars <= 2:
            return GapType.BREAKAWAY
        # Mid-move on moderate volume is the runaway/measuring gap.
        if 3 < move_bars <= 40:
            if vol_ratio >= self.heavy_volume * 1.6:
                # A volume blow-off late in a move points to exhaustion.
                return GapType.EXHAUSTION
            return GapType.RUNAWAY
        if move_bars > 40:
            return GapType.EXHAUSTION
        return GapType.COMMON

    def _check_fills(self, bar: Bar) -> None:
        for g in self._gaps:
            if not g.filled:
                # Filled when the bar's range trades back through the whole area.
                if g.up and bar.low <= g.low:
                    g.filled = True
                    g.filled_at = self._index
                elif not g.up and bar.high >= g.high:
                    g.filled = True
                    g.filled_at = self._index
            if not g.closed_through:
                # Murphy: a CLOSE back through a gap is the meaningful signal,
                # and is bearish for an up gap regardless of the gap's type.
                if g.up and bar.close < g.low:
                    g.closed_through = True
                elif not g.up and bar.close > g.high:
                    g.closed_through = True

    # -------------------------------------------------------------- accessors
    @property
    def gaps(self) -> list[Gap]:
        return list(self._gaps)

    @property
    def unfilled(self) -> list[Gap]:
        return [g for g in self._gaps if not g.filled]

    def nearest_support_gap(self, price: float) -> Gap | None:
        below = [g for g in self.unfilled if g.up and g.high <= price]
        return max(below, key=lambda g: g.high) if below else None

    def nearest_resistance_gap(self, price: float) -> Gap | None:
        above = [g for g in self.unfilled if not g.up and g.low >= price]
        return min(above, key=lambda g: g.low) if above else None

    def measuring_objective(self, move_start_price: float) -> float | None:
        """Target from the most recent runaway gap.

        Murphy's rule: a measuring gap tends to fall about halfway through the
        move, so double the distance already travelled from the move's origin.
        """
        runaways = [g for g in self._gaps if g.gap_type is GapType.RUNAWAY]
        if not runaways:
            return None
        g = runaways[-1]
        travelled = g.midpoint - move_start_price
        return move_start_price + 2.0 * travelled

    def island_reversal(self, lookback: int = 20) -> Gap | None:
        """Detect an island reversal: opposing gaps close together in time.

        Returns the second (breakaway) gap if the pattern is present.
        """
        recent = [g for g in self._gaps if self._index - g.index <= lookback]
        for a, b in zip(recent, recent[1:]):
            if a.up != b.up and (b.index - a.index) <= lookback:
                return b
        return None


# ------------------------------------------------------------- reversal bars


class ReversalKind(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"


@dataclass(slots=True)
class ReversalBar:
    """A single-bar reversal signal."""

    index: int
    ts: datetime
    kind: ReversalKind
    outside: bool
    """Range engulfed the prior bar's: adds significance per Murphy."""
    volume_ratio: float
    range_ratio: float
    """Bar range relative to its recent average."""
    new_extreme: bool
    """Whether the bar set a new extreme for the move, as the definition requires."""

    @property
    def is_climax(self) -> bool:
        """A bottom reversal on heavy volume is Murphy's selling climax."""
        return self.kind is ReversalKind.BOTTOM and self.volume_ratio >= 1.8

    @property
    def significance(self) -> float:
        """Composite weight in [0, 1] from range, volume and outside status.

        Murphy: the wider the range and the heavier the volume, the more the
        signal means.
        """
        score = 0.25
        score += min(0.3, 0.15 * self.range_ratio)
        score += min(0.3, 0.15 * self.volume_ratio)
        if self.outside:
            score += 0.15
        return max(0.0, min(1.0, score))


def detect_reversal_bar(
    index: int,
    bar: Bar,
    prev: Bar,
    extreme_high: float,
    extreme_low: float,
    avg_volume: float,
    avg_range: float,
) -> ReversalBar | None:
    """Detect a top or bottom reversal bar.

    ``extreme_high``/``extreme_low`` are the running extremes of the current move,
    used to enforce Murphy's requirement that the bar set a *new extreme for the
    move* before reversing. Without that check any down-close after an up bar
    would qualify, which would make the signal meaningless.
    """
    rng = bar.high - bar.low
    vol_ratio = (bar.volume / avg_volume) if avg_volume > 0 else 1.0
    rng_ratio = (rng / avg_range) if avg_range > 0 else 1.0
    outside = bar.high > prev.high and bar.low < prev.low

    made_high = bar.high >= extreme_high - 1e-12
    made_low = bar.low <= extreme_low + 1e-12

    if made_high and bar.close < prev.close:
        return ReversalBar(
            index=index,
            ts=bar.ts,
            kind=ReversalKind.TOP,
            outside=outside,
            volume_ratio=vol_ratio,
            range_ratio=rng_ratio,
            new_extreme=True,
        )
    if made_low and bar.close > prev.close:
        return ReversalBar(
            index=index,
            ts=bar.ts,
            kind=ReversalKind.BOTTOM,
            outside=outside,
            volume_ratio=vol_ratio,
            range_ratio=rng_ratio,
            new_extreme=True,
        )
    return None
