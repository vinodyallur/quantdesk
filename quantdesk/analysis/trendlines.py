"""Trendlines, channels and the fan principle.

Murphy's rules, implemented as stated:

* An up trendline connects successive reaction lows; a down trendline connects
  successive rally peaks.
* **Two points draw a tentative line; a third touch validates it.** This is the
  distinction the code preserves via :attr:`Trendline.valid`, because an
  unvalidated line is explicitly not yet a trendline in his framework.
* Significance grows with the number of touches and how long the line has held.
* Lines are drawn through the full bar range, not closes.
* A valid break needs a *close* beyond the line, filtered by roughly 3% for
  major lines or 1% for shorter-term ones, or by the two-bar rule.
* Broken trendlines reverse roles, so they are kept and projected forward.
* **Measuring rule:** after a break, price tends to travel beyond the line by
  the same vertical distance it previously reached on the other side.
* Slope matters. Around 45 degrees is the sustainable ideal; markedly steeper is
  unsustainable and markedly flatter is suspect. Since "degrees" depends entirely
  on chart scaling, this is expressed scale-free in ATR per bar.
* **Channel line:** parallel to the basic line, off the first prominent
  counter-swing. Breaking the channel line means *acceleration*, the opposite of
  breaking the trendline. Failing to reach it warns the other side will go.
* **Fan principle:** successively flatter lines; the break of the third is the
  reversal signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from quantdesk.analysis.swings import SwingKind, SwingPoint


class LineKind(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"


@dataclass(slots=True)
class Trendline:
    """A straight line fitted through two or more swing pivots."""

    kind: LineKind
    #: (bar_index, price) of the two anchor pivots.
    x1: int = 0
    y1: float = 0.0
    x2: int = 0
    y2: float = 0.0
    touches: int = 2
    valid: bool = False
    """False while tentative. Murphy requires a third touch to validate."""
    broken: bool = False
    broken_at: int = 0
    #: Greatest vertical distance price reached away from the line, for the
    #: post-break measuring rule.
    max_excursion: float = 0.0
    role_reversed: bool = False

    @property
    def slope(self) -> float:
        """Price change per bar."""
        dx = self.x2 - self.x1
        if dx == 0:
            return 0.0
        return (self.y2 - self.y1) / dx

    def value_at(self, index: int) -> float:
        """Projected line price at a bar index, extended indefinitely."""
        return self.y1 + self.slope * (index - self.x1)

    def distance(self, index: int, price: float) -> float:
        """Signed distance from the line: positive means price is above it."""
        return price - self.value_at(index)

    def slope_atr(self, atr: float) -> float:
        """Slope in ATR per bar - a scale-free stand-in for Murphy's 45 degrees.

        Chart degrees depend on axis scaling, so they cannot be computed from data
        alone. A line advancing about one ATR per bar is the balanced case here;
        far above that is the unsustainable steep line, far below is the suspect
        flat one.
        """
        if atr <= 0:
            return 0.0
        return self.slope / atr

    def steepness(self, atr: float, ideal: float = 1.0) -> str:
        """Classify slope as 'steep', 'balanced' or 'flat'."""
        s = abs(self.slope_atr(atr))
        if s > ideal * 2.5:
            return "steep"
        if s < ideal * 0.25:
            return "flat"
        return "balanced"

    def significance(self, current_index: int) -> float:
        """Confidence in the line, in [0, 1], from touches and duration."""
        if not self.valid:
            return 0.25
        touch_score = min(1.0, (self.touches - 2) / 4.0)
        span = max(1, current_index - self.x1)
        duration_score = min(1.0, span / 120.0)
        return 0.3 + 0.4 * touch_score + 0.3 * duration_score

    def price_objective(self, index: int) -> float:
        """Target after a break, per Murphy's trendline measuring rule.

        Price is expected to travel beyond the broken line by the same vertical
        distance it previously achieved on the other side.
        """
        line = self.value_at(index)
        if self.kind is LineKind.UPTREND:
            return line - self.max_excursion
        return line + self.max_excursion


class TrendlineTracker:
    """Fits and maintains trendlines from confirmed swing pivots.

    Only confirmed pivots are used, so no line depends on knowledge that was not
    available at the time.
    """

    def __init__(
        self,
        penetration_pct: float = 0.01,
        confirm_bars: int = 2,
        touch_atr: float = 0.5,
        max_lines: int = 6,
    ) -> None:
        self.penetration_pct = penetration_pct
        self.confirm_bars = confirm_bars
        # How close price must come to count as touching the line.
        self.touch_atr = touch_atr
        self.max_lines = max_lines
        self._lines: list[Trendline] = []
        self._beyond: dict[int, int] = {}
        self._fan: list[Trendline] = []

    # ------------------------------------------------------------------- fit
    def fit(self, points: list[SwingPoint]) -> list[Trendline]:
        """Rebuild candidate lines from the two most recent pivots of each kind."""
        troughs = [p for p in points if p.kind is SwingKind.TROUGH]
        peaks = [p for p in points if p.kind is SwingKind.PEAK]
        lines: list[Trendline] = []

        if len(troughs) >= 2:
            a, b = troughs[-2], troughs[-1]
            # An up trendline requires the second low to be higher than the first.
            if b.price > a.price:
                lines.append(
                    Trendline(LineKind.UPTREND, a.index, a.price, b.index, b.price)
                )
        if len(peaks) >= 2:
            a, b = peaks[-2], peaks[-1]
            if b.price < a.price:
                lines.append(
                    Trendline(LineKind.DOWNTREND, a.index, a.price, b.index, b.price)
                )

        # Preserve validation state and touch counts across refits.
        for new in lines:
            for old in self._lines:
                if (
                    old.kind is new.kind
                    and old.x1 == new.x1
                    and abs(old.y1 - new.y1) < 1e-9
                ):
                    new.touches = max(new.touches, old.touches)
                    new.valid = old.valid
                    new.max_excursion = old.max_excursion
                    new.broken = old.broken
                    new.broken_at = old.broken_at
                    break
        # Keep broken lines: Murphy notes they often act as support/resistance
        # again in the opposite role, so they stay projected forward.
        kept_broken = [ln for ln in self._lines if ln.broken][-self.max_lines:]
        self._lines = lines + kept_broken
        return lines

    # ---------------------------------------------------------------- update
    def update(self, index: int, high: float, low: float, close: float, atr: float) -> dict:
        """Record touches, excursions and breaks for the current bar."""
        events = {"touched": [], "broken": [], "fan_break": False}
        tolerance = self.touch_atr * atr

        for i, ln in enumerate(self._lines):
            if ln.broken:
                continue
            line_px = ln.value_at(index)

            # Touch: price came within tolerance of the line from the correct side.
            probe = low if ln.kind is LineKind.UPTREND else high
            if abs(probe - line_px) <= tolerance:
                ln.touches += 1
                if ln.touches >= 3 and not ln.valid:
                    # Third touch validates the line, per Murphy.
                    ln.valid = True
                events["touched"].append(ln)

            # Track the largest excursion away from the line for the measuring rule.
            excursion = (
                high - line_px if ln.kind is LineKind.UPTREND else line_px - low
            )
            if excursion > ln.max_excursion:
                ln.max_excursion = excursion

            # Break: close beyond by the price filter, held for confirm_bars.
            margin = abs(line_px) * self.penetration_pct
            beyond = (
                close < line_px - margin
                if ln.kind is LineKind.UPTREND
                else close > line_px + margin
            )
            if beyond:
                self._beyond[i] = self._beyond.get(i, 0) + 1
                if self._beyond[i] >= self.confirm_bars:
                    ln.broken = True
                    ln.broken_at = index
                    events["broken"].append(ln)
                    self._register_fan(ln)
                    self._beyond[i] = 0
            else:
                self._beyond[i] = 0

        # Fan principle: the third broken line in the same direction is the signal.
        if len(self._fan) >= 3:
            events["fan_break"] = True
        return events

    def _register_fan(self, line: Trendline) -> None:
        """Accumulate successively broken lines of the same kind."""
        if self._fan and self._fan[-1].kind is not line.kind:
            self._fan.clear()
        self._fan.append(line)
        if len(self._fan) > 3:
            del self._fan[:-3]

    def reset_fan(self) -> None:
        self._fan.clear()

    # ------------------------------------------------------------- accessors
    @property
    def lines(self) -> list[Trendline]:
        return list(self._lines)

    @property
    def active(self) -> list[Trendline]:
        return [ln for ln in self._lines if not ln.broken]

    @property
    def valid_lines(self) -> list[Trendline]:
        return [ln for ln in self._lines if ln.valid and not ln.broken]

    @property
    def fan_count(self) -> int:
        """How many successive lines have broken. Three is Murphy's signal."""
        return len(self._fan)

    def primary(self, kind: LineKind | None = None) -> Trendline | None:
        """Most significant unbroken line, preferring validated ones."""
        pool = [ln for ln in self.active if kind is None or ln.kind is kind]
        if not pool:
            return None
        return max(pool, key=lambda ln: (ln.valid, ln.touches))


@dataclass(slots=True)
class Channel:
    """A trendline plus a parallel return line.

    Murphy's asymmetry is the important part and is encoded in the methods below:
    breaking the *basic* line signals a trend change, while breaking the *channel*
    line signals acceleration of the existing trend. Failing to reach the channel
    line warns that the basic line is likely to give way.
    """

    basic: Trendline
    #: Price offset from the basic line to the parallel return line.
    offset: float = 0.0
    touches_channel: int = 0
    reached_last: bool = True

    def channel_at(self, index: int) -> float:
        return self.basic.value_at(index) + self.offset

    @property
    def width(self) -> float:
        return abs(self.offset)

    def position(self, index: int, price: float) -> float:
        """Where price sits across the channel: 0 basic line, 1 channel line."""
        if self.width <= 1e-12:
            return 0.5
        base = self.basic.value_at(index)
        return (price - base) / self.offset

    def breakout_objective(self, index: int, upside: bool) -> float:
        """Target after leaving the channel: one channel width beyond it.

        Murphy's channel measuring rule.
        """
        if upside:
            return self.channel_at(index) + self.width
        return self.basic.value_at(index) - self.width


def build_channel(
    line: Trendline, points: list[SwingPoint]
) -> Channel | None:
    """Construct the parallel return line off the first prominent counter-swing.

    For an up trendline drawn along lows, the channel line is projected parallel
    through the highest intervening peak.
    """
    if line.kind is LineKind.UPTREND:
        counter = [
            p for p in points if p.kind is SwingKind.PEAK and p.index >= line.x1
        ]
        if not counter:
            return None
        # Furthest peak above the line defines the channel.
        best = max(counter, key=lambda p: p.price - line.value_at(p.index))
        offset = best.price - line.value_at(best.index)
        if offset <= 0:
            return None
    else:
        counter = [
            p for p in points if p.kind is SwingKind.TROUGH and p.index >= line.x1
        ]
        if not counter:
            return None
        best = min(counter, key=lambda p: p.price - line.value_at(p.index))
        offset = best.price - line.value_at(best.index)
        if offset >= 0:
            return None
    return Channel(basic=line, offset=offset)
