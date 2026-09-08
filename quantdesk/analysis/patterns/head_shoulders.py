"""Head and shoulders, and its inverse.

Murphy calls this the best known and most reliable of the major reversal
patterns, and it is really just his trend definition breaking down in slow
motion: the left shoulder and head are the last higher high, the failure of the
right shoulder to exceed the head is the first lower high, and the neckline break
completes "descending peaks and troughs" - a new downtrend by definition.

The five-pivot signature, top version:

* left shoulder (peak), reaction low, **head** (higher peak), reaction low,
  right shoulder (peak at roughly the left shoulder's height)
* neckline drawn through the two reaction lows, typically with a slight upward
  slope at tops
* completed by a decisive *close* below the neckline
* minimum objective = vertical distance head-to-neckline, projected down from the
  break
* volume tends to thin out on each successive peak, and a light-volume break
  makes the return move to the neckline more likely

Bottoms are the mirror image, with one asymmetry Murphy insists on: at bottoms
the volume expansion is essential, while at tops it is merely desirable.
"""

from __future__ import annotations

from quantdesk.analysis.patterns.base import (
    Bias,
    DetectorContext,
    Pattern,
    PatternDetector,
    PatternKind,
    PatternStage,
    PenetrationFilter,
)
from quantdesk.analysis.patterns.geometry import line_through
from quantdesk.analysis.patterns.lifecycle import advance
from quantdesk.analysis.swings import SwingKind, SwingPoint


class HeadShouldersDetector(PatternDetector):
    """Finds head and shoulders tops and bottoms in confirmed pivots.

    Parameters
    ----------
    shoulder_tol_atr:
        How unequal the two shoulders may be and still count as "about the same
        height". Murphy's own examples include shoulders that differ visibly, so
        this is deliberately loose.
    min_head_atr:
        How far the head must clear both shoulders. Without this, any three
        peaks of similar height register as a head and shoulders.
    scan_pivots:
        How far back in the pivot list to look for the shape. The pattern needs
        five pivots; scanning a few extra windows lets an older formation be
        picked up if a later one did not qualify.
    """

    kinds = (PatternKind.HEAD_SHOULDERS_TOP, PatternKind.HEAD_SHOULDERS_BOTTOM)

    def __init__(
        self,
        shoulder_tol_atr: float = 1.2,
        min_head_atr: float = 0.8,
        penetration_pct: float = 0.01,
        confirm_bars: int = 1,
        expire_after: int = 80,
        scan_pivots: int = 9,
        max_tracked: int = 6,
    ) -> None:
        self.shoulder_tol_atr = shoulder_tol_atr
        self.min_head_atr = min_head_atr
        self.expire_after = expire_after
        self.scan_pivots = scan_pivots
        self.max_tracked = max_tracked
        self._filter = PenetrationFilter(pct=penetration_pct, confirm_bars=confirm_bars)
        self._tracked: dict[tuple, Pattern] = {}

    # ----------------------------------------------------------------- update
    def update(self, ctx: DetectorContext) -> list[Pattern]:
        changed: list[Pattern] = []

        for pattern in list(self._tracked.values()):
            if advance(
                pattern, ctx, self._filter, expire_after=self.expire_after
            ):
                changed.append(pattern)

        if self.pivots_changed(ctx):
            for pattern in self._recognise(ctx):
                if not self.accepts(pattern, self._tracked):
                    continue
                self._tracked[pattern.key] = pattern
                changed.append(pattern)

        self._evict()
        return changed

    def _evict(self) -> None:
        """Drop dead patterns, oldest first, and cap how many are tracked."""
        for key, pattern in list(self._tracked.items()):
            if pattern.stage.is_dead:
                del self._tracked[key]
                self._filter.reset(key)
        if len(self._tracked) > self.max_tracked:
            ordered = sorted(
                self._tracked.items(), key=lambda kv: kv[1].detected_index
            )
            for key, _ in ordered[: len(self._tracked) - self.max_tracked]:
                del self._tracked[key]
                self._filter.reset(key)

    @property
    def active(self) -> list[Pattern]:
        return [p for p in self._tracked.values() if not p.stage.is_dead]

    # ------------------------------------------------------------ recognition
    def _recognise(self, ctx: DetectorContext) -> list[Pattern]:
        pivots = ctx.confirmed[-self.scan_pivots:]
        if len(pivots) < 5:
            return []

        found: list[Pattern] = []
        for start in range(len(pivots) - 5, -1, -1):
            window = pivots[start : start + 5]
            if not _alternating(window):
                continue
            pattern = self._try_window(window, ctx)
            if pattern is not None:
                found.append(pattern)
        return found

    def _try_window(
        self, window: list[SwingPoint], ctx: DetectorContext
    ) -> Pattern | None:
        left, low_a, head, low_b, right = window
        atr = ctx.atr_or()
        is_top = left.kind is SwingKind.PEAK

        # The head must genuinely dominate both shoulders.
        if is_top:
            if head.price < left.price + self.min_head_atr * atr:
                return None
            if head.price < right.price + self.min_head_atr * atr:
                return None
        else:
            if head.price > left.price - self.min_head_atr * atr:
                return None
            if head.price > right.price - self.min_head_atr * atr:
                return None

        # Shoulders at roughly the same height.
        if abs(left.price - right.price) > self.shoulder_tol_atr * atr:
            return None

        neckline = line_through(low_a, low_b)
        # The right shoulder has to sit on the correct side of the neckline,
        # otherwise the neckline has already been breached and there is no
        # pattern left to complete.
        rs_side = neckline.distance(right.index, right.price)
        if is_top and rs_side <= 0:
            return None
        if not is_top and rs_side >= 0:
            return None

        height = abs(head.price - neckline.value_at(head.index))
        if height < 1.5 * atr:
            # Too shallow to be a meaningful reversal, and the measured objective
            # would be inside the noise.
            return None

        detected = max(p.confirmed_index for p in window)
        kind = (
            PatternKind.HEAD_SHOULDERS_TOP if is_top else PatternKind.HEAD_SHOULDERS_BOTTOM
        )
        bias = Bias.BEARISH if is_top else Bias.BULLISH

        pattern = Pattern(
            kind=kind,
            symbol=ctx.symbol,
            bias=bias,
            pivots=list(window),
            start_index=left.index,
            start_ts=left.ts,
            detected_index=detected,
            height=height,
            boundary=neckline.value_at(detected),
            boundary_slope=neckline.slope,
            stage=PatternStage.FORMING,
            # Murphy's stop for a top sits above the right shoulder: price back
            # through it says the lower-high premise was wrong.
            stop=right.price,
            notes=self._notes(window, neckline, atr, is_top),
        )
        pattern.formation_volume_ratio = ctx.volume_ctx.formation_ratio(
            pattern.start_index, pattern.end_index, ctx.index
        )
        return pattern

    def _notes(
        self, window: list[SwingPoint], neckline, atr: float, is_top: bool
    ) -> str:
        left, _, head, _, right = window
        bits: list[str] = []

        # Volume thinning across the peaks is Murphy's confirming signature.
        if left.volume > 0 and head.volume > 0:
            if head.volume < left.volume and right.volume < head.volume:
                bits.append("volume thinning across peaks")
            elif right.volume > head.volume:
                bits.append("volume rising into right shoulder (weak)")

        slope_atr = neckline.slope_atr(atr)
        if is_top:
            if slope_atr < -0.02:
                # He notes a downward-tilting neckline at a top is less common
                # and is the weaker, more bearish configuration.
                bits.append("downward-sloping neckline (weaker top)")
        else:
            if slope_atr > 0.02:
                bits.append("upward-sloping neckline (weaker bottom)")

        span = right.index - left.index
        bits.append(f"{span} bars wide")
        return "; ".join(bits)


def _alternating(window: list[SwingPoint]) -> bool:
    """Pivots must strictly alternate peak/trough to form the shape."""
    return all(
        a.kind is not b.kind for a, b in zip(window, window[1:])
    )
