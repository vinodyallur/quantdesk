"""Saucers and spikes: the two reversals with no clean boundary.

Murphy is candid that these resist mechanical treatment, and the code says so
rather than inventing precision he does not claim.

**Saucer / rounding bottom.** "A very slow and very gradual turn from down to
sideways to up. It is difficult to tell exactly when the saucer has been completed
or to measure how far prices will travel." So the defining test here is
*gradualness*, not a shape template: shallow swings spread over a long span with
the extreme near the middle. Since he gives no completion rule, this module picks
one and labels it as a choice - a close above the highest interior peak - so no
one mistakes it for his.

**Spike / V.** "Happens very quickly with little or no transition period ... A
daily or weekly reversal, on very heavy volume, is sometimes the only warning."
There is no formation to wait for, so a spike is emitted already COMPLETE. It
reuses :func:`~quantdesk.analysis.gaps.detect_reversal_bar`, which already
encodes his reversal-bar definition including the new-extreme requirement, and
adds the precondition he names: the market must have gotten badly overextended
first.
"""

from __future__ import annotations

from quantdesk.analysis.gaps import ReversalKind, detect_reversal_bar
from quantdesk.analysis.patterns.base import (
    Bias,
    DetectorContext,
    Pattern,
    PatternDetector,
    PatternKind,
    PatternStage,
    PenetrationFilter,
)
from quantdesk.analysis.patterns.lifecycle import advance
from quantdesk.analysis.swings import SwingKind, SwingPoint
from quantdesk.core.types import Bar
from quantdesk.features.indicators import RollingWindow


class RoundingDetector(PatternDetector):
    """Saucer bottoms and rounding tops.

    Parameters
    ----------
    min_pivots:
        How many same-side pivots must make up the curve. More pivots means a
        better-established rounding.
    max_slope_atr:
        Ceiling on the average per-bar slope of the legs, in ATR. This is what
        encodes "slow and gradual" - a sharp V fails it, which is correct, since
        that is a spike and a different pattern.
    min_span_bars:
        Saucers are long-timeframe formations; short ones are not saucers.
    """

    kinds = (PatternKind.ROUNDING_BOTTOM, PatternKind.ROUNDING_TOP)

    def __init__(
        self,
        min_pivots: int = 4,
        max_slope_atr: float = 0.25,
        min_span_bars: int = 40,
        penetration_pct: float = 0.01,
        confirm_bars: int = 1,
        expire_after: int = 120,
        scan_pivots: int = 12,
        max_tracked: int = 4,
    ) -> None:
        self.min_pivots = min_pivots
        self.max_slope_atr = max_slope_atr
        self.min_span_bars = min_span_bars
        self.expire_after = expire_after
        self.scan_pivots = scan_pivots
        self.max_tracked = max_tracked
        self._filter = PenetrationFilter(pct=penetration_pct, confirm_bars=confirm_bars)
        self._tracked: dict[tuple, Pattern] = {}

    def update(self, ctx: DetectorContext) -> list[Pattern]:
        changed: list[Pattern] = []
        for pattern in list(self._tracked.values()):
            if advance(pattern, ctx, self._filter, expire_after=self.expire_after):
                changed.append(pattern)
        if self.pivots_changed(ctx):
            for pattern in self._recognise(ctx):
                if not self.accepts(pattern, self._tracked):
                    continue
                self._tracked[pattern.key] = pattern
                changed.append(pattern)
        for key, pattern in list(self._tracked.items()):
            if pattern.stage.is_dead:
                del self._tracked[key]
                self._filter.reset(key)
        return changed

    @property
    def active(self) -> list[Pattern]:
        return [p for p in self._tracked.values() if not p.stage.is_dead]

    def _recognise(self, ctx: DetectorContext) -> list[Pattern]:
        pivots = ctx.confirmed[-self.scan_pivots:]
        if len(pivots) < self.min_pivots * 2 - 1:
            return []
        out: list[Pattern] = []
        for kind in (SwingKind.TROUGH, SwingKind.PEAK):
            pattern = self._try_curve(pivots, kind, ctx)
            if pattern is not None:
                out.append(pattern)
        return out

    def _try_curve(
        self, pivots: list[SwingPoint], side: SwingKind, ctx: DetectorContext
    ) -> Pattern | None:
        same = [p for p in pivots if p.kind is side][-self.min_pivots - 2:]
        if len(same) < self.min_pivots:
            return None
        atr = ctx.atr_or()

        span = same[-1].index - same[0].index
        if span < self.min_span_bars:
            return None

        # Gradualness: no leg may be steep. This is the saucer's whole character.
        for a, b in zip(same, same[1:]):
            bars = max(1, b.index - a.index)
            if abs(b.price - a.price) / bars > self.max_slope_atr * atr:
                return None

        # The extreme should sit in the interior, giving the U (or arch) its
        # symmetry. A monotonic run of pivots is a trend, not a rounding.
        is_bottom = side is SwingKind.TROUGH
        extreme = min(same, key=lambda p: p.price) if is_bottom else max(
            same, key=lambda p: p.price
        )
        if extreme is same[0] or extreme is same[-1]:
            return None

        # Descend then ascend (or the reverse), allowing the flat middle that
        # makes it a saucer rather than a V.
        before = same[: same.index(extreme) + 1]
        after = same[same.index(extreme) :]
        if len(before) < 2 or len(after) < 2:
            return None
        if is_bottom:
            if not _monotone(before, rising=False) or not _monotone(after, rising=True):
                return None
        else:
            if not _monotone(before, rising=True) or not _monotone(after, rising=False):
                return None

        interior = [p for p in pivots if p.kind is not side and same[0].index <= p.index <= same[-1].index]
        if not interior:
            return None
        boundary_pivot = (
            max(interior, key=lambda p: p.price)
            if is_bottom
            else min(interior, key=lambda p: p.price)
        )
        height = abs(boundary_pivot.price - extreme.price)
        if height < 1.5 * atr:
            return None

        detected = max(p.confirmed_index for p in same + [boundary_pivot])
        return Pattern(
            kind=PatternKind.ROUNDING_BOTTOM if is_bottom else PatternKind.ROUNDING_TOP,
            symbol=ctx.symbol,
            bias=Bias.BULLISH if is_bottom else Bias.BEARISH,
            pivots=list(same),
            start_index=same[0].index,
            start_ts=same[0].ts,
            detected_index=detected,
            height=height,
            boundary=boundary_pivot.price,
            boundary_slope=0.0,
            stage=PatternStage.FORMING,
            stop=extreme.price,
            notes=(
                f"gradual turn over {span} bars; completion rule is this desk's "
                "choice, Murphy gives none; objective unreliable by his own account"
            ),
        )


def _monotone(points: list[SwingPoint], rising: bool) -> bool:
    """Weakly monotonic, so a flat middle section is allowed."""
    for a, b in zip(points, points[1:]):
        if rising and b.price < a.price:
            return False
        if not rising and b.price > a.price:
            return False
    return True


class SpikeDetector(PatternDetector):
    """V tops and bottoms.

    Emitted already COMPLETE: by Murphy's description there is no consolidation
    to wait through, so requiring a boundary break would mean never catching one.

    Parameters
    ----------
    min_extension_atr:
        How far the move must have travelled from its last pivot before a
        reversal bar counts as a spike rather than ordinary two-way trade. This
        is his "so overextended in one direction" precondition.
    """

    kinds = (PatternKind.SPIKE_TOP, PatternKind.SPIKE_BOTTOM)

    def __init__(
        self,
        min_extension_atr: float = 4.0,
        volume_period: int = 30,
        max_tracked: int = 4,
    ) -> None:
        self.min_extension_atr = min_extension_atr
        self._range = RollingWindow(volume_period)
        self._prev: Bar | None = None
        self._hi = float("-inf")
        self._lo = float("inf")
        self._patterns: list[Pattern] = []
        self.max_tracked = max_tracked

    def update(self, ctx: DetectorContext) -> list[Pattern]:
        bar = Bar(
            symbol=ctx.symbol,
            ts=ctx.ts,
            open=ctx.open,
            high=ctx.high,
            low=ctx.low,
            close=ctx.close,
            volume=ctx.volume,
        )
        # Running extremes of the current move, reset at each confirmed pivot so
        # "new extreme for the move" means what Murphy means by it.
        if ctx.confirmed and ctx.confirmed[-1].confirmed_index == ctx.index:
            self._hi, self._lo = ctx.high, ctx.low
        else:
            self._hi = max(self._hi, ctx.high)
            self._lo = min(self._lo, ctx.low)

        prev = self._prev
        self._prev = bar
        avg_range = self._range.mean
        self._range.update(ctx.high - ctx.low)
        if prev is None or avg_range <= 0:
            return []

        rb = detect_reversal_bar(
            index=ctx.index,
            bar=bar,
            prev=prev,
            extreme_high=self._hi,
            extreme_low=self._lo,
            avg_volume=ctx.volume_ctx.average,
            avg_range=avg_range,
        )
        if rb is None:
            return []

        atr = ctx.atr_or()
        anchor = self._anchor(ctx, rb.kind)
        if anchor is None:
            return []
        extension = abs(ctx.close - anchor.price) / atr
        if extension < self.min_extension_atr:
            return []

        is_top = rb.kind is ReversalKind.TOP
        # No measuring rule exists for a spike. The nearest thing Murphy offers
        # is his general tendency for markets to retrace a third to a half of the
        # prior move, so that is used and labelled as the inference it is.
        move = abs(ctx.close - anchor.price)
        objective = ctx.close - 0.5 * move if is_top else ctx.close + 0.5 * move

        pattern = Pattern(
            kind=PatternKind.SPIKE_TOP if is_top else PatternKind.SPIKE_BOTTOM,
            symbol=ctx.symbol,
            bias=Bias.BEARISH if is_top else Bias.BULLISH,
            pivots=[anchor],
            start_index=anchor.index,
            start_ts=anchor.ts,
            detected_index=ctx.index,
            height=0.5 * move,
            boundary=ctx.close,
            stage=PatternStage.COMPLETE,
            breakout_index=ctx.index,
            breakout_ts=ctx.ts,
            breakout_price=ctx.close,
            breakout_volume_ratio=rb.volume_ratio,
            objective=objective,
            stop=self._hi if is_top else self._lo,
            notes=(
                f"{extension:.1f} ATR extended; reversal bar vol x{rb.volume_ratio:.1f}"
                + ("; outside bar" if rb.outside else "")
                + "; objective is a 50% retracement inference, not Murphy's rule"
            ),
        )
        self._patterns.append(pattern)
        if len(self._patterns) > self.max_tracked:
            del self._patterns[: len(self._patterns) - self.max_tracked]
        return [pattern]

    def _anchor(self, ctx: DetectorContext, kind: ReversalKind) -> SwingPoint | None:
        """Pivot the current move started from."""
        want = SwingKind.TROUGH if kind is ReversalKind.TOP else SwingKind.PEAK
        for p in reversed(ctx.confirmed):
            if p.kind is want:
                return p
        return None

    @property
    def active(self) -> list[Pattern]:
        return list(self._patterns)
