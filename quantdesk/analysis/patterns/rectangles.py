"""Rectangles and the measured move.

Two formations that share a chapter but not much else.

**Rectangle.** A pause between two parallel horizontal lines - Murphy notes it is
also called a trading range or congestion area, and is "a line" in Dow Theory
terms. He treats it as the symmetrical triangle with flat instead of converging
boundaries: same forecasting value, resolved in the direction of the trend that
preceded it. Measuring is the height of the band from the breakout. Because the
shape is identical to a double or triple top until it breaks, those detectors will
often fire on the same pivots; which reading was right is decided by which way it
breaks, and that is exactly how a chartist would treat it too.

**Measured move.** Not a shape with a breakout but a projection: an advance splits
into two equal, parallel legs. AB runs, BC corrects "about a third to a half" of
it, then CD "duplicates the size and slope" of AB. Implemented as a pattern whose
boundary is B, so it completes when price closes back through the top of the first
leg, with the objective measured from the correction low - *not* from the boundary,
which is why it sets its own target.
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
    similar,
)
from quantdesk.analysis.patterns.lifecycle import advance
from quantdesk.analysis.swings import SwingKind, SwingPoint


class RectangleDetector(PatternDetector):
    """Horizontal congestion bands.

    Parameters
    ----------
    level_tol_atr:
        How much the peaks (and troughs) may scatter and still count as one flat
        boundary.
    min_touches:
        Touches per boundary. Two each is the minimum that defines the band.
    """

    kinds = (PatternKind.RECTANGLE,)

    def __init__(
        self,
        level_tol_atr: float = 0.7,
        min_touches: int = 2,
        min_height_atr: float = 1.5,
        min_span_bars: int = 12,
        penetration_pct: float = 0.01,
        confirm_bars: int = 1,
        expire_after: int = 90,
        scan_pivots: int = 10,
        max_tracked: int = 6,
    ) -> None:
        self.level_tol_atr = level_tol_atr
        self.min_touches = min_touches
        self.min_height_atr = min_height_atr
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
        for key, p in list(self._tracked.items()):
            if p.stage.is_dead:
                del self._tracked[key]
                self._filter.reset(key)
        if len(self._tracked) > self.max_tracked:
            ordered = sorted(self._tracked.items(), key=lambda kv: kv[1].detected_index)
            for key, _ in ordered[: len(self._tracked) - self.max_tracked]:
                del self._tracked[key]
                self._filter.reset(key)
        return changed

    @property
    def active(self) -> list[Pattern]:
        return [p for p in self._tracked.values() if not p.stage.is_dead]

    def _recognise(self, ctx: DetectorContext) -> list[Pattern]:
        pivots = ctx.confirmed[-self.scan_pivots:]
        peaks = [p for p in pivots if p.kind is SwingKind.PEAK][-4:]
        troughs = [p for p in pivots if p.kind is SwingKind.TROUGH][-4:]
        if len(peaks) < self.min_touches or len(troughs) < self.min_touches:
            return []

        atr = ctx.atr_or()
        tol = self.level_tol_atr * atr
        # Both boundaries must be flat: every touch at effectively one level.
        if not all(similar(peaks[0].price, p.price, tol) for p in peaks[1:]):
            return []
        if not all(similar(troughs[0].price, p.price, tol) for p in troughs[1:]):
            return []

        top = sum(p.price for p in peaks) / len(peaks)
        bottom = sum(p.price for p in troughs) / len(troughs)
        height = top - bottom
        if height < self.min_height_atr * atr:
            return []

        used = sorted(peaks + troughs, key=lambda p: p.index)
        span = used[-1].index - used[0].index
        if span < self.min_span_bars:
            return []

        detected = max(p.confirmed_index for p in used)
        # Resolved in the direction of the preceding trend. With no prior trend
        # there is no forecast, so both resolutions are tracked.
        if ctx.prior_trend is Bias.BULLISH:
            biases: tuple[Bias, ...] = (Bias.BULLISH,)
        elif ctx.prior_trend is Bias.BEARISH:
            biases = (Bias.BEARISH,)
        else:
            biases = (Bias.BULLISH, Bias.BEARISH)

        out: list[Pattern] = []
        for bias in biases:
            up = bias is Bias.BULLISH
            pattern = Pattern(
                kind=PatternKind.RECTANGLE,
                symbol=ctx.symbol,
                bias=bias,
                pivots=used,
                start_index=used[0].index,
                start_ts=used[0].ts,
                detected_index=detected,
                height=height,
                boundary=top if up else bottom,
                boundary_slope=0.0,
                stage=PatternStage.FORMING,
                stop=bottom if up else top,
                notes=(
                    f"{len(peaks)}+{len(troughs)} touches over {span} bars"
                    + ("; no prior trend, tracking both resolutions" if len(biases) > 1 else "")
                ),
            )
            pattern.formation_volume_ratio = ctx.volume_ctx.formation_ratio(
                pattern.start_index, pattern.end_index, ctx.index
            )
            out.append(pattern)
        return out


class MeasuredMoveDetector(PatternDetector):
    """Two equal legs separated by a one-third-to-one-half correction.

    Parameters
    ----------
    min_retrace, max_retrace:
        Murphy's "a third to a half" window for the BC correction. Slightly
        widened by default because he describes it as a tendency, not a rule, and
        because his own retracement chapter treats 38.2% and 50% as the live
        zone.
    """

    kinds = (PatternKind.MEASURED_MOVE,)

    def __init__(
        self,
        min_retrace: float = 0.30,
        max_retrace: float = 0.55,
        min_leg_atr: float = 3.0,
        penetration_pct: float = 0.005,
        confirm_bars: int = 1,
        expire_after: int = 60,
        max_tracked: int = 4,
    ) -> None:
        self.min_retrace = min_retrace
        self.max_retrace = max_retrace
        self.min_leg_atr = min_leg_atr
        self.expire_after = expire_after
        self.max_tracked = max_tracked
        self._filter = PenetrationFilter(pct=penetration_pct, confirm_bars=confirm_bars)
        self._tracked: dict[tuple, Pattern] = {}

    def update(self, ctx: DetectorContext) -> list[Pattern]:
        changed: list[Pattern] = []
        for pattern in list(self._tracked.values()):
            if advance(pattern, ctx, self._filter, expire_after=self.expire_after):
                changed.append(pattern)
        if self.pivots_changed(ctx):
            pattern = self._recognise(ctx)
            if pattern is not None and self.accepts(pattern, self._tracked):
                self._tracked[pattern.key] = pattern
                changed.append(pattern)
        for key, p in list(self._tracked.items()):
            if p.stage.is_dead:
                del self._tracked[key]
                self._filter.reset(key)
        if len(self._tracked) > self.max_tracked:
            ordered = sorted(self._tracked.items(), key=lambda kv: kv[1].detected_index)
            for key, _ in ordered[: len(self._tracked) - self.max_tracked]:
                del self._tracked[key]
                self._filter.reset(key)
        return changed

    @property
    def active(self) -> list[Pattern]:
        return [p for p in self._tracked.values() if not p.stage.is_dead]

    def _recognise(self, ctx: DetectorContext) -> Pattern | None:
        pivots = ctx.confirmed[-3:]
        if len(pivots) < 3:
            return None
        a, b, c = pivots
        if a.kind is b.kind or b.kind is c.kind:
            return None

        atr = ctx.atr_or()
        leg = abs(b.price - a.price)
        if leg < self.min_leg_atr * atr:
            return None

        retrace = abs(b.price - c.price) / leg
        if not (self.min_retrace <= retrace <= self.max_retrace):
            return None

        up = b.price > a.price
        # CD duplicates AB, projected from the correction low at C.
        objective = c.price + leg if up else c.price - leg
        bars_ab = max(1, b.index - a.index)

        return Pattern(
            kind=PatternKind.MEASURED_MOVE,
            symbol=ctx.symbol,
            bias=Bias.BULLISH if up else Bias.BEARISH,
            pivots=[a, b, c],
            start_index=a.index,
            start_ts=a.ts,
            detected_index=max(p.confirmed_index for p in (a, b, c)),
            height=leg,
            # The first leg's extreme is the level that has to give way for the
            # second leg to be underway.
            boundary=b.price,
            boundary_slope=0.0,
            stage=PatternStage.FORMING,
            stop=c.price,
            objective=objective,
            notes=(
                f"AB {leg / atr:.1f} ATR over {bars_ab} bars; "
                f"BC retraced {retrace:.0%}; CD duplicates AB in size and slope, "
                f"so expect roughly {bars_ab} bars to target"
            ),
        )
