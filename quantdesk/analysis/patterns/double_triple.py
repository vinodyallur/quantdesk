"""Double and triple tops and bottoms.

One detector for both, because they are the same idea at different counts: price
tests a level twice or three times, fails, and the pattern completes when the
reaction level *between* the tests gives way.

Murphy's specifics, all enforced below:

* The peaks (or troughs) sit at about the same level. He notes the second peak
  often does not quite reach the first, so exact equality is not required.
* **Time matters.** "Most valid double tops or bottoms should have at least a
  month between the two peaks or troughs." On a daily chart that is roughly
  twenty bars, which is how ``min_separation_bars`` is scaled here - twenty bars
  of whatever timeframe is being charted, since the pattern's degree is always
  relative to the chart.
* The signal is the breaking of the intervening reaction level, not the second
  peak itself. For a triple top there are two intervening troughs, so the
  operative level is the lower of them - price has to clear the whole base.
* A close beyond, not an intraday penetration.
* Objective = pattern height projected from the break.
* Bigger and longer means a bigger expected reversal, which flows through
  :meth:`Pattern.significance`.

Murphy also observes the triple top "resembles a rectangle" and is a variation of
head and shoulders with a level head. Both detectors can therefore fire on the
same pivots; that is faithful rather than a bug, and the weight-of-evidence layer
is where agreement between overlapping readings gets resolved.
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


class DoubleTripleDetector(PatternDetector):
    """Finds double and triple tops and bottoms.

    Parameters
    ----------
    equal_tol_atr:
        How far apart the tested extremes may be and still count as the same
        level.
    min_separation_bars:
        Minimum bars between the first and last test. Murphy's "at least a month"
        translated to bar count; raise it on faster timeframes if the desk starts
        seeing trivially small formations.
    """

    kinds = (
        PatternKind.DOUBLE_TOP,
        PatternKind.DOUBLE_BOTTOM,
        PatternKind.TRIPLE_TOP,
        PatternKind.TRIPLE_BOTTOM,
    )

    def __init__(
        self,
        equal_tol_atr: float = 0.9,
        min_separation_bars: int = 20,
        min_height_atr: float = 1.5,
        penetration_pct: float = 0.01,
        confirm_bars: int = 1,
        expire_after: int = 80,
        scan_pivots: int = 9,
        max_tracked: int = 8,
    ) -> None:
        self.equal_tol_atr = equal_tol_atr
        self.min_separation_bars = min_separation_bars
        self.min_height_atr = min_height_atr
        self.expire_after = expire_after
        self.scan_pivots = scan_pivots
        self.max_tracked = max_tracked
        self._filter = PenetrationFilter(pct=penetration_pct, confirm_bars=confirm_bars)
        self._tracked: dict[tuple, Pattern] = {}

    # ----------------------------------------------------------------- update
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

        self._evict()
        return changed

    def _evict(self) -> None:
        for key, pattern in list(self._tracked.items()):
            if pattern.stage.is_dead:
                del self._tracked[key]
                self._filter.reset(key)
        if len(self._tracked) > self.max_tracked:
            ordered = sorted(self._tracked.items(), key=lambda kv: kv[1].detected_index)
            for key, _ in ordered[: len(self._tracked) - self.max_tracked]:
                del self._tracked[key]
                self._filter.reset(key)

    @property
    def active(self) -> list[Pattern]:
        return [p for p in self._tracked.values() if not p.stage.is_dead]

    # ------------------------------------------------------------ recognition
    def _recognise(self, ctx: DetectorContext) -> list[Pattern]:
        pivots = ctx.confirmed[-self.scan_pivots:]
        # Score every candidate window, then keep only the best reading per side.
        # Registering every window that qualifies double-counts the same price
        # action - a triple top contains two double tops, and each start offset
        # produces another near-duplicate - which inverts the natural frequency of
        # the formations and floods the book. Murphy's own tiebreak applies: more
        # tests over a longer span is the more significant formation.
        best: dict[bool, tuple[tuple[int, int], Pattern]] = {}
        for size in (5, 3):
            if len(pivots) < size:
                continue
            for start in range(len(pivots) - size, -1, -1):
                window = pivots[start : start + size]
                if not _alternating(window):
                    continue
                pattern = self._try_window(window, ctx)
                if pattern is None:
                    continue
                is_top = pattern.kind.is_top
                score = (len(pattern.pivots), pattern.bars)
                current = best.get(is_top)
                if current is None or score > current[0]:
                    best[is_top] = (score, pattern)
        return [entry[1] for entry in best.values()]

    def _try_window(
        self, window: list[SwingPoint], ctx: DetectorContext
    ) -> Pattern | None:
        atr = ctx.atr_or()
        is_top = window[0].kind is SwingKind.PEAK
        tests = window[::2]      # the repeated tests of the level
        reactions = window[1::2]  # the intervening counter-swings
        if len(tests) < 2 or not reactions:
            return None

        # All tests must sit at effectively the same level.
        tol = self.equal_tol_atr * atr
        anchor = tests[0].price
        if not all(similar(anchor, t.price, tol) for t in tests[1:]):
            return None

        # Murphy's time filter: the tests must be meaningfully far apart.
        span = tests[-1].index - tests[0].index
        if span < self.min_separation_bars:
            return None

        # The operative level is the reaction extreme that price must clear.
        # With two reactions (triple), the whole base has to give way, so it is
        # the lower trough at a top and the higher peak at a bottom.
        if is_top:
            boundary_pivot = min(reactions, key=lambda p: p.price)
        else:
            boundary_pivot = max(reactions, key=lambda p: p.price)
        level = boundary_pivot.price

        peak_level = max(t.price for t in tests) if is_top else min(t.price for t in tests)
        height = abs(peak_level - level)
        if height < self.min_height_atr * atr:
            return None

        detected = max(p.confirmed_index for p in window)
        triple = len(tests) >= 3
        if is_top:
            kind = PatternKind.TRIPLE_TOP if triple else PatternKind.DOUBLE_TOP
            bias = Bias.BEARISH
            stop = max(t.price for t in tests)
        else:
            kind = PatternKind.TRIPLE_BOTTOM if triple else PatternKind.DOUBLE_BOTTOM
            bias = Bias.BULLISH
            stop = min(t.price for t in tests)

        pattern = Pattern(
            kind=kind,
            symbol=ctx.symbol,
            bias=bias,
            pivots=list(window),
            start_index=window[0].index,
            start_ts=window[0].ts,
            detected_index=detected,
            height=height,
            # A horizontal boundary: the reaction level itself, no slope.
            boundary=level,
            boundary_slope=0.0,
            stage=PatternStage.FORMING,
            stop=stop,
            notes=self._notes(tests, span, is_top),
        )
        pattern.formation_volume_ratio = ctx.volume_ctx.formation_ratio(
            pattern.start_index, pattern.end_index, ctx.index
        )
        return pattern

    def _notes(self, tests: list[SwingPoint], span: int, is_top: bool) -> str:
        bits = [f"{len(tests)} tests over {span} bars"]
        # Murphy: the second peak sometimes falls short of the first, which is
        # itself a sign the buyers are weakening.
        if len(tests) == 2:
            first, second = tests
            if is_top and second.price < first.price:
                bits.append("second peak fell short")
            elif not is_top and second.price > first.price:
                bits.append("second trough held higher")
        if tests[0].volume > 0 and tests[-1].volume > 0:
            if tests[-1].volume < tests[0].volume:
                bits.append("lighter volume on the retest")
        return "; ".join(bits)


def _alternating(window: list[SwingPoint]) -> bool:
    return all(a.kind is not b.kind for a, b in zip(window, window[1:]))
