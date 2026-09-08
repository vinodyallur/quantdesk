"""Triangles, wedges and the broadening formation.

Five formations, one detector, because they are all *two boundary lines fitted
through pivots* and differ only in how those lines are oriented. Splitting them
would mean fitting the same lines five times and risking five slightly different
answers.

The classification, straight from chapter 6:

============================ ================================ ==============
shape                        lines                            resolution
============================ ================================ ==============
symmetrical triangle         upper down, lower up             prior trend
ascending triangle           upper flat, lower up             bullish
descending triangle          upper down, lower flat           bearish
wedge                        both slope the same way          against slant
broadening formation         lines diverge                    bearish
============================ ================================ ==============

Requirements enforced:

* **Minimum four reversal points.** "It always takes two points to draw a
  trendline. Therefore, in order to draw two converging trendlines, each line
  must be touched at least twice."
* **The apex is a deadline.** Expect the breakout between two-thirds and
  three-quarters of the base-to-apex width; beyond three-quarters the triangle
  "begins to lose its potency". :attr:`Pattern.notes` records where in that
  window the shape currently sits, and the pattern expires past the apex.
* **Measuring.** Height of the base projected from the breakout point - the
  method Murphy says he prefers.
* **Volume.** Contracts as the swings narrow, expands on the break. Captured by
  ``formation_volume_ratio`` and ``breakout_volume_ratio``.
* **The wedge slants against the prevailing trend**, which is why a falling
  wedge is bullish and a rising wedge bearish. Its bias comes from its own slant,
  not from context.
"""

from __future__ import annotations

from collections import deque

from quantdesk.analysis.patterns.base import (
    Bias,
    DetectorContext,
    Pattern,
    PatternDetector,
    PatternKind,
    PatternStage,
    PenetrationFilter,
)
from quantdesk.analysis.patterns.geometry import (
    APEX_WINDOW_END,
    APEX_WINDOW_START,
    Line,
    apex_progress,
    best_fit_line,
    converging,
    diverging,
    fit_error,
    gap_at,
    intersection,
    slant,
)
from quantdesk.analysis.patterns.lifecycle import advance
from quantdesk.analysis.swings import SwingKind, SwingPoint


class TriangleDetector(PatternDetector):
    """Fits converging and diverging boundary pairs and classifies them.

    Parameters
    ----------
    max_fit_atr:
        Largest mean deviation of pivots from their fitted line, in ATR, before
        the shape is rejected. This is the mechanical stand-in for a chartist
        deciding the pivots do not really line up.
    flat_tol_atr:
        Slope below which a boundary counts as horizontal, which is what
        separates ascending and descending triangles from the symmetrical case.
    wedge_slant_atr:
        Average slope above which two same-direction converging lines are a
        wedge rather than a badly fitted triangle.
    """

    kinds = (
        PatternKind.SYMMETRICAL_TRIANGLE,
        PatternKind.ASCENDING_TRIANGLE,
        PatternKind.DESCENDING_TRIANGLE,
        PatternKind.RISING_WEDGE,
        PatternKind.FALLING_WEDGE,
        PatternKind.BROADENING,
    )

    def __init__(
        self,
        max_fit_atr: float = 0.6,
        flat_tol_atr: float = 0.06,
        wedge_slant_atr: float = 0.12,
        min_convergence: float = 0.02,
        min_height_atr: float = 2.5,
        max_apex_ratio: float = 3.0,
        min_broadening_points: int = 5,
        containment_atr: float = 0.6,
        max_escape_frac: float = 0.10,
        bar_history: int = 300,
        penetration_pct: float = 0.01,
        confirm_bars: int = 1,
        expire_after: int = 90,
        scan_pivots: int = 9,
        max_tracked: int = 8,
    ) -> None:
        self.max_fit_atr = max_fit_atr
        self.flat_tol_atr = flat_tol_atr
        self.wedge_slant_atr = wedge_slant_atr
        self.min_convergence = min_convergence
        self.min_height_atr = min_height_atr
        self.max_apex_ratio = max_apex_ratio
        self.min_broadening_points = min_broadening_points
        self.containment_atr = containment_atr
        self.max_escape_frac = max_escape_frac
        #: (bar index, high, low) history, for the containment test.
        self._bars: deque[tuple[int, float, float]] = deque(maxlen=bar_history)
        self.expire_after = expire_after
        self.scan_pivots = scan_pivots
        self.max_tracked = max_tracked
        self._filter = PenetrationFilter(pct=penetration_pct, confirm_bars=confirm_bars)
        self._tracked: dict[tuple, Pattern] = {}

    # ----------------------------------------------------------------- update
    def update(self, ctx: DetectorContext) -> list[Pattern]:
        self._bars.append((ctx.index, ctx.high, ctx.low))
        changed: list[Pattern] = []
        for pattern in list(self._tracked.values()):
            if self._past_apex(pattern, ctx):
                pattern.stage = PatternStage.EXPIRED
                pattern.failure_reason = "drifted past the apex without breaking out"
                changed.append(pattern)
                continue
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
        if len(self._tracked) > self.max_tracked:
            ordered = sorted(self._tracked.items(), key=lambda kv: kv[1].detected_index)
            for key, _ in ordered[: len(self._tracked) - self.max_tracked]:
                del self._tracked[key]
                self._filter.reset(key)
        return changed

    def _past_apex(self, pattern: Pattern, ctx: DetectorContext) -> bool:
        """A still-forming converging pattern that ran out of room is finished."""
        if pattern.stage is not PatternStage.FORMING:
            return False
        if pattern.apex_index is None:
            return False
        return ctx.index > pattern.apex_index

    @property
    def active(self) -> list[Pattern]:
        return [p for p in self._tracked.values() if not p.stage.is_dead]

    # ------------------------------------------------------------ recognition
    def _recognise(self, ctx: DetectorContext) -> list[Pattern]:
        pivots = ctx.confirmed[-self.scan_pivots:]
        atr = ctx.atr_or()
        # Try the widest formation first, then drop the oldest pivot and retry.
        # A single pivot predating the triangle drags the fitted boundary badly
        # off, and a chartist simply would not draw the line through it - the
        # triangle starts where the consolidation starts, not where the scan
        # window happens to begin. Widest-first keeps Murphy's preference for the
        # larger, more significant reading.
        # Take the *best* window, not the first that fits. Trying nine windows and
        # accepting any success turns the scan into a multiple-comparisons machine
        # that finds a triangle on almost every new pivot. Scoring them and
        # keeping one reading encodes Murphy's own preference: more reversal
        # points and a wider base mean a more significant formation.
        best: list[Pattern] = []
        best_score = (-1, -1)
        for drop in range(len(pivots)):
            window = pivots[drop:]
            peaks = [p for p in window if p.kind is SwingKind.PEAK]
            troughs = [p for p in window if p.kind is SwingKind.TROUGH]
            # Murphy's minimum: two touches per line, so four reversal points.
            if len(peaks) < 2 or len(troughs) < 2:
                break
            found = self._build(peaks, troughs, atr, ctx)
            if not found:
                continue
            score = (len(found[0].pivots), found[0].bars)
            if score > best_score:
                best_score = score
                best = found
        return best

    def _build(
        self,
        peaks: list[SwingPoint],
        troughs: list[SwingPoint],
        atr: float,
        ctx: DetectorContext,
    ) -> list[Pattern]:
        upper = best_fit_line(peaks)
        lower = best_fit_line(troughs)
        if upper is None or lower is None:
            return []
        if fit_error(upper, peaks, atr) > self.max_fit_atr:
            return []
        if fit_error(lower, troughs, atr) > self.max_fit_atr:
            return []

        used = sorted(peaks + troughs, key=lambda p: p.index)
        base_index = used[0].index
        height = gap_at(upper, lower, base_index)
        if height < self.min_height_atr * atr:
            return []
        # Boundaries must not already have crossed; that is not a triangle, it is
        # a badly fitted pair of lines.
        if gap_at(upper, lower, used[-1].index) <= 0:
            return []
        # The decisive test. With the four-point minimum each line is fitted
        # through exactly two pivots, so it passes through them perfectly and the
        # fit error says nothing at all. What actually distinguishes a triangle
        # from two arbitrary converging lines is that price *stayed inside them*.
        # Without this check, any two descending peaks plus two rising troughs
        # register as a coil, which on noisy data is most of the time.
        if not self._contained(upper, lower, base_index, used[-1].index, atr):
            return []

        # Murphy identifies triangles by the *progression of the reversal points*,
        # not by an average slope: point 3 lower than point 1, point 4 higher than
        # point 2. A fitted line can slope down while the peaks themselves jitter
        # up and down, and that is not a triangle, so the touches are tested for
        # strict monotonicity directly.
        peaks_dir = _touch_direction(peaks, atr, self.flat_tol_atr)
        troughs_dir = _touch_direction(troughs, atr, self.flat_tol_atr)
        if peaks_dir is None or troughs_dir is None:
            return []

        classified = self._classify(
            upper, lower, atr, ctx.prior_trend, len(used), peaks_dir, troughs_dir
        )
        if classified is None:
            return []
        kind, biases = classified

        cross = intersection(upper, lower)
        apex_index = int(round(cross[0])) if cross is not None else None
        # The apex is a deadline, so it has to be a *near* one. Lines that meet
        # hundreds of bars out are near-parallel noise, not a coil, and would
        # make Murphy's two-thirds timing rule meaningless.
        if apex_index is not None and kind is not PatternKind.BROADENING:
            width = max(1, used[-1].index - base_index)
            if apex_index - base_index > self.max_apex_ratio * width:
                return []

        detected = max(p.confirmed_index for p in used)
        progress = apex_progress(upper, lower, base_index, ctx.index)

        out: list[Pattern] = []
        for bias in biases:
            line = upper if bias is Bias.BULLISH else lower
            out.append(
                Pattern(
                    kind=kind,
                    symbol=ctx.symbol,
                    bias=bias,
                    pivots=used,
                    start_index=base_index,
                    start_ts=used[0].ts,
                    detected_index=detected,
                    height=height,
                    boundary=line.value_at(detected),
                    boundary_slope=line.slope,
                    stage=PatternStage.FORMING,
                    apex_index=apex_index,
                    stop=lower.value_at(detected)
                    if bias is Bias.BULLISH
                    else upper.value_at(detected),
                    notes=self._notes(kind, progress, len(peaks), len(troughs), biases),
                )
            )
        for pattern in out:
            pattern.formation_volume_ratio = ctx.volume_ctx.formation_ratio(
                pattern.start_index, pattern.end_index, ctx.index
            )
        return out

    def _contained(
        self,
        upper: Line,
        lower: Line,
        base_index: int,
        end_index: int,
        atr: float,
    ) -> bool:
        """Did price stay between the boundaries across the formation?

        A tolerance is allowed because the lines are fitted through the extremes,
        so roughly half the extremes sit marginally outside by construction, and
        Murphy draws through the full range by eye rather than to the tick. A
        small fraction of genuine escapes is also tolerated - real triangles have
        the odd overshoot - but a shape price repeatedly traded outside of is not
        a triangle.
        """
        tol = self.containment_atr * atr
        escapes = 0
        total = 0
        for index, high, low in self._bars:
            if index < base_index or index > end_index:
                continue
            total += 1
            if high > upper.value_at(index) + tol or low < lower.value_at(index) - tol:
                escapes += 1
        if total < 6:
            return False
        return escapes / total <= self.max_escape_frac

    def _classify(
        self,
        upper: Line,
        lower: Line,
        atr: float,
        prior: Bias,
        n_points: int,
        peaks_dir: int,
        troughs_dir: int,
    ) -> tuple[PatternKind, tuple[Bias, ...]] | None:
        """Name the shape from how its reversal points progress.

        ``peaks_dir`` and ``troughs_dir`` are +1 rising, -1 falling, 0 level.
        Every one of Murphy's five variants is a distinct combination of the two,
        which makes the classification a lookup rather than a pile of slope
        heuristics.
        """
        if diverging(upper, lower, atr, self.min_convergence):
            # The megaphone. Murphy calls it "relatively rare" and specifically a
            # *top*, so it needs more evidence than the others: extra reversal
            # points, and an uptrend to top out of. Without those gates a
            # diverging pair of lines is one of the easiest things to find in
            # noise, and it swamps everything else.
            if n_points < self.min_broadening_points:
                return None
            if prior is not Bias.BULLISH:
                return None
            # The megaphone widens: higher highs against lower lows.
            if not (peaks_dir > 0 and troughs_dir < 0):
                return None
            return PatternKind.BROADENING, (Bias.BEARISH,)

        if not converging(upper, lower, atr, self.min_convergence):
            return None

        # Flat top, rising bottom.
        if peaks_dir == 0 and troughs_dir > 0:
            return PatternKind.ASCENDING_TRIANGLE, (Bias.BULLISH,)
        # Falling top, flat bottom.
        if peaks_dir < 0 and troughs_dir == 0:
            return PatternKind.DESCENDING_TRIANGLE, (Bias.BEARISH,)

        # Both boundaries leaning the same way is a wedge. Murphy: it "slants
        # against the prevailing trend", so a falling wedge is bullish and a
        # rising wedge bearish - the bias is the opposite of the slant, and comes
        # from the shape itself rather than from context.
        if peaks_dir > 0 and troughs_dir > 0:
            if abs(slant(upper, lower, atr)) < self.wedge_slant_atr:
                return None
            return PatternKind.RISING_WEDGE, (Bias.BEARISH,)
        if peaks_dir < 0 and troughs_dir < 0:
            if abs(slant(upper, lower, atr)) < self.wedge_slant_atr:
                return None
            return PatternKind.FALLING_WEDGE, (Bias.BULLISH,)

        # Anything left that is not falling-top-over-rising-bottom is not a
        # triangle at all.
        if not (peaks_dir < 0 and troughs_dir > 0):
            return None

        # Symmetrical: resolves with the prior trend. Without a prior trend there
        # is no forecast, so both resolutions are tracked and whichever breaks
        # first is the one that mattered.
        if prior is Bias.BULLISH:
            return PatternKind.SYMMETRICAL_TRIANGLE, (Bias.BULLISH,)
        if prior is Bias.BEARISH:
            return PatternKind.SYMMETRICAL_TRIANGLE, (Bias.BEARISH,)
        return PatternKind.SYMMETRICAL_TRIANGLE, (Bias.BULLISH, Bias.BEARISH)

    def _notes(
        self,
        kind: PatternKind,
        progress: float,
        n_peaks: int,
        n_troughs: int,
        biases: tuple[Bias, ...],
    ) -> str:
        bits = [f"{n_peaks + n_troughs} reversal points"]
        if progress != float("inf"):
            bits.append(f"{progress:.0%} to apex")
            if APEX_WINDOW_START <= progress <= APEX_WINDOW_END:
                bits.append("in Murphy's 2/3-3/4 breakout window")
            elif progress > APEX_WINDOW_END:
                bits.append("past 3/4, losing potency")
        if len(biases) > 1:
            bits.append("no prior trend, tracking both resolutions")
        return "; ".join(bits)


def _touch_direction(
    points: list[SwingPoint], atr: float, flat_tol: float
) -> int | None:
    """How a boundary's touches progress: +1 rising, -1 falling, 0 level.

    Returns None when they do not progress consistently, which disqualifies the
    shape. This is the structural test Murphy applies by eye when he checks that
    each successive peak is lower than the last and each trough higher.

    ``flat_tol`` is in ATR per bar, so the level case scales with how far apart
    the touches are: two peaks twenty bars apart may drift further and still count
    as one flat boundary than two peaks three bars apart.
    """
    if len(points) < 2:
        return None
    rising = falling = level = True
    for a, b in zip(points, points[1:]):
        bars = max(1, b.index - a.index)
        tol = flat_tol * atr * bars
        delta = b.price - a.price
        if delta <= tol:
            rising = False
        if delta >= -tol:
            falling = False
        if abs(delta) > tol:
            level = False
    if level:
        return 0
    if rising:
        return 1
    if falling:
        return -1
    return None
