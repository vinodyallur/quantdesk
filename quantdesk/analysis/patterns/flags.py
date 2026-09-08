"""Flags and pennants: the half-mast patterns.

These are the shortest formations Murphy covers and the only ones that cannot be
built from swing pivots, because the whole point is that they are *too small and
too brief* to register as swings. So this detector works from a rolling buffer of
raw bars instead.

His description, which is what the tests below encode:

* A **flagpole** first - "the prior sharp advance or decline". No steep move, no
  flag.
* Then a pause of one to three weeks on "very light volume". Downtrend versions
  run even shorter, one to two weeks.
* The **flag** is a parallelogram slanting against the prevailing trend; the
  **pennant** is a small symmetrical triangle. Same meaning, different shape, so
  one detector classifies both.
* Completion is a close through the boundary on heavier volume.
* Measuring: they "fly at half-mast" - the move after the pattern duplicates the
  flagpole. The objective is therefore the pole height projected from the
  breakout, not the pattern's own tiny height.

Durations are expressed in bars rather than weeks. On a daily chart Murphy's one
to three weeks is roughly five to fifteen bars, which is the default; on other
timeframes the pattern's degree scales with the chart, as it does throughout his
framework.
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
from quantdesk.analysis.patterns.lifecycle import advance


class _BarRow:
    """Minimal bar record for the rolling buffer."""

    __slots__ = ("index", "high", "low", "close", "volume")

    def __init__(self, index: int, high: float, low: float, close: float, volume: float):
        self.index = index
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class FlagPennantDetector(PatternDetector):
    """Finds flags and pennants from a rolling window of bars.

    Parameters
    ----------
    min_pole_atr:
        How far the flagpole must travel to count as a "sharp" move.
    min_pole_efficiency:
        Net move divided by the sum of bar ranges over the pole. Near 1.0 means a
        straight run; this is what distinguishes a genuine pole from a choppy
        drift that happens to end higher.
    min_bars, max_bars:
        Consolidation length. Murphy's one to three weeks in bar terms.
    max_depth:
        Ceiling on how much of the pole the consolidation may retrace, as a
        fraction. A pause that gives back most of the pole is not a flag.
    """

    kinds = (PatternKind.FLAG, PatternKind.PENNANT)

    def __init__(
        self,
        pole_lookback: int = 12,
        min_pole_atr: float = 3.0,
        min_pole_efficiency: float = 0.55,
        min_bars: int = 4,
        max_bars: int = 15,
        max_depth: float = 0.5,
        max_formation_volume: float = 1.0,
        penetration_pct: float = 0.005,
        confirm_bars: int = 1,
        max_tracked: int = 6,
    ) -> None:
        self.pole_lookback = pole_lookback
        self.min_pole_atr = min_pole_atr
        self.min_pole_efficiency = min_pole_efficiency
        self.min_bars = min_bars
        self.max_bars = max_bars
        self.max_depth = max_depth
        self.max_formation_volume = max_formation_volume
        self._filter = PenetrationFilter(pct=penetration_pct, confirm_bars=confirm_bars)
        self._buf: deque[_BarRow] = deque(maxlen=pole_lookback + max_bars + 4)
        self._tracked: dict[tuple, Pattern] = {}
        self.max_tracked = max_tracked

    # ----------------------------------------------------------------- update
    def update(self, ctx: DetectorContext) -> list[Pattern]:
        self._buf.append(
            _BarRow(ctx.index, ctx.high, ctx.low, ctx.close, ctx.volume)
        )
        changed: list[Pattern] = []

        for pattern in list(self._tracked.values()):
            # Flags are short-lived by definition, so they expire fast.
            if advance(
                pattern, ctx, self._filter, expire_after=self.max_bars + 5
            ):
                changed.append(pattern)

        # A flag is a single brief pause, so at most one can be live per
        # direction. Without this the sliding pole window re-registers the same
        # consolidation under a new key on every bar.
        live_biases = {p.bias for p in self._tracked.values() if not p.stage.is_dead}
        pattern = self._recognise(ctx)
        if (
            pattern is not None
            and pattern.bias not in live_biases
            and pattern.key not in self._tracked
        ):
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

    # ------------------------------------------------------------ recognition
    def _recognise(self, ctx: DetectorContext) -> Pattern | None:
        atr = ctx.atr_or()
        rows = list(self._buf)
        # Need a pole plus a consolidation of at least the minimum length.
        if len(rows) < self.pole_lookback + self.min_bars:
            return None

        for pause_len in range(self.min_bars, self.max_bars + 1):
            if len(rows) < self.pole_lookback + pause_len:
                continue
            pause = rows[-pause_len:]
            pole = rows[-(pause_len + self.pole_lookback) : -pause_len]
            if len(pole) < 3:
                continue
            found = self._try(pole, pause, atr, ctx)
            if found is not None:
                return found
        return None

    def _try(
        self,
        pole: list[_BarRow],
        pause: list[_BarRow],
        atr: float,
        ctx: DetectorContext,
    ) -> Pattern | None:
        net = pole[-1].close - pole[0].close
        travel = sum(r.high - r.low for r in pole)
        if travel <= 0:
            return None
        if abs(net) < self.min_pole_atr * atr:
            return None
        # Efficiency: a pole is a straight line, not a round trip.
        if abs(net) / travel < self.min_pole_efficiency:
            return None

        up = net > 0
        pole_height = abs(net)
        pause_high = max(r.high for r in pause)
        pause_low = min(r.low for r in pause)
        depth = (
            (pole[-1].close - pause_low) if up else (pause_high - pole[-1].close)
        )
        if depth > self.max_depth * pole_height:
            return None
        # The consolidation must be tight relative to the pole; otherwise this is
        # a new trend leg, not a pause.
        if (pause_high - pause_low) > 0.6 * pole_height:
            return None

        # Volume should dry up during the pause. Murphy calls it "very light".
        pole_vol = sum(r.volume for r in pole) / len(pole)
        pause_vol = sum(r.volume for r in pause) / len(pause)
        vol_ratio = pause_vol / pole_vol if pole_vol > 0 else 1.0
        if vol_ratio > self.max_formation_volume:
            return None

        kind = self._classify(pause)
        bias = Bias.BULLISH if up else Bias.BEARISH
        boundary = pause_high if up else pause_low
        stop = pause_low if up else pause_high

        # Half-mast: the objective is the pole duplicated from the breakout, so it
        # is set here rather than derived from the pattern's own height.
        objective = boundary + pole_height if up else boundary - pole_height

        slope_note = self._slope_note(pause, up)
        return Pattern(
            kind=kind,
            symbol=ctx.symbol,
            bias=bias,
            pivots=[],
            start_index=pole[0].index,
            detected_index=ctx.index,
            height=pole_height,
            boundary=boundary,
            boundary_slope=0.0,
            stage=PatternStage.FORMING,
            stop=stop,
            objective=objective,
            formation_volume_ratio=vol_ratio,
            notes=(
                f"pole {pole_height / atr:.1f} ATR over {len(pole)} bars; "
                f"pause {len(pause)} bars at x{vol_ratio:.2f} volume; "
                f"half-mast target{slope_note}"
            ),
        )

    def _classify(self, pause: list[_BarRow]) -> PatternKind:
        """Narrowing range means a pennant; parallel edges mean a flag."""
        mid = len(pause) // 2
        if mid < 2:
            return PatternKind.FLAG
        early = max(r.high for r in pause[:mid]) - min(r.low for r in pause[:mid])
        late = max(r.high for r in pause[mid:]) - min(r.low for r in pause[mid:])
        if early > 0 and late / early < 0.7:
            return PatternKind.PENNANT
        return PatternKind.FLAG

    def _slope_note(self, pause: list[_BarRow], up: bool) -> str:
        """Murphy expects the pause to slant against the trend."""
        mid = len(pause) // 2
        if mid < 2:
            return ""
        early = sum(r.close for r in pause[:mid]) / mid
        late = sum(r.close for r in pause[mid:]) / (len(pause) - mid)
        counter = late < early if up else late > early
        return "; slants against trend" if counter else "; slants with trend (atypical)"
