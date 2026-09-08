"""The candle reader: one bar in, validated candle signals out.

Holds the rolling scale that makes "long" and "small" meaningful, and enforces the
rule that governs the whole chapter:

    "You cannot have a bullish reversal pattern in an uptrend. You can have a
    series of candlesticks that resemble the bullish pattern, but if the trend is
    up, it is not a bullish Japanese candle pattern."

So a shape that looks like a hammer in an uptrend is not reported as a hammer. This
is the single most common way candlestick screens generate noise, and it is why
:meth:`CandleReader.update` requires the prevailing trend as an argument rather
than treating it as optional context.
"""

from __future__ import annotations

from collections import deque

from quantdesk.analysis.candles.geometry import (
    Candle,
    CandleKind,
    CandleSignal,
    SizeContext,
)
from quantdesk.analysis.candles.pairs import pair_strength, read_pairs
from quantdesk.analysis.candles.single import read_single, single_strength
from quantdesk.analysis.candles.triples import read_triples, triple_strength
from quantdesk.analysis.patterns.base import Bias
from quantdesk.core.types import Bar
from quantdesk.features.indicators import ATR, RollingWindow


class CandleReader:
    """Incremental candlestick reader for one symbol.

    Parameters
    ----------
    scale_period:
        Lookback for the average body, range and volume that define "long" and
        "small". Long enough to be stable, short enough to track a volatility
        regime change.
    require_trend_opposition:
        Murphy's rule. Leave on. Turning it off produces many more signals, almost
        all of them meaningless, and is provided only for measuring how much the
        rule is actually filtering.
    min_strength:
        Floor on reported signal quality, to keep marginal shapes out of the book.
    """

    def __init__(
        self,
        symbol: str,
        scale_period: int = 20,
        atr_period: int = 14,
        require_trend_opposition: bool = True,
        min_strength: float = 0.35,
        history: int = 40,
    ) -> None:
        self.symbol = symbol
        self.require_trend_opposition = require_trend_opposition
        self.min_strength = min_strength
        self._body = RollingWindow(scale_period)
        self._range = RollingWindow(scale_period)
        self._volume = RollingWindow(scale_period)
        self._atr = ATR(atr_period)
        self._candles: deque[Candle] = deque(maxlen=4)
        self._index = -1
        self._signals: deque[CandleSignal] = deque(maxlen=history)
        self._last: list[CandleSignal] = []

    # ----------------------------------------------------------------- update
    def update(self, bar: Bar, prior_trend: Bias = Bias.NEUTRAL) -> list[CandleSignal]:
        """Feed one closed bar plus the prevailing trend. Returns new signals."""
        self._index += 1
        candle = Candle.from_bar(self._index, bar)

        # Scale is measured on history *before* this bar, so the current candle is
        # judged against what came before it rather than partly against itself.
        size = SizeContext(
            avg_body=self._body.mean,
            avg_range=self._range.mean,
            atr=self._atr.value,
            avg_volume=self._volume.mean,
        )
        self._body.update(candle.body)
        self._range.update(candle.range)
        self._volume.update(candle.volume)
        self._atr.update(bar.high, bar.low, bar.close)
        self._candles.append(candle)

        if size.avg_body <= 0:
            self._last = []
            return []

        found: list[tuple[CandleKind, float]] = []

        for kind in read_single(candle, size, prior_trend):
            found.append((kind, single_strength(kind, candle, size)))

        if len(self._candles) >= 2:
            prev = self._candles[-2]
            for kind in read_pairs(prev, candle, size):
                found.append((kind, pair_strength(kind, prev, candle, size)))

        if len(self._candles) >= 3:
            a, b, c = self._candles[-3], self._candles[-2], candle
            for kind in read_triples(a, b, c, size):
                found.append((kind, triple_strength(kind, a, b, c, size)))

        out: list[CandleSignal] = []
        vol_ratio = size.volume_ratio(candle)
        for kind, strength in found:
            signal = CandleSignal(
                kind=kind,
                symbol=self.symbol,
                index=self._index,
                ts=bar.ts,
                bias=kind.bias,
                prior_trend=prior_trend,
                strength=strength,
                sessions=kind.sessions,
                volume_ratio=vol_ratio,
            )
            if not self._admits(signal):
                continue
            self._finalise(signal)
            if signal.strength < self.min_strength:
                continue
            out.append(signal)
            self._signals.append(signal)

        # Strongest first, and prefer multi-session readings on ties: a three-day
        # star is more evidence than a one-day shadow.
        out.sort(key=lambda s: (s.strength, s.sessions), reverse=True)
        self._last = out
        return out

    # ------------------------------------------------------------------ gating
    def _admits(self, signal: CandleSignal) -> bool:
        """Apply Murphy's trend rule and drop the merely decorative."""
        if signal.bias is Bias.NEUTRAL:
            # Indecision candles carry no direction, so the trend rule cannot
            # apply. They are kept because a doji at a resistance level is real
            # information, just not directional on its own.
            return True
        if not self.require_trend_opposition:
            return True
        if signal.kind.is_reversal:
            # A reversal must have something to reverse.
            return signal.prior_trend is signal.bias.opposite
        # Continuation patterns must agree with the trend they continue.
        return signal.prior_trend is signal.bias

    def _finalise(self, signal: CandleSignal) -> None:
        """Adjust strength for volume and record why."""
        notes: list[str] = []
        # Murphy's asymmetry from the price-pattern chapters carries over: volume
        # matters more on the upside, so a bullish reversal on heavy trade is worth
        # more, while a bearish one does not need it.
        if signal.volume_ratio >= 1.5:
            signal.strength = min(1.0, signal.strength * 1.15)
            notes.append(f"volume x{signal.volume_ratio:.1f}")
        elif signal.volume_ratio < 0.7 and signal.bias is Bias.BULLISH:
            signal.strength *= 0.85
            notes.append("light volume on a bullish signal")
        if signal.sessions >= 3:
            notes.append(f"{signal.sessions}-session pattern")
        signal.notes = "; ".join(notes)

    # -------------------------------------------------------------- accessors
    @property
    def last(self) -> list[CandleSignal]:
        """Signals from the most recent bar."""
        return list(self._last)

    @property
    def history(self) -> list[CandleSignal]:
        return list(self._signals)

    @property
    def bar_index(self) -> int:
        return self._index

    def recent(self, within_bars: int = 5) -> list[CandleSignal]:
        """Signals from the last few bars, since candle signals decay quickly."""
        cutoff = self._index - within_bars
        return [s for s in self._signals if s.index >= cutoff]

    def net_bias(self, within_bars: int = 3) -> float:
        """Strength-weighted candle read in [-1, 1] over the recent window.

        Candlestick signals are short-horizon by nature - Murphy places them among
        the near-term tools - so the default window is deliberately tight.
        """
        signals = [s for s in self.recent(within_bars) if s.bias is not Bias.NEUTRAL]
        if not signals:
            return 0.0
        total = sum(s.strength * s.bias.sign for s in signals)
        weight = sum(s.strength for s in signals)
        if weight <= 0:
            return 0.0
        return max(-1.0, min(1.0, total / weight))

    def summary(self) -> list[str]:
        return [s.describe() for s in self._last]
