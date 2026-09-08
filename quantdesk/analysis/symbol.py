"""Per-symbol analysis: the whole Murphy stack behind one ``update(bar)`` call.

Every technique in the book depends on the ones before it - patterns need pivots,
pivots need ATR, divergences need both an oscillator and the pivots to compare it
against - so the update order here is a dependency order, not an arbitrary one:

1. ATR and the oscillators, which need only the bar.
2. Swing pivots, which need ATR to scale their threshold.
3. Trend, levels, trendlines and retracements, all of which read pivots.
4. Patterns, candles and Elliott, which read all of the above.
5. Divergence, which needs oscillator readings sampled *at* pivots.

Divergence deserves a note. Murphy treats it as one of the most valuable oscillator
signals, and it cannot be computed from an oscillator alone: it is a disagreement
between successive *price* pivots and the oscillator's value at those same pivots.
So this class samples each oscillator when a pivot is confirmed, which is the only
way to compare like with like.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quantdesk.analysis.candles import CandleReader, CandleSignal
from quantdesk.analysis.dow import TrendState, classify_degree, classify_trend
from quantdesk.analysis.elliott import ElliottCounter
from quantdesk.analysis.gaps import GapTracker
from quantdesk.analysis.levels import LevelBook
from quantdesk.analysis.patterns import Bias, Pattern, PatternScanner
from quantdesk.analysis.retracement import RetracementMap, build_retracements
from quantdesk.analysis.swings import SwingDetector, SwingKind, SwingPoint
from quantdesk.analysis.timeframes import Timeframe, TimeframeStack
from quantdesk.analysis.trendlines import TrendlineTracker
from quantdesk.core.types import Bar
from quantdesk.features.indicators import (
    ATR,
    DMI,
    MACD,
    OBV,
    RSI,
    SMA,
    BollingerBands,
    RollingWindow,
    Stochastic,
)


@dataclass(slots=True)
class DivergenceRead:
    """A disagreement between price pivots and an oscillator at those pivots."""

    name: str
    bias: Bias
    kind: str
    """'regular' when the oscillator fails to confirm a new price extreme."""
    price_from: float = 0.0
    price_to: float = 0.0
    osc_from: float = 0.0
    osc_to: float = 0.0

    def describe(self) -> str:
        return (
            f"{self.name} {self.kind} {self.bias.value} divergence "
            f"(price {self.price_from:.4g}->{self.price_to:.4g}, "
            f"osc {self.osc_from:.1f}->{self.osc_to:.1f})"
        )


class SymbolAnalysis:
    """The full analytical picture for one instrument, updated bar by bar."""

    def __init__(
        self,
        symbol: str,
        atr_period: int = 14,
        timeframes: list[Timeframe] | None = None,
        base_timeframe: Timeframe = Timeframe.M5,
        ma_fast: int = 20,
        ma_slow: int = 50,
    ) -> None:
        self.symbol = symbol
        # --- primitives
        self.atr = ATR(atr_period)
        self.rsi = RSI(14)
        self.macd = MACD(12, 26, 9)
        self.stoch = Stochastic(14, 3, 3)
        self.dmi = DMI(14)
        self.bbands = BollingerBands(20, 2.0)
        self.obv = OBV()
        self.ma_fast = SMA(ma_fast)
        self.ma_slow = SMA(ma_slow)
        self._volume = RollingWindow(30)
        # --- structure
        self.swings = SwingDetector(atr_period=atr_period)
        self.levels = LevelBook()
        self.trendlines = TrendlineTracker()
        self.gaps = GapTracker()
        # --- higher-order readings
        self.patterns = PatternScanner(symbol, atr_period=atr_period, swings=self.swings)
        self.candles = CandleReader(symbol)
        self.elliott = ElliottCounter(symbol)
        self.timeframes = TimeframeStack(
            symbol,
            timeframes or [base_timeframe, Timeframe.H1, Timeframe.H4],
            base=base_timeframe,
        )
        # --- state
        self.trend: TrendState | None = None
        self.retracements: RetracementMap | None = None
        self.divergences: list[DivergenceRead] = []
        self.last_bar: Bar | None = None
        self._index = -1
        self._prev_bar: Bar | None = None
        #: Oscillator readings sampled at each confirmed pivot, for divergence.
        self._pivot_osc: list[tuple[SwingPoint, dict[str, float]]] = []
        self._ma_fast_prev = 0.0
        self._ma_slow_prev = 0.0

    # ----------------------------------------------------------------- update
    def update(self, bar: Bar) -> None:
        """Feed one closed bar through the whole stack in dependency order."""
        self._index += 1
        self.last_bar = bar

        # 1. Bar-only indicators.
        atr = self.atr.update(bar.high, bar.low, bar.close)
        self.rsi.update(bar.close)
        self.macd.update(bar.close)
        self.stoch.update(bar.high, bar.low, bar.close)
        self.dmi.update(bar.high, bar.low, bar.close)
        self.bbands.update(bar.close)
        self.obv.update(bar.close, bar.volume)
        self._ma_fast_prev = self.ma_fast.value
        self._ma_slow_prev = self.ma_slow.value
        self.ma_fast.update(bar.close)
        self.ma_slow.update(bar.close)
        self._volume.update(bar.volume)

        # 2. Pivots, which the pattern scanner shares rather than duplicating.
        pivot = self.swings.update(bar)
        confirmed = self.swings.confirmed

        # 3. Structure built on pivots.
        self.trend = classify_trend(confirmed, bar.close)
        if pivot is not None:
            self.levels.add_pivot(pivot, atr)
            self._sample_oscillators(pivot)
        self.levels.update_price(bar.close, self._index)
        self.trendlines.fit(confirmed)
        self.trendlines.update(self._index, bar.high, bar.low, bar.close, atr)
        self.gaps.update(bar, atr, self._volume.mean)
        if len(confirmed) >= 2:
            self.retracements = build_retracements(confirmed[-2], confirmed[-1])

        # 4. Higher-order readings.
        self.patterns.update(bar)
        self.candles.update(bar, self.trend_bias)
        self.elliott.update(self._index, confirmed)
        self.timeframes.update(bar)

        # 5. Divergence needs the pivot-sampled oscillator history.
        self.divergences = self._read_divergences()
        self._prev_bar = bar

    def _sample_oscillators(self, pivot: SwingPoint) -> None:
        """Record oscillator values at a confirmed pivot, for later comparison."""
        self._pivot_osc.append(
            (
                pivot,
                {
                    "RSI": self.rsi.value,
                    "MACD": self.macd.macd,
                    "Stochastic": self.stoch.k,
                },
            )
        )
        if len(self._pivot_osc) > 40:
            del self._pivot_osc[:-40]

    def _read_divergences(self) -> list[DivergenceRead]:
        """Compare the last two same-side pivots against the oscillators.

        Bearish: price makes a higher peak, the oscillator makes a lower one - the
        advance is losing momentum. Bullish is the mirror at troughs.
        """
        out: list[DivergenceRead] = []
        for kind, bias_if_diverging in (
            (SwingKind.PEAK, Bias.BEARISH),
            (SwingKind.TROUGH, Bias.BULLISH),
        ):
            same = [(p, o) for p, o in self._pivot_osc if p.kind is kind][-2:]
            if len(same) < 2:
                continue
            (prev_p, prev_o), (last_p, last_o) = same
            for name in ("RSI", "MACD", "Stochastic"):
                pv, lv = prev_o.get(name, 0.0), last_o.get(name, 0.0)
                if kind is SwingKind.PEAK:
                    diverging = last_p.price > prev_p.price and lv < pv
                else:
                    diverging = last_p.price < prev_p.price and lv > pv
                if diverging:
                    out.append(
                        DivergenceRead(
                            name=name,
                            bias=bias_if_diverging,
                            kind="regular",
                            price_from=prev_p.price,
                            price_to=last_p.price,
                            osc_from=pv,
                            osc_to=lv,
                        )
                    )
        return out

    # -------------------------------------------------------------- accessors
    @property
    def index(self) -> int:
        return self._index

    @property
    def price(self) -> float:
        return self.last_bar.close if self.last_bar else 0.0

    @property
    def trend_bias(self) -> Bias:
        if self.trend is None:
            return Bias.NEUTRAL
        return Bias.from_sign(self.trend.direction.sign)

    @property
    def degree(self) -> str:
        """Murphy's major / intermediate / minor trend degree.

        His boundaries are stated in *time* - roughly six months for major, three
        weeks to six months for intermediate - so converting a bar count into a
        degree needs to know how many bars make a day on this timeframe.
        """
        if self.trend is None:
            return "unknown"
        return classify_degree(self.trend.span_bars, self.bars_per_day).value

    @property
    def bars_per_day(self) -> float:
        """Bars per 24 hours on the base timeframe. Crypto trades continuously."""
        return 1440.0 / max(1, self.timeframes.base.minutes)

    @property
    def ma_slope_bias(self) -> Bias:
        """"Which way are the moving averages pointing?" - checklist item."""
        if not self.ma_slow.ready:
            return Bias.NEUTRAL
        fast_up = self.ma_fast.value > self._ma_fast_prev
        slow_up = self.ma_slow.value > self._ma_slow_prev
        stacked_up = self.ma_fast.value > self.ma_slow.value
        if fast_up and slow_up and stacked_up:
            return Bias.BULLISH
        if not fast_up and not slow_up and not stacked_up:
            return Bias.BEARISH
        return Bias.NEUTRAL

    @property
    def volume_avg(self) -> float:
        return self._volume.mean

    @property
    def volume_ratio(self) -> float:
        if self.last_bar is None or self._volume.mean <= 0:
            return 1.0
        return self.last_bar.volume / self._volume.mean

    @property
    def actionable_patterns(self) -> list[Pattern]:
        return self.patterns.actionable()

    @property
    def candle_signals(self) -> list[CandleSignal]:
        return self.candles.last

    @property
    def ready(self) -> bool:
        """Enough history for the stack to be worth reading."""
        return self.atr.ready and self.ma_slow.ready and len(self.swings) >= 2

    def headroom(self, long_side: bool) -> float:
        """ATR of clear space to the next opposing level."""
        return self.levels.headroom(self.price, self.atr.value, long_side)
