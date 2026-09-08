"""Multi-timeframe analysis: aggregation and top-down alignment.

Murphy's procedural rule, stated twice in chapter 16 and running through chapter 8:

    "Work from the long term to the short term."
    "Use intraday charts to fine-tune entry and exit."

That is a hierarchy, not a vote. The higher timeframe decides *what direction is
permitted*; the lower timeframe only decides *when to act on it*. A desk that lets a
five-minute signal trade against the daily trend has inverted his method. So
:class:`TimeframeStack` exposes :meth:`permitted_direction` - derived from the higher
timeframes alone - separately from the timing read.

**The aggregation must not leak.** A higher-timeframe bar is only complete once the
next period has started. Emitting a forming hourly bar to a strategy hands it
knowledge of the future, and it is a subtle bug because the forming bar looks
perfectly ordinary. :class:`BarAggregator` therefore emits a period's bar only when a
base bar from a *later* period arrives, and exposes the in-progress bar separately
under a name that cannot be mistaken for a closed one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from quantdesk.analysis.dow import TrendDirection, TrendState, classify_trend
from quantdesk.analysis.patterns.base import Bias
from quantdesk.analysis.swings import SwingDetector
from quantdesk.core.types import Bar
from quantdesk.features.indicators import ATR


class Timeframe(str, Enum):
    """Timeframes the desk works in, with their length in minutes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

    @property
    def minutes(self) -> int:
        return _MINUTES[self]

    @property
    def delta(self) -> timedelta:
        return timedelta(minutes=self.minutes)

    def floor(self, ts: datetime) -> datetime:
        """Start of the period containing ``ts``.

        Weeks anchor to Monday and days to midnight UTC; everything below that is a
        fixed-width bucket from the epoch, which keeps buckets stable across
        restarts and matches how exchanges label candles.
        """
        if self is Timeframe.W1:
            midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
            return midnight - timedelta(days=midnight.weekday())
        if self is Timeframe.D1:
            return ts.replace(hour=0, minute=0, second=0, microsecond=0)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        elapsed = int((ts - epoch).total_seconds() // 60)
        bucket = elapsed - (elapsed % self.minutes)
        return epoch + timedelta(minutes=bucket)

    def __lt__(self, other: "Timeframe") -> bool:  # type: ignore[override]
        return self.minutes < other.minutes


_MINUTES: dict[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
    Timeframe.W1: 10080,
}


class BarAggregator:
    """Rolls base bars up into one higher timeframe.

    Emits a completed bar only when a base bar belonging to a later period arrives.
    The consequence is a one-base-bar reporting lag, which is correct: at the moment
    an hourly candle's last minute closes, you do not yet know no more trades are
    coming in that hour.
    """

    def __init__(self, timeframe: Timeframe) -> None:
        self.timeframe = timeframe
        self._bucket: datetime | None = None
        self._open = 0.0
        self._high = float("-inf")
        self._low = float("inf")
        self._close = 0.0
        self._volume = 0.0
        self._symbol = ""
        self._count = 0

    def update(self, bar: Bar) -> Bar | None:
        """Feed one base bar. Returns a completed higher-timeframe bar, or None."""
        bucket = self.timeframe.floor(bar.ts)
        completed: Bar | None = None

        if self._bucket is None:
            self._start(bucket, bar)
            return None

        if bucket > self._bucket:
            # A new period has begun, so the previous one is now final.
            completed = self._emit()
            self._start(bucket, bar)
            return completed

        if bucket < self._bucket:
            # Out-of-order bar. Dropping it is safer than rewriting a closed period,
            # which would retroactively change history a strategy already saw.
            return None

        self._high = max(self._high, bar.high)
        self._low = min(self._low, bar.low)
        self._close = bar.close
        self._volume += bar.volume
        self._count += 1
        return None

    def _start(self, bucket: datetime, bar: Bar) -> None:
        self._bucket = bucket
        self._symbol = bar.symbol
        self._open = bar.open
        self._high = bar.high
        self._low = bar.low
        self._close = bar.close
        self._volume = bar.volume
        self._count = 1

    def _emit(self) -> Bar:
        return Bar(
            symbol=self._symbol,
            ts=self._bucket,  # type: ignore[arg-type]
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
        )

    @property
    def forming(self) -> Bar | None:
        """The in-progress period. NOT a closed bar - never feed this to a strategy."""
        if self._bucket is None:
            return None
        return self._emit()

    @property
    def bars_in_period(self) -> int:
        return self._count


@dataclass(slots=True)
class TimeframeRead:
    """Trend state for one timeframe."""

    timeframe: Timeframe
    trend: TrendState | None = None
    atr: float = 0.0
    bars_seen: int = 0
    last_close: float = 0.0

    @property
    def bias(self) -> Bias:
        if self.trend is None:
            return Bias.NEUTRAL
        return Bias.from_sign(self.trend.direction.sign)

    @property
    def ready(self) -> bool:
        """Enough structure to have an opinion."""
        return self.trend is not None and self.trend.pivots_used >= 2


class TimeframeStack:
    """Runs the same structural analysis across several timeframes at once.

    Parameters
    ----------
    timeframes:
        Ordered low to high. The base bar stream feeds the lowest directly and is
        aggregated up into the rest.
    timing_count:
        How many of the lowest timeframes count as "timing" rather than "direction".
        Murphy's split: the long term sets direction, the short term fine-tunes.
    """

    def __init__(
        self,
        symbol: str,
        timeframes: list[Timeframe] | None = None,
        base: Timeframe = Timeframe.M5,
        timing_count: int = 1,
        atr_period: int = 14,
    ) -> None:
        self.symbol = symbol
        self.base = base
        self.timeframes = timeframes or [Timeframe.M5, Timeframe.H1, Timeframe.H4]
        self.timing_count = max(0, min(timing_count, len(self.timeframes) - 1))
        self._aggs: dict[Timeframe, BarAggregator] = {
            tf: BarAggregator(tf) for tf in self.timeframes if tf is not base
        }
        self._swings: dict[Timeframe, SwingDetector] = {
            tf: SwingDetector(atr_period=atr_period) for tf in self.timeframes
        }
        self._atrs: dict[Timeframe, ATR] = {
            tf: ATR(atr_period) for tf in self.timeframes
        }
        self._reads: dict[Timeframe, TimeframeRead] = {
            tf: TimeframeRead(timeframe=tf) for tf in self.timeframes
        }

    # ----------------------------------------------------------------- update
    def update(self, bar: Bar) -> dict[Timeframe, Bar]:
        """Feed one base bar. Returns the higher-timeframe bars that closed."""
        closed: dict[Timeframe, Bar] = {}
        if self.base in self._reads:
            self._ingest(self.base, bar)
            closed[self.base] = bar
        for tf, agg in self._aggs.items():
            done = agg.update(bar)
            if done is not None:
                self._ingest(tf, done)
                closed[tf] = done
        return closed

    def _ingest(self, tf: Timeframe, bar: Bar) -> None:
        atr = self._atrs[tf].update(bar.high, bar.low, bar.close)
        swings = self._swings[tf]
        swings.update(bar)
        read = self._reads[tf]
        read.trend = classify_trend(swings.confirmed, bar.close)
        read.atr = atr
        read.bars_seen += 1
        read.last_close = bar.close

    # -------------------------------------------------------------- alignment
    @property
    def reads(self) -> dict[Timeframe, TimeframeRead]:
        return dict(self._reads)

    @property
    def direction_frames(self) -> list[Timeframe]:
        """The higher timeframes that set the permitted direction."""
        return self.timeframes[self.timing_count :]

    @property
    def timing_frames(self) -> list[Timeframe]:
        """The lower timeframes used only to time entries."""
        return self.timeframes[: self.timing_count]

    def permitted_direction(self) -> Bias:
        """What the higher timeframes allow. This is the gate, not a suggestion.

        Requires the direction-setting timeframes to agree. Disagreement means
        NEUTRAL - stand aside - rather than deferring to the faster one, because
        deferring to the faster one is exactly the inversion Murphy warns against.
        """
        frames = [self._reads[tf] for tf in self.direction_frames]
        ready = [r for r in frames if r.ready]
        if not ready:
            return Bias.NEUTRAL
        biases = {r.bias for r in ready}
        if len(biases) == 1:
            return biases.pop()
        # Mixed. If the very highest has a view and nothing directly contradicts it,
        # follow it; otherwise stand aside.
        highest = ready[-1]
        opposed = any(
            r.bias is highest.bias.opposite for r in ready if r is not highest
        )
        if highest.bias is not Bias.NEUTRAL and not opposed:
            return highest.bias
        return Bias.NEUTRAL

    def alignment(self) -> float:
        """How much all timeframes agree, in [-1, 1], weighted toward the slower.

        Weighting by timeframe length rather than equally is the whole point: a
        daily trend is not one vote among many alongside a five-minute wiggle.
        """
        total = 0.0
        weight = 0.0
        for tf in self.timeframes:
            read = self._reads[tf]
            if not read.ready:
                continue
            w = float(tf.minutes)
            total += w * read.bias.sign
            weight += w
        if weight <= 0:
            return 0.0
        return max(-1.0, min(1.0, total / weight))

    def agrees(self, side: Bias) -> bool:
        """Is trading ``side`` allowed by the higher-timeframe structure?"""
        permitted = self.permitted_direction()
        if permitted is Bias.NEUTRAL:
            return False
        return permitted is side

    def timing_bias(self) -> Bias:
        """Read from the fast timeframes, for entry timing only."""
        frames = [self._reads[tf] for tf in self.timing_frames if self._reads[tf].ready]
        if not frames:
            return Bias.NEUTRAL
        biases = {r.bias for r in frames}
        return biases.pop() if len(biases) == 1 else Bias.NEUTRAL

    def summary(self) -> list[str]:
        out: list[str] = []
        for tf in reversed(self.timeframes):
            read = self._reads[tf]
            role = "direction" if tf in self.direction_frames else "timing"
            trend = read.trend.direction.value if read.trend else "unknown"
            health = read.trend.health.value if read.trend else "-"
            out.append(
                f"{tf.value:>4} [{role:>9}] {trend:<8} {health:<10} "
                f"bars {read.bars_seen:<5} atr {read.atr:.4g}"
            )
        out.append(
            f"permitted: {self.permitted_direction().value} | "
            f"alignment {self.alignment():+.2f}"
        )
        return out
