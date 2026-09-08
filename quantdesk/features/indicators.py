"""Streaming indicators.

Every indicator here is fed one bar at a time and carries its own state. This is
what lets the backtester and the live desk share strategy code exactly: there is
no "recompute the whole dataframe" path that could accidentally see the future.

On performance vs numerical robustness: EMA/RSI/ATR are genuinely recursive so
they are true O(1). Windowed statistics (mean, stdev) keep a deque and use numpy
over the window instead of running sum-of-squares. Sum-of-squares is O(1) but
loses precision badly when the mean is large relative to the variance, which is
exactly the case for asset prices (mean ~79,000, stdev ~100). Windows here are
under a few hundred elements, so the O(n) cost is irrelevant next to being right.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Literal

import numpy as np


class Indicator:
    """Common shape: feed values in, read ``value`` and ``ready`` out."""

    __slots__ = ()

    @property
    def ready(self) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def value(self) -> float:  # pragma: no cover - overridden
        raise NotImplementedError


class EMA(Indicator):
    """Exponential moving average, seeded with an SMA of the first `period` values."""

    __slots__ = ("period", "alpha", "_value", "_seed", "_n")

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("EMA period must be >= 1")
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self._value = 0.0
        self._seed = 0.0
        self._n = 0

    def update(self, x: float) -> float:
        self._n += 1
        if self._n <= self.period:
            # Seed with a simple average so early values aren't dominated by x[0].
            self._seed += x
            self._value = self._seed / self._n
        else:
            self._value += self.alpha * (x - self._value)
        return self._value

    @property
    def ready(self) -> bool:
        return self._n >= self.period

    @property
    def value(self) -> float:
        return self._value


class RollingWindow(Indicator):
    """Fixed-length window exposing robust summary statistics."""

    __slots__ = ("period", "_buf", "_arr", "_dirty")

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("window period must be >= 1")
        self.period = period
        self._buf: Deque[float] = deque(maxlen=period)
        self._arr: np.ndarray = np.empty(0, dtype=np.float64)
        self._dirty = True

    def update(self, x: float) -> float:
        self._buf.append(float(x))
        self._dirty = True
        return x

    def _array(self) -> np.ndarray:
        if self._dirty:
            self._arr = np.fromiter(self._buf, np.float64, len(self._buf))
            self._dirty = False
        return self._arr

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.period

    @property
    def value(self) -> float:
        return self._buf[-1] if self._buf else 0.0

    @property
    def mean(self) -> float:
        arr = self._array()
        return float(arr.mean()) if arr.size else 0.0

    @property
    def std(self) -> float:
        arr = self._array()
        # ddof=1: we're estimating population stdev from a sample.
        return float(arr.std(ddof=1)) if arr.size > 1 else 0.0

    @property
    def maximum(self) -> float:
        arr = self._array()
        return float(arr.max()) if arr.size else 0.0

    @property
    def minimum(self) -> float:
        arr = self._array()
        return float(arr.min()) if arr.size else 0.0

    def zscore(self, x: float | None = None) -> float:
        """How many stdevs ``x`` sits from the window mean."""
        x = self.value if x is None else x
        sd = self.std
        if sd <= 1e-12:
            return 0.0
        return (x - self.mean) / sd

    def percentile_rank(self, x: float | None = None) -> float:
        """Fraction of the window at or below ``x``, in [0, 1]."""
        arr = self._array()
        if arr.size == 0:
            return 0.5
        x = self.value if x is None else x
        return float((arr <= x).sum()) / arr.size


class RSI(Indicator):
    """Wilder's Relative Strength Index."""

    __slots__ = ("period", "_avg_gain", "_avg_loss", "_prev", "_n")

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._prev: float | None = None
        self._n = 0

    def update(self, close: float) -> float:
        if self._prev is None:
            self._prev = close
            return 50.0
        change = close - self._prev
        self._prev = close
        gain = max(0.0, change)
        loss = max(0.0, -change)
        self._n += 1
        if self._n <= self.period:
            # Simple average for the seed period, then Wilder smoothing.
            self._avg_gain += (gain - self._avg_gain) / self._n
            self._avg_loss += (loss - self._avg_loss) / self._n
        else:
            k = 1.0 / self.period
            self._avg_gain += k * (gain - self._avg_gain)
            self._avg_loss += k * (loss - self._avg_loss)
        if self._avg_loss <= 1e-12:
            return 100.0 if self._avg_gain > 0 else 50.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @property
    def ready(self) -> bool:
        return self._n >= self.period

    @property
    def value(self) -> float:
        if self._avg_loss <= 1e-12:
            return 100.0 if self._avg_gain > 0 else 50.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


class ATR(Indicator):
    """Wilder's Average True Range. Fed high/low/close rather than a single value."""

    __slots__ = ("period", "_atr", "_prev_close", "_n")

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._atr = 0.0
        self._prev_close: float | None = None
        self._n = 0

    def update(self, high: float, low: float, close: float) -> float:
        if self._prev_close is None:
            tr = high - low
        else:
            # True range accounts for gaps between bars, not just the bar's range.
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        self._prev_close = close
        self._n += 1
        if self._n <= self.period:
            self._atr += (tr - self._atr) / self._n
        else:
            self._atr += (tr - self._atr) / self.period
        return self._atr

    @property
    def ready(self) -> bool:
        return self._n >= self.period

    @property
    def value(self) -> float:
        return self._atr


class RollingExtreme(Indicator):
    """Sliding-window max or min in amortised O(1) using a monotonic deque."""

    __slots__ = ("period", "mode", "_dq", "_i", "_count")

    def __init__(self, period: int, mode: Literal["max", "min"] = "max") -> None:
        self.period = period
        self.mode = mode
        self._dq: Deque[tuple[int, float]] = deque()
        self._i = 0
        self._count = 0

    def update(self, x: float) -> float:
        self._i += 1
        self._count = min(self._count + 1, self.period)
        # Drop tail entries that can never be the extreme again.
        if self.mode == "max":
            while self._dq and self._dq[-1][1] <= x:
                self._dq.pop()
        else:
            while self._dq and self._dq[-1][1] >= x:
                self._dq.pop()
        self._dq.append((self._i, x))
        # Expire entries that have fallen out of the window.
        while self._dq and self._dq[0][0] <= self._i - self.period:
            self._dq.popleft()
        return self._dq[0][1]

    @property
    def ready(self) -> bool:
        return self._count >= self.period

    @property
    def value(self) -> float:
        return self._dq[0][1] if self._dq else 0.0


class ROC(Indicator):
    """Rate of change over ``period`` bars, as a fraction."""

    __slots__ = ("period", "_buf")

    def __init__(self, period: int = 12) -> None:
        self.period = period
        self._buf: Deque[float] = deque(maxlen=period + 1)

    def update(self, x: float) -> float:
        self._buf.append(float(x))
        return self.value

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.period + 1

    @property
    def value(self) -> float:
        if len(self._buf) < 2:
            return 0.0
        base = self._buf[0]
        if abs(base) < 1e-12:
            return 0.0
        return (self._buf[-1] - base) / base


# ===========================================================================
# Murphy indicator set
# ---------------------------------------------------------------------------
# The indicators below follow "Technical Analysis of the Financial Markets"
# (John J. Murphy, 1999). Where the book states a parameter explicitly, that
# value is the default and the docstring says so. Where it does not, the
# docstring says the default is a convention rather than the book's number, so
# it is always clear which knobs are sourced and which are chosen.
# ===========================================================================


class SMA(Indicator):
    """Simple moving average."""

    __slots__ = ("period", "_buf", "_sum")

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("SMA period must be >= 1")
        self.period = period
        self._buf: Deque[float] = deque(maxlen=period)
        self._sum = 0.0

    def update(self, x: float) -> float:
        if len(self._buf) == self.period:
            self._sum -= self._buf[0]
        self._buf.append(float(x))
        self._sum += float(x)
        return self.value

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.period

    @property
    def value(self) -> float:
        return self._sum / len(self._buf) if self._buf else 0.0


class WMA(Indicator):
    """Linearly weighted moving average.

    Murphy's construction: the most recent value carries weight equal to the
    period, the one before it period-1, and so on down to 1, divided by the sum
    of those weights. For a 10-period average the divisor is 55.
    """

    __slots__ = ("period", "_buf", "_divisor")

    def __init__(self, period: int) -> None:
        self.period = period
        self._buf: Deque[float] = deque(maxlen=period)
        self._divisor = period * (period + 1) / 2.0

    def update(self, x: float) -> float:
        self._buf.append(float(x))
        return self.value

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.period

    @property
    def value(self) -> float:
        if not self._buf:
            return 0.0
        n = len(self._buf)
        divisor = n * (n + 1) / 2.0
        return sum(v * (i + 1) for i, v in enumerate(self._buf)) / divisor


class Momentum(Indicator):
    """Price now minus price N periods ago, oscillating around zero.

    Default 10 periods is Murphy's stated common choice, which he ties to the
    28-period trading cycle: oscillator lengths are set to half a dominant
    cycle. Above zero means the current close is above the close N back
    (near-term uptrend); below zero, the reverse.
    """

    __slots__ = ("period", "_buf")

    def __init__(self, period: int = 10) -> None:
        self.period = period
        self._buf: Deque[float] = deque(maxlen=period + 1)

    def update(self, x: float) -> float:
        self._buf.append(float(x))
        return self.value

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.period + 1

    @property
    def value(self) -> float:
        if len(self._buf) < 2:
            return 0.0
        return self._buf[-1] - self._buf[0]


class MACD(Indicator):
    """Moving Average Convergence/Divergence.

    Murphy's stated parameters: the MACD line is the difference of 12- and
    26-period exponential averages of the close, and the signal line is a
    9-period exponential average of the MACD line. The histogram is the gap
    between the two, which Murphy notes can turn several periods ahead of the
    crossover and so acts as an early warning.

    Interpretation, in Murphy's order of importance: divergence against price
    while the lines sit well away from zero is the meaningful signal; the
    crossover is the trigger.
    """

    __slots__ = ("fast", "slow", "signal_period", "_ema_fast", "_ema_slow",
                 "_signal", "_n", "_macd", "_hist", "_prev_hist", "_prev_macd",
                 "_prev_signal")

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self._ema_fast = EMA(fast)
        self._ema_slow = EMA(slow)
        self._signal = EMA(signal)
        self._n = 0
        self._macd = 0.0
        self._hist = 0.0
        self._prev_hist = 0.0
        self._prev_macd = 0.0
        self._prev_signal = 0.0

    def update(self, close: float) -> float:
        self._n += 1
        f = self._ema_fast.update(close)
        s = self._ema_slow.update(close)
        self._prev_macd = self._macd
        self._prev_signal = self.signal
        self._macd = f - s
        # Only start the signal average once the slow EMA is meaningful,
        # otherwise the signal line is chasing EMA seeding noise.
        if self._ema_slow.ready:
            self._signal.update(self._macd)
        self._prev_hist = self._hist
        self._hist = self._macd - self.signal
        return self._macd

    @property
    def ready(self) -> bool:
        return self._n >= self.slow + self.signal_period

    @property
    def value(self) -> float:
        return self._macd

    @property
    def macd(self) -> float:
        return self._macd

    @property
    def signal(self) -> float:
        return self._signal.value

    @property
    def histogram(self) -> float:
        return self._hist

    @property
    def crossed_up(self) -> bool:
        """MACD line crossed above the signal line on this bar."""
        return self._prev_macd <= self._prev_signal and self._macd > self.signal

    @property
    def crossed_down(self) -> bool:
        return self._prev_macd >= self._prev_signal and self._macd < self.signal

    @property
    def histogram_turned_up(self) -> bool:
        return self._hist > self._prev_hist

    @property
    def above_zero(self) -> bool:
        return self._macd > 0.0


class Stochastic(Indicator):
    """Stochastic oscillator (%K / %D), George Lane's construction.

    Measures where the close sits inside the recent high-low range, scaled 0 to
    100. Murphy's stated parameters and semantics:

    * %K is the raw position of the close in the N-period range.
    * Fast %D is a 3-period average of %K.
    * Slow stochastics take a further 3-period average, and Murphy notes most
      traders prefer slow because the signals are more reliable. This class
      exposes both and defaults ``use_slow=True`` for that reason.
    * Extremes are the 80 and 20 lines, not 70/30 as with RSI.

    The setup Murphy describes is divergence while %D is beyond an extreme; the
    trigger is %K crossing %D.
    """

    __slots__ = ("period", "smooth_k", "smooth_d", "use_slow", "_hi", "_lo",
                 "_raw_k", "_fast_d", "_slow_d", "_n", "_prev_k", "_prev_d")

    def __init__(
        self,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
        use_slow: bool = True,
    ) -> None:
        self.period = period
        self.smooth_k = smooth_k
        self.smooth_d = smooth_d
        self.use_slow = use_slow
        self._hi = RollingExtreme(period, "max")
        self._lo = RollingExtreme(period, "min")
        self._raw_k = 50.0
        self._fast_d = SMA(smooth_k)
        self._slow_d = SMA(smooth_d)
        self._n = 0
        self._prev_k = 50.0
        self._prev_d = 50.0

    def update(self, high: float, low: float, close: float) -> float:
        self._n += 1
        self._prev_k = self.k
        self._prev_d = self.d

        hh = self._hi.update(high)
        ll = self._lo.update(low)
        span = hh - ll
        # A flat range carries no information; hold at the neutral midpoint
        # rather than dividing by zero or jumping to an extreme.
        self._raw_k = 50.0 if span <= 1e-12 else 100.0 * (close - ll) / span
        self._fast_d.update(self._raw_k)
        self._slow_d.update(self._fast_d.value)
        return self.k

    @property
    def ready(self) -> bool:
        return self._n >= self.period + self.smooth_k + (self.smooth_d if self.use_slow else 0)

    @property
    def k(self) -> float:
        """%K. For slow stochastics this is the smoothed %K (= fast %D)."""
        return self._fast_d.value if self.use_slow else self._raw_k

    @property
    def d(self) -> float:
        """%D, the signal line."""
        return self._slow_d.value if self.use_slow else self._fast_d.value

    @property
    def value(self) -> float:
        return self.d

    @property
    def overbought(self) -> bool:
        return self.d > 80.0

    @property
    def oversold(self) -> bool:
        return self.d < 20.0

    @property
    def crossed_up(self) -> bool:
        """%K crossed above %D: Murphy's actual buy trigger."""
        return self._prev_k <= self._prev_d and self.k > self.d

    @property
    def crossed_down(self) -> bool:
        return self._prev_k >= self._prev_d and self.k < self.d


class WilliamsR(Indicator):
    """Larry Williams %R.

    The close measured down from the period high rather than up from the low,
    which makes it an inverted stochastic running from -100 to 0. Murphy notes
    charting packages usually plot it inverted to avoid the confusion; this
    class keeps the true sign and exposes :attr:`inverted` for display.

    Period default of 14 matches the stochastic length; the book does not fix a
    single value for %R.
    """

    __slots__ = ("period", "_hi", "_lo", "_value", "_n")

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._hi = RollingExtreme(period, "max")
        self._lo = RollingExtreme(period, "min")
        self._value = -50.0
        self._n = 0

    def update(self, high: float, low: float, close: float) -> float:
        self._n += 1
        hh = self._hi.update(high)
        ll = self._lo.update(low)
        span = hh - ll
        self._value = -50.0 if span <= 1e-12 else -100.0 * (hh - close) / span
        return self._value

    @property
    def ready(self) -> bool:
        return self._n >= self.period

    @property
    def value(self) -> float:
        return self._value

    @property
    def inverted(self) -> float:
        """0..100 scale, oriented like a stochastic."""
        return self._value + 100.0

    @property
    def overbought(self) -> bool:
        return self._value > -20.0

    @property
    def oversold(self) -> bool:
        return self._value < -80.0


class CCI(Indicator):
    """Commodity Channel Index (Donald Lambert), cited by Murphy in Ch. 10.

    Typical price is (H+L+C)/3. The index is that price's deviation from its
    own moving average, divided by mean absolute deviation and scaled by 0.015
    so that most readings land inside +/-100.

    The 20-period default and the 0.015 constant are the standard published
    values; Murphy discusses the indicator without pinning the length.
    """

    __slots__ = ("period", "constant", "_buf", "_sma")

    def __init__(self, period: int = 20, constant: float = 0.015) -> None:
        self.period = period
        self.constant = constant
        self._buf: Deque[float] = deque(maxlen=period)
        self._sma = SMA(period)

    def update(self, high: float, low: float, close: float) -> float:
        tp = (high + low + close) / 3.0
        self._buf.append(tp)
        self._sma.update(tp)
        return self.value

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.period

    @property
    def value(self) -> float:
        if not self._buf:
            return 0.0
        mean = self._sma.value
        # Mean absolute deviation, not standard deviation: that is what the
        # original formula specifies and it makes the 0.015 constant correct.
        mad = sum(abs(v - mean) for v in self._buf) / len(self._buf)
        if mad <= 1e-12:
            return 0.0
        return (self._buf[-1] - mean) / (self.constant * mad)

    @property
    def overbought(self) -> bool:
        return self.value > 100.0

    @property
    def oversold(self) -> bool:
        return self.value < -100.0


class DMI(Indicator):
    """Wilder's Directional Movement, giving +DI, -DI and ADX.

    Murphy's stated interpretation, which is what makes this the most useful
    single indicator in the set:

    * +DI crossing above -DI is a buy signal; below, a sell.
    * ADX rates trendiness on a 0-100 scale. A **rising** ADX means the market
      is trending and suits trend-following tools; a **falling** ADX means it is
      not, and oscillators are the better tool. This is a regime switch, not a
      direction call.
    * ADX turning down from above 40 is an early warning the trend is tiring.
    * ADX pushing back above 20 often marks the start of a new trend.

    Murphy also cites Wilder's estimate that markets trend strongly only around
    30% of the time, which is the whole reason a regime filter matters.

    Period default 14 is Wilder's.
    """

    __slots__ = ("period", "_prev_high", "_prev_low", "_prev_close",
                 "_sm_tr", "_sm_plus", "_sm_minus", "_adx", "_n",
                 "_prev_plus_di", "_prev_minus_di", "_prev_adx")

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._prev_high: float | None = None
        self._prev_low: float | None = None
        self._prev_close: float | None = None
        # Wilder-smoothed accumulators for true range and directional movement.
        self._sm_tr = 0.0
        self._sm_plus = 0.0
        self._sm_minus = 0.0
        self._adx = 0.0
        self._n = 0
        self._prev_plus_di = 0.0
        self._prev_minus_di = 0.0
        self._prev_adx = 0.0

    def update(self, high: float, low: float, close: float) -> float:
        if self._prev_high is None:
            self._prev_high, self._prev_low, self._prev_close = high, low, close
            return 0.0

        # Capture the previous DI readings before the accumulators advance,
        # otherwise the crossover test compares this bar against itself.
        self._prev_plus_di = self.plus_di
        self._prev_minus_di = self.minus_di

        up_move = high - self._prev_high
        down_move = self._prev_low - low

        # Directional movement is exclusive: only the larger side counts, and
        # only when it is positive. Inside bars produce no directional movement.
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        tr = max(
            high - low,
            abs(high - self._prev_close),
            abs(low - self._prev_close),
        )

        self._prev_high, self._prev_low, self._prev_close = high, low, close
        self._n += 1

        p = self.period
        if self._n <= p:
            # Seed with running sums, as Wilder does.
            self._sm_tr += tr
            self._sm_plus += plus_dm
            self._sm_minus += minus_dm
        else:
            self._sm_tr = self._sm_tr - (self._sm_tr / p) + tr
            self._sm_plus = self._sm_plus - (self._sm_plus / p) + plus_dm
            self._sm_minus = self._sm_minus - (self._sm_minus / p) + minus_dm

        plus_di = self.plus_di
        minus_di = self.minus_di
        total = plus_di + minus_di
        dx = 0.0 if total <= 1e-12 else 100.0 * abs(plus_di - minus_di) / total

        self._prev_adx = self._adx
        if self._n <= p * 2:
            # Average the first block of DX values, then Wilder-smooth.
            k = max(1, self._n - p + 1)
            self._adx += (dx - self._adx) / k
        else:
            self._adx += (dx - self._adx) / p
        return self._adx

    @property
    def plus_di(self) -> float:
        if self._sm_tr <= 1e-12:
            return 0.0
        return 100.0 * self._sm_plus / self._sm_tr

    @property
    def minus_di(self) -> float:
        if self._sm_tr <= 1e-12:
            return 0.0
        return 100.0 * self._sm_minus / self._sm_tr

    @property
    def adx(self) -> float:
        return self._adx

    @property
    def value(self) -> float:
        return self._adx

    @property
    def ready(self) -> bool:
        return self._n >= self.period * 2

    @property
    def trending(self) -> bool:
        """ADX above the 20 level Murphy cites as the trend threshold."""
        return self._adx > 20.0

    @property
    def adx_rising(self) -> bool:
        return self._adx > self._prev_adx

    @property
    def trend_exhausting(self) -> bool:
        """ADX rolling over from above 40: Murphy's early warning."""
        return self._prev_adx > 40.0 and self._adx < self._prev_adx

    @property
    def direction(self) -> int:
        """+1 when +DI leads, -1 when -DI leads, 0 when indistinguishable."""
        p, m = self.plus_di, self.minus_di
        if abs(p - m) < 1e-9:
            return 0
        return 1 if p > m else -1

    @property
    def crossed_up(self) -> bool:
        """+DI crossed above -DI on this bar: Murphy's DMI buy signal."""
        return (
            self._prev_plus_di <= self._prev_minus_di
            and self.plus_di > self.minus_di
        )

    @property
    def crossed_down(self) -> bool:
        """-DI crossed above +DI on this bar: the DMI sell signal."""
        return (
            self._prev_plus_di >= self._prev_minus_di
            and self.plus_di < self.minus_di
        )

    @property
    def spread(self) -> float:
        """+DI minus -DI, a continuous measure of directional pressure."""
        return self.plus_di - self.minus_di


class ParabolicSAR(Indicator):
    """Wilder's Parabolic stop-and-reverse.

    Always in the market: the dots trail below price in an uptrend and above it
    in a downtrend, and a touch flips the position. Murphy stresses that the
    acceleration factor starts slow to give a trend room to establish itself,
    then speeds up until it catches price.

    He is equally clear about the failure mode: it whipsaws relentlessly in
    sideways markets, which is exactly why it should be paired with an ADX
    regime filter rather than traded on its own.

    The acceleration constants (0.02 start and step, 0.20 cap) are Wilder's
    published defaults; Murphy describes the mechanism without listing them.
    """

    __slots__ = ("af_start", "af_step", "af_max", "_sar", "_ep", "_af",
                 "_long", "_n", "_prev_high", "_prev_low", "_prior_high",
                 "_prior_low", "_reversed")

    def __init__(
        self,
        af_start: float = 0.02,
        af_step: float = 0.02,
        af_max: float = 0.20,
    ) -> None:
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max
        self._sar = 0.0
        self._ep = 0.0
        self._af = af_start
        self._long = True
        self._n = 0
        self._prev_high = 0.0
        self._prev_low = 0.0
        self._prior_high = 0.0
        self._prior_low = 0.0
        self._reversed = False

    def update(self, high: float, low: float, close: float) -> float:
        self._n += 1
        self._reversed = False

        if self._n == 1:
            self._sar = low
            self._ep = high
            self._long = True
            self._prev_high = self._prior_high = high
            self._prev_low = self._prior_low = low
            return self._sar

        # Advance the stop toward the extreme point.
        sar = self._sar + self._af * (self._ep - self._sar)

        if self._long:
            # Never let the stop rise above the last two lows.
            sar = min(sar, self._prev_low, self._prior_low)
            if low <= sar:
                # Stopped out: flip short, anchor the new stop at the old extreme.
                self._long = False
                self._reversed = True
                sar = self._ep
                self._ep = low
                self._af = self.af_start
            else:
                if high > self._ep:
                    self._ep = high
                    self._af = min(self._af + self.af_step, self.af_max)
        else:
            sar = max(sar, self._prev_high, self._prior_high)
            if high >= sar:
                self._long = True
                self._reversed = True
                sar = self._ep
                self._ep = high
                self._af = self.af_start
            else:
                if low < self._ep:
                    self._ep = low
                    self._af = min(self._af + self.af_step, self.af_max)

        self._sar = sar
        self._prior_high, self._prior_low = self._prev_high, self._prev_low
        self._prev_high, self._prev_low = high, low
        return self._sar

    @property
    def ready(self) -> bool:
        return self._n >= 5

    @property
    def value(self) -> float:
        return self._sar

    @property
    def is_long(self) -> bool:
        return self._long

    @property
    def flipped(self) -> bool:
        """True on the bar where the system reversed."""
        return self._reversed

    @property
    def acceleration(self) -> float:
        return self._af


class BollingerBands(Indicator):
    """Bollinger Bands.

    Murphy's stated parameters: a 20-period moving average with bands at two
    standard deviations. His interpretation notes, which the properties below
    encode:

    * The bands act as price targets; a move off one band tends to carry to the
      other.
    * Band **width** measures volatility, and the extremes are informative in
      opposite directions: unusually narrow bands tend to precede a sharp move
      (volatility contraction resolving), while unusually wide bands suggest a
      move is mature and volatility is likely to subside.
    * Unlike percentage envelopes, the bands adapt to volatility automatically.

    Note the standard deviation here is the population form (ddof=0), which is
    the convention for Bollinger Bands, in contrast to the sample form used by
    :class:`RollingWindow`.
    """

    __slots__ = ("period", "num_std", "_win", "_width_hist")

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        self.period = period
        self.num_std = num_std
        self._win: Deque[float] = deque(maxlen=period)
        # Track bandwidth history so "narrow" and "wide" are relative to this
        # instrument's own past, not an absolute number that cannot generalise.
        self._width_hist = RollingWindow(period * 5)

    def update(self, close: float) -> float:
        self._win.append(float(close))
        if len(self._win) >= 2:
            self._width_hist.update(self.bandwidth)
        return self.middle

    @property
    def ready(self) -> bool:
        return len(self._win) >= self.period

    @property
    def middle(self) -> float:
        return sum(self._win) / len(self._win) if self._win else 0.0

    @property
    def std(self) -> float:
        n = len(self._win)
        if n < 2:
            return 0.0
        mean = self.middle
        var = sum((v - mean) ** 2 for v in self._win) / n
        return math.sqrt(var)

    @property
    def upper(self) -> float:
        return self.middle + self.num_std * self.std

    @property
    def lower(self) -> float:
        return self.middle - self.num_std * self.std

    @property
    def value(self) -> float:
        return self.middle

    @property
    def bandwidth(self) -> float:
        """Band separation as a fraction of the middle band."""
        mid = self.middle
        if mid <= 1e-12:
            return 0.0
        return (self.upper - self.lower) / mid

    @property
    def percent_b(self) -> float:
        """Where price sits across the bands: 0 at the lower, 1 at the upper."""
        lo, up = self.lower, self.upper
        if up - lo <= 1e-12:
            return 0.5
        return (self._win[-1] - lo) / (up - lo)

    def width_percentile(self) -> float:
        """Current bandwidth's rank within its own history, in [0, 1]."""
        return self._width_hist.percentile_rank(self.bandwidth)

    @property
    def squeeze(self) -> bool:
        """Bandwidth in the bottom decile of its history: a move may be coming."""
        return self._width_hist.ready and self.width_percentile() < 0.10


class KeltnerChannel(Indicator):
    """Keltner Channels, Linda Raschke's ATR variant as given in Appendix A.

    Construction: a 20-period exponential average, with a 10-period ATR doubled
    and added or subtracted.

    The interpretation is a **breakout** system and is the opposite of STARC
    bands despite the near-identical construction. A close above the upper band
    is a sign of strength (an expansion in upward volatility) and a close below
    the lower band a sign of weakness. The signal persists until price closes
    beyond the opposite band, and in a rising market the centre EMA should act
    as support.
    """

    __slots__ = ("ema_period", "atr_period", "multiplier", "_ema", "_atr",
                 "_signal", "_n")

    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> None:
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier
        self._ema = EMA(ema_period)
        self._atr = ATR(atr_period)
        self._signal = 0
        self._n = 0

    def update(self, high: float, low: float, close: float) -> float:
        self._n += 1
        self._ema.update(close)
        self._atr.update(high, low, close)
        if self.ready:
            # Signal latches until price closes beyond the opposite band.
            if close > self.upper:
                self._signal = 1
            elif close < self.lower:
                self._signal = -1
        return self.middle

    @property
    def ready(self) -> bool:
        return self._n >= max(self.ema_period, self.atr_period)

    @property
    def middle(self) -> float:
        return self._ema.value

    @property
    def upper(self) -> float:
        return self._ema.value + self.multiplier * self._atr.value

    @property
    def lower(self) -> float:
        return self._ema.value - self.multiplier * self._atr.value

    @property
    def value(self) -> float:
        return self.middle

    @property
    def signal(self) -> int:
        """Latched breakout state: +1 strength, -1 weakness, 0 undecided."""
        return self._signal


class StarcBands(Indicator):
    """STARC bands (Stoller Average Range Channels), per Appendix A.

    Construction: a 6-period moving average with a 15-period ATR doubled and
    added or subtracted.

    The interpretation is a **risk filter** and is the opposite of Keltner
    Channels. Trading outside the bands is uncommon and marks an extreme: near
    or above the upper band it is a high-risk moment to buy and a low-risk
    moment to sell, and at or below the lower band the reverse. Used as a
    veto on chasing, not as a breakout trigger.
    """

    __slots__ = ("ma_period", "atr_period", "multiplier", "_ma", "_atr", "_n",
                 "_last_close")

    def __init__(
        self,
        ma_period: int = 6,
        atr_period: int = 15,
        multiplier: float = 2.0,
    ) -> None:
        self.ma_period = ma_period
        self.atr_period = atr_period
        self.multiplier = multiplier
        self._ma = SMA(ma_period)
        self._atr = ATR(atr_period)
        self._n = 0
        self._last_close = 0.0

    def update(self, high: float, low: float, close: float) -> float:
        self._n += 1
        self._ma.update(close)
        self._atr.update(high, low, close)
        self._last_close = close
        return self.middle

    @property
    def ready(self) -> bool:
        return self._n >= max(self.ma_period, self.atr_period)

    @property
    def middle(self) -> float:
        return self._ma.value

    @property
    def upper(self) -> float:
        return self._ma.value + self.multiplier * self._atr.value

    @property
    def lower(self) -> float:
        return self._ma.value - self.multiplier * self._atr.value

    @property
    def value(self) -> float:
        return self.middle

    @property
    def position(self) -> float:
        """Where the close sits across the bands, 0 at lower and 1 at upper."""
        lo, up = self.lower, self.upper
        if up - lo <= 1e-12:
            return 0.5
        return (self._last_close - lo) / (up - lo)

    @property
    def high_risk_to_buy(self) -> bool:
        return self.ready and self._last_close >= self.upper

    @property
    def high_risk_to_sell(self) -> bool:
        return self.ready and self._last_close <= self.lower


class Envelope(Indicator):
    """Percentage envelope around a moving average (Murphy Ch. 9).

    Fixed percentage bands rather than volatility-scaled ones. Murphy notes the
    appropriate percentage varies by market and timeframe, so the 3% default is
    a placeholder to be tuned per instrument, not a value from the book.
    """

    __slots__ = ("period", "pct", "_ma")

    def __init__(self, period: int = 20, pct: float = 0.03) -> None:
        self.period = period
        self.pct = pct
        self._ma = SMA(period)

    def update(self, close: float) -> float:
        return self._ma.update(close)

    @property
    def ready(self) -> bool:
        return self._ma.ready

    @property
    def middle(self) -> float:
        return self._ma.value

    @property
    def upper(self) -> float:
        return self._ma.value * (1.0 + self.pct)

    @property
    def lower(self) -> float:
        return self._ma.value * (1.0 - self.pct)

    @property
    def value(self) -> float:
        return self._ma.value


class OBV(Indicator):
    """On Balance Volume (Joseph Granville), Murphy Ch. 7.

    A running total that adds the bar's whole volume when the close is up and
    subtracts it when the close is down. Murphy is explicit that only the
    *direction* of the resulting line matters, since the absolute level depends
    on where the series happened to start.

    He also names the flaw: assigning an entire bar's volume a single sign is
    crude, since a close a tick higher counts the same as a limit-up day. The
    ``signed_volume`` variant here weights by where the close falls within the
    bar's range, which is one of the refinements he alludes to. OBV remains the
    default because it is what the book specifies.
    """

    __slots__ = ("_obv", "_prev_close", "_n", "_slope", "_hist", "_signed")

    def __init__(self, slope_window: int = 20) -> None:
        self._obv = 0.0
        self._prev_close: float | None = None
        self._n = 0
        self._signed = 0.0
        self._hist = RollingWindow(slope_window)
        self._slope = 0.0

    def update(self, close: float, volume: float) -> float:
        self._n += 1
        if self._prev_close is not None:
            if close > self._prev_close:
                self._obv += volume
            elif close < self._prev_close:
                self._obv -= volume
            # An unchanged close leaves OBV alone, per the original definition.
        self._prev_close = close
        self._hist.update(self._obv)
        return self._obv

    def update_range(
        self, high: float, low: float, close: float, volume: float
    ) -> float:
        """OBV plus the range-weighted variant in ``signed_volume``."""
        span = high - low
        if span > 1e-12:
            # +1 at the high, -1 at the low, linear in between.
            location = (2.0 * (close - low) / span) - 1.0
        else:
            location = 0.0
        self._signed += location * volume
        return self.update(close, volume)

    @property
    def ready(self) -> bool:
        return self._n >= 2

    @property
    def value(self) -> float:
        return self._obv

    @property
    def signed_volume(self) -> float:
        """Range-weighted accumulation. Only valid if ``update_range`` is used."""
        return self._signed

    @property
    def rising(self) -> bool:
        """Is the OBV line trending up across its window?"""
        if not self._hist.ready:
            return self._obv > 0
        return self._obv > self._hist.mean
