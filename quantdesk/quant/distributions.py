"""Return distribution analysis.

The purpose is to replace an assumption with a measurement. Almost every standard risk
number - Sharpe, volatility targeting, normal VaR - quietly assumes returns are
Gaussian, and financial returns are not. They are fat-tailed and often skewed, which
means the assumption fails in exactly the situations the numbers exist to protect
against.

So this module measures the shape rather than assuming it:

* **Kurtosis and the Jarque-Bera test** say how far from normal the returns actually
  are. Crypto typically shows excess kurtosis well above zero.
* **Historical VaR and CVaR** are read off the empirical distribution rather than
  computed from a normal quantile. The gap between the two is worth looking at: when
  historical VaR is much worse than normal VaR, a normal-based risk limit is
  understating the danger.
* **CVaR (expected shortfall)** matters more than VaR, because VaR tells you a
  threshold and CVaR tells you how bad it is once you are past it. Position sizing
  cares about the second.
* **The Hurst exponent** distinguishes trending from mean-reverting behaviour, which is
  a direct statistical test of whether Murphy's trend-following premise applies to a
  given instrument at a given timeframe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(slots=True)
class DistributionStats:
    """Measured shape of a return series."""

    count: int
    mean: float
    std: float
    skew: float
    excess_kurtosis: float
    """Zero for a normal distribution. Positive means fat tails."""
    jarque_bera_p: float
    """Probability the sample came from a normal distribution."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    normal_var_95: float
    """VaR implied by a normal fit, for comparison with the historical figure."""
    normal_var_99: float
    best: float
    worst: float
    hurst: float

    @property
    def is_normal(self) -> bool:
        """Can normality be rejected at the 5% level?"""
        return self.jarque_bera_p >= 0.05

    @property
    def fat_tailed(self) -> bool:
        return self.excess_kurtosis > 1.0

    @property
    def tail_understatement(self) -> float:
        """How much a normal assumption understates the 99% loss, as a ratio.

        Measured at 99% rather than 95% deliberately. Fat tails live in the extreme,
        and at 95% a normal fit is often *more* conservative than reality - because the
        fat tails inflate the fitted standard deviation, which widens the normal
        quantile everywhere. Comparing at 95% therefore hides the very effect this is
        meant to expose.

        Above 1.0 means real losses at that confidence are worse than a normal model
        predicts, so a risk limit calibrated on normality is too loose.
        """
        if abs(self.normal_var_99) < 1e-12:
            return 1.0
        return abs(self.var_99) / abs(self.normal_var_99)

    @property
    def regime(self) -> str:
        """Trending, mean-reverting, or random, from the Hurst exponent."""
        if self.hurst > 0.55:
            return "trending"
        if self.hurst < 0.45:
            return "mean-reverting"
        return "random walk"

    def summary(self) -> list[str]:
        return [
            f"n            {self.count}",
            f"mean/std     {self.mean:+.5f} / {self.std:.5f}",
            f"skew         {self.skew:+.3f}",
            f"excess kurt  {self.excess_kurtosis:+.3f}"
            + ("  FAT TAILS" if self.fat_tailed else ""),
            f"normality    Jarque-Bera p={self.jarque_bera_p:.4g}"
            + ("  (cannot reject normal)" if self.is_normal else "  (NOT normal)"),
            f"VaR 95/99    {self.var_95:+.4%} / {self.var_99:+.4%}",
            f"CVaR 95/99   {self.cvar_95:+.4%} / {self.cvar_99:+.4%}",
            f"vs normal    normal VaR99 {self.normal_var_99:+.4%}, "
            f"real is {self.tail_understatement:.2f}x as bad",
            f"worst/best   {self.worst:+.4%} / {self.best:+.4%}",
            f"hurst        {self.hurst:.3f}  ({self.regime})",
        ]


def analyse(returns: np.ndarray | list[float]) -> DistributionStats:
    """Measure the shape of a return series."""
    data = np.asarray(returns, dtype=float)
    data = data[np.isfinite(data)]
    n = data.size
    if n < 8:
        raise ValueError(f"need at least 8 returns to characterise a distribution, got {n}")

    mean = float(data.mean())
    std = float(data.std(ddof=1))
    skew = float(stats.skew(data))
    kurt = float(stats.kurtosis(data))  # already excess
    try:
        _, jb_p = stats.jarque_bera(data)
    except Exception:  # pragma: no cover - tiny samples
        jb_p = float("nan")

    return DistributionStats(
        count=n,
        mean=mean,
        std=std,
        skew=skew,
        excess_kurtosis=kurt,
        jarque_bera_p=float(jb_p),
        var_95=historical_var(data, 0.95),
        var_99=historical_var(data, 0.99),
        cvar_95=conditional_var(data, 0.95),
        cvar_99=conditional_var(data, 0.99),
        normal_var_95=float(mean + std * stats.norm.ppf(0.05)),
        normal_var_99=float(mean + std * stats.norm.ppf(0.01)),
        best=float(data.max()),
        worst=float(data.min()),
        # Hurst describes how a *level* series scales, so it is computed on the
        # cumulative path these returns imply. Running it on the returns themselves is a
        # category error that silently produces a number near zero and reads as
        # "strongly mean-reverting" for any return series at all.
        hurst=hurst_exponent(np.cumsum(data)),
    )


def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Value at risk from the empirical distribution, as a negative return.

    Empirical rather than parametric on purpose: the whole reason to compute this is
    that the parametric version is wrong in the tail.
    """
    data = np.asarray(returns, dtype=float)
    if data.size == 0:
        return 0.0
    return float(np.quantile(data, 1.0 - confidence))


def conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Expected shortfall: the average loss given that VaR was breached.

    More useful than VaR for sizing, because it answers "how bad is the bad case?"
    rather than "where does the bad case begin?".
    """
    data = np.asarray(returns, dtype=float)
    if data.size == 0:
        return 0.0
    threshold = historical_var(data, confidence)
    tail = data[data <= threshold]
    if tail.size == 0:
        return float(threshold)
    return float(tail.mean())


def hurst_exponent(series: np.ndarray, max_lag: int = 64) -> float:
    """Hurst exponent via rescaled-range on lagged differences.

    0.5 is a random walk, above 0.5 is persistent (trending), below is
    anti-persistent (mean-reverting). This is a direct statistical check on whether a
    trend-following premise is justified for an instrument, rather than assuming it.

    **It measures persistence of increments, not drift.** A random walk with a large
    constant drift still scores 0.5, because adding a constant to every increment
    changes their mean but not their variance, and this estimator reads variance
    scaling. That is the correct definition but a common source of confusion: a chart
    that visibly slopes upward is not necessarily persistent in the Hurst sense. Use it
    to ask "do moves tend to continue?", not "is price going up?" - Dow trend
    classification answers the second question.

    Returns 0.5 when there is too little data to say, which fails to the neutral
    answer rather than inventing a signal.
    """
    data = np.asarray(series, dtype=float)
    n = data.size
    if n < 32:
        return 0.5
    lags = range(2, min(max_lag, n // 2))
    taus: list[float] = []
    used: list[int] = []
    for lag in lags:
        diff = data[lag:] - data[:-lag]
        if diff.size < 2:
            continue
        tau = float(np.sqrt(np.std(diff, ddof=1)))
        if tau > 0 and np.isfinite(tau):
            taus.append(tau)
            used.append(lag)
    if len(taus) < 4:
        return 0.5
    # Slope of log(tau) against log(lag); Hurst is twice that slope.
    slope = float(np.polyfit(np.log(used), np.log(taus), 1)[0])
    return max(0.0, min(1.0, slope * 2.0))


def drawdown_series(equity: np.ndarray | list[float]) -> np.ndarray:
    """Drawdown at each point, as a negative fraction of the running peak."""
    data = np.asarray(equity, dtype=float)
    if data.size == 0:
        return np.array([])
    peaks = np.maximum.accumulate(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peaks > 0, (data - peaks) / peaks, 0.0)
    return dd


def tail_ratio(returns: np.ndarray | list[float], pct: float = 0.05) -> float:
    """Size of the right tail over the left tail.

    Above 1.0 means the good surprises are bigger than the bad ones, which is the
    payoff profile trend following is supposed to produce and a useful check that it
    actually is.
    """
    data = np.asarray(returns, dtype=float)
    if data.size < 20:
        return 1.0
    right = float(np.quantile(data, 1.0 - pct))
    left = float(abs(np.quantile(data, pct)))
    return right / left if left > 1e-12 else 1.0
