"""Is the edge real, or is it luck?

This is the most important module in the quant layer, because a backtest result on its
own cannot answer that question. A Sharpe of 0.8 from 200 trades and a Sharpe of 0.8
from 20 trades are different claims, and the usual presentation makes them look
identical.

Three tools, each attacking a different way a backtest lies.

**Bootstrap confidence intervals.** Resample the returns and recompute the statistic
thousands of times. If the 95% interval on Sharpe includes zero, the strategy has not
demonstrated an edge - whatever the point estimate says.

**Permutation testing.** Shuffle the *timing* of the signals while keeping the returns,
then see how often random timing does as well. This directly tests whether the entry
logic contains information, and it is brutal: many strategies that look profitable turn
out to be indistinguishable from trading the same market at random times.

**Walk-forward splits.** Fit or tune on one window, evaluate on the next, never
overlapping. The only honest way to use in-sample data. :func:`walk_forward_splits`
generates the windows; the caller does the fitting, because what "fit" means depends on
the model.

Also here: deflated Sharpe, which corrects for the fact that trying many strategy
variants and reporting the best one inflates the apparent result. If parameters were
tuned at all, the raw Sharpe is optimistic and this says by how much.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(slots=True)
class BootstrapResult:
    """A statistic with a confidence interval earned by resampling."""

    statistic: float
    lower: float
    upper: float
    p_value: float
    """Fraction of resamples at or below zero. Small means a real effect."""
    resamples: int

    @property
    def significant(self) -> bool:
        """Does the interval exclude zero?"""
        return self.lower > 0.0

    def line(self, name: str = "statistic") -> str:
        verdict = "significant" if self.significant else "NOT distinguishable from zero"
        return (
            f"{name} {self.statistic:+.3f}  95% CI [{self.lower:+.3f}, {self.upper:+.3f}]  "
            f"p={self.p_value:.4f}  {verdict}"
        )


def sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    """Annualised Sharpe from per-bar returns."""
    data = np.asarray(returns, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 2:
        return 0.0
    sd = data.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(data.mean() / sd * math.sqrt(periods_per_year))


def bootstrap_sharpe(
    returns: np.ndarray | list[float],
    periods_per_year: float,
    resamples: int = 2000,
    seed: int = 7,
) -> BootstrapResult:
    """Confidence interval on Sharpe by resampling returns with replacement.

    Note the limitation honestly: this resamples independently, so it does not preserve
    autocorrelation. For strategies whose returns are serially correlated the interval
    will be too narrow, and a block bootstrap would be needed. It is still far better
    than reporting a point estimate alone.
    """
    data = np.asarray(returns, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 10:
        return BootstrapResult(0.0, 0.0, 0.0, 1.0, 0)

    rng = np.random.default_rng(seed)
    observed = sharpe(data, periods_per_year)
    draws = np.empty(resamples)
    n = data.size
    for i in range(resamples):
        sample = data[rng.integers(0, n, n)]
        draws[i] = sharpe(sample, periods_per_year)

    lower, upper = np.quantile(draws, [0.025, 0.975])
    # One-sided: how often did resampling produce a non-positive Sharpe?
    p_value = float((draws <= 0).mean())
    return BootstrapResult(
        statistic=observed,
        lower=float(lower),
        upper=float(upper),
        p_value=p_value,
        resamples=resamples,
    )


def permutation_test(
    strategy_returns: np.ndarray | list[float],
    market_returns: np.ndarray | list[float],
    resamples: int = 2000,
    seed: int = 7,
) -> BootstrapResult:
    """Does the strategy beat the same market traded at random times?

    Keeps the market's returns and shuffles the strategy's exposure pattern. The null
    hypothesis is that entry timing carries no information, and a strategy that cannot
    reject it is riding the market rather than reading it.
    """
    strat = np.asarray(strategy_returns, dtype=float)
    market = np.asarray(market_returns, dtype=float)
    size = min(strat.size, market.size)
    if size < 20:
        return BootstrapResult(0.0, 0.0, 0.0, 1.0, 0)
    strat, market = strat[:size], market[:size]

    # Recover the exposure that would have produced these returns from this market.
    with np.errstate(divide="ignore", invalid="ignore"):
        exposure = np.where(np.abs(market) > 1e-12, strat / market, 0.0)
    exposure = np.nan_to_num(exposure, nan=0.0, posinf=0.0, neginf=0.0)

    observed = float(strat.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for i in range(resamples):
        draws[i] = float((rng.permutation(exposure) * market).sum())

    lower, upper = np.quantile(draws, [0.025, 0.975])
    # How often does random timing match or beat the real thing?
    p_value = float((draws >= observed).mean())
    return BootstrapResult(
        statistic=observed,
        lower=float(lower),
        upper=float(upper),
        p_value=p_value,
        resamples=resamples,
    )


def deflated_sharpe(
    observed_sharpe: float,
    trials: int,
    sample_size: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Probability the Sharpe is real after accounting for how many variants were tried.

    Testing fifty parameter combinations and reporting the best one produces an
    impressive number from noise alone. This adjusts for that selection, and for
    non-normal returns, following the deflated-Sharpe idea: compare the observed value
    against the maximum you would *expect* from the same number of trials on worthless
    strategies.

    Returns a probability. Below 0.5 means the result is more likely selection
    artefact than edge.
    """
    if sample_size < 10 or trials < 1:
        return 0.0

    # Expected maximum Sharpe from `trials` independent worthless strategies.
    euler = 0.5772156649
    if trials > 1:
        z1 = stats.norm.ppf(1.0 - 1.0 / trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (trials * math.e))
        expected_max = (1.0 - euler) * z1 + euler * z2
    else:
        expected_max = 0.0

    # Standard error of Sharpe, corrected for skew and fat tails.
    variance = (
        1.0
        - skew * observed_sharpe
        + (excess_kurtosis / 4.0) * observed_sharpe ** 2
    ) / (sample_size - 1)
    if variance <= 0:
        return 0.0
    se = math.sqrt(variance)
    return float(stats.norm.cdf((observed_sharpe - expected_max) / se))


@dataclass(slots=True)
class Split:
    """One walk-forward window."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


def walk_forward_splits(
    n: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
    anchored: bool = False,
    gap: int = 0,
) -> list[Split]:
    """Generate non-overlapping out-of-sample windows.

    ``anchored`` keeps the training window's start fixed and lets it grow, which is how
    a live system actually accumulates history. Rolling instead keeps the window a fixed
    length, which adapts faster to regime change.

    ``gap`` leaves bars between train and test. Worth using whenever features look back
    or labels look forward, because without it the last training bars and the first test
    bars share information and the out-of-sample result is contaminated.
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must both be positive")
    step = step or test_size
    splits: list[Split] = []
    train_start = 0
    train_end = train_size
    while train_end + gap + test_size <= n:
        test_start = train_end + gap
        splits.append(
            Split(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_start + test_size,
            )
        )
        train_end += step
        if not anchored:
            train_start += step
    return splits


def summarise(
    returns: np.ndarray | list[float],
    periods_per_year: float,
    trials: int = 1,
    market_returns: np.ndarray | list[float] | None = None,
) -> list[str]:
    """One-shot verdict on whether a return series demonstrates an edge."""
    data = np.asarray(returns, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 20:
        return [f"only {data.size} observations: nothing can be concluded"]

    boot = bootstrap_sharpe(data, periods_per_year)
    lines = [boot.line("sharpe")]

    dsr = deflated_sharpe(
        boot.statistic,
        trials=trials,
        sample_size=data.size,
        skew=float(stats.skew(data)),
        excess_kurtosis=float(stats.kurtosis(data)),
    )
    lines.append(
        f"deflated sharpe probability {dsr:.3f} after {trials} trial(s)"
        + ("  (likely real)" if dsr > 0.5 else "  (likely selection artefact)")
    )

    if market_returns is not None:
        perm = permutation_test(data, market_returns)
        lines.append(
            f"permutation: random timing beat this strategy "
            f"{perm.p_value:.1%} of the time"
            + ("  (timing carries information)" if perm.p_value < 0.05 else
               "  (timing NOT distinguishable from random)")
        )
    return lines
