"""Statistical and probability models.

Where Murphy's material reads charts and Douglas's reads discipline, this layer asks
whether the numbers survive scrutiny::

    from quantdesk.quant import analyse, bootstrap_sharpe, fit_vol_regime

    shape = analyse(returns)              # fat tails? which regime? real VaR?
    verdict = bootstrap_sharpe(returns, periods_per_year=8760)
    if not verdict.significant:
        print("no demonstrated edge, whatever the point estimate says")

:mod:`validation` is the important one. A backtest result without a confidence interval
is a claim without evidence, and the permutation test in particular is unforgiving
about strategies whose entry timing turns out to carry no information.
"""

from quantdesk.quant.distributions import (
    DistributionStats,
    analyse,
    conditional_var,
    drawdown_series,
    historical_var,
    hurst_exponent,
    tail_ratio,
)
from quantdesk.quant.kelly import (
    KellySizing,
    from_edge_stats,
    growth_rate,
    kelly_fraction,
    kelly_position,
)
from quantdesk.quant.regime import (
    TrendRegime,
    VolRegime,
    classify_trend_regime,
    fit_vol_regime,
)
from quantdesk.quant.validation import (
    BootstrapResult,
    Split,
    bootstrap_sharpe,
    deflated_sharpe,
    permutation_test,
    sharpe,
    summarise,
    walk_forward_splits,
)

__all__ = [
    "BootstrapResult",
    "DistributionStats",
    "KellySizing",
    "Split",
    "TrendRegime",
    "VolRegime",
    "analyse",
    "bootstrap_sharpe",
    "classify_trend_regime",
    "conditional_var",
    "deflated_sharpe",
    "drawdown_series",
    "fit_vol_regime",
    "from_edge_stats",
    "growth_rate",
    "historical_var",
    "hurst_exponent",
    "kelly_fraction",
    "kelly_position",
    "permutation_test",
    "sharpe",
    "summarise",
    "tail_ratio",
    "walk_forward_splits",
]
