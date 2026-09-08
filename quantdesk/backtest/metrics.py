"""Backtest metrics.

Chosen to answer the questions that decide whether a strategy is worth running, not
to fill a report. A few notes on why these and not others:

* **Sharpe needs the right periods-per-year.** Annualising a 5-minute strategy with
  252 is off by two orders of magnitude, so ``bars_per_year`` comes from configuration
  and is passed in rather than assumed.
* **Max drawdown is measured on the bar-by-bar curve**, not on closed trades. What
  matters is the worst the account actually looked, not the worst it looked at the
  moments trades happened to close.
* **Profit factor and expectancy** are reported alongside hit rate, because a 30% hit
  rate with a 5:1 payoff is a good strategy and a hit-rate-only view calls it bad.
  Murphy's point in chapter 16 is exactly this: "because most trades are losers, the
  only way to come out ahead is to ensure that the dollar amount of the winning trades
  is greater than that of the losing trades."
* **Rejection reasons are counted.** A strategy that declined 900 trades and took 4 is
  telling you something important about its gates, and no return metric shows it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from quantdesk.core.types import Fill

if TYPE_CHECKING:  # pragma: no cover
    from quantdesk.backtest.engine import TradeRecord
    from quantdesk.pipeline import BarDecision


@dataclass(slots=True)
class EquityPoint:
    ts: datetime
    equity: float
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    open_positions: int = 0


@dataclass
class BacktestResult:
    """Everything worth knowing about a run."""

    starting_equity: float
    final_equity: float
    total_return: float
    annualised_return: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_bars: int
    calmar: float
    trades: int
    wins: int
    losses: int
    hit_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    total_fees: float
    fills: int
    bars: int
    exposure: float
    """Average gross exposure as a fraction of equity."""
    mean_r: float = 0.0
    """Plain average of per-trade R. Only meaningful when risk per trade is even."""
    risk_weighted_r: float = 0.0
    """Total P&L over total risk taken. The R figure that reconciles with the money.

    When every trade risks the same amount this equals :attr:`mean_r`. When it does not,
    the plain mean is dominated by the trades that risked least - a trade risking 1
    currency unit and making 3 contributes +3R to the average with the same weight as a
    trade risking 500 and making 1,500. This aggregate weights each trade by the risk it
    actually took, which is why it cannot disagree in sign with the P&L.
    """
    median_risk: float = 0.0
    min_risk: float = 0.0
    max_risk: float = 0.0
    risk_dispersion: float = 0.0
    """Largest risk over smallest, across trades. 1.0 is perfectly even sizing.

    The single number that says whether R statistics can be trusted. Anything above
    roughly 10 means per-trade R values are not comparable with each other.
    """
    trades_without_risk: int = 0
    """Trades whose planned risk never arrived, and so are excluded from R."""
    rejections: dict[str, int] = field(default_factory=dict)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    trade_list: list["TradeRecord"] = field(default_factory=list)

    @property
    def profitable(self) -> bool:
        return self.final_equity > self.starting_equity

    @property
    def r_is_trustworthy(self) -> bool:
        """Whether per-trade R averages describe the strategy or the sizing.

        The threshold is a judgement, not a law: a 10x spread in risk per trade already
        means the biggest bet carries ten times the influence of the smallest on any
        plain average, which is enough to flip the sign of a reported expectancy.
        """
        return self.risk_dispersion <= 10.0 and self.trades_without_risk == 0

    def summary(self) -> list[str]:
        lines = [
            f"return       {self.total_return:+.2%}  "
            f"({self.starting_equity:,.0f} -> {self.final_equity:,.0f})",
            f"annualised   {self.annualised_return:+.2%}  vol {self.volatility:.2%}",
            f"sharpe       {self.sharpe:+.2f}   sortino {self.sortino:+.2f}   "
            f"calmar {self.calmar:+.2f}",
            f"max drawdown {self.max_drawdown:.2%} over {self.max_drawdown_bars} bars",
            f"trades       {self.trades}  hit rate {self.hit_rate:.1%}  "
            f"profit factor {self.profit_factor:.2f}",
            f"expectancy   {self.expectancy:+,.2f} per trade  "
            f"(avg win {self.avg_win:+,.0f} / avg loss {self.avg_loss:+,.0f})",
            f"best {self.largest_win:+,.0f}  worst {self.largest_loss:+,.0f}  "
            f"fees {self.total_fees:,.0f}",
            f"bars {self.bars}  fills {self.fills}  avg gross exposure {self.exposure:.1%}",
            f"risk/trade   median {self.median_risk:,.2f}  "
            f"range {self.min_risk:,.2f} to {self.max_risk:,.2f}  "
            f"spread {self.risk_dispersion:,.0f}x",
            f"expectancy   {self.risk_weighted_r:+.3f}R risk-weighted  "
            f"({self.mean_r:+.3f}R unweighted)",
        ]
        if not self.r_is_trustworthy:
            lines.append(
                "  WARNING: risk per trade is uneven, so the unweighted R figures "
                "describe the sizing, not the edge. Use the risk-weighted number."
            )
        if self.trades_without_risk:
            lines.append(
                f"  {self.trades_without_risk} trade(s) had no recorded risk and are "
                f"excluded from R"
            )
        if self.rejections:
            lines.append("why trades were declined:")
            for reason, count in sorted(
                self.rejections.items(), key=lambda kv: -kv[1]
            )[:6]:
                lines.append(f"  {count:6d}  {reason}")
        return lines


def compute_metrics(
    equity_curve: list[EquityPoint],
    trades: list["TradeRecord"],
    fills: list[Fill],
    decisions: list["BarDecision"],
    bars_per_year: float,
    starting_equity: float,
) -> BacktestResult:
    """Turn a run's raw history into metrics."""
    equities = [p.equity for p in equity_curve]
    final = equities[-1] if equities else starting_equity
    total_return = (final / starting_equity - 1.0) if starting_equity > 0 else 0.0

    returns = _bar_returns(equities)
    vol = _stdev(returns)
    mean = sum(returns) / len(returns) if returns else 0.0

    # Annualise from the actual bar frequency, not a hardcoded trading-day count.
    ann_factor = math.sqrt(bars_per_year) if bars_per_year > 0 else 0.0
    volatility = vol * ann_factor
    sharpe = (mean / vol) * ann_factor if vol > 0 else 0.0

    downside = [r for r in returns if r < 0]
    down_dev = _stdev(downside) if downside else 0.0
    sortino = (mean / down_dev) * ann_factor if down_dev > 0 else 0.0

    max_dd, dd_bars = _max_drawdown(equities)
    periods = len(returns) / bars_per_year if bars_per_year > 0 else 0.0
    annualised = _annualise(final / starting_equity, periods) if starting_equity > 0 else 0.0
    calmar = (annualised / max_dd) if max_dd > 1e-12 else 0.0

    closed = [t for t in trades if not t.is_open]
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    hit_rate = len(wins) / len(closed) if closed else 0.0
    # A strategy with no losses has an undefined profit factor; report it as infinite
    # rather than silently dividing by zero or reporting a misleading 0.
    profit_factor = (gross_win / gross_loss) if gross_loss > 1e-12 else (
        float("inf") if gross_win > 0 else 0.0
    )
    expectancy = (sum(t.pnl for t in closed) / len(closed)) if closed else 0.0

    rejections: dict[str, int] = {}
    for decision in decisions:
        for reason in decision.rejected:
            rejections[reason] = rejections.get(reason, 0) + 1

    exposure = 0.0
    if equity_curve:
        exposure = sum(
            p.gross_exposure / p.equity for p in equity_curve if p.equity > 0
        ) / len(equity_curve)

    # R statistics, and the dispersion figure that says whether to believe them.
    # Only trades whose planned risk actually arrived can contribute: a trade with no
    # recorded risk reports 0.0R, which would otherwise be averaged in as a scratch.
    measurable = [t for t in closed if t.risk_known]
    risks = sorted(t.initial_risk for t in measurable)
    total_risk = sum(risks)
    mean_r = (
        sum(t.r_multiple for t in measurable) / len(measurable) if measurable else 0.0
    )
    risk_weighted_r = (
        sum(t.pnl for t in measurable) / total_risk if total_risk > 1e-12 else 0.0
    )
    median_risk = _median(risks)
    min_risk = risks[0] if risks else 0.0
    max_risk = risks[-1] if risks else 0.0
    dispersion = (max_risk / min_risk) if min_risk > 1e-12 else 0.0

    return BacktestResult(
        starting_equity=starting_equity,
        final_equity=final,
        total_return=total_return,
        annualised_return=annualised,
        volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        max_drawdown_bars=dd_bars,
        calmar=calmar,
        trades=len(closed),
        wins=len(wins),
        losses=len(losses),
        hit_rate=hit_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_win=(gross_win / len(wins)) if wins else 0.0,
        avg_loss=(-gross_loss / len(losses)) if losses else 0.0,
        largest_win=max((t.pnl for t in closed), default=0.0),
        largest_loss=min((t.pnl for t in closed), default=0.0),
        total_fees=sum(f.commission for f in fills),
        fills=len(fills),
        bars=len(equity_curve),
        exposure=exposure,
        mean_r=mean_r,
        risk_weighted_r=risk_weighted_r,
        median_risk=median_risk,
        min_risk=min_risk,
        max_risk=max_risk,
        risk_dispersion=dispersion,
        trades_without_risk=len(closed) - len(measurable),
        rejections=rejections,
        equity_curve=equity_curve,
        trade_list=closed,
    )


def _median(sorted_values: list[float]) -> float:
    """Median of an already-sorted list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _annualise(growth: float, periods: float) -> float:
    """Compound a total growth factor over a fractional number of years.

    Done in log space because the direct form overflows. Annualising a two-bar sample of
    5-minute crypto data raises a growth factor to the power of a hundred thousand, and
    ``**`` raises ``OverflowError`` rather than returning infinity - which crashed metrics
    for any very short run instead of reporting a meaningless number as meaningless.

    Returns infinity, rather than raising, when the extrapolation genuinely does exceed
    what a float can hold. A sample that short cannot support an annual figure at all, and
    an obviously absurd value is more honest than a plausible-looking one.
    """
    if periods <= 0 or growth <= 0:
        return 0.0
    exponent = math.log(growth) / periods
    if exponent > 709.0:
        return float("inf")
    if exponent < -709.0:
        return -1.0
    return math.exp(exponent) - 1.0

def _bar_returns(equities: list[float]) -> list[float]:
    out: list[float] = []
    for prev, curr in zip(equities, equities[1:]):
        if prev > 0:
            out.append(curr / prev - 1.0)
    return out


def _stdev(values: list[float]) -> float:
    """Sample standard deviation, guarding the single-observation case."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(0.0, var))


def _max_drawdown(equities: list[float]) -> tuple[float, int]:
    """Worst peak-to-trough fall, and how many bars it lasted."""
    peak = float("-inf")
    peak_index = 0
    worst = 0.0
    worst_bars = 0
    for i, value in enumerate(equities):
        if value > peak:
            peak = value
            peak_index = i
        if peak > 0:
            dd = (peak - value) / peak
            if dd > worst:
                worst = dd
                worst_bars = i - peak_index
    return worst, worst_bars
