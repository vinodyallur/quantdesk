"""Kelly sizing, and why the desk does not use full Kelly.

Kelly gives the position fraction that maximises the long-run growth rate of capital.
It is the mathematically optimal answer to "how much?" - and it is almost always the
wrong number to trade, for three reasons that matter more than the optimality:

**It maximises growth, not survival.** Full Kelly produces drawdowns that approach
100% with probability one over a long enough horizon. Murphy's chapter 16 limits exist
precisely to prevent that, which is why :func:`kelly_position` is capped by the money
manager rather than overriding it.

**It assumes you know the true win rate.** You have an *estimate* from a finite sample,
and Kelly is extremely sensitive to it: overestimating the edge by a little
overbets by a lot. :func:`kelly_fraction` therefore takes the *lower* confidence bound
on the win rate, not the point estimate. Sizing off the optimistic end of a confidence
interval is how accounts die.

**It assumes fixed odds.** Trading payoffs vary per trade; the payoff ratio here is an
average over past trades and the next one may not resemble it.

So the desk uses fractional Kelly - typically a quarter to a half - as an *upper bound*
that Murphy's limits then bind further. Kelly's real use is diagnostic: if Kelly says
zero or negative, the measured edge does not justify any position at all, and that is
worth knowing regardless of what the signal says.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantdesk.psychology.edge import EdgeStats


@dataclass(slots=True)
class KellySizing:
    """Kelly's answer, and the fraction of it the desk will actually use."""

    full_kelly: float
    """Optimal growth fraction. Rarely tradable."""
    fraction_used: float
    recommended: float
    """``full_kelly * fraction_used``, floored at zero."""
    win_rate_used: float
    """The conservative win rate the calculation was based on."""
    payoff_ratio: float
    reliable: bool
    note: str = ""

    @property
    def positive_edge(self) -> bool:
        return self.full_kelly > 0

    def summary(self) -> list[str]:
        lines = [
            f"full kelly   {self.full_kelly:+.1%} of equity",
            f"using        {self.fraction_used:.0%} kelly -> {self.recommended:.1%}",
            f"based on     win rate {self.win_rate_used:.1%} (lower bound), "
            f"payoff {self.payoff_ratio:.2f}:1",
        ]
        if not self.reliable:
            lines.append("             SAMPLE TOO SMALL - treat as indicative only")
        if self.note:
            lines.append(f"             {self.note}")
        return lines


def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """The Kelly formula for a two-outcome bet.

    ``f = p - q/b`` where p is the win probability, q is 1-p, and b is the payoff
    ratio. Returns zero rather than a negative number when the edge is negative: a
    negative Kelly technically means bet the other way, but for a directional strategy
    it means "do not trade this", and returning a negative size invites a caller to
    accidentally invert it.
    """
    p = max(0.0, min(1.0, win_rate))
    if payoff_ratio <= 0:
        return 0.0
    f = p - (1.0 - p) / payoff_ratio
    return max(0.0, f)


def from_edge_stats(
    stats: EdgeStats,
    fraction: float = 0.25,
    use_lower_bound: bool = True,
) -> KellySizing:
    """Kelly sizing from measured trade history.

    ``use_lower_bound`` is on by default and matters more than the fraction. Kelly's
    sensitivity to the win rate estimate means using the point estimate systematically
    overbets - the estimate is as likely to be too high as too low, but the cost of
    being too high is far greater than the benefit of being right.
    """
    payoff = stats.payoff_ratio
    if use_lower_bound:
        win_rate, _ = stats.win_rate_interval()
    else:
        win_rate = stats.win_rate

    full = kelly_fraction(win_rate, payoff)
    fraction = max(0.0, min(1.0, fraction))
    note = ""
    if payoff <= 0:
        note = "no completed losing trades yet, so the payoff ratio is undefined"
    elif full <= 0:
        note = (
            f"a {win_rate:.0%} win rate at {payoff:.2f}:1 does not justify a position; "
            f"breakeven needs {stats.breakeven_win_rate():.0%}"
        )

    return KellySizing(
        full_kelly=full,
        fraction_used=fraction,
        recommended=full * fraction,
        win_rate_used=win_rate,
        payoff_ratio=payoff,
        reliable=stats.reliable,
        note=note,
    )


def kelly_position(
    stats: EdgeStats,
    equity: float,
    risk_per_unit: float,
    murphy_risk_cap: float = 0.05,
    fraction: float = 0.25,
) -> tuple[float, str]:
    """Position size in units, with Kelly capped by Murphy's risk limit.

    Returns ``(qty, binding_constraint)``. Murphy's 5% cap wins whenever it is tighter,
    and it usually is - which is the intended relationship. Kelly is here to say when
    the edge does *not* justify the full allowance, not to authorise exceeding it.
    """
    if equity <= 0 or risk_per_unit <= 0:
        return 0.0, "no equity or no defined risk"

    sizing = from_edge_stats(stats, fraction=fraction)
    # Reliability is checked *first*. An unreliable sample cannot support any verdict
    # about the edge, including a negative one - five wins and no losses is not
    # evidence of anything, and reading it as "no measurable payoff, size zero" would
    # stop the desk trading precisely when it has too little history to know better.
    if not sizing.reliable:
        qty = (murphy_risk_cap * equity) / risk_per_unit
        return qty, f"sample too small for Kelly, using the {murphy_risk_cap:.0%} cap"
    if not sizing.positive_edge:
        return 0.0, sizing.note or "no measured edge"

    kelly_risk = min(sizing.recommended, murphy_risk_cap)
    binding = (
        "fractional Kelly"
        if sizing.recommended < murphy_risk_cap
        else f"Murphy's {murphy_risk_cap:.0%} risk cap"
    )
    return (kelly_risk * equity) / risk_per_unit, binding


def growth_rate(win_rate: float, payoff_ratio: float, fraction: float) -> float:
    """Expected log growth per trade at a given bet fraction.

    Useful for showing the shape of the curve: it is flat near the optimum and falls
    off a cliff beyond it, which is the real argument for betting under Kelly rather
    than over. Half Kelly gives roughly three quarters of the growth at far less than
    half the risk of ruin.
    """
    import math

    p = max(0.0, min(1.0, win_rate))
    q = 1.0 - p
    if fraction <= 0:
        return 0.0
    win_leg = 1.0 + fraction * payoff_ratio
    loss_leg = 1.0 - fraction
    if win_leg <= 0 or loss_leg <= 0:
        # Beyond this fraction a single loss is ruinous; log growth is undefined.
        return float("-inf")
    return p * math.log(win_leg) + q * math.log(loss_leg)
