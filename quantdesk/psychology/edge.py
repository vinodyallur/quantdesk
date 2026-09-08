"""Edge statistics: is the sample big enough to mean anything?

Douglas's central mechanical claim: "Events that have probable outcomes can produce
consistent results, if you can get the odds in your favor and there is a large enough
sample size." He illustrates it with blackjack, where the house edge of about 4.5%
only shows up over many hands and says nothing about the next one.

Two things follow, and both are implemented here.

**A losing streak is not evidence.** He points out you might know that "out of the next
20 trades, 12 will be winners and 8 will be losers", but not the sequence. So when the
desk has five losses in a row, the useful question is not "is the edge broken?" but
"how surprising is a run of five at this win rate?" :func:`streak_probability` answers
it, and the answer is usually "not at all surprising" - which is exactly the
intervention that panic would prevent.

**Small samples cannot be judged.** :meth:`EdgeStats.reliable` refuses to draw a
conclusion below a minimum count, and the confidence interval on the win rate is
reported alongside the point estimate so a 60% win rate from 10 trades is visibly
different from 60% from 400.

Expectancy is reported in R - multiples of the risk taken - because that is the only
scale on which trades of different sizes and volatilities are comparable, and it is
what makes "the dollar amount of the winning trades is greater than that of the
losing trades" measurable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(slots=True)
class TradeOutcome:
    """One completed trade, reduced to what matters for edge measurement."""

    symbol: str
    r_multiple: float
    """Profit or loss in units of the risk originally taken."""
    pnl: float = 0.0
    agent: str = ""
    won: bool = False

    def __post_init__(self) -> None:
        self.won = self.r_multiple > 0


@dataclass
class EdgeStats:
    """Measured behaviour of an edge over a sample of trades."""

    count: int = 0
    wins: int = 0
    losses: int = 0
    total_r: float = 0.0
    win_r: float = 0.0
    loss_r: float = 0.0
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    current_streak: int = 0
    """Positive for consecutive wins, negative for consecutive losses."""
    min_sample: int = 30

    @property
    def win_rate(self) -> float:
        return self.wins / self.count if self.count else 0.0

    @property
    def expectancy_r(self) -> float:
        """Average R per trade. The number that decides whether the edge is real."""
        return self.total_r / self.count if self.count else 0.0

    @property
    def avg_win_r(self) -> float:
        return self.win_r / self.wins if self.wins else 0.0

    @property
    def avg_loss_r(self) -> float:
        return self.loss_r / self.losses if self.losses else 0.0

    @property
    def payoff_ratio(self) -> float:
        """Average win over average loss. Murphy's asymmetry, measured."""
        avg_loss = abs(self.avg_loss_r)
        return self.avg_win_r / avg_loss if avg_loss > 1e-12 else 0.0

    @property
    def reliable(self) -> bool:
        """Is the sample large enough to say anything at all?"""
        return self.count >= self.min_sample

    @property
    def profitable(self) -> bool:
        return self.expectancy_r > 0

    def win_rate_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Wilson score interval on the win rate.

        Reported next to the point estimate so a small sample looks small. At 10
        trades the interval is so wide as to be useless, which is the honest answer.

        Wilson rather than the textbook normal approximation, because the normal
        approximation collapses to zero width at a 0% or 100% observed rate - it would
        claim certainty from nine wins out of nine, which is precisely the small-sample
        overconfidence this method exists to prevent.
        """
        n = self.count
        if n <= 0:
            return 0.0, 1.0
        p = self.win_rate
        z2 = z * z
        denominator = 1.0 + z2 / n
        centre = (p + z2 / (2.0 * n)) / denominator
        margin = (
            z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
        ) / denominator
        return max(0.0, centre - margin), min(1.0, centre + margin)

    def breakeven_win_rate(self) -> float:
        """Win rate needed to break even at the observed payoff ratio.

        The comparison that matters: a 35% win rate is excellent at 3:1 and fatal at
        1:1, and looking at the hit rate alone cannot tell them apart.
        """
        payoff = self.payoff_ratio
        if payoff <= 0:
            return 1.0
        return 1.0 / (1.0 + payoff)

    @property
    def edge_margin(self) -> float:
        """Observed win rate minus the win rate needed to break even."""
        return self.win_rate - self.breakeven_win_rate()

    def summary(self) -> list[str]:
        lo, hi = self.win_rate_interval()
        lines = [
            f"sample       {self.count} trades"
            + ("" if self.reliable else f"  (below {self.min_sample}: NOT reliable)"),
            f"win rate     {self.win_rate:.1%}  (95% CI {lo:.1%} to {hi:.1%})",
            f"expectancy   {self.expectancy_r:+.3f}R per trade",
            f"payoff       {self.payoff_ratio:.2f} : 1  "
            f"(avg win {self.avg_win_r:+.2f}R, avg loss {self.avg_loss_r:+.2f}R)",
            f"breakeven    needs {self.breakeven_win_rate():.1%} at this payoff, "
            f"margin {self.edge_margin:+.1%}",
            f"streaks      longest {self.longest_win_streak}W / "
            f"{self.longest_loss_streak}L, currently {self.current_streak:+d}",
        ]
        return lines


def measure(outcomes: list[TradeOutcome], min_sample: int = 30) -> EdgeStats:
    """Reduce a list of trade outcomes to edge statistics."""
    stats = EdgeStats(min_sample=min_sample)
    streak = 0
    for outcome in outcomes:
        stats.count += 1
        stats.total_r += outcome.r_multiple
        if outcome.won:
            stats.wins += 1
            stats.win_r += outcome.r_multiple
            streak = streak + 1 if streak > 0 else 1
            stats.longest_win_streak = max(stats.longest_win_streak, streak)
        else:
            stats.losses += 1
            stats.loss_r += outcome.r_multiple
            streak = streak - 1 if streak < 0 else -1
            stats.longest_loss_streak = max(stats.longest_loss_streak, -streak)
    stats.current_streak = streak
    return stats


def streak_probability(length: int, win_rate: float, trades: int) -> float:
    """Rough chance of seeing a losing run of at least ``length`` in ``trades``.

    The point of this function is to defuse panic. A 45% win rate over 200 trades makes
    a run of eight losses more likely than not, so treating one as evidence the edge
    has broken is a misreading of Douglas's third truth: "there is a random
    distribution between wins and losses for any given set of variables that define an
    edge."

    Uses the standard approximation - expected number of runs times the probability of
    a run starting - which is accurate enough for the judgement being made and honest
    about being an approximation.
    """
    if length <= 0 or trades <= 0:
        return 1.0
    loss_rate = 1.0 - max(0.0, min(1.0, win_rate))
    if loss_rate <= 0:
        return 0.0
    if loss_rate >= 1:
        return 1.0
    p_run = loss_rate ** length
    opportunities = max(1, trades - length + 1)
    # Probability at least one such run occurs, treating starts as near-independent.
    return 1.0 - (1.0 - p_run) ** opportunities


def expected_worst_streak(win_rate: float, trades: int) -> int:
    """Longest losing run you should *expect* over a sample.

    Useful as a drawdown expectation before the fact. Knowing in advance that nine
    consecutive losses are normal is what makes the ninth one survivable.
    """
    loss_rate = 1.0 - max(1e-9, min(1.0 - 1e-9, win_rate))
    if trades <= 0:
        return 0
    # Expected maximum run length for a Bernoulli sequence.
    return max(1, int(math.log(trades) / math.log(1.0 / loss_rate)))


def is_streak_alarming(
    observed_streak: int, win_rate: float, trades: int, threshold: float = 0.05
) -> tuple[bool, str]:
    """Should a losing run actually cause concern?

    Returns ``(alarming, explanation)``. Deliberately hard to alarm: the default
    threshold means the run has to be in the least likely 5% of outcomes before the
    desk says anything, because the far more common error is abandoning a working edge
    after a normal run of losses.
    """
    if observed_streak <= 0:
        return False, "no losing streak in progress"
    probability = streak_probability(observed_streak, win_rate, max(trades, observed_streak))
    expected = expected_worst_streak(win_rate, max(trades, 1))
    if observed_streak <= expected:
        return False, (
            f"{observed_streak} losses in a row is within the {expected} expected at a "
            f"{win_rate:.0%} win rate over {trades} trades"
        )
    if probability >= threshold:
        return False, (
            f"a run of {observed_streak} had a {probability:.0%} chance of appearing in "
            f"{trades} trades, which is not unusual"
        )
    return True, (
        f"a run of {observed_streak} had only a {probability:.1%} chance in "
        f"{trades} trades, which is worth investigating"
    )
