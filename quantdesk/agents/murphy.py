"""Agents that trade Murphy's method directly.

Two agents here, and they answer different questions.

:class:`MurphyChecklistAgent` is the flagship. It does not invent a strategy - it
reads the chapter 19 checklist verdict and trades it. That is deliberate: the whole
point of the checklist is that no single technique is trusted alone, so an agent
that consults twenty-three questions and weighs them is a more faithful
implementation of the book than twenty-three agents each shouting one answer.

:class:`StructureAgent` trades the chapter 4 material on its own - trend direction,
trendline integrity, room to the next level, and Murphy's 40-60% retracement entry
zone. It exists separately because structure is the part of his framework with the
best signal-to-noise ratio, and keeping it independent lets the blender see when
structure and the wider checklist disagree.

Both refuse to trade against the higher timeframe. That is not a tunable
preference; it is his stated procedure.
"""

from __future__ import annotations

from quantdesk.agents.base import AgentContext, SignalAgent, linear_map
from quantdesk.analysis.patterns import Bias
from quantdesk.core.types import Signal


class MurphyChecklistAgent(SignalAgent):
    """Trades the weight of evidence from Murphy's chapter 19 checklist."""

    name = "murphy_checklist"
    weight = 2.0
    warmup = 60

    def __init__(
        self,
        min_conviction: float = 0.20,
        min_coverage: float = 0.55,
        name: str | None = None,
        weight: float | None = None,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.min_conviction = min_conviction
        self.min_coverage = min_coverage

    def evaluate(self, ctx: AgentContext) -> Signal | None:
        verdict = ctx.verdict
        if verdict is None:
            return None

        # Blockers are hard. A verdict opposing the higher timeframes, or built from
        # a mostly-unanswered checklist, is not a weak signal - it is not a signal.
        if verdict.blockers:
            return None
        if verdict.bias is Bias.NEUTRAL:
            return None
        if verdict.coverage < self.min_coverage:
            return None
        if verdict.conviction < self.min_conviction:
            return None

        # Score carries direction and strength of evidence; confidence carries how
        # much of the checklist agreed. Murphy's synthesis is exactly this split -
        # what the evidence says, and how unanimous it was.
        score = verdict.score
        confidence = min(1.0, 0.35 + 0.65 * verdict.agreement)

        bull = len(verdict.bullish)
        bear = len(verdict.bearish)
        return self.signal(
            ctx,
            score=score,
            confidence=confidence,
            horizon_bars=48,
            rationale=(
                f"checklist {verdict.bias.value}: {bull} bullish vs {bear} bearish "
                f"answers, agreement {verdict.agreement:.0%}, "
                f"coverage {verdict.coverage:.0%}"
            ),
            checklist_score=verdict.score,
            conviction=verdict.conviction,
            agreement=verdict.agreement,
            coverage=verdict.coverage,
        )


class StructureAgent(SignalAgent):
    """Trades trend structure: Dow direction, trendlines, levels and retracements."""

    name = "murphy_structure"
    weight = 1.5
    warmup = 60

    def __init__(
        self,
        min_headroom_atr: float = 1.5,
        name: str | None = None,
        weight: float | None = None,
    ) -> None:
        super().__init__(name=name, weight=weight)
        #: Murphy's tactic is to buy where there is room to the next resistance, not
        #: where price is pressed against it.
        self.min_headroom_atr = min_headroom_atr

    def evaluate(self, ctx: AgentContext) -> Signal | None:
        sa = ctx.analysis
        if sa is None or not sa.ready or sa.trend is None:
            return None

        bias = sa.trend_bias
        if bias is Bias.NEUTRAL:
            return None

        # "Work from the long term to the short term." The higher timeframes decide
        # what is permitted before anything else is considered.
        if not sa.timeframes.agrees(bias):
            return None

        long_side = bias is Bias.BULLISH
        reasons: list[str] = [f"{sa.trend.direction.value} trend ({sa.degree})"]

        # Trend health. Murphy's early-warning states - a failed peak, a broken
        # trendline - are reasons to stand aside, not to reverse.
        health = sa.trend.health.value
        if health not in ("intact", "confirmed"):
            return None
        strength = sa.trend.confidence

        # Room to run. A trade with no headroom has no reward to balance its risk.
        headroom = sa.headroom(long_side)
        if headroom < self.min_headroom_atr:
            return None
        if headroom != float("inf"):
            reasons.append(f"{headroom:.1f} ATR of headroom")
        else:
            reasons.append("no level in the way")

        # Trendline confirmation adds weight; a broken one against us removes it.
        valid_lines = sa.trendlines.valid_lines
        line_bonus = 0.0
        for line in valid_lines:
            line_up = line.kind.value == "uptrend"
            if line_up is long_side:
                line_bonus = min(0.25, 0.08 * line.touches)
                reasons.append(f"{line.touches}-touch trendline holding")
                break
        # The fan principle: three broken lines is his reversal signal, so it argues
        # against continuing to trade the old direction.
        if sa.trendlines.fan_count >= 3:
            return None

        # Retracement entry. His 40-60% pullback zone is the preferred entry, and
        # past 66% the move is a reversal rather than a correction.
        retr_bonus = 0.0
        if sa.retracements is not None:
            frac = sa.retracements.signed_retracement_of(sa.price)
            leg_up = sa.retracements.is_up_leg
            if leg_up is long_side and 0.0 <= frac <= 0.66:
                if 0.40 <= frac <= 0.60:
                    retr_bonus = 0.25
                    reasons.append(f"in the {frac:.0%} retracement entry zone")
                else:
                    retr_bonus = 0.10
                    reasons.append(f"pulled back {frac:.0%}")
            elif leg_up is long_side and frac > 0.66:
                # Beyond two thirds the premise has failed.
                return None

        # ADX: Wilder's observation, which Murphy quotes, is that markets trend
        # strongly only about 30% of the time. Below 20 there is no trend to follow.
        adx_scalar = 0.4
        if sa.dmi.ready:
            if sa.dmi.adx < 20:
                return None
            adx_scalar = linear_map(sa.dmi.adx, 20.0, 40.0, 0.4, 1.0)
            reasons.append(f"ADX {sa.dmi.adx:.0f}")

        score = bias.sign * min(1.0, 0.45 + line_bonus + retr_bonus)
        confidence = max(0.0, min(1.0, strength * adx_scalar))
        if confidence <= 0.05:
            return None

        return self.signal(
            ctx,
            score=score,
            confidence=confidence,
            horizon_bars=36,
            rationale="; ".join(reasons),
            adx=sa.dmi.adx,
            headroom_atr=0.0 if headroom == float("inf") else headroom,
            trend_confidence=strength,
        )


class DivergenceAgent(SignalAgent):
    """Trades oscillator divergence against an overextended move.

    Murphy rates divergence among the most valuable oscillator signals: price makes a
    new extreme, the oscillator does not, and the move is running out of participants.

    This is the one Murphy agent that deliberately trades *against* the prevailing
    trend, so it is the most dangerous, and it is gated hardest: divergence must
    appear on more than one oscillator, and the move must already be extended. He is
    clear that in a strong trend an overbought oscillator can stay overbought for a
    long time, so divergence alone is a warning rather than a trade.
    """

    name = "murphy_divergence"
    weight = 1.0
    warmup = 60

    def __init__(
        self,
        min_oscillators: int = 2,
        name: str | None = None,
        weight: float | None = None,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.min_oscillators = min_oscillators

    def evaluate(self, ctx: AgentContext) -> Signal | None:
        sa = ctx.analysis
        if sa is None or not sa.ready or not sa.divergences:
            return None

        bullish = [d for d in sa.divergences if d.bias is Bias.BULLISH]
        bearish = [d for d in sa.divergences if d.bias is Bias.BEARISH]
        side, group = (
            (Bias.BULLISH, bullish) if len(bullish) > len(bearish) else (Bias.BEARISH, bearish)
        )
        if len(group) < self.min_oscillators:
            return None

        # Divergence only means something against a move that has actually run.
        if side is Bias.BEARISH and sa.trend_bias is not Bias.BULLISH:
            return None
        if side is Bias.BULLISH and sa.trend_bias is not Bias.BEARISH:
            return None

        # Require the oscillator to be at an extreme as well, which is what makes it
        # a warning about exhaustion rather than ordinary noise.
        rsi = sa.rsi.value
        if side is Bias.BEARISH and rsi < 60:
            return None
        if side is Bias.BULLISH and rsi > 40:
            return None

        names = ", ".join(d.name for d in group)
        confidence = min(0.7, 0.25 + 0.15 * len(group))
        return self.signal(
            ctx,
            score=side.sign * 0.5,
            confidence=confidence,
            horizon_bars=18,
            rationale=(
                f"{len(group)}-oscillator {side.value} divergence ({names}) "
                f"against a {sa.trend_bias.value} move, RSI {rsi:.0f}"
            ),
            divergence_count=float(len(group)),
            rsi=rsi,
        )
