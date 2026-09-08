"""Agents that trade formations: chart patterns, candlesticks and Elliott waves.

Three agents on three different horizons, which is why they are separate rather than
one "pattern" agent:

* :class:`PatternAgent` trades completed chart patterns. Intermediate horizon,
  measured objectives, and a real 3:1 reward-to-risk gate.
* :class:`CandleAgent` trades candlestick reversals. Short horizon by nature, so its
  signals decay fast and its weight is modest.
* :class:`ElliottAgent` trades wave position. Longest horizon, lowest confidence,
  because wave counting is the most subjective technique in the book and this agent
  refuses to act on a count with rule violations.

All three inherit the same non-negotiable: nothing trades against the higher
timeframe, and nothing trades a formation that has not completed. Murphy is
consistent that a forming pattern is not a pattern.
"""

from __future__ import annotations

from quantdesk.agents.base import AgentContext, SignalAgent
from quantdesk.analysis.patterns import Bias, PatternStage, volume_confirms
from quantdesk.core.types import Signal


class PatternAgent(SignalAgent):
    """Trades completed chart patterns with measured objectives."""

    name = "murphy_patterns"
    weight = 2.0
    warmup = 60

    def __init__(
        self,
        min_significance: float = 0.35,
        min_reward_risk: float = 3.0,
        require_volume: bool = True,
        max_age_bars: int = 20,
        name: str | None = None,
        weight: float | None = None,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.min_significance = min_significance
        #: Murphy's chapter 16 yardstick, applied to the pattern's own objective.
        self.min_reward_risk = min_reward_risk
        self.require_volume = require_volume
        #: A breakout twenty bars ago is history, not a signal.
        self.max_age_bars = max_age_bars

    def evaluate(self, ctx: AgentContext) -> Signal | None:
        sa = ctx.analysis
        if sa is None or not sa.ready:
            return None

        candidates = sa.patterns.actionable(
            min_significance=self.min_significance,
            min_reward_risk=self.min_reward_risk,
            require_volume=self.require_volume,
        )
        # Freshness. The measured move is expected *from* the breakout, so a stale
        # breakout has already given away most of the opportunity.
        candidates = [
            p for p in candidates if sa.index - p.breakout_index <= self.max_age_bars
        ]
        # And it must not already have reached its objective.
        candidates = [p for p in candidates if not self._objective_met(p, sa)]
        if not candidates:
            return None

        best = candidates[0]
        if not sa.timeframes.agrees(best.bias):
            return None

        significance = best.significance(sa.index, sa.atr.value)
        # Patterns pointing the other way argue against this one. Murphy's own
        # treatment expects formations to overlap and sometimes conflict, so the net
        # of them is more honest than the single strongest.
        agreeing = sum(
            p.significance(sa.index, sa.atr.value)
            for p in candidates
            if p.bias is best.bias
        )
        opposing = sum(
            p.significance(sa.index, sa.atr.value)
            for p in candidates
            if p.bias is best.bias.opposite
        )
        total = agreeing + opposing
        if total <= 0:
            return None
        consensus = (agreeing - opposing) / total
        if consensus <= 0:
            return None

        stage_bonus = 0.1 if best.stage is PatternStage.RETURN_MOVE else 0.0
        confidence = max(0.0, min(1.0, significance * consensus + stage_bonus))
        if confidence < 0.1:
            return None

        return self.signal(
            ctx,
            score=best.bias.sign * min(1.0, 0.5 + 0.5 * consensus),
            confidence=confidence,
            horizon_bars=max(12, best.bars),
            rationale=(
                f"{best.kind.label} {best.stage.value}, target {best.objective:.6g}, "
                f"stop {best.stop:.6g}, R:R {best.reward_risk:.1f}, "
                f"vol x{best.breakout_volume_ratio:.1f}"
                + (f"; {len(candidates)} patterns live" if len(candidates) > 1 else "")
            ),
            objective=best.objective,
            stop=best.stop,
            reward_risk=best.reward_risk,
            significance=significance,
            consensus=consensus,
        )

    def _objective_met(self, pattern, sa) -> bool:
        bar = sa.last_bar
        if bar is None or pattern.objective <= 0:
            return False
        if pattern.bias is Bias.BULLISH:
            return bar.high >= pattern.objective
        return bar.low <= pattern.objective

    def plan_for(self, ctx: AgentContext):
        """The best tradable pattern, for the money manager to size.

        Exposed separately because a pattern carries its own stop and objective, and
        throwing those away to size off a generic ATR stop would discard the most
        useful thing the formation gives you.
        """
        sa = ctx.analysis
        if sa is None:
            return None
        candidates = sa.patterns.actionable(
            min_significance=self.min_significance,
            min_reward_risk=self.min_reward_risk,
            require_volume=self.require_volume,
        )
        for pattern in candidates:
            if sa.index - pattern.breakout_index > self.max_age_bars:
                continue
            if not sa.timeframes.agrees(pattern.bias):
                continue
            if pattern.stop > 0 and pattern.objective > 0:
                return pattern
        return None


class CandleAgent(SignalAgent):
    """Trades candlestick reversals, gated by the trend they must oppose."""

    name = "murphy_candles"
    weight = 0.8
    warmup = 40

    def __init__(
        self,
        min_strength: float = 0.55,
        min_sessions: int = 2,
        within_bars: int = 3,
        require_level: bool = True,
        name: str | None = None,
        weight: float | None = None,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.min_strength = min_strength
        #: Single-session shapes are the weakest evidence, so two sessions minimum by
        #: default. Murphy calls the three-day stars the ones that "work exceptionally
        #: well".
        self.min_sessions = min_sessions
        self.within_bars = within_bars
        #: A reversal candle matters far more at a support or resistance zone than in
        #: open space. This is not stated as a rule but follows directly from his
        #: treatment of levels.
        self.require_level = require_level

    def evaluate(self, ctx: AgentContext) -> Signal | None:
        sa = ctx.analysis
        if sa is None or not sa.ready:
            return None

        recent = [
            s
            for s in sa.candles.recent(self.within_bars)
            if s.bias is not Bias.NEUTRAL
            and s.strength >= self.min_strength
            and s.sessions >= self.min_sessions
        ]
        if not recent:
            return None

        best = max(recent, key=lambda s: (s.strength, s.sessions))
        # The reader already enforced that a reversal opposes the trend; this checks
        # the trade is allowed by the slower timeframes.
        if not sa.timeframes.agrees(best.bias):
            return None

        reasons = [f"{best.kind.label} (strength {best.strength:.2f})"]

        if self.require_level:
            level, distance = sa.levels.proximity(sa.price, sa.atr.value)
            if level is None or distance > 1.0:
                return None
            reasons.append(f"at a {level.role.value} zone {distance:.1f} ATR away")

        confidence = min(1.0, best.strength * (0.7 + 0.1 * best.sessions))
        return self.signal(
            ctx,
            score=best.bias.sign * min(1.0, 0.4 + 0.4 * best.strength),
            confidence=confidence,
            # Candle signals are near-term. Murphy places them among the short-term
            # timing tools, not the directional ones.
            horizon_bars=8,
            rationale="; ".join(reasons),
            candle_strength=best.strength,
            sessions=float(best.sessions),
            volume_ratio=best.volume_ratio,
        )


class ElliottAgent(SignalAgent):
    """Trades wave position, and only from a count with no rule violations."""

    name = "murphy_elliott"
    weight = 0.7
    warmup = 80

    def __init__(
        self,
        min_confidence: float = 0.6,
        name: str | None = None,
        weight: float | None = None,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.min_confidence = min_confidence

    def evaluate(self, ctx: AgentContext) -> Signal | None:
        sa = ctx.analysis
        if sa is None or not sa.ready:
            return None

        best = sa.elliott.best
        if best is None:
            return None
        # A count that breaks Elliott's rules is not a weaker count, it is the wrong
        # count. There is no confidence discount that makes it tradable.
        if not best.valid:
            return None
        if best.confidence < self.min_confidence:
            return None

        bias = sa.elliott.bias()
        if bias is Bias.NEUTRAL:
            return None
        if not sa.timeframes.agrees(bias):
            return None

        targets = sa.elliott.targets()
        target_note = ""
        for key in ("wave3_min", "wave5_min"):
            if key in targets:
                target_note = f", {key} {targets[key]:.6g}"
                break

        return self.signal(
            ctx,
            score=bias.sign * 0.5,
            confidence=min(1.0, best.confidence * 0.8),
            horizon_bars=60,
            rationale=f"{sa.elliott.position()}{target_note}",
            wave_confidence=best.confidence,
            waves=float(len(best.waves)),
        )


def murphy_agents() -> list[SignalAgent]:
    """The full Murphy agent set, weighted by how much the book trusts each idea.

    Weights follow his emphasis rather than being uniform: the checklist and chart
    patterns carry most, structure close behind, candlesticks and Elliott least -
    the first because they are near-term timing tools, the second because he is
    candid about how subjective wave counting is.
    """
    from quantdesk.agents.murphy import (
        DivergenceAgent,
        MurphyChecklistAgent,
        StructureAgent,
    )

    return [
        MurphyChecklistAgent(),
        PatternAgent(),
        StructureAgent(),
        DivergenceAgent(),
        CandleAgent(),
        ElliottAgent(),
    ]
