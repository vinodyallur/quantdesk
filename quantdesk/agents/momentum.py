"""Trend-following agent."""

from __future__ import annotations

from quantdesk.agents.base import AgentContext, SignalAgent, linear_map, squash
from quantdesk.core.types import Signal


class MomentumAgent(SignalAgent):
    """Buys established up-trends and sells established down-trends.

    The raw view is the fast/slow EMA spread, but expressed in ATR units rather
    than percent. A 1% spread means something very different in a 0.5%-vol asset
    than in a 5%-vol asset, and scoring on raw percent would make the agent
    permanently over-committed to whatever is most volatile.

    Conviction is then adjusted for two things that make a trend signal less
    trustworthy:

    * Disagreement between the EMA spread and the slope of the slow EMA, which
      usually means the trend is rolling over.
    * Stretched RSI, where a trend is most vulnerable to a sharp mean reversion.
      This trims size rather than flipping direction; fading strong trends outright
      is a well-known way to lose money slowly.
    """

    name = "momentum"
    weight = 1.0

    def __init__(
        self,
        spread_scale: float = 1.5,
        rsi_fade_start: float = 75.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.spread_scale = spread_scale
        self.rsi_fade_start = rsi_fade_start

    def evaluate(self, ctx: AgentContext) -> Signal | None:
        f = ctx.snapshot
        if f.atr_pct <= 1e-9:
            return None

        # EMA spread measured in ATR units: a vol-normalised trend strength.
        spread_atr = f.ema_spread / f.atr_pct
        score = squash(spread_atr, self.spread_scale)
        if abs(score) < 0.05:
            return None

        # Slope agreement: is the slow EMA still moving our way?
        slope_atr = f.trend_slope / f.atr_pct
        agrees = (slope_atr > 0) == (score > 0)
        confidence = 0.85 if agrees else 0.40

        # Longer-horizon rate of change as a second confirmation.
        if (f.roc_slow > 0) == (score > 0):
            confidence = min(1.0, confidence + 0.15)

        # Trim conviction when RSI is stretched in our direction of travel.
        rsi_extreme = f.rsi if score > 0 else (100.0 - f.rsi)
        if rsi_extreme > self.rsi_fade_start:
            confidence *= linear_map(rsi_extreme, self.rsi_fade_start, 95.0, 1.0, 0.35)

        return self.signal(
            ctx,
            score=score,
            confidence=confidence,
            horizon_bars=24,
            rationale=(
                f"EMA spread {spread_atr:+.2f} ATR, slope "
                f"{'with' if agrees else 'against'} trend, RSI {f.rsi:.0f}"
            ),
            spread_atr=spread_atr,
            slope_atr=slope_atr,
            rsi=f.rsi,
        )
