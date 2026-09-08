"""Mean-reversion agent."""

from __future__ import annotations

from quantdesk.agents.base import AgentContext, SignalAgent, linear_map, squash
from quantdesk.core.types import Signal


class MeanReversionAgent(SignalAgent):
    """Buys statistically cheap prices and sells rich ones, but only in ranges.

    Mean reversion and trend following disagree by construction, so this agent
    gates itself hard: if the vol-normalised EMA spread says a trend is in force,
    it stands down entirely rather than fighting it. Buying dips in a downtrend is
    the single most reliable way to lose money with this style.

    Within a range, the view is the negative price z-score, reinforced when RSI
    confirms an extreme. Conviction is cut when volatility is in its upper
    percentiles, because a vol expansion is more often the start of a breakout
    than a snap back.
    """

    name = "mean_reversion"
    weight = 0.9

    def __init__(
        self,
        z_scale: float = 1.6,
        trend_veto_atr: float = 1.2,
        min_abs_z: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.z_scale = z_scale
        self.trend_veto_atr = trend_veto_atr
        self.min_abs_z = min_abs_z

    def evaluate(self, ctx: AgentContext) -> Signal | None:
        f = ctx.snapshot
        if f.atr_pct <= 1e-9:
            return None

        # Stand down in a trending regime: this style has no edge there.
        spread_atr = abs(f.ema_spread / f.atr_pct)
        if spread_atr > self.trend_veto_atr:
            return None

        if abs(f.zscore) < self.min_abs_z:
            return None

        # Cheap (negative z) means buy, hence the sign flip.
        score = -squash(f.zscore, self.z_scale)

        # RSI agreement: oversold supports a long, overbought supports a short.
        if score > 0:
            rsi_support = linear_map(f.rsi, 45.0, 20.0, 0.5, 1.0)
        else:
            rsi_support = linear_map(f.rsi, 55.0, 80.0, 0.5, 1.0)
        confidence = rsi_support

        # High realised vol favours continuation over reversion: cut size.
        if f.vol_percentile > 0.75:
            confidence *= linear_map(f.vol_percentile, 0.75, 1.0, 1.0, 0.3)

        # A quieter regime than usual is the ideal setting for this style.
        if f.vol_percentile < 0.4:
            confidence = min(1.0, confidence * 1.15)

        if confidence < 0.15:
            return None

        return self.signal(
            ctx,
            score=score,
            confidence=confidence,
            horizon_bars=8,
            rationale=(
                f"z={f.zscore:+.2f} in range (trend {spread_atr:.2f} ATR), "
                f"RSI {f.rsi:.0f}, vol pct {f.vol_percentile:.0%}"
            ),
            zscore=f.zscore,
            rsi=f.rsi,
            vol_percentile=f.vol_percentile,
        )
