"""Volatility regime agent."""

from __future__ import annotations

from quantdesk.agents.base import AgentContext, RegimeAgent, RegimeView, linear_map


class VolRegimeAgent(RegimeAgent):
    """Cuts the desk's risk appetite when volatility spikes.

    This agent has no view on direction, which is why it is a regime agent rather
    than a signal agent. Its only job is to answer "is now a sane time to be
    carrying full size?".

    The behaviour that matters: when realised volatility jumps into the top of its
    own historical range, size down. Volatility clusters, so a spike is a decent
    forecast of more spikes, and a fixed-notional book takes its worst losses
    exactly then. This is a blunt instrument compared to a proper vol forecast, but
    it captures most of the benefit.

    Note this is separate from the volatility *targeting* in the position sizer.
    Sizing scales each position by its own vol continuously; this agent applies a
    discrete risk-off brake across the whole book.
    """

    name = "vol_regime"

    def __init__(
        self,
        calm_pct: float = 0.70,
        panic_pct: float = 0.97,
        floor: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.calm_pct = calm_pct
        self.panic_pct = panic_pct
        self.floor = floor

    def evaluate(self, ctx: AgentContext) -> RegimeView | None:
        f = ctx.snapshot
        pct = f.vol_percentile

        if pct <= self.calm_pct:
            appetite = 1.0
            label = "normal"
        else:
            # Taper linearly from full size at calm_pct down to the floor at panic_pct.
            appetite = linear_map(pct, self.calm_pct, self.panic_pct, 1.0, self.floor)
            label = "elevated" if pct < self.panic_pct else "stressed"

        return RegimeView(
            agent=self.name,
            ts=f.ts,
            risk_appetite=appetite,
            label=label,
            rationale=(
                f"{ctx.symbol} vol {f.realized_vol:.1%} ann "
                f"({pct:.0%} pctile) -> {label}, appetite {appetite:.0%}"
            ),
        )
