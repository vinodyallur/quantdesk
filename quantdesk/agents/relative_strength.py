"""Cross-sectional relative strength agent."""

from __future__ import annotations

from quantdesk.agents.base import AgentContext, SignalAgent, linear_map
from quantdesk.core.types import Signal


class RelativeStrengthAgent(SignalAgent):
    """Ranks the universe against itself and leans long the leaders, short the laggards.

    This is the only agent here that is genuinely cross-sectional. The others ask
    "is this symbol going up?"; this one asks "is this symbol going up *more than
    its peers*?". That distinction matters in a correlated universe like crypto,
    where every absolute-momentum agent piles into the same direction at the same
    time and the desk ends up with one big beta bet wearing four different hats.

    Ranking uses risk-adjusted momentum (return over volatility) so a symbol does
    not top the table purely by being the most volatile thing on the screen.

    Scores are centred on the cross-section, which means they sum to roughly zero.
    That is the point: it pushes the portfolio toward relative bets rather than
    directional ones.
    """

    name = "rel_strength"
    weight = 0.8

    def __init__(
        self,
        min_symbols: int = 3,
        meaningful_spread: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.min_symbols = min_symbols
        # Minimum cross-sectional dispersion in risk-adjusted momentum units for
        # the ranking to be treated as information rather than rounding noise.
        self.meaningful_spread = meaningful_spread

    def evaluate(self, ctx: AgentContext) -> Signal | None:
        # Only rank against peers that are warm and have usable volatility.
        peers = {
            sym: snap
            for sym, snap in ctx.all_snapshots.items()
            if snap.ready and snap.realized_vol > 1e-9
        }
        if len(peers) < self.min_symbols or ctx.symbol not in peers:
            return None

        scores = {
            sym: snap.roc_slow / snap.realized_vol for sym, snap in peers.items()
        }
        ordered = sorted(scores, key=lambda s: scores[s])
        n = len(ordered)
        rank = ordered.index(ctx.symbol)

        # Map rank onto [-1, 1]: bottom of the table is -1, top is +1.
        score = (rank / (n - 1)) * 2.0 - 1.0

        values = list(scores.values())
        spread = max(values) - min(values)
        own = scores[ctx.symbol]
        mean = sum(values) / n

        # Confidence needs two things to be true, and they are independent.
        #
        # 1. This symbol stands apart from the pack (relative dispersion). Without
        #    this, mid-table symbols would be traded as hard as the extremes.
        # 2. The pack is actually spread out in absolute terms (absolute
        #    dispersion). Relative dispersion alone is scale-free, so a universe
        #    whose risk-adjusted momentum is identical to three decimal places
        #    would still hand full confidence to whichever symbol rounded highest.
        #    That is ranking noise, and this gate is what suppresses it.
        relative = abs(own - mean) / spread if spread > 1e-12 else 0.0
        absolute = linear_map(
            spread, self.meaningful_spread, self.meaningful_spread * 4.0, 0.0, 1.0
        )
        confidence = min(1.0, relative * 2.0) * absolute

        if confidence < 0.15 or abs(score) < 0.2:
            return None

        return self.signal(
            ctx,
            score=score,
            confidence=confidence,
            horizon_bars=32,
            rationale=(
                f"rank {rank + 1}/{n} on risk-adj momentum "
                f"({own:+.3f} vs pack {mean:+.3f}, spread {spread:.3f})"
            ),
            rank=float(rank + 1),
            universe_size=float(n),
            risk_adj_momentum=own,
            dispersion=spread,
        )
