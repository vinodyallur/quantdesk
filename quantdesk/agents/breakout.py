"""Donchian breakout agent."""

from __future__ import annotations

from quantdesk.agents.base import AgentContext, SignalAgent, linear_map
from quantdesk.core.types import Signal


class BreakoutAgent(SignalAgent):
    """Trades pushes through the edges of the recent range.

    Position within the Donchian channel drives direction: pressed against the
    high is bullish, against the low is bearish. Only the outer bands count, since
    mid-channel prices carry no structural information.

    Two filters do the real work here, because unfiltered channel breaks are
    mostly noise:

    * Volume confirmation. A break on below-average volume is usually a wick that
      gets retraced. Volume z-score scales conviction and a clearly weak tape
      suppresses the signal entirely.
    * Channel width relative to ATR. When the range is narrow in ATR terms, price
      is constantly poking the edges and "breakout" means nothing. Requiring a
      reasonably wide channel keeps this agent quiet in chop.
    """

    name = "breakout"
    weight = 0.85

    def __init__(
        self,
        edge: float = 0.90,
        min_volume_z: float = -0.25,
        min_channel_atr: float = 2.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.edge = edge
        self.min_volume_z = min_volume_z
        self.min_channel_atr = min_channel_atr

    def evaluate(self, ctx: AgentContext) -> Signal | None:
        f = ctx.snapshot
        if f.atr <= 1e-12 or f.price <= 0:
            return None

        # Is the range wide enough for a break to be informative?
        channel_atr = (f.donchian_high - f.donchian_low) / f.atr
        if channel_atr < self.min_channel_atr:
            return None

        pos = f.channel_pos
        if pos >= self.edge:
            direction = 1.0
            strength = linear_map(pos, self.edge, 1.0, 0.35, 1.0)
        elif pos <= (1.0 - self.edge):
            direction = -1.0
            strength = linear_map(pos, 1.0 - self.edge, 0.0, 0.35, 1.0)
        else:
            return None

        # Require the tape to not be actively weak.
        if f.volume_z < self.min_volume_z:
            return None

        score = direction * strength
        # Volume z of 0 is average (0.6 confidence), +2 is a strong tape (1.0).
        confidence = linear_map(f.volume_z, -0.25, 2.0, 0.45, 1.0)

        # A break that also aligns with the prevailing trend is worth more.
        if (f.ema_spread > 0) == (direction > 0):
            confidence = min(1.0, confidence + 0.15)

        return self.signal(
            ctx,
            score=score,
            confidence=confidence,
            horizon_bars=16,
            rationale=(
                f"{'upper' if direction > 0 else 'lower'} band "
                f"(pos {pos:.0%}, width {channel_atr:.1f} ATR, vol z {f.volume_z:+.1f})"
            ),
            channel_pos=pos,
            channel_atr=channel_atr,
            volume_z=f.volume_z,
        )
