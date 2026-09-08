"""Support and resistance.

Murphy's treatment, implemented directly:

* Support is a prior reaction low, resistance a prior peak. So levels are built
  from confirmed swing pivots.
* **Significance** comes from three things he names explicitly: how much time was
  spent at the level, the volume traded there, and how recently. All three feed
  the strength score below.
* **Role reversal.** Once penetrated by a significant margin, support becomes
  resistance and vice versa. He gives concrete filters: roughly 3% for major
  levels, nearer 1% for shorter-term ones, or a two-bar close-beyond time
  filter. Both are implemented; the price filter is the default.
* Levels only reverse roles once price has moved far enough to convince
  participants they were wrong, which is why a marginal poke through does not
  count.
* **Round numbers** act as psychological levels in their own right, and he
  advises against resting stops exactly on them.

Nearby pivots are clustered into zones rather than kept as individual prices. A
level is an area, not a line, and treating each pivot as its own level produces a
useless thicket of near-duplicates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from quantdesk.analysis.swings import SwingKind, SwingPoint


class LevelRole(str, Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"

    @property
    def opposite(self) -> "LevelRole":
        return (
            LevelRole.RESISTANCE if self is LevelRole.SUPPORT else LevelRole.SUPPORT
        )


@dataclass(slots=True)
class Level:
    """A support or resistance zone built from one or more pivots."""

    price: float
    role: LevelRole
    touches: int = 1
    first_index: int = 0
    last_index: int = 0
    volume: float = 0.0
    #: Half-width of the zone. A level is an area, not an exact price.
    width: float = 0.0
    #: True once price has decisively penetrated and the role has flipped.
    reversed_role: bool = False
    origin_role: LevelRole | None = None
    last_ts: datetime | None = None

    def __post_init__(self) -> None:
        if self.origin_role is None:
            self.origin_role = self.role

    @property
    def low(self) -> float:
        return self.price - self.width

    @property
    def high(self) -> float:
        return self.price + self.width

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def distance_pct(self, price: float) -> float:
        """Signed distance from ``price`` to this level, as a fraction of price."""
        if price <= 0:
            return 0.0
        return (self.price - price) / price

    def strength(self, current_index: int, recency_halflife: float = 200.0) -> float:
        """Composite significance in [0, 1].

        Combines Murphy's three factors. Touches dominate, since a level tested
        repeatedly is the clearest evidence that participants care about it.
        Recency decays exponentially: he is explicit that recent activity is more
        potent, because the level's power comes from traders' memory of positions
        taken there.
        """
        touch_score = min(1.0, math.log1p(self.touches) / math.log(6.0))
        age = max(0, current_index - self.last_index)
        recency = math.pow(0.5, age / max(1.0, recency_halflife))
        # Time spent: how long the level has been part of the structure.
        span = max(0, self.last_index - self.first_index)
        duration = min(1.0, span / 60.0)
        score = 0.5 * touch_score + 0.3 * recency + 0.2 * duration
        # A level that has already flipped role has proven itself twice over.
        if self.reversed_role:
            score = min(1.0, score * 1.1)
        return max(0.0, min(1.0, score))


class LevelBook:
    """Maintains clustered support/resistance zones from swing pivots.

    Parameters
    ----------
    cluster_atr:
        Pivots within this many ATR of an existing level are merged into it
        rather than creating a new one. Murphy describes levels as areas without
        quantifying the width, so this is a chosen parameter.
    penetration_pct:
        Fractional move beyond a level required to flip its role. Murphy's 1-3%
        guidance: 3% for major levels, ~1% for shorter-term. Default 0.01 suits
        intraday work; raise it for daily/weekly structure.
    """

    def __init__(
        self,
        cluster_atr: float = 0.6,
        penetration_pct: float = 0.01,
        max_levels: int = 40,
        confirm_bars: int = 2,
    ) -> None:
        self.cluster_atr = cluster_atr
        self.penetration_pct = penetration_pct
        self.max_levels = max_levels
        self.confirm_bars = confirm_bars
        self._levels: list[Level] = []
        self._beyond_count: dict[int, int] = {}

    # ------------------------------------------------------------------ build
    def add_pivot(self, pivot: SwingPoint, atr: float) -> Level:
        """Fold a confirmed pivot into the level book."""
        role = (
            LevelRole.RESISTANCE if pivot.kind is SwingKind.PEAK else LevelRole.SUPPORT
        )
        tolerance = max(self.cluster_atr * atr, pivot.price * 0.0005)

        for lvl in self._levels:
            if abs(lvl.price - pivot.price) <= tolerance:
                # Merge: volume-weight the price so heavily traded pivots pull
                # the zone toward themselves.
                total_v = lvl.volume + pivot.volume
                if total_v > 0:
                    lvl.price = (
                        lvl.price * lvl.volume + pivot.price * pivot.volume
                    ) / total_v
                else:
                    lvl.price = (lvl.price + pivot.price) / 2.0
                lvl.touches += 1
                lvl.volume = total_v
                lvl.last_index = max(lvl.last_index, pivot.index)
                lvl.last_ts = pivot.ts
                lvl.width = max(lvl.width, tolerance)
                return lvl

        lvl = Level(
            price=pivot.price,
            role=role,
            touches=1,
            first_index=pivot.index,
            last_index=pivot.index,
            volume=pivot.volume,
            width=tolerance,
            last_ts=pivot.ts,
        )
        self._levels.append(lvl)
        self._prune()
        return lvl

    def _prune(self) -> None:
        if len(self._levels) <= self.max_levels:
            return
        # Drop the least significant levels, judged at the newest index we know.
        newest = max((v.last_index for v in self._levels), default=0)
        self._levels.sort(key=lambda v: v.strength(newest), reverse=True)
        del self._levels[self.max_levels:]

    # ------------------------------------------------------------ role reversal
    def update_price(self, close: float, index: int) -> list[Level]:
        """Check for decisive penetrations and flip roles. Returns flipped levels.

        A penetration counts only when price closes beyond the level by more than
        ``penetration_pct``, for ``confirm_bars`` consecutive bars. That is
        Murphy's price filter and two-bar time filter applied together, which is
        what keeps a single intrabar poke from rewriting the structure.
        """
        flipped: list[Level] = []
        for i, lvl in enumerate(self._levels):
            margin = lvl.price * self.penetration_pct
            beyond = (
                close > lvl.price + margin
                if lvl.role is LevelRole.RESISTANCE
                else close < lvl.price - margin
            )
            if beyond:
                self._beyond_count[i] = self._beyond_count.get(i, 0) + 1
                if self._beyond_count[i] >= self.confirm_bars and not lvl.reversed_role:
                    lvl.role = lvl.role.opposite
                    lvl.reversed_role = True
                    lvl.last_index = index
                    flipped.append(lvl)
                    self._beyond_count[i] = 0
            else:
                self._beyond_count[i] = 0
        return flipped

    # --------------------------------------------------------------- accessors
    @property
    def levels(self) -> list[Level]:
        return list(self._levels)

    def nearest(
        self, price: float, role: LevelRole | None = None, above: bool | None = None
    ) -> Level | None:
        """Closest level to ``price``, optionally constrained by role or side."""
        candidates = [
            v
            for v in self._levels
            if (role is None or v.role is role)
            and (
                above is None
                or (v.price > price if above else v.price < price)
            )
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda v: abs(v.price - price))

    def overhead_resistance(self, price: float) -> Level | None:
        return self.nearest(price, LevelRole.RESISTANCE, above=True)

    def underlying_support(self, price: float) -> Level | None:
        return self.nearest(price, LevelRole.SUPPORT, above=False)

    def proximity(self, price: float, atr: float) -> tuple[Level | None, float]:
        """Nearest level of any kind, and its distance in ATR units.

        Distance in ATR rather than percent so "close to resistance" means the
        same thing across instruments of different volatility.
        """
        lvl = self.nearest(price)
        if lvl is None or atr <= 0:
            return lvl, float("inf")
        return lvl, abs(lvl.price - price) / atr

    def headroom(self, price: float, atr: float, long_side: bool) -> float:
        """ATR of clear space before the next opposing level.

        Directly useful for reward-to-risk: Murphy's tactic is to buy when there
        is room to the next resistance, not when price is pressed against it.
        """
        lvl = (
            self.overhead_resistance(price) if long_side else self.underlying_support(price)
        )
        if lvl is None or atr <= 0:
            return float("inf")
        return abs(lvl.price - price) / atr


# --------------------------------------------------------------- round numbers


def round_number_levels(price: float, count: int = 3) -> list[float]:
    """Psychologically significant round numbers bracketing ``price``.

    Murphy notes traders think in round numbers and that these stall advances and
    declines. The increment scales with the price's magnitude so it produces
    sensible levels whether the instrument trades at 55 or 79,000.
    """
    if price <= 0:
        return []
    magnitude = math.pow(10.0, math.floor(math.log10(price)))
    out: list[float] = []
    # Halves and quarters of the decade matter too (Murphy cites 25, 50, 75).
    for step in (magnitude, magnitude / 2.0, magnitude / 4.0):
        base = math.floor(price / step) * step
        for k in range(-count, count + 2):
            lvl = base + k * step
            if lvl > 0:
                out.append(round(lvl, 8))
    return sorted(set(out))


def nearest_round_number(price: float) -> tuple[float, float]:
    """Closest round number and its distance as a fraction of price."""
    levels = round_number_levels(price)
    if not levels:
        return 0.0, float("inf")
    nearest = min(levels, key=lambda v: abs(v - price))
    return nearest, abs(nearest - price) / price if price > 0 else float("inf")


def offset_stop_from_round(
    stop: float, long_side: bool, buffer_pct: float = 0.002
) -> float:
    """Nudge a stop clear of a round number.

    Murphy's tactical advice: protective stops on longs belong *below* round
    numbers and on shorts *above* them, because everyone else's orders cluster at
    the round number and price often reverses right there. This shifts a stop that
    sits too close, and leaves it alone otherwise.
    """
    rn, dist = nearest_round_number(stop)
    if dist > buffer_pct or rn <= 0:
        return stop
    shift = rn * buffer_pct
    return rn - shift if long_side else rn + shift
