"""Candlestick anatomy and the shared vocabulary for candle patterns.

Murphy's chapter 12 treatment rests on the body/shadow proportions, so those are
computed once here rather than recalculated inside every pattern test. The body is
the open-to-close range, white when the close is higher and black when lower, and
the shadows are the wicks above and below it.

Two structural decisions worth stating.

**Sizes are relative, never absolute.** "Long" and "small" mean nothing in
isolation - a 200-point body is enormous on a quiet day and unremarkable in a
panic. Every proportion here is measured against either the candle's own range or
a rolling average body, so the same thresholds work on any instrument at any
volatility.

**Trend context is part of the pattern, not a filter on it.** Murphy is explicit:
"You cannot have a bullish reversal pattern in an uptrend. You can have a series
of candlesticks that resemble the bullish pattern, but if the trend is up, it is
not a bullish Japanese candle pattern." That is why :class:`CandleSignal` carries
the trend it was read against, and why the reader refuses to emit a reversal that
points the same way as the move preceding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from quantdesk.analysis.patterns.base import Bias
from quantdesk.core.types import Bar


class CandleKind(str, Enum):
    """Patterns from Murphy's chapter 12 reference table.

    The number in his table is how many sessions the pattern spans, which is also
    how this package is organised: :mod:`single`, :mod:`pairs`, :mod:`triples`.
    """

    # --- one session
    DOJI = "doji"
    LONG_LEGGED_DOJI = "long_legged_doji"
    GRAVESTONE_DOJI = "gravestone_doji"
    DRAGONFLY_DOJI = "dragonfly_doji"
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    INVERTED_HAMMER = "inverted_hammer"
    MARUBOZU_WHITE = "marubozu_white"
    MARUBOZU_BLACK = "marubozu_black"
    SPINNING_TOP = "spinning_top"
    BELT_HOLD_BULLISH = "belt_hold_bullish"
    BELT_HOLD_BEARISH = "belt_hold_bearish"
    # --- two sessions
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    HARAMI_BULLISH = "harami_bullish"
    HARAMI_BEARISH = "harami_bearish"
    HARAMI_CROSS_BULLISH = "harami_cross_bullish"
    HARAMI_CROSS_BEARISH = "harami_cross_bearish"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    DOJI_STAR_BULLISH = "doji_star_bullish"
    DOJI_STAR_BEARISH = "doji_star_bearish"
    KICKING_BULLISH = "kicking_bullish"
    KICKING_BEARISH = "kicking_bearish"
    TWEEZER_BOTTOM = "tweezer_bottom"
    TWEEZER_TOP = "tweezer_top"
    # --- three sessions
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    MORNING_DOJI_STAR = "morning_doji_star"
    EVENING_DOJI_STAR = "evening_doji_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    THREE_INSIDE_UP = "three_inside_up"
    THREE_INSIDE_DOWN = "three_inside_down"
    THREE_OUTSIDE_UP = "three_outside_up"
    THREE_OUTSIDE_DOWN = "three_outside_down"

    @property
    def sessions(self) -> int:
        if self in _THREE_SESSION:
            return 3
        if self in _TWO_SESSION:
            return 2
        return 1

    @property
    def bias(self) -> Bias:
        """Which way the pattern points. NEUTRAL means indecision, not no signal."""
        return _BIAS.get(self, Bias.NEUTRAL)

    @property
    def is_reversal(self) -> bool:
        """Most candle patterns are reversal patterns; a few signal continuation."""
        return self not in _CONTINUATION

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


_TWO_SESSION = frozenset(
    {
        CandleKind.ENGULFING_BULLISH,
        CandleKind.ENGULFING_BEARISH,
        CandleKind.HARAMI_BULLISH,
        CandleKind.HARAMI_BEARISH,
        CandleKind.HARAMI_CROSS_BULLISH,
        CandleKind.HARAMI_CROSS_BEARISH,
        CandleKind.PIERCING_LINE,
        CandleKind.DARK_CLOUD_COVER,
        CandleKind.DOJI_STAR_BULLISH,
        CandleKind.DOJI_STAR_BEARISH,
        CandleKind.KICKING_BULLISH,
        CandleKind.KICKING_BEARISH,
        CandleKind.TWEEZER_BOTTOM,
        CandleKind.TWEEZER_TOP,
    }
)

_THREE_SESSION = frozenset(
    {
        CandleKind.MORNING_STAR,
        CandleKind.EVENING_STAR,
        CandleKind.MORNING_DOJI_STAR,
        CandleKind.EVENING_DOJI_STAR,
        CandleKind.THREE_WHITE_SOLDIERS,
        CandleKind.THREE_BLACK_CROWS,
        CandleKind.THREE_INSIDE_UP,
        CandleKind.THREE_INSIDE_DOWN,
        CandleKind.THREE_OUTSIDE_UP,
        CandleKind.THREE_OUTSIDE_DOWN,
    }
)

#: Three soldiers and three crows extend a move rather than reverse one.
_CONTINUATION = frozenset(
    {CandleKind.THREE_WHITE_SOLDIERS, CandleKind.THREE_BLACK_CROWS}
)

_BIAS: dict[CandleKind, Bias] = {
    CandleKind.DRAGONFLY_DOJI: Bias.BULLISH,
    CandleKind.GRAVESTONE_DOJI: Bias.BEARISH,
    CandleKind.HAMMER: Bias.BULLISH,
    CandleKind.INVERTED_HAMMER: Bias.BULLISH,
    CandleKind.HANGING_MAN: Bias.BEARISH,
    CandleKind.SHOOTING_STAR: Bias.BEARISH,
    CandleKind.MARUBOZU_WHITE: Bias.BULLISH,
    CandleKind.MARUBOZU_BLACK: Bias.BEARISH,
    CandleKind.BELT_HOLD_BULLISH: Bias.BULLISH,
    CandleKind.BELT_HOLD_BEARISH: Bias.BEARISH,
    CandleKind.ENGULFING_BULLISH: Bias.BULLISH,
    CandleKind.ENGULFING_BEARISH: Bias.BEARISH,
    CandleKind.HARAMI_BULLISH: Bias.BULLISH,
    CandleKind.HARAMI_BEARISH: Bias.BEARISH,
    CandleKind.HARAMI_CROSS_BULLISH: Bias.BULLISH,
    CandleKind.HARAMI_CROSS_BEARISH: Bias.BEARISH,
    CandleKind.PIERCING_LINE: Bias.BULLISH,
    CandleKind.DARK_CLOUD_COVER: Bias.BEARISH,
    CandleKind.DOJI_STAR_BULLISH: Bias.BULLISH,
    CandleKind.DOJI_STAR_BEARISH: Bias.BEARISH,
    CandleKind.KICKING_BULLISH: Bias.BULLISH,
    CandleKind.KICKING_BEARISH: Bias.BEARISH,
    CandleKind.TWEEZER_BOTTOM: Bias.BULLISH,
    CandleKind.TWEEZER_TOP: Bias.BEARISH,
    CandleKind.MORNING_STAR: Bias.BULLISH,
    CandleKind.EVENING_STAR: Bias.BEARISH,
    CandleKind.MORNING_DOJI_STAR: Bias.BULLISH,
    CandleKind.EVENING_DOJI_STAR: Bias.BEARISH,
    CandleKind.THREE_WHITE_SOLDIERS: Bias.BULLISH,
    CandleKind.THREE_BLACK_CROWS: Bias.BEARISH,
    CandleKind.THREE_INSIDE_UP: Bias.BULLISH,
    CandleKind.THREE_INSIDE_DOWN: Bias.BEARISH,
    CandleKind.THREE_OUTSIDE_UP: Bias.BULLISH,
    CandleKind.THREE_OUTSIDE_DOWN: Bias.BEARISH,
    # Doji, spinning top and long-legged doji are indecision, not direction.
}


@dataclass(slots=True)
class Candle:
    """One session's anatomy, with proportions precomputed."""

    index: int
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @classmethod
    def from_bar(cls, index: int, bar: Bar) -> "Candle":
        return cls(index, bar.ts, bar.open, bar.high, bar.low, bar.close, bar.volume)

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_white(self) -> bool:
        """Close above open. Murphy notes the West prints this hollow, not white."""
        return self.close > self.open

    @property
    def is_black(self) -> bool:
        return self.close < self.open

    @property
    def body_top(self) -> float:
        return max(self.open, self.close)

    @property
    def body_bottom(self) -> float:
        return min(self.open, self.close)

    @property
    def midpoint(self) -> float:
        """Midpoint of the *body*, which is what piercing and dark cloud test."""
        return (self.open + self.close) / 2.0

    @property
    def upper_shadow(self) -> float:
        return self.high - self.body_top

    @property
    def lower_shadow(self) -> float:
        return self.body_bottom - self.low

    @property
    def body_frac(self) -> float:
        """Body as a fraction of the whole range, in [0, 1]."""
        r = self.range
        if r <= 0:
            return 0.0
        return self.body / r

    def engulfs(self, other: "Candle") -> bool:
        """Does this body completely cover the other's body?"""
        return (
            self.body_top >= other.body_top and self.body_bottom <= other.body_bottom
        )

    def inside(self, other: "Candle") -> bool:
        """Is this body contained within the other's? The harami condition."""
        return (
            self.body_top <= other.body_top and self.body_bottom >= other.body_bottom
        )


@dataclass(slots=True)
class CandleSignal:
    """A recognised candle pattern, with the context that makes it valid."""

    kind: CandleKind
    symbol: str
    index: int
    ts: datetime
    bias: Bias
    #: Trend the pattern was read against. Murphy requires a reversal pattern to
    #: oppose the preceding trend, so this is evidence, not decoration.
    prior_trend: Bias = Bias.NEUTRAL
    #: 0-1 quality from the pattern's proportions and its volume.
    strength: float = 0.5
    sessions: int = 1
    volume_ratio: float = 1.0
    notes: str = ""

    @property
    def confirmed_by_trend(self) -> bool:
        """A reversal must oppose the move before it to mean anything."""
        if not self.kind.is_reversal:
            return self.prior_trend is self.bias
        return self.prior_trend is self.bias.opposite

    def describe(self) -> str:
        parts = [self.kind.label, self.bias.value, f"strength {self.strength:.2f}"]
        if self.notes:
            parts.append(self.notes)
        return " | ".join(parts)


@dataclass(slots=True)
class SizeContext:
    """Rolling scale for judging "long" and "small" bodies.

    Murphy's descriptions are all comparative - a "long white candlestick", a
    "small body" - so the thresholds have to come from recent history rather than
    a constant.
    """

    avg_body: float = 0.0
    avg_range: float = 0.0
    atr: float = 0.0
    avg_volume: float = 0.0

    #: Body below this fraction of the average body counts as small.
    small_body: float = 0.5
    #: Body above this multiple of the average body counts as long.
    long_body: float = 1.3
    #: Body/range fraction below this counts as a doji.
    doji_frac: float = 0.08

    def is_small_body(self, c: Candle) -> bool:
        if self.avg_body <= 0:
            return c.body_frac < 0.3
        return c.body <= self.small_body * self.avg_body

    def is_long_body(self, c: Candle) -> bool:
        if self.avg_body <= 0:
            return c.body_frac > 0.6
        return c.body >= self.long_body * self.avg_body

    def is_doji(self, c: Candle) -> bool:
        """Open and close effectively equal.

        Murphy: "there is some consideration as to whether the open and close
        price must be exactly equal. This is a time when the prices must be almost
        equal, especially when dealing with large price movements." So this is a
        proportion of the session's range, not an exact match.
        """
        if c.range <= 0:
            return True
        return c.body_frac <= self.doji_frac

    def volume_ratio(self, c: Candle) -> float:
        if self.avg_volume <= 0:
            return 1.0
        return c.volume / self.avg_volume
