"""Single-session candle patterns.

The one-session entries from Murphy's table: hammer, hanging man, shooting star,
inverted hammer, belt hold, plus the doji family, marubozu and spinning top.

The hammer/hanging man pair is the clearest illustration of why trend context is
structural rather than cosmetic. They are *the same candle* - small body at the top
of the range, long lower shadow. The only thing that separates a bullish hammer
from a bearish hanging man is whether it appears after a decline or after an
advance. The same is true of the shooting star and inverted hammer. So these
functions take the prior trend and return the correct one of the pair, rather than
returning a shape and leaving the caller to interpret it.
"""

from __future__ import annotations

from quantdesk.analysis.candles.geometry import (
    Candle,
    CandleKind,
    SizeContext,
)
from quantdesk.analysis.patterns.base import Bias

#: A shadow this many times the body makes it the dominant feature of the session.
SHADOW_BODY_RATIO = 2.0
#: Opposite shadow must be no more than this fraction of the range to be "absent".
NEGLIGIBLE_SHADOW = 0.1


def classify_doji(c: Candle, size: SizeContext) -> CandleKind | None:
    """Doji, and which of the three meaningful variants it is.

    Murphy names them by shadow arrangement: long-legged has both shadows long and
    "reflects considerable indecision"; the gravestone has only an upper shadow and
    the longer it is the more bearish; the dragonfly is its mirror and "usually
    considered quite bullish".
    """
    if not size.is_doji(c):
        return None
    r = c.range
    if r <= 0:
        return CandleKind.DOJI

    upper = c.upper_shadow / r
    lower = c.lower_shadow / r

    if upper <= NEGLIGIBLE_SHADOW and lower >= 0.6:
        return CandleKind.DRAGONFLY_DOJI
    if lower <= NEGLIGIBLE_SHADOW and upper >= 0.6:
        return CandleKind.GRAVESTONE_DOJI
    if upper >= 0.3 and lower >= 0.3:
        return CandleKind.LONG_LEGGED_DOJI
    return CandleKind.DOJI


def classify_umbrella(
    c: Candle, size: SizeContext, prior_trend: Bias
) -> CandleKind | None:
    """The hammer / hanging man pair: small body up top, long lower shadow.

    Identical shapes; the preceding trend decides which it is. A hammer hammers out
    a bottom after a decline. A hanging man appears after an advance and warns the
    advance is in trouble.
    """
    if size.is_doji(c):
        return None
    if c.body <= 0 or c.range <= 0:
        return None
    if c.lower_shadow < SHADOW_BODY_RATIO * c.body:
        return None
    if c.upper_shadow > NEGLIGIBLE_SHADOW * c.range:
        return None
    if not size.is_small_body(c):
        return None

    if prior_trend is Bias.BEARISH:
        return CandleKind.HAMMER
    if prior_trend is Bias.BULLISH:
        return CandleKind.HANGING_MAN
    # Without a trend to reverse, the shape carries no reversal meaning.
    return None


def classify_inverted(
    c: Candle, size: SizeContext, prior_trend: Bias
) -> CandleKind | None:
    """The shooting star / inverted hammer pair: long upper shadow, body at the low.

    Same shape, opposite readings depending on what preceded it. The shooting star
    shoots up out of an uptrend and fails; the inverted hammer does the same after a
    decline and hints the sellers have lost control.
    """
    if size.is_doji(c):
        return None
    if c.body <= 0 or c.range <= 0:
        return None
    if c.upper_shadow < SHADOW_BODY_RATIO * c.body:
        return None
    if c.lower_shadow > NEGLIGIBLE_SHADOW * c.range:
        return None
    if not size.is_small_body(c):
        return None

    if prior_trend is Bias.BULLISH:
        return CandleKind.SHOOTING_STAR
    if prior_trend is Bias.BEARISH:
        return CandleKind.INVERTED_HAMMER
    return None


def classify_marubozu(c: Candle, size: SizeContext) -> CandleKind | None:
    """A long body with effectively no shadows: one side controlled all session."""
    if not size.is_long_body(c):
        return None
    r = c.range
    if r <= 0:
        return None
    if c.upper_shadow > NEGLIGIBLE_SHADOW * r or c.lower_shadow > NEGLIGIBLE_SHADOW * r:
        return None
    return CandleKind.MARUBOZU_WHITE if c.is_white else CandleKind.MARUBOZU_BLACK


def classify_belt_hold(c: Candle, size: SizeContext) -> CandleKind | None:
    """A long body that opens at its extreme and closes near the other.

    Bullish belt hold opens on the low and drives up; bearish opens on the high and
    sells off. One shadow is missing, unlike the marubozu where both are.
    """
    if not size.is_long_body(c):
        return None
    r = c.range
    if r <= 0:
        return None
    if c.is_white and c.lower_shadow <= NEGLIGIBLE_SHADOW * r:
        return CandleKind.BELT_HOLD_BULLISH
    if c.is_black and c.upper_shadow <= NEGLIGIBLE_SHADOW * r:
        return CandleKind.BELT_HOLD_BEARISH
    return None


def classify_spinning_top(c: Candle, size: SizeContext) -> CandleKind | None:
    """Small body with both shadows longer than it.

    Murphy: "The body color is relatively unimportant ... These candlesticks are
    considered as days of indecision." So no directional bias is attached.
    """
    if size.is_doji(c):
        return None
    if c.body <= 0:
        return None
    if not size.is_small_body(c):
        return None
    if c.upper_shadow <= c.body or c.lower_shadow <= c.body:
        return None
    return CandleKind.SPINNING_TOP


def read_single(
    c: Candle, size: SizeContext, prior_trend: Bias
) -> list[CandleKind]:
    """All single-session patterns present on this candle.

    Ordered most specific first, since a session can legitimately satisfy several
    descriptions and the caller usually wants the strongest reading.
    """
    out: list[CandleKind] = []
    for fn in (classify_umbrella, classify_inverted):
        kind = fn(c, size, prior_trend)
        if kind is not None:
            out.append(kind)
    doji = classify_doji(c, size)
    if doji is not None:
        out.append(doji)
    for fn2 in (classify_marubozu, classify_belt_hold, classify_spinning_top):
        kind = fn2(c, size)
        if kind is not None:
            out.append(kind)
    return out


def single_strength(kind: CandleKind, c: Candle, size: SizeContext) -> float:
    """Quality of a one-session pattern from its own proportions.

    The shadow patterns get stronger the more extreme the shadow-to-body ratio,
    which is Murphy's own comment about the gravestone: "the longer the upper
    shadow, the more bearish the interpretation."
    """
    r = c.range
    if r <= 0:
        return 0.3

    if kind in (CandleKind.HAMMER, CandleKind.HANGING_MAN, CandleKind.DRAGONFLY_DOJI):
        return _clamp(c.lower_shadow / r)
    if kind in (
        CandleKind.SHOOTING_STAR,
        CandleKind.INVERTED_HAMMER,
        CandleKind.GRAVESTONE_DOJI,
    ):
        return _clamp(c.upper_shadow / r)
    if kind in (CandleKind.MARUBOZU_WHITE, CandleKind.MARUBOZU_BLACK):
        base = c.body / size.avg_body if size.avg_body > 0 else 1.0
        return _clamp(0.4 + 0.3 * base)
    if kind in (CandleKind.BELT_HOLD_BULLISH, CandleKind.BELT_HOLD_BEARISH):
        return _clamp(c.body_frac)
    # Indecision patterns are informative but weak on their own.
    if kind in (
        CandleKind.DOJI,
        CandleKind.LONG_LEGGED_DOJI,
        CandleKind.SPINNING_TOP,
    ):
        return 0.3
    return 0.5


def _clamp(x: float, low: float = 0.2, high: float = 1.0) -> float:
    return max(low, min(high, x))
