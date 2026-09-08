"""Two-session candle patterns.

Murphy describes two of these in detail, and his descriptions are followed to the
letter because the details are what make them work.

**Dark cloud cover** - "The first day of this pattern is a long white candlestick.
This reflects the current trend of the market ... The next day opens above the high
price of the previous day, again adding to the bullishness. However, trading for the
rest of the day is lower with a close price at least below the midpoint of the body
of the first day."

Three conditions there, all enforced below: the first session must be a *long*
white body, the second must open above the prior *high* (not merely above the prior
close), and it must close below the *midpoint of the first body*. A close that only
dips into the first body is not a dark cloud.

**Piercing line** - the mirror image: long black first session, second opens at a
new low and closes above the midpoint of the first body.

The rest come from his reference table: engulfing, harami, harami cross, doji star,
kicking and tweezers.
"""

from __future__ import annotations

from quantdesk.analysis.candles.geometry import Candle, CandleKind, SizeContext
from quantdesk.analysis.patterns.base import Bias


def classify_engulfing(
    prev: Candle, cur: Candle, size: SizeContext
) -> CandleKind | None:
    """A body that swallows the previous one whole, closing the opposite way.

    Requires opposite colours - the point is that one side decisively took over -
    and a real body on the first session, since engulfing a doji is a harami cross
    situation rather than an engulfing.
    """
    if size.is_doji(prev) or size.is_doji(cur):
        return None
    if not cur.engulfs(prev):
        return None
    if cur.body <= prev.body:
        return None
    if cur.is_white and prev.is_black:
        return CandleKind.ENGULFING_BULLISH
    if cur.is_black and prev.is_white:
        return CandleKind.ENGULFING_BEARISH
    return None


def classify_harami(
    prev: Candle, cur: Candle, size: SizeContext
) -> CandleKind | None:
    """The inside day: a small body contained within the previous large one.

    Momentum has stalled. A doji inside the prior body is the harami *cross*, which
    Murphy lists separately and which is the stronger warning.
    """
    if not size.is_long_body(prev):
        return None
    if not cur.inside(prev):
        return None
    if not size.is_small_body(cur):
        return None

    cross = size.is_doji(cur)
    if prev.is_black:
        # Contained after a long black session: potential bullish turn.
        return CandleKind.HARAMI_CROSS_BULLISH if cross else CandleKind.HARAMI_BULLISH
    if prev.is_white:
        return CandleKind.HARAMI_CROSS_BEARISH if cross else CandleKind.HARAMI_BEARISH
    return None


def classify_dark_cloud(
    prev: Candle, cur: Candle, size: SizeContext
) -> CandleKind | None:
    """Dark cloud cover, exactly as Murphy specifies it."""
    if not size.is_long_body(prev) or not prev.is_white:
        return None
    if not cur.is_black:
        return None
    # "opens above the high price of the previous day"
    if cur.open <= prev.high:
        return None
    # "close price at least below the midpoint of the body of the first day"
    if cur.close > prev.midpoint:
        return None
    # It must not fully engulf, or it is a bearish engulfing - a different pattern.
    if cur.close < prev.body_bottom:
        return None
    return CandleKind.DARK_CLOUD_COVER


def classify_piercing(
    prev: Candle, cur: Candle, size: SizeContext
) -> CandleKind | None:
    """Piercing line: the bullish mirror of dark cloud cover."""
    if not size.is_long_body(prev) or not prev.is_black:
        return None
    if not cur.is_white:
        return None
    # "prices open at a new low"
    if cur.open >= prev.low:
        return None
    # "close above the midpoint of the first candlestick's body"
    if cur.close < prev.midpoint:
        return None
    if cur.close > prev.body_top:
        return None
    return CandleKind.PIERCING_LINE


def classify_doji_star(
    prev: Candle, cur: Candle, size: SizeContext
) -> CandleKind | None:
    """A doji gapping away from a long body: the market has stopped dead.

    The gap is what makes it a star. Without separation from the prior body it is
    just an indecisive session following a decisive one.
    """
    if not size.is_long_body(prev):
        return None
    if not size.is_doji(cur):
        return None
    if prev.is_white and cur.body_bottom > prev.body_top:
        return CandleKind.DOJI_STAR_BEARISH
    if prev.is_black and cur.body_top < prev.body_bottom:
        return CandleKind.DOJI_STAR_BULLISH
    return None


def classify_kicking(
    prev: Candle, cur: Candle, size: SizeContext
) -> CandleKind | None:
    """Two opposite marubozu separated by a gap - a violent about-face.

    One of the strongest two-session reversals precisely because neither session
    showed any hesitation and the market gapped between them.
    """
    if not (size.is_long_body(prev) and size.is_long_body(cur)):
        return None
    r_prev, r_cur = prev.range, cur.range
    if r_prev <= 0 or r_cur <= 0:
        return None
    # Both must be near-shadowless.
    if max(prev.upper_shadow, prev.lower_shadow) > 0.1 * r_prev:
        return None
    if max(cur.upper_shadow, cur.lower_shadow) > 0.1 * r_cur:
        return None
    if prev.is_black and cur.is_white and cur.low > prev.high:
        return CandleKind.KICKING_BULLISH
    if prev.is_white and cur.is_black and cur.high < prev.low:
        return CandleKind.KICKING_BEARISH
    return None


def classify_tweezers(
    prev: Candle, cur: Candle, size: SizeContext, tolerance_atr: float = 0.1
) -> CandleKind | None:
    """Two sessions sharing a high or a low almost exactly.

    A matched extreme means the same level rejected price twice in a row, which is
    support or resistance forming in miniature.
    """
    tol = tolerance_atr * size.atr if size.atr > 0 else 0.0
    if tol <= 0:
        return None
    if abs(cur.low - prev.low) <= tol and cur.low < min(prev.close, cur.close):
        return CandleKind.TWEEZER_BOTTOM
    if abs(cur.high - prev.high) <= tol and cur.high > max(prev.close, cur.close):
        return CandleKind.TWEEZER_TOP
    return None


def read_pairs(
    prev: Candle, cur: Candle, size: SizeContext
) -> list[CandleKind]:
    """Every two-session pattern present, strongest and most specific first."""
    out: list[CandleKind] = []
    for fn in (
        classify_kicking,
        classify_dark_cloud,
        classify_piercing,
        classify_engulfing,
        classify_doji_star,
        classify_harami,
        classify_tweezers,
    ):
        kind = fn(prev, cur, size)
        if kind is not None:
            out.append(kind)
    return out


def pair_strength(
    kind: CandleKind, prev: Candle, cur: Candle, size: SizeContext
) -> float:
    """Quality from how emphatically the second session reversed the first."""
    if kind in (CandleKind.ENGULFING_BULLISH, CandleKind.ENGULFING_BEARISH):
        # The more it engulfs, the more convincing the takeover.
        if prev.body <= 0:
            return 0.5
        return _clamp(0.4 + 0.3 * (cur.body / prev.body - 1.0))
    if kind in (CandleKind.DARK_CLOUD_COVER, CandleKind.PIERCING_LINE):
        # How deep into the prior body it closed. Murphy's minimum is the midpoint;
        # deeper is stronger, and a full engulf would be a different pattern.
        if prev.body <= 0:
            return 0.5
        if kind is CandleKind.DARK_CLOUD_COVER:
            depth = (prev.body_top - cur.close) / prev.body
        else:
            depth = (cur.close - prev.body_bottom) / prev.body
        return _clamp(0.3 + 0.7 * depth)
    if kind in (CandleKind.KICKING_BULLISH, CandleKind.KICKING_BEARISH):
        return 0.9
    if kind in (CandleKind.HARAMI_CROSS_BULLISH, CandleKind.HARAMI_CROSS_BEARISH):
        return 0.65
    if kind in (CandleKind.HARAMI_BULLISH, CandleKind.HARAMI_BEARISH):
        return 0.45
    if kind in (CandleKind.DOJI_STAR_BULLISH, CandleKind.DOJI_STAR_BEARISH):
        return 0.6
    if kind in (CandleKind.TWEEZER_BOTTOM, CandleKind.TWEEZER_TOP):
        return 0.4
    return 0.5


def _clamp(x: float, low: float = 0.2, high: float = 1.0) -> float:
    return max(low, min(high, x))
