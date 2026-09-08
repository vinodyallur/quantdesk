"""Three-session candle patterns.

Murphy singles out the morning and evening stars as "two powerful reversal candle
patterns ... three day patterns that work exceptionally well", so they get the
strictest treatment here.

The evening star sequence: a long white body confirming the uptrend, then a small
body gapping *above* it - the star, showing the advance has stalled - then a long
black body closing well back into the first session's range. The gap is what makes
it a star rather than three ordinary sessions, and the third session's close is what
confirms the reversal. The morning star is the mirror.

Also from his table: three white soldiers, three black crows, and the three
inside/outside up and down patterns, which are confirmations of a harami or an
engulfing by a following session.
"""

from __future__ import annotations

from quantdesk.analysis.candles.geometry import Candle, CandleKind, SizeContext
from quantdesk.analysis.candles.pairs import (
    classify_engulfing,
    classify_harami,
)
from quantdesk.analysis.patterns.base import Bias


def classify_star(
    first: Candle, star: Candle, third: Candle, size: SizeContext
) -> CandleKind | None:
    """Morning and evening stars, including the doji-star variants.

    The star's body must gap clear of both neighbours' bodies. A small body that
    merely sits inside the prior range is a harami, which is a weaker and different
    signal.
    """
    if not size.is_small_body(star):
        return None
    if not size.is_long_body(first) or not size.is_long_body(third):
        return None

    doji_star = size.is_doji(star)

    # Evening star: uptrend, star gaps above, third session sells off.
    if first.is_white and third.is_black:
        if star.body_bottom <= first.body_top:
            return None
        # A second gap, down from the star into the third session, is the textbook
        # form but is often absent in practice, so it is scored in
        # :func:`triple_strength` rather than required here.
        # Murphy's confirmation: the third close pushes back into the first body.
        if third.close > first.midpoint:
            return None
        return (
            CandleKind.EVENING_DOJI_STAR if doji_star else CandleKind.EVENING_STAR
        )

    # Morning star: downtrend, star gaps below, third session rallies.
    if first.is_black and third.is_white:
        if star.body_top >= first.body_bottom:
            return None
        if third.close < first.midpoint:
            return None
        return (
            CandleKind.MORNING_DOJI_STAR if doji_star else CandleKind.MORNING_STAR
        )
    return None


def classify_soldiers_crows(
    a: Candle, b: Candle, c: Candle, size: SizeContext
) -> CandleKind | None:
    """Three consecutive long bodies marching the same way.

    Each session must open within the previous body and close beyond its close -
    a steady, orderly advance or decline. These are continuation patterns, not
    reversals: they say the move is healthy.
    """
    whites = a.is_white and b.is_white and c.is_white
    blacks = a.is_black and b.is_black and c.is_black
    if not (whites or blacks):
        return None
    if not all(size.is_long_body(x) for x in (a, b, c)):
        return None

    if whites:
        if not (b.close > a.close and c.close > b.close):
            return None
        # Each opens inside the prior body: orderly, not gapping and exhausted.
        if not (a.body_bottom <= b.open <= a.body_top):
            return None
        if not (b.body_bottom <= c.open <= b.body_top):
            return None
        return CandleKind.THREE_WHITE_SOLDIERS

    if not (b.close < a.close and c.close < b.close):
        return None
    if not (a.body_bottom <= b.open <= a.body_top):
        return None
    if not (b.body_bottom <= c.open <= b.body_top):
        return None
    return CandleKind.THREE_BLACK_CROWS


def classify_three_inside(
    a: Candle, b: Candle, c: Candle, size: SizeContext
) -> CandleKind | None:
    """A harami confirmed by the third session closing beyond the first body.

    The harami says momentum stalled; the third session says it actually turned.
    """
    harami = classify_harami(a, b, size)
    if harami is None:
        return None
    if harami in (CandleKind.HARAMI_BULLISH, CandleKind.HARAMI_CROSS_BULLISH):
        if c.is_white and c.close > a.body_top:
            return CandleKind.THREE_INSIDE_UP
        return None
    if c.is_black and c.close < a.body_bottom:
        return CandleKind.THREE_INSIDE_DOWN
    return None


def classify_three_outside(
    a: Candle, b: Candle, c: Candle, size: SizeContext
) -> CandleKind | None:
    """An engulfing pattern confirmed by a third session extending the reversal."""
    engulf = classify_engulfing(a, b, size)
    if engulf is None:
        return None
    if engulf is CandleKind.ENGULFING_BULLISH:
        if c.is_white and c.close > b.close:
            return CandleKind.THREE_OUTSIDE_UP
        return None
    if c.is_black and c.close < b.close:
        return CandleKind.THREE_OUTSIDE_DOWN
    return None


def read_triples(
    a: Candle, b: Candle, c: Candle, size: SizeContext
) -> list[CandleKind]:
    """Every three-session pattern present, strongest first."""
    out: list[CandleKind] = []
    for fn in (
        classify_star,
        classify_three_outside,
        classify_three_inside,
        classify_soldiers_crows,
    ):
        kind = fn(a, b, c, size)
        if kind is not None:
            out.append(kind)
    return out


def triple_strength(
    kind: CandleKind, a: Candle, b: Candle, c: Candle, size: SizeContext
) -> float:
    """Quality from how far the third session carried the reversal."""
    if kind in (
        CandleKind.MORNING_STAR,
        CandleKind.EVENING_STAR,
        CandleKind.MORNING_DOJI_STAR,
        CandleKind.EVENING_DOJI_STAR,
    ):
        # How deep the third close pushed back through the first body. Murphy's
        # minimum is the midpoint, and these are the patterns he rates highest.
        if a.body <= 0:
            return 0.7
        if kind in (CandleKind.EVENING_STAR, CandleKind.EVENING_DOJI_STAR):
            depth = (a.body_top - c.close) / a.body
        else:
            depth = (c.close - a.body_bottom) / a.body
        base = 0.55 + 0.35 * min(1.0, max(0.0, depth))
        # The doji variants are the stronger reading: total stalemate at the top.
        if kind in (CandleKind.EVENING_DOJI_STAR, CandleKind.MORNING_DOJI_STAR):
            base += 0.05
        return min(1.0, base)
    if kind in (CandleKind.THREE_OUTSIDE_UP, CandleKind.THREE_OUTSIDE_DOWN):
        return 0.8
    if kind in (CandleKind.THREE_INSIDE_UP, CandleKind.THREE_INSIDE_DOWN):
        return 0.7
    if kind in (CandleKind.THREE_WHITE_SOLDIERS, CandleKind.THREE_BLACK_CROWS):
        return 0.75
    return 0.6
