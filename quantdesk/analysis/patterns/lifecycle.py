"""The state machine every boundary-based pattern shares.

Head and shoulders, double and triple tops, triangles, rectangles, flags and
wedges are recognised differently but *behave* identically once found, because
Murphy describes the same sequence for all of them:

1. The shape builds. It is not a pattern yet and not tradable.
2. A decisive close through the boundary completes it. That is the signal.
3. Price often returns to the broken boundary, which should now hold as
   support or resistance in its new role. He calls this the return move and
   treats it as a second, better entry.
4. The move resumes toward the measured objective.

Plus two ways to die:

* **Failure.** "A return move ... should not recross the neckline once it has
  been broken." A decisive close back inside invalidates the pattern. Applied to
  every formation, not just head and shoulders.
* **Expiry.** A shape that never breaks out stops meaning anything. For
  triangles he is specific: past three-quarters of the way to the apex it "begins
  to lose its potency."

Keeping this in one place means a detector only has to answer "is this shape
present?", and the trading semantics are guaranteed to be consistent across all
twenty formations.
"""

from __future__ import annotations

from quantdesk.analysis.patterns.base import (
    Bias,
    DetectorContext,
    Pattern,
    PatternStage,
    PenetrationFilter,
)


def objective_from_boundary(pattern: Pattern, break_index: int) -> float:
    """Murphy's measuring rule: pattern height projected from the break point.

    The projection starts at the *boundary* where it was pierced, not at the
    closing price of the breakout bar. He is explicit for head and shoulders -
    "the vertical distance from the head to the neckline projected downward from
    the breaking of the neckline" - and the same construction applies to
    triangles ("project that vertical distance from the breakout point") and
    rectangles. Using the close instead would make the target drift with however
    far the breakout bar happened to run.
    """
    origin = pattern.boundary_at(break_index)
    if pattern.bias is Bias.BULLISH:
        return origin + pattern.height
    return origin - pattern.height


def fallback_stop(pattern: Pattern) -> float:
    """Protective stop from the shape itself, when the detector set none.

    For a top, the stop belongs above the last rally peak in the formation - the
    right shoulder for head and shoulders, the second peak for a double top -
    because a move back through it says the pattern was wrong.
    """
    if not pattern.pivots:
        return 0.0
    if pattern.bias is Bias.BULLISH:
        return min(p.price for p in pattern.pivots[-2:])
    return max(p.price for p in pattern.pivots[-2:])


def advance(
    pattern: Pattern,
    ctx: DetectorContext,
    filt: PenetrationFilter,
    *,
    expire_after: int = 80,
    return_move_atr: float = 0.5,
) -> bool:
    """Move one pattern forward by one bar. Returns True if the stage changed.

    ``filt`` supplies the close-beyond, price-filter and time-filter tests, so a
    marginal intrabar poke through the boundary cannot complete a pattern.
    """
    if pattern.stage.is_dead:
        return False

    atr = ctx.atr_or()
    upside = pattern.bias is Bias.BULLISH
    boundary = pattern.boundary_at(ctx.index)

    if pattern.stage is PatternStage.FORMING:
        return _advance_forming(
            pattern, ctx, filt, boundary, upside, atr, expire_after
        )

    # ---------------------------------------------------------- after breakout
    # Murphy's invalidation, applied at every post-breakout stage: a decisive
    # close back through the boundary means the pattern has failed.
    recrossed = filt.beyond(ctx.close, boundary, not upside, atr)
    if recrossed:
        pattern.stage = PatternStage.FAILED
        pattern.failure_reason = "closed back through the boundary"
        return True

    if pattern.stage is PatternStage.COMPLETE:
        # Return move: price comes back to touch the broken boundary, which has
        # now reversed role. Lighter breakout volume makes this more likely, per
        # chapter 5, so it is expected rather than treated as weakness.
        probe = ctx.low if upside else ctx.high
        if abs(probe - boundary) <= return_move_atr * atr:
            pattern.stage = PatternStage.RETURN_MOVE
            pattern.return_move_low = probe
            return True
        return False

    if pattern.stage is PatternStage.RETURN_MOVE:
        # Holding the boundary and pushing past the breakout price resumes the
        # move. This is the entry Murphy prefers, since risk is defined tightly
        # against a boundary that has just proved itself.
        beyond_breakout = (
            ctx.close > pattern.breakout_price
            if upside
            else ctx.close < pattern.breakout_price
        )
        if beyond_breakout:
            pattern.stage = PatternStage.RESUMED
            return True
        return False

    return False


def _advance_forming(
    pattern: Pattern,
    ctx: DetectorContext,
    filt: PenetrationFilter,
    boundary: float,
    upside: bool,
    atr: float,
    expire_after: int,
) -> bool:
    """FORMING: watch for the completing break, invalidation, or a timeout."""
    # Invalidation while still forming. A top that trades decisively above its
    # own highest point is no longer a top; the same in reverse for a bottom.
    if pattern.pivots:
        if upside:
            extreme = min(p.price for p in pattern.pivots)
            broke_wrong_way = ctx.close < extreme - filt.margin(extreme, atr)
        else:
            extreme = max(p.price for p in pattern.pivots)
            broke_wrong_way = ctx.close > extreme + filt.margin(extreme, atr)
        if broke_wrong_way:
            pattern.stage = PatternStage.EXPIRED
            pattern.failure_reason = "price left the formation the wrong way"
            return True

    if ctx.index - pattern.detected_index > expire_after:
        pattern.stage = PatternStage.EXPIRED
        pattern.failure_reason = f"no breakout within {expire_after} bars"
        return True

    if not filt.check(pattern.key, ctx.close, boundary, upside, atr):
        return False

    pattern.stage = PatternStage.COMPLETE
    pattern.breakout_index = ctx.index
    pattern.breakout_ts = ctx.ts
    pattern.breakout_price = ctx.close
    pattern.breakout_volume_ratio = ctx.volume_ctx.ratio(ctx.volume)
    pattern.formation_volume_ratio = ctx.volume_ctx.formation_ratio(
        pattern.start_index, pattern.end_index, ctx.index
    )
    # Respect an objective the detector already computed. Most formations measure
    # height-from-the-boundary, but a few do not - the measured move projects from
    # the correction low, and flags project the flagpole - so a pre-set target is
    # taken as authoritative.
    if pattern.objective <= 0:
        pattern.objective = objective_from_boundary(pattern, ctx.index)
    if pattern.stop <= 0:
        pattern.stop = fallback_stop(pattern)
    return True


def volume_confirms(pattern: Pattern, threshold: float = 1.2) -> bool:
    """Did the breakout carry the volume Murphy wants to see?

    Asymmetric on purpose. "Volume is more important on the upside" and at
    bottoms "the volume pickup is absolutely essential", whereas markets can fall
    of their own weight, so a downside break without expansion is still
    tradable.
    """
    if pattern.bias is Bias.BULLISH:
        return pattern.breakout_volume_ratio >= threshold
    return pattern.breakout_volume_ratio >= 0.8


def objective_met(pattern: Pattern, high: float, low: float) -> bool:
    """Has price reached the measured target?"""
    if pattern.objective <= 0 or not pattern.stage.is_actionable:
        return False
    if pattern.bias is Bias.BULLISH:
        return high >= pattern.objective
    return low <= pattern.objective
