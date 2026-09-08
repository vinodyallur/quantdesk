"""Shared vocabulary for chart patterns.

Murphy's chapters 5 and 6 describe roughly twenty formations, but they are all
assembled from the same four ingredients, so those ingredients live here rather
than being reimplemented per pattern:

1. **A shape made of confirmed pivots.** Every formation is a specific
   arrangement of peaks and troughs. Detectors therefore consume
   :class:`~quantdesk.analysis.swings.SwingPoint` lists and nothing else.
2. **A boundary whose closing violation completes the pattern.** Neckline,
   trough between two tops, triangle trendline, flag channel: same idea each
   time. Until that close happens the pattern is *forming*, and Murphy is blunt
   that a forming pattern is not yet tradable.
3. **A measuring rule.** Almost always the pattern's own height projected from
   the breakout point.
4. **A volume signature.** Contraction while the pattern builds, expansion on
   the breakout. He stresses this is far more critical on the upside than the
   downside, and *essential* at bottoms.

The two things this module exists to prevent:

* **Silent lookahead.** A pattern is only knowable once its last pivot is
  confirmed, which is strictly later than that pivot's own bar.
  :attr:`Pattern.detected_index` records when the shape became visible, and
  :meth:`Pattern.was_knowable_at` is the guard downstream code uses.
* **Marginal breakouts counting as breakouts.** Murphy names three filters -
  close beyond rather than intraday, a 1-3% price filter, and a two-bar time
  filter. :class:`PenetrationFilter` implements all three in one place so every
  detector applies them identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from quantdesk.analysis.swings import SwingKind, SwingPoint
from quantdesk.features.indicators import RollingWindow


class Bias(str, Enum):
    """Which way a pattern resolves once complete."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

    @property
    def sign(self) -> int:
        if self is Bias.BULLISH:
            return 1
        if self is Bias.BEARISH:
            return -1
        return 0

    @property
    def opposite(self) -> "Bias":
        if self is Bias.BULLISH:
            return Bias.BEARISH
        if self is Bias.BEARISH:
            return Bias.BULLISH
        return Bias.NEUTRAL

    @classmethod
    def from_sign(cls, value: float) -> "Bias":
        if value > 0:
            return cls.BULLISH
        if value < 0:
            return cls.BEARISH
        return cls.NEUTRAL


class PatternFamily(str, Enum):
    REVERSAL = "reversal"
    CONTINUATION = "continuation"


class PatternKind(str, Enum):
    """Every formation implemented from Murphy chapters 5 and 6."""

    # Reversal (ch. 5)
    HEAD_SHOULDERS_TOP = "head_shoulders_top"
    HEAD_SHOULDERS_BOTTOM = "head_shoulders_bottom"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ROUNDING_TOP = "rounding_top"
    ROUNDING_BOTTOM = "rounding_bottom"
    SPIKE_TOP = "spike_top"
    SPIKE_BOTTOM = "spike_bottom"
    # Continuation (ch. 6)
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    BROADENING = "broadening"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    FLAG = "flag"
    PENNANT = "pennant"
    RECTANGLE = "rectangle"
    MEASURED_MOVE = "measured_move"

    @property
    def family(self) -> PatternFamily:
        return (
            PatternFamily.REVERSAL
            if self in _REVERSAL_KINDS
            else PatternFamily.CONTINUATION
        )

    @property
    def innate_bias(self) -> Bias:
        """Bias fixed by the shape itself, ignoring context.

        Head and shoulders tops are bearish whichever trend precedes them.
        Wedges too: Murphy states a falling wedge is bullish and a rising wedge
        bearish. Triangles, rectangles and flags have no innate bias - they
        inherit the prior trend - so they return NEUTRAL and the detector fills
        the bias in from context. :attr:`inherits_trend` distinguishes the cases.
        """
        return _INNATE_BIAS.get(self, Bias.NEUTRAL)

    @property
    def inherits_trend(self) -> bool:
        """True when the pattern resolves in the direction of the prior trend."""
        return self in _TREND_INHERITING

    @property
    def is_top(self) -> bool:
        return self in _TOP_KINDS

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


_REVERSAL_KINDS = frozenset(
    {
        PatternKind.HEAD_SHOULDERS_TOP,
        PatternKind.HEAD_SHOULDERS_BOTTOM,
        PatternKind.DOUBLE_TOP,
        PatternKind.DOUBLE_BOTTOM,
        PatternKind.TRIPLE_TOP,
        PatternKind.TRIPLE_BOTTOM,
        PatternKind.ROUNDING_TOP,
        PatternKind.ROUNDING_BOTTOM,
        PatternKind.SPIKE_TOP,
        PatternKind.SPIKE_BOTTOM,
        # Murphy flags the broadening formation as a "megaphone top" - the one
        # triangle variant that usually marks a reversal rather than a pause.
        PatternKind.BROADENING,
    }
)

_TOP_KINDS = frozenset(
    {
        PatternKind.HEAD_SHOULDERS_TOP,
        PatternKind.DOUBLE_TOP,
        PatternKind.TRIPLE_TOP,
        PatternKind.ROUNDING_TOP,
        PatternKind.SPIKE_TOP,
        PatternKind.BROADENING,
    }
)

_INNATE_BIAS: dict[PatternKind, Bias] = {
    PatternKind.HEAD_SHOULDERS_TOP: Bias.BEARISH,
    PatternKind.HEAD_SHOULDERS_BOTTOM: Bias.BULLISH,
    PatternKind.DOUBLE_TOP: Bias.BEARISH,
    PatternKind.DOUBLE_BOTTOM: Bias.BULLISH,
    PatternKind.TRIPLE_TOP: Bias.BEARISH,
    PatternKind.TRIPLE_BOTTOM: Bias.BULLISH,
    PatternKind.ROUNDING_TOP: Bias.BEARISH,
    PatternKind.ROUNDING_BOTTOM: Bias.BULLISH,
    PatternKind.SPIKE_TOP: Bias.BEARISH,
    PatternKind.SPIKE_BOTTOM: Bias.BULLISH,
    PatternKind.BROADENING: Bias.BEARISH,
    PatternKind.RISING_WEDGE: Bias.BEARISH,
    PatternKind.FALLING_WEDGE: Bias.BULLISH,
}

_TREND_INHERITING = frozenset(
    {
        PatternKind.SYMMETRICAL_TRIANGLE,
        PatternKind.ASCENDING_TRIANGLE,
        PatternKind.DESCENDING_TRIANGLE,
        PatternKind.FLAG,
        PatternKind.PENNANT,
        PatternKind.RECTANGLE,
        PatternKind.MEASURED_MOVE,
    }
)


class PatternStage(str, Enum):
    """Lifecycle of a formation.

    Murphy's sequence exactly: the shape builds (FORMING), a decisive close
    through the boundary completes it (COMPLETE), price frequently bounces back
    to the broken boundary (RETURN_MOVE) which should hold, and then the move
    proceeds (RESUMED). FAILED covers a completed pattern that recrossed its own
    boundary - his explicit invalidation - and EXPIRED covers a shape that ran
    out of time, such as a triangle drifting past its apex.
    """

    FORMING = "forming"
    COMPLETE = "complete"
    RETURN_MOVE = "return_move"
    RESUMED = "resumed"
    FAILED = "failed"
    EXPIRED = "expired"

    @property
    def is_actionable(self) -> bool:
        """Only a broken-out pattern is tradable."""
        return self in (
            PatternStage.COMPLETE,
            PatternStage.RETURN_MOVE,
            PatternStage.RESUMED,
        )

    @property
    def is_dead(self) -> bool:
        return self in (PatternStage.FAILED, PatternStage.EXPIRED)


@dataclass(slots=True)
class Pattern:
    """A detected formation and everything needed to trade or display it."""

    kind: PatternKind
    symbol: str
    bias: Bias
    pivots: list[SwingPoint]
    start_index: int
    detected_index: int
    """Bar on which this shape first became knowable. Never earlier than the
    ``confirmed_index`` of its last pivot."""
    height: float = 0.0
    """Price height used by the measuring rule."""
    boundary: float = 0.0
    """Level whose closing violation completes the pattern, evaluated at
    :attr:`detected_index`. Sloped boundaries also carry ``boundary_slope``."""
    boundary_slope: float = 0.0
    stage: PatternStage = PatternStage.FORMING
    breakout_index: int = 0
    breakout_ts: datetime | None = None
    breakout_price: float = 0.0
    breakout_volume_ratio: float = 0.0
    """Breakout-bar volume over the pattern's own average. >1 is expansion."""
    formation_volume_ratio: float = 1.0
    """Late-formation volume over early-formation volume. <1 is Murphy's
    contraction-into-the-pattern signature."""
    objective: float = 0.0
    stop: float = 0.0
    apex_index: int | None = None
    """Where converging boundaries meet, i.e. the pattern's time limit."""
    return_move_low: float = 0.0
    notes: str = ""
    failure_reason: str = ""
    start_ts: datetime | None = None

    # ------------------------------------------------------------------ identity
    @property
    def key(self) -> tuple:
        """Stable identity so a shape is tracked, not re-emitted every bar.

        Bias is part of the identity because the two-sided formations - a
        symmetrical triangle or a rectangle with no clear prior trend - are
        tracked once per possible resolution, and those readings must not
        collide.
        """
        return (
            self.kind,
            self.bias,
            self.start_index,
            tuple(p.index for p in self.pivots),
        )

    @property
    def end_index(self) -> int:
        return self.pivots[-1].index if self.pivots else self.start_index

    @property
    def bars(self) -> int:
        """Width of the formation in bars, base to last pivot."""
        return max(0, self.end_index - self.start_index)

    def was_knowable_at(self, index: int) -> bool:
        """Guard against acting on a shape before it existed."""
        return index >= self.detected_index

    # ------------------------------------------------------------- measurement
    def boundary_at(self, index: int) -> float:
        """Boundary price projected to ``index``, for sloped necklines and lines."""
        return self.boundary + self.boundary_slope * (index - self.detected_index)

    @property
    def risk(self) -> float:
        """Distance from breakout to protective stop."""
        if self.breakout_price <= 0 or self.stop <= 0:
            return 0.0
        return abs(self.breakout_price - self.stop)

    @property
    def reward(self) -> float:
        if self.breakout_price <= 0 or self.objective <= 0:
            return 0.0
        return abs(self.objective - self.breakout_price)

    @property
    def reward_risk(self) -> float:
        """Murphy's chapter 16 gate: he wants at least 3:1 before committing."""
        r = self.risk
        if r <= 1e-12:
            return 0.0
        return self.reward / r

    @property
    def height_pct(self) -> float:
        ref = self.breakout_price or self.boundary
        if ref <= 0:
            return 0.0
        return self.height / ref

    # -------------------------------------------------------------- confidence
    def significance(self, current_index: int, atr: float = 0.0) -> float:
        """Composite quality in [0, 1].

        Weighted from the factors Murphy actually names as determining a
        pattern's forecasting power:

        * **Size.** "The longer the time period ... and the greater the height of
          the pattern, the greater the potential impending reversal. This is true
          of all chart patterns." Duration and height both feed in.
        * **Volume.** Expansion on the breakout, contraction during formation.
          Weighted more heavily for bullish resolutions, since he calls upside
          volume critical and downside volume merely desirable.
        * **Stage.** A forming shape is capped low, because it is not yet a
          pattern. A pattern that failed scores zero.
        * **Freshness.** Signals decay; a breakout twenty bars ago is stale.
        """
        if self.stage.is_dead:
            return 0.0

        duration = min(1.0, self.bars / 40.0)
        if atr > 0:
            size = min(1.0, self.height / (4.0 * atr))
        else:
            size = min(1.0, self.height_pct / 0.06)

        vol_expansion = min(1.0, max(0.0, self.breakout_volume_ratio / 2.0))
        vol_contraction = min(1.0, max(0.0, (1.4 - self.formation_volume_ratio) / 1.4))
        volume_score = 0.65 * vol_expansion + 0.35 * vol_contraction
        # Upside breakouts must show the volume; downside ones need not.
        volume_weight = 0.28 if self.bias is Bias.BULLISH else 0.14

        core = (
            0.30 * duration
            + 0.28 * size
            + volume_weight * volume_score
        )
        core /= 0.30 + 0.28 + volume_weight

        if self.stage is PatternStage.FORMING:
            return min(0.45, core * 0.5)

        age = max(0, current_index - self.breakout_index)
        freshness = max(0.0, 1.0 - age / 25.0)
        score = core * (0.55 + 0.45 * freshness)
        if self.stage is PatternStage.RETURN_MOVE:
            # The pullback to a broken boundary is Murphy's preferred entry, so
            # it is not penalised.
            score *= 1.05
        return max(0.0, min(1.0, score))

    def describe(self) -> str:
        parts = [self.kind.label, self.stage.value]
        if self.stage.is_actionable:
            parts.append(f"target {self.objective:.4g} stop {self.stop:.4g}")
            parts.append(f"R:R {self.reward_risk:.1f}")
        if self.notes:
            parts.append(self.notes)
        return " | ".join(parts)


# --------------------------------------------------------------------- filters


class PenetrationFilter:
    """Murphy's three breakout filters, applied together.

    He lists them in chapter 5 as the ways to cut false signals: require a
    *close* beyond the level rather than an intraday poke, add a price filter of
    roughly 1% (short-term) to 3% (major), and add a time filter of two
    consecutive closes beyond. Defaults here are the short-term settings, since
    the desk runs on intraday crypto bars.

    Stateful and keyed, so one instance can police many levels at once. The
    consecutive-bar counter resets the moment price falls back inside, which is
    what makes a one-bar spike through fail to qualify.
    """

    def __init__(
        self,
        pct: float = 0.01,
        confirm_bars: int = 1,
        atr_mult: float = 0.0,
    ) -> None:
        self.pct = pct
        self.confirm_bars = max(1, confirm_bars)
        #: Optional ATR-based margin, used instead of ``pct`` when larger.
        self.atr_mult = atr_mult
        self._streak: dict[object, int] = {}

    def margin(self, level: float, atr: float = 0.0) -> float:
        by_pct = abs(level) * self.pct
        by_atr = self.atr_mult * atr if self.atr_mult > 0 else 0.0
        return max(by_pct, by_atr)

    def beyond(
        self, close: float, level: float, upside: bool, atr: float = 0.0
    ) -> bool:
        """Single-bar test: is this close decisively beyond the level?"""
        m = self.margin(level, atr)
        return close > level + m if upside else close < level - m

    def check(
        self,
        key: object,
        close: float,
        level: float,
        upside: bool,
        atr: float = 0.0,
    ) -> bool:
        """Stateful test including the consecutive-close time filter."""
        if self.beyond(close, level, upside, atr):
            self._streak[key] = self._streak.get(key, 0) + 1
            if self._streak[key] >= self.confirm_bars:
                return True
        else:
            self._streak[key] = 0
        return False

    def reset(self, key: object) -> None:
        self._streak.pop(key, None)

    def forget(self, keys: set) -> None:
        """Drop state for keys no longer tracked, so the dict cannot grow forever."""
        for k in list(self._streak):
            if k not in keys:
                del self._streak[k]


class VolumeContext:
    """Rolling volume statistics used for pattern confirmation.

    Two distinct questions, both of which Murphy asks of every formation:
    did activity *dry up* while the pattern built, and did it *expand* on the
    breakout? Kept as one object so detectors do not each maintain their own.
    """

    def __init__(self, period: int = 60) -> None:
        self._win = RollingWindow(period)
        self._history: list[float] = []
        self._cap = max(period, 400)

    def update(self, volume: float) -> None:
        self._win.update(volume)
        self._history.append(volume)
        if len(self._history) > self._cap:
            del self._history[: len(self._history) - self._cap]

    @property
    def average(self) -> float:
        # RollingWindow.mean is a property, not a method.
        return self._win.mean

    def ratio(self, volume: float) -> float:
        """Volume relative to the recent average. >1 means expansion."""
        avg = self.average
        if avg <= 0:
            return 1.0
        return volume / avg

    def window(self, start_index: int, end_index: int, current_index: int) -> list[float]:
        """Volumes for an absolute bar range, mapped into the local history."""
        offset = current_index - (len(self._history) - 1)
        lo = max(0, start_index - offset)
        hi = max(lo, min(len(self._history), end_index - offset + 1))
        return self._history[lo:hi]

    def formation_ratio(
        self, start_index: int, end_index: int, current_index: int
    ) -> float:
        """Late-half volume over early-half volume inside a formation.

        Below 1.0 is the contraction Murphy expects as swings narrow within a
        consolidation. Returns 1.0 (neutral) when there is too little data to
        judge, so a short pattern is neither rewarded nor punished.
        """
        vols = self.window(start_index, end_index, current_index)
        if len(vols) < 6:
            return 1.0
        mid = len(vols) // 2
        early = sum(vols[:mid]) / max(1, mid)
        late = sum(vols[mid:]) / max(1, len(vols) - mid)
        if early <= 0:
            return 1.0
        return late / early


# --------------------------------------------------------------------- context


@dataclass(slots=True)
class DetectorContext:
    """Everything a detector may look at on the current bar.

    Deliberately narrow. Detectors get ``confirmed`` pivots only - never the
    tentative extreme - so it is not possible for a formation to be built from a
    pivot that had not yet happened.
    """

    symbol: str
    index: int
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    atr: float
    confirmed: list[SwingPoint]
    volume_ctx: VolumeContext
    prior_trend: Bias = Bias.NEUTRAL
    """Trend preceding the formation, from :mod:`quantdesk.analysis.dow`. Used by
    the patterns that inherit their direction rather than owning one."""

    def peaks(self, n: int | None = None) -> list[SwingPoint]:
        out = [p for p in self.confirmed if p.kind is SwingKind.PEAK]
        return out if n is None else out[-n:]

    def troughs(self, n: int | None = None) -> list[SwingPoint]:
        out = [p for p in self.confirmed if p.kind is SwingKind.TROUGH]
        return out if n is None else out[-n:]

    def atr_or(self, fallback_pct: float = 0.01) -> float:
        """ATR, falling back to a fraction of price before it warms up."""
        if self.atr > 0:
            return self.atr
        return max(1e-9, self.close * fallback_pct)

    @property
    def newest_pivot_index(self) -> int:
        """Bar index of the most recent confirmed pivot, or -1 if there are none.

        Monotonically increasing, which makes it a cheap signature for "has the
        pivot set changed?" - the question that decides whether a pivot-based
        detector needs to re-scan at all.
        """
        return self.confirmed[-1].index if self.confirmed else -1


class PatternDetector:
    """Base class for detectors.

    Contract: :meth:`update` is called once per bar with a
    :class:`DetectorContext` and returns patterns whose state changed on this
    bar. Detectors own their in-progress shapes and are responsible for
    advancing them through :class:`PatternStage`.
    """

    kinds: tuple[PatternKind, ...] = ()

    #: Newest pivot index at the last re-scan. See :meth:`pivots_changed`.
    _last_pivot_seen: int = -2

    def pivots_changed(self, ctx: DetectorContext) -> bool:
        """Has a new pivot been confirmed since the last re-scan?

        Formations built from pivots can only change when the pivot set does, so
        re-scanning on every bar is both wasted work and a source of spurious
        near-duplicate detections: the same shape gets re-registered under a
        slightly different key as ATR drifts and thresholds shift underneath it.
        Detectors that work from raw bars rather than pivots - flags and pennants -
        do not use this.
        """
        newest = ctx.newest_pivot_index
        if newest == self._last_pivot_seen:
            return False
        self._last_pivot_seen = newest
        return True

    def accepts(self, pattern: "Pattern", tracked: dict) -> bool:
        """Is this a genuinely new formation, or a re-reading of a live one?

        Keeps the earliest detection rather than the latest. The earliest is also
        the earliest signal, so preferring it avoids quietly backdating a pattern
        the desk could not have acted on yet.
        """
        if pattern.key in tracked:
            return False
        for other in tracked.values():
            if not other.stage.is_dead and overlaps(pattern, other):
                return False
        return True

    def update(self, ctx: DetectorContext) -> list[Pattern]:  # pragma: no cover
        raise NotImplementedError

    @property
    def active(self) -> list[Pattern]:  # pragma: no cover
        return []


def overlaps(a: Pattern, b: Pattern, min_share: float = 0.5) -> bool:
    """Do two patterns describe substantially the same price action?

    As the pivot window slides forward, each newly confirmed pivot produces a
    formation sharing most of its pivots with the previous one. Those are not new
    formations, they are the same one re-read a bar later, and treating them as
    distinct floods the book with near-duplicates and inverts how often each
    pattern appears to occur.

    Patterns built from raw bars rather than pivots - flags and pennants - are
    compared on their spans instead.
    """
    if a.kind is not b.kind or a.bias is not b.bias:
        return False
    ai = {p.index for p in a.pivots}
    bi = {p.index for p in b.pivots}
    if not ai or not bi:
        return not (a.end_index < b.start_index or b.end_index < a.start_index)
    shared = len(ai & bi)
    return shared / min(len(ai), len(bi)) >= min_share


def similar(a: float, b: float, tolerance: float) -> bool:
    """Are two prices at effectively the same level?

    Used for the "peaks at about the same height" tests in double and triple
    tops, and for detecting a flat triangle boundary. Tolerance is an absolute
    price distance, normally derived from ATR by the caller.
    """
    return abs(a - b) <= tolerance
