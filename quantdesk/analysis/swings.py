"""Swing pivot detection.

Murphy's entire framework rests on one observation: markets move in zigzags, and
the *direction of the successive peaks and troughs* is what defines trend. Almost
everything else in the book - Dow trend classification, support and resistance,
trendlines, retracements, chart patterns, Elliott waves - is built on top of
those pivots. So this module has to be right before any of it means anything.

The detector is a threshold zigzag: track the running extreme in the current
direction, and confirm a reversal only once price has retraced from that extreme
by more than a threshold. The threshold is expressed in ATR multiples by default
rather than a fixed percentage, so the same code behaves sensibly on a 0.5%-vol
instrument and a 5%-vol one.

**The lookahead trap.** A zigzag pivot is only knowable in hindsight: the peak
was the peak because price later fell away from it. Naively drawing zigzags on a
chart and then "trading the pivots" is one of the most common ways to build a
backtest that cannot be traded, because at the time of the pivot bar you did not
yet know it was a pivot. This implementation is explicit about it:

* :attr:`SwingPoint.confirmed_index` records the bar on which the pivot became
  knowable, which is always later than the pivot's own bar.
* :meth:`SwingDetector.confirmed` only ever returns pivots whose confirmation
  bar has already passed.
* The unconfirmed running extreme is available separately as
  :attr:`SwingDetector.tentative`, clearly labelled, so callers cannot mistake
  it for established structure.

Anything downstream that needs pivots must use ``confirmed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterator

from quantdesk.core.types import Bar
from quantdesk.features.indicators import ATR


class SwingKind(str, Enum):
    PEAK = "peak"
    TROUGH = "trough"

    @property
    def opposite(self) -> "SwingKind":
        return SwingKind.TROUGH if self is SwingKind.PEAK else SwingKind.PEAK

    @property
    def sign(self) -> int:
        return 1 if self is SwingKind.PEAK else -1


@dataclass(slots=True)
class SwingPoint:
    """A confirmed turning point in the price series."""

    index: int
    """Bar index of the pivot itself."""
    ts: datetime
    price: float
    kind: SwingKind
    confirmed_index: int
    """Bar index on which this pivot became knowable. Always > ``index``."""
    volume: float = 0.0
    #: How far price moved away from the pivot to confirm it, in ATR units.
    confirm_atr: float = 0.0

    @property
    def is_peak(self) -> bool:
        return self.kind is SwingKind.PEAK

    @property
    def bars_to_confirm(self) -> int:
        return self.confirmed_index - self.index


@dataclass(slots=True)
class TentativeSwing:
    """The running extreme that has not yet been confirmed as a pivot.

    Deliberately a separate type from :class:`SwingPoint` so it is impossible to
    pass an unconfirmed extreme into code expecting established structure.
    """

    index: int
    ts: datetime
    price: float
    kind: SwingKind
    retrace_atr: float = 0.0


class SwingDetector:
    """Threshold zigzag over a stream of bars.

    Parameters
    ----------
    atr_mult:
        Reversal threshold in ATR multiples. Murphy gives no number here - he
        identifies pivots by eye - so this is a chosen parameter. Around 1.5 ATR
        filters intrabar noise while keeping genuine intermediate swings.
    atr_period:
        Lookback for the ATR used to scale the threshold.
    min_pct:
        Floor on the threshold as a fraction of price, so a collapse in ATR
        cannot make the detector hypersensitive.
    use_extremes:
        When True, pivots sit on bar highs/lows, which is what Murphy specifies
        for trendlines ("include all price action"). When False, closes are used.
    """

    def __init__(
        self,
        atr_mult: float = 1.5,
        atr_period: int = 14,
        min_pct: float = 0.002,
        use_extremes: bool = True,
        max_points: int = 200,
    ) -> None:
        self.atr_mult = atr_mult
        self.min_pct = min_pct
        self.use_extremes = use_extremes
        self.max_points = max_points

        self._atr = ATR(atr_period)
        self._points: list[SwingPoint] = []
        self._index = -1
        self._direction: int = 0          # +1 seeking a peak, -1 seeking a trough
        self._extreme_price = 0.0
        self._extreme_index = -1
        self._extreme_ts: datetime | None = None
        self._extreme_volume = 0.0
        self._last_bar: Bar | None = None

        # Bootstrap trackers, used only until the first direction is established.
        self._boot_hi: float | None = None
        self._boot_lo: float | None = None
        self._boot_hi_idx = -1
        self._boot_lo_idx = -1
        self._boot_hi_ts: datetime | None = None
        self._boot_lo_ts: datetime | None = None
        self._boot_hi_vol = 0.0
        self._boot_lo_vol = 0.0

    # ------------------------------------------------------------------ update
    def update(self, bar: Bar) -> SwingPoint | None:
        """Feed one bar. Returns a pivot if this bar confirmed one."""
        self._index += 1
        self._last_bar = bar
        atr = self._atr.update(bar.high, bar.low, bar.close)

        hi = bar.high if self.use_extremes else bar.close
        lo = bar.low if self.use_extremes else bar.close

        threshold = max(self.atr_mult * atr, self.min_pct * max(bar.close, 1e-9))
        if threshold <= 0:
            return None

        # Bootstrap. Direction is unknown at the start, so track the running high
        # and low independently and let whichever threshold breaks first decide.
        # Anchoring on a single seed price would be biased, and comparing against
        # the current bar's own close would only ever measure its wick.
        if self._direction == 0:
            if self._boot_hi is None:
                self._boot_hi = self._boot_lo = bar.close
                self._boot_hi_idx = self._boot_lo_idx = self._index
                self._boot_hi_ts = self._boot_lo_ts = bar.ts
                self._boot_hi_vol = self._boot_lo_vol = bar.volume
                return None

            if hi > self._boot_hi:
                self._boot_hi, self._boot_hi_idx = hi, self._index
                self._boot_hi_ts, self._boot_hi_vol = bar.ts, bar.volume
            if lo < self._boot_lo:
                self._boot_lo, self._boot_lo_idx = lo, self._index
                self._boot_lo_ts, self._boot_lo_vol = bar.ts, bar.volume

            fell_from_high = self._boot_hi - lo >= threshold
            rose_from_low = hi - self._boot_lo >= threshold

            if fell_from_high and self._boot_hi_idx <= self._boot_lo_idx:
                # We made a high, then dropped away from it: that high was a peak.
                self._set_extreme(
                    self._boot_hi_idx, self._boot_hi_ts, self._boot_hi, self._boot_hi_vol
                )
                pivot = self._emit(SwingKind.PEAK, threshold, atr)
                self._direction = -1
                self._set_extreme(self._index, bar.ts, lo, bar.volume)
                return pivot
            if rose_from_low:
                self._set_extreme(
                    self._boot_lo_idx, self._boot_lo_ts, self._boot_lo, self._boot_lo_vol
                )
                pivot = self._emit(SwingKind.TROUGH, threshold, atr)
                self._direction = 1
                self._set_extreme(self._index, bar.ts, hi, bar.volume)
                return pivot
            if fell_from_high:
                self._set_extreme(
                    self._boot_hi_idx, self._boot_hi_ts, self._boot_hi, self._boot_hi_vol
                )
                pivot = self._emit(SwingKind.PEAK, threshold, atr)
                self._direction = -1
                self._set_extreme(self._index, bar.ts, lo, bar.volume)
                return pivot
            return None

        if self._direction > 0:
            # Riding up toward a peak.
            if hi > self._extreme_price:
                self._set_extreme(self._index, bar.ts, hi, bar.volume)
                return None
            if self._extreme_price - lo >= threshold:
                # Retraced far enough: the prior extreme was a genuine peak.
                pivot = self._emit(SwingKind.PEAK, threshold, atr)
                self._direction = -1
                self._set_extreme(self._index, bar.ts, lo, bar.volume)
                return pivot
        else:
            if lo < self._extreme_price:
                self._set_extreme(self._index, bar.ts, lo, bar.volume)
                return None
            if hi - self._extreme_price >= threshold:
                pivot = self._emit(SwingKind.TROUGH, threshold, atr)
                self._direction = 1
                self._set_extreme(self._index, bar.ts, hi, bar.volume)
                return pivot
        return None

    def _set_extreme(
        self, index: int, ts: datetime, price: float, volume: float
    ) -> None:
        self._extreme_index = index
        self._extreme_ts = ts
        self._extreme_price = price
        self._extreme_volume = volume

    def _emit(self, kind: SwingKind, threshold: float, atr: float) -> SwingPoint:
        pivot = SwingPoint(
            index=self._extreme_index,
            ts=self._extreme_ts or (self._last_bar.ts if self._last_bar else None),  # type: ignore[arg-type]
            price=self._extreme_price,
            kind=kind,
            confirmed_index=self._index,
            volume=self._extreme_volume,
            confirm_atr=(threshold / atr) if atr > 0 else 0.0,
        )
        self._points.append(pivot)
        if len(self._points) > self.max_points:
            del self._points[: len(self._points) - self.max_points]
        return pivot

    # -------------------------------------------------------------- accessors
    @property
    def confirmed(self) -> list[SwingPoint]:
        """All confirmed pivots, oldest first. Safe to use for decisions."""
        return list(self._points)

    def last(self, kind: SwingKind | None = None, n: int = 1) -> list[SwingPoint]:
        """Most recent ``n`` pivots, optionally filtered to peaks or troughs."""
        pts = self._points if kind is None else [p for p in self._points if p.kind is kind]
        return pts[-n:]

    @property
    def peaks(self) -> list[SwingPoint]:
        return [p for p in self._points if p.kind is SwingKind.PEAK]

    @property
    def troughs(self) -> list[SwingPoint]:
        return [p for p in self._points if p.kind is SwingKind.TROUGH]

    @property
    def tentative(self) -> TentativeSwing | None:
        """The unconfirmed running extreme. NOT established structure.

        Useful for display and for measuring how far an in-progress leg has run,
        but must never be treated as a pivot for signal generation.
        """
        if self._direction == 0 or self._extreme_ts is None:
            return None
        return TentativeSwing(
            index=self._extreme_index,
            ts=self._extreme_ts,
            price=self._extreme_price,
            kind=SwingKind.PEAK if self._direction > 0 else SwingKind.TROUGH,
        )

    @property
    def direction(self) -> int:
        """+1 if currently rising toward a potential peak, -1 if falling."""
        return self._direction

    @property
    def bar_index(self) -> int:
        return self._index

    @property
    def ready(self) -> bool:
        """At least two pivots, the minimum to say anything about structure."""
        return len(self._points) >= 2

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self) -> Iterator[SwingPoint]:
        return iter(self._points)


@dataclass(slots=True)
class SwingLeg:
    """A directional move between two consecutive pivots."""

    start: SwingPoint
    end: SwingPoint

    @property
    def is_up(self) -> bool:
        return self.end.price > self.start.price

    @property
    def height(self) -> float:
        return abs(self.end.price - self.start.price)

    @property
    def bars(self) -> int:
        return self.end.index - self.start.index

    @property
    def pct(self) -> float:
        if self.start.price <= 0:
            return 0.0
        return (self.end.price - self.start.price) / self.start.price

    def retracement_of(self, price: float) -> float:
        """How much of this leg ``price`` has given back, as a fraction.

        0.0 means price is still at the leg's end; 1.0 means the whole leg has
        been retraced back to its origin. Values above 1.0 mean price has moved
        beyond the leg's origin.
        """
        if self.height <= 1e-12:
            return 0.0
        return abs(self.end.price - price) / self.height


def legs(points: list[SwingPoint]) -> list[SwingLeg]:
    """Consecutive pivots paired into legs."""
    return [SwingLeg(a, b) for a, b in zip(points, points[1:])]
