"""The single entry point: feed it bars, get patterns.

Every detector needs the same inputs - confirmed pivots, ATR, rolling volume, and
the prevailing trend - so the scanner computes them once per bar and fans them
out. That also guarantees all detectors see an identical view of the market, which
matters when their readings get combined later.

Two things worth being explicit about.

**Prior trend is an approximation.** The continuation patterns resolve "in the
direction of the market trend that preceded" them, so they need the trend from
*before* the consolidation. What the scanner supplies is
:func:`~quantdesk.analysis.dow.classify_trend` over the confirmed pivots as of
now. While a consolidation is young that still reflects the move that led into it,
which is the intent; once a long range has built its own pivots the reading decays
toward sideways, and the affected patterns then track both resolutions rather than
guessing. That is the conservative failure mode.

**Nothing here looks ahead.** The scanner owns its
:class:`~quantdesk.analysis.swings.SwingDetector` and passes only
``confirmed`` pivots into the context. The tentative running extreme is never
exposed to a detector, so no formation can be built from a pivot that had not yet
happened.
"""

from __future__ import annotations

from quantdesk.analysis.dow import TrendState, classify_trend
from quantdesk.analysis.patterns.base import (
    Bias,
    DetectorContext,
    Pattern,
    PatternDetector,
    PatternFamily,
    PatternKind,
    VolumeContext,
)
from quantdesk.analysis.patterns.double_triple import DoubleTripleDetector
from quantdesk.analysis.patterns.flags import FlagPennantDetector
from quantdesk.analysis.patterns.head_shoulders import HeadShouldersDetector
from quantdesk.analysis.patterns.rectangles import (
    MeasuredMoveDetector,
    RectangleDetector,
)
from quantdesk.analysis.patterns.saucer_spike import RoundingDetector, SpikeDetector
from quantdesk.analysis.patterns.triangles import TriangleDetector
from quantdesk.analysis.swings import SwingDetector, SwingPoint
from quantdesk.core.types import Bar
from quantdesk.features.indicators import ATR


def default_detectors() -> list[PatternDetector]:
    """One instance of every formation in Murphy chapters 5 and 6."""
    return [
        HeadShouldersDetector(),
        DoubleTripleDetector(),
        RoundingDetector(),
        SpikeDetector(),
        TriangleDetector(),
        FlagPennantDetector(),
        RectangleDetector(),
        MeasuredMoveDetector(),
    ]


class PatternScanner:
    """Runs every pattern detector over one symbol's bar stream.

    Parameters
    ----------
    swings:
        Optionally share an existing detector, so the scanner and the rest of the
        analysis stack agree on exactly one set of pivots rather than each
        maintaining its own.
    detectors:
        Override the default set, e.g. to run only reversal patterns.
    """

    def __init__(
        self,
        symbol: str,
        atr_period: int = 14,
        volume_period: int = 60,
        swings: SwingDetector | None = None,
        detectors: list[PatternDetector] | None = None,
        flat_tolerance: float = 0.0025,
        strict: bool = False,
    ) -> None:
        self.symbol = symbol
        self.flat_tolerance = flat_tolerance
        #: Raise on detector errors instead of recording them. Use in tests and
        #: backtests, where a broken detector should fail loudly.
        self.strict = strict
        #: Recent detector failures, newest last. Surfaced in the terminal UI so
        #: a degraded read is visible rather than looking like a quiet market.
        self.errors: list[str] = []
        self._swings = swings if swings is not None else SwingDetector(atr_period=atr_period)
        self._owns_swings = swings is None
        self._atr = ATR(atr_period)
        self._volume = VolumeContext(volume_period)
        self._detectors = detectors if detectors is not None else default_detectors()
        self._index = -1
        self._trend: TrendState | None = None
        self._last_changed: list[Pattern] = []

    # ----------------------------------------------------------------- update
    def update(self, bar: Bar) -> list[Pattern]:
        """Feed one closed bar. Returns patterns whose state changed on this bar."""
        self._index += 1
        atr = self._atr.update(bar.high, bar.low, bar.close)
        self._volume.update(bar.volume)
        # Only advance the swing detector if we own it; a shared one is driven by
        # its owner, and double-feeding would corrupt its bar indexing.
        if self._owns_swings:
            self._swings.update(bar)

        confirmed = self._swings.confirmed
        self._trend = classify_trend(confirmed, bar.close, self.flat_tolerance)

        ctx = DetectorContext(
            symbol=self.symbol,
            index=self._index,
            ts=bar.ts,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            atr=atr,
            confirmed=confirmed,
            volume_ctx=self._volume,
            prior_trend=Bias.from_sign(self._trend.direction.sign),
        )

        changed: list[Pattern] = []
        for detector in self._detectors:
            # One misbehaving detector must not take the whole read down, so
            # failures are contained - but they are *recorded*, never swallowed.
            # A silently skipped detector looks identical to a quiet market,
            # which is how a real bug once hid here.
            try:
                changed.extend(detector.update(ctx))
            except Exception as exc:  # pragma: no cover - defensive
                if self.strict:
                    raise
                self.errors.append(
                    f"bar {self._index}: {type(detector).__name__}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if len(self.errors) > 50:
                    del self.errors[:-50]
        self._last_changed = changed
        return changed

    # -------------------------------------------------------------- accessors
    @property
    def bar_index(self) -> int:
        return self._index

    @property
    def atr(self) -> float:
        # ATR.value is a property, not a method.
        return self._atr.value

    @property
    def trend(self) -> TrendState | None:
        """Current Dow-style trend classification."""
        return self._trend

    @property
    def pivots(self) -> list[SwingPoint]:
        return self._swings.confirmed

    @property
    def changed(self) -> list[Pattern]:
        """Patterns that changed state on the most recent bar."""
        return list(self._last_changed)

    @property
    def active(self) -> list[Pattern]:
        """Every live pattern across all detectors, newest first."""
        out: list[Pattern] = []
        for detector in self._detectors:
            out.extend(detector.active)
        out.sort(key=lambda p: p.detected_index, reverse=True)
        return out

    def forming(self) -> list[Pattern]:
        """Shapes still building. Watch list only - not tradable."""
        return [p for p in self.active if p.stage.value == "forming"]

    def actionable(
        self,
        min_significance: float = 0.0,
        min_reward_risk: float = 0.0,
        require_volume: bool = False,
        family: PatternFamily | None = None,
    ) -> list[Pattern]:
        """Completed patterns passing the requested quality gates.

        ``min_reward_risk`` is where Murphy's chapter 16 discipline plugs in: he
        wants at least 3:1 before a trade is worth taking. It is off by default so
        the scanner reports honestly and the *caller* decides what to trade.

        ``require_volume`` applies the asymmetric volume test - expansion demanded
        on upside breakouts, tolerated as absent on downside ones.
        """
        from quantdesk.analysis.patterns.lifecycle import volume_confirms

        out: list[Pattern] = []
        for pattern in self.active:
            if not pattern.stage.is_actionable:
                continue
            if family is not None and pattern.kind.family is not family:
                continue
            if pattern.significance(self._index, self.atr) < min_significance:
                continue
            if min_reward_risk > 0 and pattern.reward_risk < min_reward_risk:
                continue
            if require_volume and not volume_confirms(pattern):
                continue
            out.append(pattern)
        out.sort(
            key=lambda p: p.significance(self._index, self.atr), reverse=True
        )
        return out

    def net_bias(self, min_significance: float = 0.15) -> float:
        """Significance-weighted directional read across all live patterns, in [-1, 1].

        A crude weight-of-evidence tally over the pattern layer alone. The real
        checklist in chapter 19 weighs trend, volume, oscillators and patterns
        together; this is only the pattern contribution to it.
        """
        total = 0.0
        weight = 0.0
        for pattern in self.actionable(min_significance=min_significance):
            sig = pattern.significance(self._index, self.atr)
            total += sig * pattern.bias.sign
            weight += sig
        if weight <= 0:
            return 0.0
        return max(-1.0, min(1.0, total / weight))

    def by_kind(self, kind: PatternKind) -> list[Pattern]:
        return [p for p in self.active if p.kind is kind]

    def summary(self) -> list[str]:
        """One line per live pattern, for the terminal UI."""
        return [
            f"{p.kind.label}: {p.describe()} "
            f"(sig {p.significance(self._index, self.atr):.2f})"
            for p in self.active
        ]
