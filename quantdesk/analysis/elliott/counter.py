"""Finding wave counts in confirmed pivots.

Wave counting is the least mechanical technique in Murphy's book - two competent
Elliotticians routinely disagree on the same chart - so this module is deliberately
honest about that rather than pretending to a single answer:

* It generates *several* candidate counts and returns them ranked, instead of
  asserting one.
* :attr:`WaveCount.violations` is populated rather than silently rejecting, so a
  count that breaks a rule is visible as a broken count instead of vanishing.
* :attr:`WaveCount.confidence` is dominated by rule compliance, and the agent layer
  is expected to size down or stand aside when the best available count is weak.

The counting itself walks back from the most recent confirmed pivot and tries to fit
the impulse and corrective templates onto alternating pivots, which is the same
thing a chartist does by eye when they ask "what if this low was wave 4?".
"""

from __future__ import annotations

from quantdesk.analysis.elliott.waves import (
    CORRECTION_LABELS,
    IMPULSE_LABELS,
    Wave,
    WaveCount,
    WaveKind,
    WaveLabel,
    classify_correction,
    project_wave3,
    project_wave5,
    retracement_targets,
    validate,
)
from quantdesk.analysis.patterns.base import Bias
from quantdesk.analysis.swings import SwingKind, SwingPoint


class ElliottCounter:
    """Maintains candidate wave counts for one symbol.

    Parameters
    ----------
    strict_overlap:
        Treat wave 4 overlapping wave 1 as fatal (Murphy's stocks rule) rather than
        a warning (his commodities caveat).
    max_pivots:
        How far back to consider. Elliott counts nest across degrees indefinitely;
        this keeps the search to the degree the desk is trading.
    """

    def __init__(
        self,
        symbol: str,
        strict_overlap: bool = False,
        max_pivots: int = 12,
        min_confidence: float = 0.0,
    ) -> None:
        self.symbol = symbol
        self.strict_overlap = strict_overlap
        self.max_pivots = max_pivots
        self.min_confidence = min_confidence
        self._counts: list[WaveCount] = []
        self._index = -1

    # ----------------------------------------------------------------- update
    def update(self, index: int, pivots: list[SwingPoint]) -> list[WaveCount]:
        """Recount from confirmed pivots. Returns candidates, best first."""
        self._index = index
        pool = pivots[-self.max_pivots :]
        candidates: list[WaveCount] = []

        # An impulse needs six pivots to define five legs; a correction needs four.
        for size, kind in ((6, WaveKind.IMPULSE), (4, WaveKind.CORRECTION)):
            for start in range(len(pool) - size, -1, -1):
                window = pool[start : start + size]
                count = self._build(window, kind)
                if count is not None:
                    candidates.append(count)

        # Partial impulses matter too: knowing we are mid-wave-3 is the most
        # actionable read Elliott offers, and waiting for all five loses it.
        for size in (5, 4, 3):
            if len(pool) < size:
                continue
            window = pool[-size:]
            count = self._build(window, WaveKind.IMPULSE, partial=True)
            if count is not None:
                candidates.append(count)

        candidates = [c for c in candidates if c.confidence >= self.min_confidence]
        candidates.sort(key=lambda c: (c.confidence, len(c.waves)), reverse=True)
        self._counts = candidates[:6]
        return self._counts

    # ------------------------------------------------------------------ build
    def _build(
        self,
        window: list[SwingPoint],
        kind: WaveKind,
        partial: bool = False,
    ) -> WaveCount | None:
        """Label a run of alternating pivots as an impulse or a correction."""
        if len(window) < 3:
            return None
        # Pivots must alternate, or they do not describe consecutive legs.
        if any(a.kind is b.kind for a, b in zip(window, window[1:])):
            return None

        labels = IMPULSE_LABELS if kind is WaveKind.IMPULSE else CORRECTION_LABELS
        legs = list(zip(window, window[1:]))
        if not partial and len(legs) != len(labels):
            return None
        if len(legs) > len(labels):
            return None

        # Direction comes from the first leg: an impulse starts with a motive wave.
        first_up = legs[0][1].price > legs[0][0].price
        waves = [
            Wave(label=labels[i], start=a, end=b)
            for i, (a, b) in enumerate(legs)
        ]

        # Motive waves must all travel the structure's way, corrective ones against.
        for w in waves:
            if w.label.is_motive and w.up is not first_up:
                return None
            if not w.label.is_motive and w.up is first_up:
                return None

        # An impulse must make net progress; otherwise it is a range, not a wave.
        net = window[-1].price - window[0].price
        if first_up and net <= 0:
            return None
        if not first_up and net >= 0:
            return None

        count = WaveCount(
            kind=kind,
            waves=waves,
            up=first_up,
            detected_index=max(p.confirmed_index for p in window),
            notes="partial count" if partial else "",
        )
        if kind is WaveKind.CORRECTION:
            count.shape = classify_correction(waves)
        count.violations = validate(count, self.strict_overlap)
        return count

    # -------------------------------------------------------------- accessors
    @property
    def counts(self) -> list[WaveCount]:
        return list(self._counts)

    @property
    def best(self) -> WaveCount | None:
        return self._counts[0] if self._counts else None

    def position(self) -> str:
        """Plain-language read of where the market sits in the best count."""
        best = self.best
        if best is None:
            return "no wave count"
        label = best.current_wave
        if label is None:
            return "no wave count"
        direction = "up" if best.up else "down"
        if best.kind is WaveKind.IMPULSE:
            return f"in wave {label.value} of a {direction} impulse"
        return f"in wave {label.value} of a {direction} correction ({best.shape.value})"

    def bias(self) -> Bias:
        """Directional read from the best count.

        The tradable expectation depends on which wave is in progress. Waves 3 and 5
        run with the impulse; waves 2 and 4 are corrections against it, and Murphy's
        point about corrections is that they set up the *next* motive wave. So a
        wave 2 or 4 in an up impulse is still ultimately bullish, which is why they
        map to the impulse direction rather than against it.
        """
        best = self.best
        if best is None or best.confidence < 0.4:
            return Bias.NEUTRAL
        if best.kind is WaveKind.IMPULSE:
            # A completed five-wave move is a warning, not a continuation.
            if best.complete and best.current_wave is WaveLabel.FIVE:
                return Bias.BEARISH if best.up else Bias.BULLISH
            return Bias.BULLISH if best.up else Bias.BEARISH
        # A correction resolves in the direction opposite to itself.
        return Bias.BEARISH if best.up else Bias.BULLISH

    def targets(self) -> dict[str, float]:
        """Fibonacci projections from the best count."""
        best = self.best
        if best is None:
            return {}
        out: dict[str, float] = {}
        w3 = project_wave3(best)
        if w3 is not None:
            out["wave3_min"] = w3
        w5 = project_wave5(best)
        if w5 is not None:
            out["wave5_min"], out["wave5_max"] = w5
        last = best.waves[-1] if best.waves else None
        if last is not None and last.label.is_motive:
            for ratio, level in retracement_targets(last).items():
                out[f"retrace_{ratio}"] = level
        return out

    def summary(self) -> list[str]:
        return [
            f"{c.describe()} conf {c.confidence:.2f}"
            + (f" -- {'; '.join(c.violations)}" if c.violations else "")
            for c in self._counts
        ]
