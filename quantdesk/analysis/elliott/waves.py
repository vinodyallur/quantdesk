"""Elliott wave structure, rules and Fibonacci ratios.

Murphy names three aspects of wave theory "in that order of importance": pattern,
ratio, and time. This module follows that ordering - the rules that define a valid
pattern are hard constraints, the Fibonacci ratios are projections layered on top,
and time relationships are recorded but not relied on, since he notes they are
"considered by some Elliotticians to be less reliable".

The inviolable rules, as checked by :func:`validate`:

* **A correction can never take place in five waves.** "Corrective waves are
  threes, never fives (with the exception of triangles)." He treats this as one of
  the most important rules, because it tells you whether a completed move is the
  whole correction or only the first leg of a larger one.
* **Wave 4 never overlaps wave 1.** Murphy flags this as "the unbreakable rule ...
  in stocks" while noting it "is not as rigid in commodities" where intraday
  penetrations occur. Both readings are supported through ``strict_overlap``.
* **Wave 2 never fully retraces wave 1.** If it did, wave 1 was not a wave 1.
* **Wave 3 is never the shortest** of waves 1, 3 and 5.

Everything is built from confirmed swing pivots, so a wave count can only ever use
turning points that had already been established.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from quantdesk.analysis.swings import SwingKind, SwingPoint

#: Ratios Murphy derives from the Fibonacci sequence. The sequence converges on
#: .618 from below and 1.618 from above, and he lists 1.00, .50 and .67 as the
#: early values that matter in their own right.
PHI = 0.618
PHI_INV = 1.618
#: "multiplying wave 1 by 3.236 (2 x 1.618)" for the wave 5 objective.
WAVE5_MULT = 3.236
RETRACEMENTS = (0.382, 0.50, 0.618, 0.667)


class WaveKind(str, Enum):
    IMPULSE = "impulse"
    """Five waves in the direction of the larger trend."""
    CORRECTION = "correction"
    """Three waves against it - a-b-c."""


class WaveLabel(str, Enum):
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    A = "a"
    B = "b"
    C = "c"

    @property
    def is_motive(self) -> bool:
        """Does this wave travel with the structure's direction?"""
        return self in (
            WaveLabel.ONE,
            WaveLabel.THREE,
            WaveLabel.FIVE,
            WaveLabel.A,
            WaveLabel.C,
        )


IMPULSE_LABELS = (
    WaveLabel.ONE,
    WaveLabel.TWO,
    WaveLabel.THREE,
    WaveLabel.FOUR,
    WaveLabel.FIVE,
)
CORRECTION_LABELS = (WaveLabel.A, WaveLabel.B, WaveLabel.C)


class CorrectiveShape(str, Enum):
    """Murphy's three classifications of corrective waves."""

    ZIGZAG = "zigzag"
    FLAT = "flat"
    TRIANGLE = "triangle"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Wave:
    """One leg of a wave structure, between two confirmed pivots."""

    label: WaveLabel
    start: SwingPoint
    end: SwingPoint

    @property
    def length(self) -> float:
        return abs(self.end.price - self.start.price)

    @property
    def bars(self) -> int:
        return max(0, self.end.index - self.start.index)

    @property
    def up(self) -> bool:
        return self.end.price > self.start.price

    @property
    def sign(self) -> int:
        return 1 if self.up else -1

    def retracement_of(self, other: "Wave") -> float:
        """This wave's length as a fraction of another's."""
        if other.length <= 1e-12:
            return 0.0
        return self.length / other.length


@dataclass(slots=True)
class WaveCount:
    """A labelled wave structure with its rule violations recorded."""

    kind: WaveKind
    waves: list[Wave]
    up: bool
    """Direction of the structure as a whole."""
    violations: list[str] = field(default_factory=list)
    shape: CorrectiveShape = CorrectiveShape.UNKNOWN
    detected_index: int = 0
    """Bar on which this count became knowable - the last pivot's confirmation."""
    notes: str = ""

    @property
    def valid(self) -> bool:
        return not self.violations

    @property
    def complete(self) -> bool:
        """Are all waves of the structure present?"""
        needed = 5 if self.kind is WaveKind.IMPULSE else 3
        return len(self.waves) >= needed

    @property
    def start_index(self) -> int:
        return self.waves[0].start.index if self.waves else 0

    @property
    def end_index(self) -> int:
        return self.waves[-1].end.index if self.waves else 0

    @property
    def current_wave(self) -> WaveLabel | None:
        return self.waves[-1].label if self.waves else None

    def wave(self, label: WaveLabel) -> Wave | None:
        for w in self.waves:
            if w.label is label:
                return w
        return None

    @property
    def confidence(self) -> float:
        """How much to trust this count, in [0, 1].

        Rule violations dominate, because Elliott's value is entirely in its
        constraints - a count that breaks them is not a weaker count, it is the
        wrong count. Completeness and wave-3 extension add to it, since Murphy
        notes wave 3 extending is the normal case in stocks.
        """
        if self.violations:
            return max(0.0, 0.35 - 0.15 * len(self.violations))
        score = 0.45
        if self.complete:
            score += 0.2
        w1, w3, w5 = (
            self.wave(WaveLabel.ONE),
            self.wave(WaveLabel.THREE),
            self.wave(WaveLabel.FIVE),
        )
        if w1 and w3 and w3.length > w1.length:
            # Wave 3 the longest is the textbook, and the most tradable, case.
            score += 0.15
            if w5 and w3.length > w5.length:
                score += 0.1
        return min(1.0, score)

    def describe(self) -> str:
        seq = "-".join(w.label.value for w in self.waves)
        direction = "up" if self.up else "down"
        state = "valid" if self.valid else f"{len(self.violations)} violation(s)"
        out = f"{self.kind.value} {direction} [{seq}] {state}"
        if self.shape is not CorrectiveShape.UNKNOWN:
            out += f" ({self.shape.value})"
        return out


def validate(count: WaveCount, strict_overlap: bool = False) -> list[str]:
    """Check a count against Elliott's hard rules. Returns violation descriptions.

    ``strict_overlap`` selects Murphy's stocks reading, where wave 4 overlapping
    wave 1 is "the unbreakable rule". Left off by default because this desk trades
    crypto around the clock, which behaves like his commodities case: "Intraday
    penetrations can occur."
    """
    problems: list[str] = []

    if count.kind is WaveKind.CORRECTION:
        # His most emphatic rule.
        if len(count.waves) > 3 and count.shape is not CorrectiveShape.TRIANGLE:
            problems.append(
                "correction has more than three waves, and corrections are "
                "threes, never fives"
            )
        return problems

    labels = [w.label for w in count.waves]
    if labels != list(IMPULSE_LABELS[: len(labels)]):
        problems.append("impulse waves are not in sequence")
        return problems

    w1 = count.wave(WaveLabel.ONE)
    w2 = count.wave(WaveLabel.TWO)
    w3 = count.wave(WaveLabel.THREE)
    w4 = count.wave(WaveLabel.FOUR)
    w5 = count.wave(WaveLabel.FIVE)

    if w1 and w2 and w2.retracement_of(w1) >= 1.0:
        problems.append("wave 2 retraced all of wave 1")

    if w1 and w3 and w5:
        if w3.length < w1.length and w3.length < w5.length:
            problems.append("wave 3 is the shortest of waves 1, 3 and 5")

    if w1 and w4:
        # Overlap: wave 4 entering wave 1's price territory.
        if count.up:
            overlap = w4.end.price < w1.end.price
        else:
            overlap = w4.end.price > w1.end.price
        if overlap:
            msg = "wave 4 overlaps wave 1"
            if strict_overlap:
                problems.append(msg)
            else:
                problems.append(msg + " (tolerated outside stocks, but a warning)")

    return problems


def project_wave3(count: WaveCount) -> float | None:
    """Minimum wave 3 objective.

    Murphy: "A minimum target for the top of wave 3 can be obtained by multiplying
    the length of wave 1 by 1.618 and adding that total to the bottom of 2."
    """
    w1 = count.wave(WaveLabel.ONE)
    w2 = count.wave(WaveLabel.TWO)
    if w1 is None or w2 is None:
        return None
    base = w2.end.price
    move = w1.length * PHI_INV
    return base + move if count.up else base - move


def project_wave5(count: WaveCount) -> tuple[float, float] | None:
    """Wave 5 objective range.

    Murphy: "The top of wave 5 can be approximated by multiplying wave 1 by 3.236
    ... and adding that value to the top or bottom of wave 1 for maximum and
    minimum targets." Two anchors, so two targets.
    """
    w1 = count.wave(WaveLabel.ONE)
    if w1 is None:
        return None
    move = w1.length * WAVE5_MULT
    lo_anchor = min(w1.start.price, w1.end.price)
    hi_anchor = max(w1.start.price, w1.end.price)
    if count.up:
        return lo_anchor + move, hi_anchor + move
    return hi_anchor - move, lo_anchor - move


def retracement_targets(wave: Wave) -> dict[str, float]:
    """Where a correction of ``wave`` would sit at each Fibonacci ratio."""
    origin = wave.end.price
    out: dict[str, float] = {}
    for ratio in RETRACEMENTS:
        move = wave.length * ratio
        out[f"{ratio:.3f}"] = origin - move if wave.up else origin + move
    return out


def classify_correction(waves: list[Wave]) -> CorrectiveShape:
    """Zigzag, flat or triangle - Murphy's three corrective classifications.

    A zigzag is a sharp 5-3-5 that makes real progress against the trend; a flat is
    a shallow sideways 3-3-5 where wave b returns near the start of a; a triangle
    keeps contracting. The discriminator used here is how far wave b retraces wave
    a, which is the practical difference a chartist sees.
    """
    if len(waves) < 2:
        return CorrectiveShape.UNKNOWN
    a, b = waves[0], waves[1]
    if a.length <= 1e-12:
        return CorrectiveShape.UNKNOWN
    b_ratio = b.retracement_of(a)
    if len(waves) >= 5:
        return CorrectiveShape.TRIANGLE
    if b_ratio >= 0.8:
        # Wave b nearly erases wave a: sideways, not corrective progress.
        return CorrectiveShape.FLAT
    if b_ratio <= 0.6:
        return CorrectiveShape.ZIGZAG
    return CorrectiveShape.UNKNOWN
