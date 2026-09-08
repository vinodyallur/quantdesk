"""Chart patterns from Murphy chapters 5 and 6.

Start with :class:`PatternScanner` - it wires every detector to one bar stream::

    scanner = PatternScanner("BTC/USD")
    for bar in bars:
        for pattern in scanner.update(bar):
            print(pattern.describe())

    tradable = scanner.actionable(min_significance=0.4, min_reward_risk=3.0)

Individual detectors are exported for the cases where only part of the set is
wanted, and the shared pieces - :class:`Pattern`, :class:`PenetrationFilter`, the
:mod:`geometry` helpers - are exported because the Elliott and candlestick layers
build on the same foundations.
"""

from quantdesk.analysis.patterns.base import (
    Bias,
    DetectorContext,
    Pattern,
    PatternDetector,
    PatternFamily,
    PatternKind,
    PatternStage,
    PenetrationFilter,
    VolumeContext,
    overlaps,
)
from quantdesk.analysis.patterns.double_triple import DoubleTripleDetector
from quantdesk.analysis.patterns.flags import FlagPennantDetector
from quantdesk.analysis.patterns.geometry import (
    APEX_WINDOW_END,
    APEX_WINDOW_START,
    Line,
    apex_progress,
    best_fit_line,
    converging,
    diverging,
    fit_error,
    intersection,
    line_through,
)
from quantdesk.analysis.patterns.head_shoulders import HeadShouldersDetector
from quantdesk.analysis.patterns.lifecycle import (
    advance,
    objective_from_boundary,
    objective_met,
    volume_confirms,
)
from quantdesk.analysis.patterns.rectangles import (
    MeasuredMoveDetector,
    RectangleDetector,
)
from quantdesk.analysis.patterns.saucer_spike import RoundingDetector, SpikeDetector
from quantdesk.analysis.patterns.scanner import PatternScanner, default_detectors
from quantdesk.analysis.patterns.triangles import TriangleDetector

__all__ = [
    "APEX_WINDOW_END",
    "APEX_WINDOW_START",
    "Bias",
    "DetectorContext",
    "DoubleTripleDetector",
    "FlagPennantDetector",
    "HeadShouldersDetector",
    "Line",
    "MeasuredMoveDetector",
    "Pattern",
    "PatternDetector",
    "PatternFamily",
    "PatternKind",
    "PatternScanner",
    "PatternStage",
    "PenetrationFilter",
    "RectangleDetector",
    "RoundingDetector",
    "SpikeDetector",
    "TriangleDetector",
    "VolumeContext",
    "advance",
    "overlaps",
    "apex_progress",
    "best_fit_line",
    "converging",
    "default_detectors",
    "diverging",
    "fit_error",
    "intersection",
    "line_through",
    "objective_from_boundary",
    "objective_met",
    "volume_confirms",
]
