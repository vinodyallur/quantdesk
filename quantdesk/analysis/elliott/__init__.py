"""Elliott Wave Theory, Murphy chapter 13.

Two modules: :mod:`waves` holds the structure, the hard rules and the Fibonacci
ratios; :mod:`counter` searches confirmed pivots for candidate counts.

Wave counting is the most subjective technique in the book, so the API returns
*ranked candidates with their rule violations attached* rather than one confident
answer::

    counter = ElliottCounter("BTC/USD")
    counter.update(bar_index, swing_detector.confirmed)
    print(counter.position())        # "in wave 3 of an up impulse"
    print(counter.bias())            # Bias.BULLISH
    print(counter.targets())         # Fibonacci projections
    for line in counter.summary():   # every candidate, with violations
        print(line)
"""

from quantdesk.analysis.elliott.counter import ElliottCounter
from quantdesk.analysis.elliott.waves import (
    CORRECTION_LABELS,
    IMPULSE_LABELS,
    PHI,
    PHI_INV,
    RETRACEMENTS,
    WAVE5_MULT,
    CorrectiveShape,
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

__all__ = [
    "CORRECTION_LABELS",
    "CorrectiveShape",
    "ElliottCounter",
    "IMPULSE_LABELS",
    "PHI",
    "PHI_INV",
    "RETRACEMENTS",
    "WAVE5_MULT",
    "Wave",
    "WaveCount",
    "WaveKind",
    "WaveLabel",
    "classify_correction",
    "project_wave3",
    "project_wave5",
    "retracement_targets",
    "validate",
]
