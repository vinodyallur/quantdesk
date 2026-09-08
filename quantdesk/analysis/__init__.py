"""Murphy's analytical framework: structure, levels, patterns.

The layering follows the book, and each layer only depends on the ones above it:

1. :mod:`swings` - peaks and troughs. Everything rests on these.
2. :mod:`dow` - trend classified from the direction of those pivots.
3. :mod:`levels` - support and resistance built from pivots, with role reversal.
4. :mod:`trendlines` - lines and channels through pivots, plus the fan principle.
5. :mod:`retracement` - percentage retracements and speed lines.
6. :mod:`gaps` - gaps, island reversals and reversal bars.
7. :mod:`patterns` - the chart formations, which combine all of the above.

Imports are kept shallow here: the pattern package is deliberately *not* pulled in
at import time, since it drags in the whole detector set. Import it directly when
it is needed.
"""

from quantdesk.analysis.dow import (
    TrendDegree,
    TrendDirection,
    TrendHealth,
    TrendState,
    classify_degree,
    classify_trend,
)
from quantdesk.analysis.gaps import (
    Gap,
    GapTracker,
    GapType,
    ReversalBar,
    ReversalKind,
    detect_reversal_bar,
)
from quantdesk.analysis.levels import (
    Level,
    LevelBook,
    LevelRole,
    nearest_round_number,
    offset_stop_from_round,
    round_number_levels,
)
from quantdesk.analysis.retracement import (
    RetracementLevel,
    RetracementMap,
    RetracementZone,
    SpeedLines,
    build_retracements,
    build_speedlines,
    classify_retracement,
    fibonacci_projection,
    in_entry_zone,
)
from quantdesk.analysis.swings import (
    SwingDetector,
    SwingKind,
    SwingLeg,
    SwingPoint,
    TentativeSwing,
    legs,
)
from quantdesk.analysis.trendlines import (
    Channel,
    LineKind,
    Trendline,
    TrendlineTracker,
    build_channel,
)

__all__ = [
    "Channel",
    "Gap",
    "GapTracker",
    "GapType",
    "Level",
    "LevelBook",
    "LevelRole",
    "LineKind",
    "RetracementLevel",
    "RetracementMap",
    "RetracementZone",
    "ReversalBar",
    "ReversalKind",
    "SpeedLines",
    "SwingDetector",
    "SwingKind",
    "SwingLeg",
    "SwingPoint",
    "TentativeSwing",
    "TrendDegree",
    "TrendDirection",
    "TrendHealth",
    "TrendState",
    "Trendline",
    "TrendlineTracker",
    "build_channel",
    "build_retracements",
    "build_speedlines",
    "classify_degree",
    "classify_retracement",
    "classify_trend",
    "detect_reversal_bar",
    "fibonacci_projection",
    "in_entry_zone",
    "legs",
    "nearest_round_number",
    "offset_stop_from_round",
    "round_number_levels",
]
