"""Trading psychology as machine-checkable rules, from Douglas's *Trading in the Zone*.

Where Murphy's material tells the desk *what* to trade, this tells it whether it
actually followed its own rules while doing so. Nothing here generates signals or
vetoes trades - it observes and reports::

    monitor = DisciplineMonitor()
    monitor.on_entry("BTC/USD", stop=94_000, side=1, agent="murphy_patterns")
    monitor.on_exit("BTC/USD", r_multiple=2.4)
    for line in monitor.summary():
        print(line)

The most useful output is usually :meth:`DisciplineMonitor.streak_verdict`, which
answers "is this losing run actually abnormal?" - almost always no, and knowing that
is what stops a working edge being abandoned after eight losses.
"""

from quantdesk.psychology.discipline import DisciplineMonitor, Intervention
from quantdesk.psychology.edge import (
    EdgeStats,
    TradeOutcome,
    expected_worst_streak,
    is_streak_alarming,
    measure,
    streak_probability,
)
from quantdesk.psychology.principles import (
    FUNDAMENTAL_TRUTHS,
    AdherenceReport,
    Principle,
    PrincipleEvent,
    Severity,
    principles_text,
)

__all__ = [
    "AdherenceReport",
    "DisciplineMonitor",
    "EdgeStats",
    "FUNDAMENTAL_TRUTHS",
    "Intervention",
    "Principle",
    "PrincipleEvent",
    "Severity",
    "TradeOutcome",
    "expected_worst_streak",
    "is_streak_alarming",
    "measure",
    "principles_text",
    "streak_probability",
]
