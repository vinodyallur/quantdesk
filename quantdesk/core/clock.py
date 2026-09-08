"""Clock abstraction.

The single most important property of this codebase is that backtest and live
trading run the *same* strategy code. Time is the main thing that differs, so it
is injected. Strategy and risk code must ask the clock for "now" and never call
``datetime.now`` directly, otherwise a backtest would silently leak future
information or stamp records with wall-clock time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from quantdesk.core.types import utcnow


class Clock(ABC):
    @abstractmethod
    def now(self) -> datetime:
        """Current time, timezone-aware UTC."""

    @property
    def is_simulated(self) -> bool:
        return False


class LiveClock(Clock):
    """Wall clock, for live/paper trading."""

    def now(self) -> datetime:
        return utcnow()


class SimClock(Clock):
    """Clock driven by replayed market data, for backtests."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or utcnow()

    def now(self) -> datetime:
        return self._now

    def set(self, ts: datetime) -> None:
        """Advance simulated time. Monotonic: never steps backwards."""
        if ts > self._now:
            self._now = ts

    @property
    def is_simulated(self) -> bool:
        return True
