"""Market data feed interface.

Two capabilities, deliberately separated:

* ``history()`` - pull a closed window of bars. Used to warm up indicators and to
  drive backtests.
* ``stream()`` - async generator of bars as they arrive. Used by the live desk.

Any feed satisfying this interface can drive the desk, which is what makes the
offline synthetic feed and the real Alpaca feed interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator

from quantdesk.core.types import Bar


class DataFeed(ABC):
    """Source of market data for a fixed universe and timeframe."""

    def __init__(self, symbols: list[str], timeframe: str) -> None:
        self.symbols = symbols
        self.timeframe = timeframe

    @abstractmethod
    def history(
        self,
        start: datetime,
        end: datetime | None = None,
        symbols: list[str] | None = None,
    ) -> dict[str, list[Bar]]:
        """Closed bars per symbol, ascending by time."""

    @abstractmethod
    async def stream(self) -> AsyncIterator[Bar]:
        """Yield bars as they become available. Runs until cancelled."""
        raise NotImplementedError
        yield  # pragma: no cover - makes this an async generator for typing

    @property
    def name(self) -> str:
        return type(self).__name__

    async def close(self) -> None:
        """Release any network resources. Safe to call more than once."""
        return None
