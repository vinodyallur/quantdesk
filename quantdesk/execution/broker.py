"""Broker interface.

Both the simulator and the Alpaca paper adapter implement this, which is what
lets the backtester and the live desk run identical strategy code and differ only
in where orders land.

Fills are pulled rather than pushed (``drain_fills``). A push callback would mean
the portfolio could be mutated from a network thread mid-decision; pulling keeps
all state changes on the desk's own loop at a well-defined point in the cycle.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod

from quantdesk.core.types import AccountSnapshot, Fill, Order, Position

_order_seq = itertools.count(1)


def next_order_id(prefix: str = "qd") -> str:
    return f"{prefix}-{next(_order_seq):06d}"


class Broker(ABC):
    """Somewhere to send orders and get fills back."""

    #: True when this broker cannot touch real money under any configuration.
    is_simulated: bool = False

    @abstractmethod
    async def submit(self, order: Order) -> Order:
        """Send an order. Returns the order with status/ids populated."""

    @abstractmethod
    async def cancel_all(self) -> int:
        """Cancel every working order. Returns how many were cancelled."""

    @abstractmethod
    def drain_fills(self) -> list[Fill]:
        """Return fills seen since the last call and clear the buffer."""

    async def sync(self) -> None:
        """Reconcile local state with the broker. No-op for the simulator."""
        return None

    async def positions(self) -> dict[str, Position]:
        return {}

    async def account(self) -> AccountSnapshot | None:
        return None

    @property
    def name(self) -> str:
        return type(self).__name__

    async def close(self) -> None:
        return None
