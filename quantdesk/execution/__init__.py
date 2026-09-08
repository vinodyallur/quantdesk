"""Order routing and broker adapters.

:class:`SimBroker` fills on the next bar's open, so a backtest cannot buy at a price
it only learned about after the fact. :class:`AlpacaPaperBroker` refuses to construct
against a live endpoint. :class:`OrderRouter` sits in front of either and enforces one
working order per symbol, which is what stops a persistent signal from stacking
duplicate positions.

The Alpaca adapter is imported lazily: market data works without any API key, so the
package must import cleanly on a machine with no credentials and no ``alpaca-py``.
"""

from quantdesk.execution.broker import Broker, next_order_id
from quantdesk.execution.router import OrderRouter, RouteResult
from quantdesk.execution.sim_broker import SimBroker

__all__ = [
    "AlpacaBrokerError",
    "AlpacaPaperBroker",
    "Broker",
    "OrderRouter",
    "RouteResult",
    "SimBroker",
    "next_order_id",
]


def __getattr__(name: str):
    """Import the Alpaca adapter only when it is actually asked for."""
    if name in ("AlpacaPaperBroker", "AlpacaBrokerError"):
        from quantdesk.execution import alpaca_broker

        return getattr(alpaca_broker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
