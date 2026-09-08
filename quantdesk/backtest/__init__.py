"""Event-driven backtesting.

The backtester runs the live :class:`~quantdesk.pipeline.Pipeline` against
:class:`~quantdesk.execution.SimBroker`, so there is only one implementation of the
strategy and no opportunity for backtest and live to drift apart::

    from quantdesk.backtest import Backtester

    result = Backtester().run(bars)
    for line in result.summary():
        print(line)
"""

from quantdesk.backtest.account import Account
from quantdesk.backtest.engine import Backtester, TradeRecord
from quantdesk.backtest.metrics import BacktestResult, EquityPoint, compute_metrics

__all__ = [
    "Account",
    "BacktestResult",
    "Backtester",
    "EquityPoint",
    "TradeRecord",
    "compute_metrics",
]
