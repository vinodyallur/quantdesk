"""Event-driven backtester.

It runs the real :class:`~quantdesk.pipeline.Pipeline` against a real
:class:`~quantdesk.execution.SimBroker`. There is no separate simulation strategy
code, because the moment there are two implementations they drift and the backtest
stops describing the thing you would actually run.

The three ways a backtest lies, and what stops each here:

**Same-bar fills.** Deciding on a bar's close and filling at that same close means
trading on information you did not have when the decision was made. ``SimBroker``
fills at the *next* bar's open, so every decision is separated from its fill by a bar.

**Survivorship in the loop.** Bars from several symbols are merged and replayed in
strict timestamp order, so no symbol ever sees a bar from the future while another is
still in the past. Feeding symbols sequentially instead would let the cross-sectional
agents peek.

**Costless execution.** Commission and slippage come from configuration and are
charged on every fill. A strategy that only works at zero cost is worth knowing about
before it trades.

Equity is marked on every bar rather than only on fills, so the drawdown figures
reflect what the account actually went through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from quantdesk.backtest.account import Account
from quantdesk.backtest.metrics import BacktestResult, EquityPoint, compute_metrics
from quantdesk.config import Settings, get_settings
from quantdesk.core.clock import SimClock
from quantdesk.core.types import Bar, Fill, Order, Position
from quantdesk.execution.sim_broker import SimBroker
from quantdesk.pipeline import BarDecision, Pipeline

log = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """A round trip, from first entry to flat, for attribution."""

    symbol: str
    opened_at: datetime
    closed_at: datetime | None = None
    side: int = 0
    entry: float = 0.0
    exit: float = 0.0
    qty: float = 0.0
    pnl: float = 0.0
    fees: float = 0.0
    bars_held: int = 0
    reason: str = ""
    initial_risk: float = 0.0
    """Currency at risk at entry, i.e. distance to the stop times size. Needed for R.

    Accumulated from the ``planned_risk`` the orders carried, so pyramid layers and
    partial fills each contribute their own share. Never reconstructed from a fill.
    """
    peak_qty: float = 0.0
    """Largest quantity this round trip ever held.

    ``qty`` is the *live* remaining quantity and is driven to zero as the position is
    closed, so by the time a trade is recorded it says nothing about how big the trade
    was. Attribution needs the size actually carried, so it is kept separately.
    """

    @property
    def won(self) -> bool:
        return self.pnl > 0

    @property
    def r_multiple(self) -> float:
        """Result in units of the risk originally taken.

        Returns 0.0 when the initial risk is unknown rather than dividing by something
        near zero: a trade that opened and closed at almost the same price has a tiny
        price difference, and using that as the denominator produces absurd R values.
        The denominator must be the *planned* risk, not the realised move.
        """
        if self.initial_risk <= 1e-9:
            return 0.0
        return self.pnl / self.initial_risk

    @property
    def risk_known(self) -> bool:
        """Whether R means anything for this trade.

        A trade whose planned risk never arrived reports 0.0R, which is indistinguishable
        from a genuine scratch. Averaging those in silently pulls expectancy toward
        zero, so they are excluded from R statistics rather than counted as flat.
        """
        return self.initial_risk > 1e-9

    @property
    def is_open(self) -> bool:
        return self.closed_at is None


class Backtester:
    """Replays historical bars through the live decision pipeline."""

    def __init__(
        self,
        settings: Settings | None = None,
        pipeline: Pipeline | None = None,
        broker: SimBroker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.pipeline = pipeline or Pipeline(self.settings)
        self.broker = broker or SimBroker(
            commission_bps=self.settings.commission_bps,
            slippage_bps=self.settings.slippage_bps,
        )
        # The broker matches orders; the account keeps cash and equity. Two objects
        # because in live trading the second comes from the broker's own books.
        self.account = Account(starting_equity=self.settings.starting_equity)
        self.clock = SimClock()
        self.equity_curve: list[EquityPoint] = []
        self.trades: list[TradeRecord] = []
        self.fills: list[Fill] = []
        self.decisions: list[BarDecision] = []
        self._open_trades: dict[str, TradeRecord] = {}
        # Order id -> planned risk *per unit of quantity*. Storing it per unit rather
        # than per order is what makes partial fills add up correctly: each fill takes
        # its own share and the shares sum to the order's total planned risk.
        self._order_risk: dict[str, float] = {}
        self._bar_index = 0

    # ------------------------------------------------------------------- run
    def run(self, bars: list[Bar] | dict[str, list[Bar]]) -> BacktestResult:
        """Replay bars and return the result.

        Accepts a flat list or a per-symbol mapping. Either way the bars are merged
        and sorted by timestamp so the whole desk advances together.
        """
        stream = _merge(bars)
        if not stream:
            raise ValueError("no bars to backtest")

        log.info(
            "backtest: %d bars, %d symbols, %s to %s",
            len(stream),
            len({b.symbol for b in stream}),
            stream[0].ts,
            stream[-1].ts,
        )

        for bar in stream:
            self._bar_index += 1
            self.clock.set(bar.ts)

            # Fills first. An order decided on the previous bar fills at this bar's
            # open, before this bar's decision is made - which is the correct
            # sequence and the reason the simulator cannot look ahead.
            new_fills = self.broker.on_bar(bar)
            if new_fills:
                self._apply_fills(new_fills, bar)

            self.account.mark(bar.symbol, bar.close)
            positions = self.account.live_positions()
            equity = self.account.equity
            self.pipeline.sync_account(equity, positions)

            decision = self.pipeline.on_bar(bar)
            self.decisions.append(decision)

            for order in decision.orders:
                self.broker.submit_sync(order)
                # After submit, because the broker is what assigns an id to an order
                # that arrived without one.
                if order.id and order.qty > 1e-12:
                    self._order_risk[order.id] = order.planned_risk / order.qty

            self._record_equity(bar, equity, positions)

        # Anything still open at the end is marked out at the last price, so the
        # result is not flattered by unrealised losses that were never closed.
        self._close_open_trades(stream[-1])
        return compute_metrics(
            equity_curve=self.equity_curve,
            trades=self.trades,
            fills=self.fills,
            decisions=self.decisions,
            bars_per_year=self.settings.bars_per_year,
            starting_equity=self.settings.starting_equity,
        )

    # ----------------------------------------------------------------- fills
    def _apply_fills(self, fills: list[Fill], bar: Bar) -> None:
        for fill in fills:
            self.account.apply(fill)
            self.fills.append(fill)
            self._track_trade(fill, bar)

    def _track_trade(self, fill: Fill, bar: Bar) -> None:
        """Fold a fill into the open round trip, opening or closing as needed."""
        existing = self._open_trades.get(fill.symbol)
        signed = fill.signed_qty
        # Risk came with the order that produced this fill. Taking this fill's share of
        # it keeps partial fills and pyramid layers correct, and never depends on what
        # the trade manager happens to be holding at fill time.
        risk_per_qty = self._order_risk.get(fill.order_id, 0.0)
        fill_risk = risk_per_qty * fill.qty

        if existing is None:
            self._open_trades[fill.symbol] = TradeRecord(
                symbol=fill.symbol,
                opened_at=fill.ts,
                side=1 if signed > 0 else -1,
                entry=fill.price,
                qty=abs(signed),
                peak_qty=abs(signed),
                fees=fill.commission,
                reason=fill.tag,
                initial_risk=fill_risk,
            )
            return

        same_way = (existing.side > 0) == (signed > 0)
        existing.fees += fill.commission
        if same_way:
            # Adding: weighted average entry, and the added risk accumulates. A pyramid
            # layer genuinely puts more currency at risk, so R must be measured against
            # the total rather than against the first layer alone.
            total = existing.qty + abs(signed)
            existing.entry = (
                existing.entry * existing.qty + fill.price * abs(signed)
            ) / total
            existing.qty = total
            existing.peak_qty = max(existing.peak_qty, total)
            existing.initial_risk += fill_risk
            return

        closing = min(existing.qty, abs(signed))
        existing.pnl += (fill.price - existing.entry) * closing * existing.side
        existing.qty -= closing
        if existing.qty <= 1e-12:
            existing.exit = fill.price
            existing.closed_at = fill.ts
            existing.bars_held = self._bar_index
            existing.pnl -= existing.fees
            self.trades.append(existing)
            del self._open_trades[fill.symbol]
            # A flip leaves residual quantity that starts a new trade. Only the residual
            # puts new risk on - the part that closed the old position carried none - so
            # the fill's risk is apportioned to the residual alone.
            residual = abs(signed) - closing
            if residual > 1e-12:
                self._open_trades[fill.symbol] = TradeRecord(
                    symbol=fill.symbol,
                    opened_at=fill.ts,
                    side=1 if signed > 0 else -1,
                    entry=fill.price,
                    qty=residual,
                    peak_qty=residual,
                    reason=fill.tag,
                    initial_risk=risk_per_qty * residual,
                )

    def _close_open_trades(self, last_bar: Bar) -> None:
        for trade in self._open_trades.values():
            price = self.account.last_price(trade.symbol) or last_bar.close
            trade.exit = price
            trade.closed_at = last_bar.ts
            trade.pnl += (price - trade.entry) * trade.qty * trade.side - trade.fees
            trade.reason = (trade.reason + " | marked out at end of test").strip(" |")
            self.trades.append(trade)
        self._open_trades.clear()

    # ---------------------------------------------------------------- equity
    def _record_equity(
        self, bar: Bar, equity: float, positions: dict[str, Position]
    ) -> None:
        gross = sum(abs(p.market_value(p.last_price)) for p in positions.values())
        net = sum(p.market_value(p.last_price) for p in positions.values())
        self.equity_curve.append(
            EquityPoint(
                ts=bar.ts,
                equity=equity,
                gross_exposure=gross,
                net_exposure=net,
                open_positions=sum(1 for p in positions.values() if not p.is_flat),
            )
        )


def _merge(bars: list[Bar] | dict[str, list[Bar]]) -> list[Bar]:
    """Flatten and sort bars into one chronological stream.

    Sorting by timestamp is what keeps the cross-sectional agents honest: every symbol
    is at the same moment when any decision is made. The symbol tiebreak keeps the
    order deterministic when several bars share a timestamp, so runs are reproducible.
    """
    if isinstance(bars, dict):
        flat: list[Bar] = []
        for series in bars.values():
            flat.extend(series)
    else:
        flat = list(bars)
    flat.sort(key=lambda b: (b.ts, b.symbol))
    return flat
