"""Fill simulator for backtests.

The single most important thing this class does is refuse to fill an order on the
same bar that generated it. Orders submitted while processing bar T are filled at
the **open of bar T+1**.

That ordering is not a detail. If you fill at bar T's close, the strategy has
already seen that close when it decided to trade, so the backtest is quietly
trading on information it could not have had. It is the most common way a
backtest ends up looking profitable and the live version does not, and because
the resulting equity curve looks entirely plausible it is very hard to spot after
the fact.

Costs modelled:

* **Commission** in basis points of notional.
* **Slippage** in basis points, always against you: buys fill above the open,
  sells below it.

Costs not modelled, which will flatter results: market impact beyond fixed
slippage, spread widening in stress, borrow costs on shorts, and the fact that a
real venue may not have liquidity at your size.
"""

from __future__ import annotations

import logging

from quantdesk.core.types import (
    Bar,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Side,
)
from quantdesk.execution.broker import Broker, next_order_id

log = logging.getLogger(__name__)


class SimBroker(Broker):
    """Deterministic next-bar-open fill simulator."""

    is_simulated = True

    def __init__(
        self,
        commission_bps: float = 1.0,
        slippage_bps: float = 2.0,
        max_participation: float | None = None,
    ) -> None:
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        # Cap fills at a fraction of the bar's volume. Disabled by default: Alpaca's
        # crypto bar volumes are frequently near zero (single-trade 5-minute bars),
        # so enabling this against that data would block essentially every fill and
        # look like a strategy problem rather than a data problem. Turn it on for
        # equities, where volume is trustworthy.
        self.max_participation = max_participation

        self._pending: dict[str, list[Order]] = {}
        self._fills: list[Fill] = []
        self.orders: list[Order] = []
        self.rejected = 0

    # ------------------------------------------------------------------ submit
    async def submit(self, order: Order) -> Order:
        if not order.id:
            order.id = next_order_id("sim")
        if order.qty <= 0:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "non-positive quantity"
            self.rejected += 1
            return order
        order.status = OrderStatus.SUBMITTED
        self._pending.setdefault(order.symbol, []).append(order)
        self.orders.append(order)
        return order

    def submit_sync(self, order: Order) -> Order:
        """Synchronous submit, for the backtest loop which has no event loop."""
        if not order.id:
            order.id = next_order_id("sim")
        if order.qty <= 0:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "non-positive quantity"
            self.rejected += 1
            return order
        order.status = OrderStatus.SUBMITTED
        self._pending.setdefault(order.symbol, []).append(order)
        self.orders.append(order)
        return order

    async def cancel_all(self) -> int:
        n = 0
        for orders in self._pending.values():
            for o in orders:
                o.status = OrderStatus.CANCELED
                n += 1
        self._pending.clear()
        return n

    def cancel_symbol(self, symbol: str) -> int:
        """Drop working orders for one symbol, used before re-targeting it."""
        orders = self._pending.pop(symbol, [])
        for o in orders:
            o.status = OrderStatus.CANCELED
        return len(orders)

    # -------------------------------------------------------------------- fill
    def on_bar(self, bar: Bar) -> list[Fill]:
        """Fill orders queued before this bar, at this bar's open."""
        pending = self._pending.pop(bar.symbol, [])
        if not pending:
            return []

        fills: list[Fill] = []
        volume_left = (
            bar.volume * self.max_participation
            if self.max_participation is not None
            else float("inf")
        )

        for order in pending:
            fill_qty = order.remaining
            if fill_qty <= 0:
                continue

            if volume_left <= 0:
                # No liquidity left this bar: requeue for the next one.
                self._pending.setdefault(bar.symbol, []).append(order)
                continue

            if fill_qty > volume_left:
                fill_qty = volume_left
            volume_left -= fill_qty

            price = self._fill_price(order, bar)
            if price <= 0:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "no valid fill price"
                self.rejected += 1
                continue

            # Limit orders only fill if the bar actually traded through the limit.
            if order.type is OrderType.LIMIT and order.limit_price is not None:
                if order.side is Side.BUY and bar.low > order.limit_price:
                    self._pending.setdefault(bar.symbol, []).append(order)
                    continue
                if order.side is Side.SELL and bar.high < order.limit_price:
                    self._pending.setdefault(bar.symbol, []).append(order)
                    continue

            commission = fill_qty * price * self.commission_bps / 10_000.0
            fill = Fill(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                qty=fill_qty,
                price=price,
                ts=bar.ts,
                commission=commission,
                slippage_bps=self.slippage_bps,
                tag=order.tag,
            )

            # Update the order's running average fill price.
            total = order.filled_qty + fill_qty
            order.avg_fill_price = (
                (order.avg_fill_price * order.filled_qty + price * fill_qty) / total
            )
            order.filled_qty = total
            order.status = (
                OrderStatus.FILLED if order.remaining <= 1e-12 else OrderStatus.PARTIAL
            )
            if order.status is OrderStatus.PARTIAL:
                self._pending.setdefault(bar.symbol, []).append(order)

            fills.append(fill)
            self._fills.append(fill)

        return fills

    def _fill_price(self, order: Order, bar: Bar) -> float:
        """Next bar's open, moved against us by the slippage assumption."""
        base = bar.open if bar.open > 0 else bar.close
        if base <= 0:
            return 0.0
        slip = base * self.slippage_bps / 10_000.0
        if order.type is OrderType.LIMIT and order.limit_price is not None:
            # Fill at the better of the limit and the slipped market price.
            market = base + slip if order.side is Side.BUY else base - slip
            return (
                min(market, order.limit_price)
                if order.side is Side.BUY
                else max(market, order.limit_price)
            )
        return base + slip if order.side is Side.BUY else base - slip

    # ------------------------------------------------------------------ drains
    def drain_fills(self) -> list[Fill]:
        out = self._fills
        self._fills = []
        return out

    @property
    def pending_count(self) -> int:
        return sum(len(v) for v in self._pending.values())

    def reset(self) -> None:
        self._pending.clear()
        self._fills.clear()
        self.orders.clear()
        self.rejected = 0
