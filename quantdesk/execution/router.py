"""Order router: the last gate between a decision and the market.

The pipeline has already sized and risk-checked, so why another layer? Because the
checks that matter here are about *the act of sending*, not about whether the trade
is a good idea:

* **One in flight per symbol.** Without this, a signal that persists across several
  bars sends the same order repeatedly while the first is still working, and the desk
  quietly ends up at three times the intended size. This is the single most common way
  an automated desk blows through its own position limits.
* **Rejections must not loop.** A symbol whose orders keep failing is put in
  timeout rather than retried forever, which stops one bad instrument from consuming
  the whole loop.
* **Closing is always allowed.** When risk halts the desk, reducing exposure must
  still get through. A gate that blocks exits as well as entries turns a drawdown
  into a trap.
* **Kill switch.** One call cancels everything working and, optionally, flattens.

The router keeps its own record of what it sent, but treats the broker as the
authority on what actually exists, reconciling on every cycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from quantdesk.core.types import Fill, Order, OrderStatus, Position, utcnow
from quantdesk.execution.broker import Broker

log = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """What happened to a batch of orders."""

    sent: list[Order] = field(default_factory=list)
    blocked: list[tuple[Order, str]] = field(default_factory=list)

    @property
    def any_sent(self) -> bool:
        return bool(self.sent)

    def summary(self) -> str:
        parts = [f"{len(self.sent)} sent"]
        if self.blocked:
            parts.append(f"{len(self.blocked)} blocked")
        return ", ".join(parts)


class OrderRouter:
    """Sends orders to a broker, one working order per symbol."""

    def __init__(
        self,
        broker: Broker,
        max_rejects: int = 3,
        timeout_seconds: float = 300.0,
        stale_after_seconds: float = 120.0,
    ) -> None:
        self.broker = broker
        self.max_rejects = max_rejects
        self.timeout_seconds = timeout_seconds
        #: A working order older than this is assumed dead rather than blocking the
        #: symbol forever, which is what happens if a fill notification is missed.
        self.stale_after_seconds = stale_after_seconds
        self._working: dict[str, Order] = {}
        self._rejects: dict[str, int] = {}
        self._timeout_until: dict[str, datetime] = {}
        self.history: list[Order] = []

    # ------------------------------------------------------------------ send
    async def route(
        self,
        orders: list[Order],
        positions: dict[str, Position] | None = None,
        allow_opening: bool = True,
        now: datetime | None = None,
    ) -> RouteResult:
        """Send what may be sent, and report what was blocked and why."""
        result = RouteResult()
        now = now or utcnow()
        positions = positions or {}
        self._expire_stale(now)

        for order in orders:
            reason = self._blocked_reason(order, positions, allow_opening, now)
            if reason is not None:
                result.blocked.append((order, reason))
                continue

            placed = await self.broker.submit(order)
            self.history.append(placed)
            if len(self.history) > 2000:
                del self.history[:-2000]

            if placed.status is OrderStatus.REJECTED:
                count = self._rejects.get(order.symbol, 0) + 1
                self._rejects[order.symbol] = count
                if count >= self.max_rejects:
                    self._timeout_until[order.symbol] = now + timedelta(
                        seconds=self.timeout_seconds
                    )
                    log.warning(
                        "%s in timeout after %d rejections: %s",
                        order.symbol,
                        count,
                        placed.reject_reason,
                    )
                result.blocked.append((placed, f"rejected: {placed.reject_reason}"))
                continue

            self._rejects.pop(order.symbol, None)
            if not placed.status.is_terminal:
                self._working[order.symbol] = placed
            result.sent.append(placed)
        return result

    def _blocked_reason(
        self,
        order: Order,
        positions: dict[str, Position],
        allow_opening: bool,
        now: datetime,
    ) -> str | None:
        if order.qty <= 0:
            return "zero quantity"

        until = self._timeout_until.get(order.symbol)
        if until is not None:
            if now < until:
                return f"symbol in timeout until {until:%H:%M:%S}"
            del self._timeout_until[order.symbol]
            self._rejects.pop(order.symbol, None)

        if order.symbol in self._working:
            return "an order is already working for this symbol"

        # Reducing exposure is never blocked. When the desk is halted this is the
        # only thing that still needs to get out.
        if not allow_opening and not self._reduces(order, positions):
            return "opening new risk is currently disallowed"
        return None

    def _reduces(self, order: Order, positions: dict[str, Position]) -> bool:
        """Would this order move the position closer to flat?"""
        pos = positions.get(order.symbol)
        if pos is None or pos.is_flat:
            return False
        # Opposite sign to the position, and no larger than it.
        return pos.qty * order.signed_qty < 0 and abs(order.signed_qty) <= abs(pos.qty) + 1e-9

    def _expire_stale(self, now: datetime) -> None:
        """Release symbols whose working order has gone quiet.

        Without this a single missed fill notification blocks a symbol permanently -
        the desk goes silent on that instrument and it looks like the strategy simply
        stopped having views.
        """
        for symbol, order in list(self._working.items()):
            age = (now - order.ts).total_seconds()
            if age > self.stale_after_seconds:
                log.info("releasing %s: working order %s went stale", symbol, order.id)
                del self._working[symbol]

    # ------------------------------------------------------------ reconciliation
    def on_fills(self, fills: list[Fill]) -> None:
        """Clear working orders that have been filled."""
        for fill in fills:
            order = self._working.get(fill.symbol)
            if order is None:
                continue
            order.filled_qty += fill.qty
            order.avg_fill_price = fill.price
            if order.filled_qty >= order.qty - 1e-9:
                order.status = OrderStatus.FILLED
                del self._working[fill.symbol]
            else:
                order.status = OrderStatus.PARTIAL

    async def reconcile(self) -> dict[str, Position]:
        """Pull broker truth and drop working orders the broker no longer has."""
        await self.broker.sync()
        fills = self.broker.drain_fills()
        if fills:
            self.on_fills(fills)
        return await self.broker.positions()

    # ------------------------------------------------------------- kill switch
    async def kill(self, flatten: bool = False) -> str:
        """Cancel everything working, and optionally close all positions."""
        cancelled = await self.broker.cancel_all()
        self._working.clear()
        detail = f"cancelled {cancelled} working order(s)"
        if flatten:
            flattener = getattr(self.broker, "flatten", None)
            if flattener is not None:
                await flattener()
                detail += " and flattened all positions"
            else:
                detail += " (broker cannot flatten)"
        log.warning("kill switch: %s", detail)
        return detail

    # -------------------------------------------------------------- accessors
    @property
    def working(self) -> dict[str, Order]:
        return dict(self._working)

    @property
    def blocked_symbols(self) -> list[str]:
        return sorted(self._timeout_until)

    def status(self) -> dict[str, object]:
        return {
            "broker": self.broker.name,
            "simulated": self.broker.is_simulated,
            "working": len(self._working),
            "in_timeout": self.blocked_symbols,
            "orders_sent": len(self.history),
        }
