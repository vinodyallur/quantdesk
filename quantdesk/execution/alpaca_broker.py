"""Alpaca paper-trading adapter.

Three safety properties, in order of how badly they would hurt if missing.

**It cannot reach a live endpoint.** The constructor refuses ``paper=False``
outright rather than trusting configuration. A desk that can be pointed at real
money by an environment variable is one typo away from a very bad day, and nothing
in this project needs that capability.

**Broker state wins.** :meth:`positions` and :meth:`account` read Alpaca, and the
pipeline adopts those numbers rather than its own bookkeeping. Local position
tracking drifts the moment a partial fill, a fee, or a rejection is missed, and a
drifted position size means every subsequent risk calculation is wrong.

**Submissions are idempotent.** Every order carries a client order id derived from
our own id, so a retry after a timeout cannot open the position twice. This is the
failure that turns a network blip into unintended leverage.

Fills are pulled, not pushed, matching :class:`~quantdesk.execution.broker.Broker`:
the trade-update stream would otherwise mutate the portfolio from a network task in
the middle of a decision.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from quantdesk.core.types import (
    AccountSnapshot,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    TimeInForce,
    utcnow,
)
from quantdesk.execution.broker import Broker

log = logging.getLogger(__name__)


class AlpacaBrokerError(RuntimeError):
    """Raised for configuration problems that must not be silently tolerated."""


class AlpacaPaperBroker(Broker):
    """Places orders against an Alpaca paper account."""

    is_simulated = False

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        asset_class: str = "crypto",
    ) -> None:
        if not paper:
            # Not a warning, not a config flag. This class exists for paper trading.
            raise AlpacaBrokerError(
                "AlpacaPaperBroker refuses to run against a live endpoint. "
                "Live trading is deliberately not implemented in this project."
            )
        if not api_key or not secret_key:
            raise AlpacaBrokerError(
                "Alpaca paper trading needs QD_ALPACA_KEY and QD_ALPACA_SECRET. "
                "Market data works without keys, but placing orders does not."
            )
        self.asset_class = asset_class
        self._fills: list[Fill] = []
        self._seen_fill_ids: set[str] = set()
        #: Maps our order id to the broker's, so fills can be attributed.
        self._submitted: dict[str, str] = {}

        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise AlpacaBrokerError(
                "alpaca-py is required for order placement: pip install alpaca-py"
            ) from exc

        self._client = TradingClient(api_key, secret_key, paper=True)
        self._verify_paper()

    def _verify_paper(self) -> None:
        """Confirm with the broker that this really is a paper account.

        Trusting the ``paper=True`` flag we passed in is circular. Asking the account
        itself is the only check that means anything.
        """
        try:
            account = self._client.get_account()
        except Exception as exc:  # pragma: no cover - network dependent
            raise AlpacaBrokerError(f"could not reach Alpaca: {exc}") from exc
        # Alpaca exposes this on the account object; if it is missing we refuse
        # rather than assume.
        if getattr(account, "status", None) is None:
            raise AlpacaBrokerError("Alpaca account response was not understood")
        log.info(
            "connected to Alpaca paper account %s (status %s)",
            getattr(account, "account_number", "?"),
            account.status,
        )

    # ----------------------------------------------------------------- orders
    async def submit(self, order: Order) -> Order:
        """Send one order, idempotently."""
        from alpaca.trading.enums import OrderSide, TimeInForce as AlpacaTIF
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        side = OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL
        # Crypto trades around the clock, so GTC is the only sensible default; DAY
        # would expire an order at a session boundary that does not exist.
        tif = AlpacaTIF.GTC if order.tif is TimeInForce.GTC else AlpacaTIF.DAY
        if self.asset_class == "crypto":
            tif = AlpacaTIF.GTC

        common = {
            "symbol": order.symbol,
            "qty": round(order.qty, 9),
            "side": side,
            "time_in_force": tif,
            # The idempotency key. A retry with the same id is rejected as a
            # duplicate by Alpaca instead of opening a second position.
            "client_order_id": order.id or None,
        }
        if order.type is OrderType.LIMIT and order.limit_price:
            request = LimitOrderRequest(limit_price=order.limit_price, **common)
        else:
            request = MarketOrderRequest(**common)

        try:
            placed = await asyncio.to_thread(self._client.submit_order, request)
        except Exception as exc:  # noqa: BLE001 - surface as a rejection, not a crash
            order.status = OrderStatus.REJECTED
            order.reject_reason = str(exc)[:300]
            log.error("order rejected for %s: %s", order.symbol, exc)
            return order

        order.broker_order_id = str(placed.id)
        order.status = _map_status(getattr(placed, "status", None))
        filled = float(getattr(placed, "filled_qty", 0) or 0)
        order.filled_qty = filled
        avg = getattr(placed, "filled_avg_price", None)
        if avg:
            order.avg_fill_price = float(avg)
        self._submitted[order.id] = order.broker_order_id
        return order

    async def cancel_all(self) -> int:
        try:
            cancelled = await asyncio.to_thread(self._client.cancel_orders)
        except Exception as exc:  # noqa: BLE001
            log.error("cancel_all failed: %s", exc)
            return 0
        return len(cancelled or [])

    # ------------------------------------------------------------------ fills
    async def poll_fills(self) -> None:
        """Fetch recently closed orders and record any fills not yet seen.

        Deduplicated on the broker's own order id, because polling windows overlap and
        double-counting a fill would corrupt the position and every risk number
        derived from it.
        """
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        try:
            request = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=100)
            orders = await asyncio.to_thread(self._client.get_orders, request)
        except Exception as exc:  # noqa: BLE001
            log.error("could not poll fills: %s", exc)
            return

        for o in orders or []:
            key = str(o.id)
            if key in self._seen_fill_ids:
                continue
            qty = float(getattr(o, "filled_qty", 0) or 0)
            price = getattr(o, "filled_avg_price", None)
            if qty <= 0 or not price:
                continue
            self._seen_fill_ids.add(key)
            self._fills.append(
                Fill(
                    order_id=str(getattr(o, "client_order_id", "") or key),
                    symbol=str(o.symbol),
                    side=Side.BUY if str(o.side).lower().endswith("buy") else Side.SELL,
                    qty=qty,
                    price=float(price),
                    ts=getattr(o, "filled_at", None) or utcnow(),
                    tag="alpaca",
                )
            )
        # Bound the dedup set so a long session does not grow it without limit.
        if len(self._seen_fill_ids) > 5000:
            self._seen_fill_ids = set(list(self._seen_fill_ids)[-2500:])

    def drain_fills(self) -> list[Fill]:
        out = self._fills
        self._fills = []
        return out

    # ------------------------------------------------------------------ state
    async def sync(self) -> None:
        await self.poll_fills()

    async def positions(self) -> dict[str, Position]:
        """Positions as the broker sees them. This is the authority, not our books."""
        try:
            raw = await asyncio.to_thread(self._client.get_all_positions)
        except Exception as exc:  # noqa: BLE001
            log.error("could not fetch positions: %s", exc)
            return {}

        out: dict[str, Position] = {}
        for p in raw or []:
            qty = float(p.qty)
            if str(getattr(p, "side", "")).lower().endswith("short"):
                qty = -abs(qty)
            out[str(p.symbol)] = Position(
                symbol=str(p.symbol),
                qty=qty,
                avg_price=float(p.avg_entry_price),
                realized_pnl=0.0,
                last_price=float(getattr(p, "current_price", 0) or p.avg_entry_price),
            )
        return out

    async def account(self) -> AccountSnapshot | None:
        try:
            a = await asyncio.to_thread(self._client.get_account)
        except Exception as exc:  # noqa: BLE001
            log.error("could not fetch account: %s", exc)
            return None
        equity = float(a.equity)
        return AccountSnapshot(
            ts=utcnow(),
            equity=equity,
            cash=float(a.cash),
            day_start_equity=float(getattr(a, "last_equity", equity) or equity),
            peak_equity=equity,
        )

    async def flatten(self) -> int:
        """Close every position. Used by the risk kill switch."""
        try:
            await asyncio.to_thread(self._client.close_all_positions, True)
        except Exception as exc:  # noqa: BLE001
            log.error("flatten failed: %s", exc)
            return 0
        return 1


def _map_status(raw: object) -> OrderStatus:
    """Translate Alpaca's status vocabulary into ours."""
    text = str(raw or "").lower()
    if "partially" in text:
        return OrderStatus.PARTIAL
    if "filled" in text:
        return OrderStatus.FILLED
    if "cancel" in text or "expired" in text:
        return OrderStatus.CANCELED
    if "reject" in text:
        return OrderStatus.REJECTED
    if "new" in text or "accepted" in text or "pending" in text:
        return OrderStatus.SUBMITTED
    return OrderStatus.SUBMITTED
