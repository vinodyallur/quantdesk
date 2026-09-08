"""Paper broker: real prices, simulated money.

``SimBroker`` is a matching engine and deliberately keeps no cash, because in live
trading the broker owns that number and inventing a second source of truth is how books
drift apart. For paper trading there is no broker to ask, so something has to hold the
ledger. This class is that something: it composes the matching engine with the same
:class:`~quantdesk.backtest.account.Account` the backtester uses, and presents the pair
through the ordinary :class:`~quantdesk.execution.broker.Broker` interface.

The point of composing rather than subclassing is that fill logic exists once. A paper
session and a backtest fill by identical rules, so a result seen here is reproducible
offline.

**Fills land at the next bar's open, never the current close.** The decision was made on
this bar's close, so filling at that same close would trade on information the strategy
did not have. Live paper trading is the easiest place for that mistake to hide, because
nothing about the output looks wrong.

**No configuration makes this touch real money.** ``is_simulated`` is True and there is no
network call anywhere in this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from quantdesk.backtest.account import Account
from quantdesk.core.types import (
    AccountSnapshot,
    Bar,
    Fill,
    Order,
    Position,
    utcnow,
)
from quantdesk.execution.broker import Broker
from quantdesk.execution.sim_broker import SimBroker

log = logging.getLogger(__name__)


@dataclass(slots=True)
class EquityTick:
    """One point on the session's equity curve, for charting and for results."""

    ts: datetime
    equity: float
    cash: float
    gross_exposure: float
    open_positions: int


@dataclass(slots=True)
class RoundTrip:
    """A completed position, from flat to flat. What "results" actually means."""

    symbol: str
    side: int
    opened_at: datetime
    closed_at: datetime
    entry: float
    exit: float
    qty: float
    pnl: float
    fees: float
    reason: str = ""

    @property
    def won(self) -> bool:
        return self.pnl > 0

    @property
    def return_pct(self) -> float:
        notional = abs(self.entry * self.qty)
        return self.pnl / notional if notional > 0 else 0.0


class PaperBroker(Broker):
    """Simulated execution against live prices, with a real ledger behind it."""

    is_simulated = True

    def __init__(
        self,
        starting_equity: float = 100_000.0,
        commission_bps: float = 1.0,
        slippage_bps: float = 2.0,
    ) -> None:
        if starting_equity <= 0:
            raise ValueError("starting equity must be positive")
        self.starting_equity = float(starting_equity)
        self._matcher = SimBroker(
            commission_bps=commission_bps, slippage_bps=slippage_bps
        )
        self.account_ledger = Account(starting_equity=self.starting_equity)
        self.equity_curve: list[EquityTick] = []
        self.round_trips: list[RoundTrip] = []
        self.all_fills: list[Fill] = []
        self._open: dict[str, dict] = {}
        self._last_ts: datetime | None = None

    # ------------------------------------------------------------------ orders
    async def submit(self, order: Order) -> Order:
        return await self._matcher.submit(order)

    def submit_sync(self, order: Order) -> Order:
        return self._matcher.submit_sync(order)

    async def cancel_all(self) -> int:
        return await self._matcher.cancel_all()

    def drain_fills(self) -> list[Fill]:
        """Fills the desk has not seen yet.

        Already applied to the ledger by :meth:`on_bar`. This hands them to the router
        for its own bookkeeping; it is not the point at which money moves.
        """
        return self._matcher.drain_fills()

    # -------------------------------------------------------------------- bars
    def on_bar(self, bar: Bar) -> list[Fill]:
        """Advance the session by one bar: fill what is pending, then mark.

        Must be called before the strategy sees this bar. Orders queued on the previous
        bar fill at this bar's open, which is what separates a decision from its fill.
        """
        self._last_ts = bar.ts
        fills = self._matcher.on_bar(bar)
        for fill in fills:
            self.account_ledger.apply(fill)
            self.all_fills.append(fill)
            self._track_round_trip(fill)

        self.account_ledger.mark(bar.symbol, bar.close)
        self.account_ledger.touch_peak()
        self.equity_curve.append(
            EquityTick(
                ts=bar.ts,
                equity=self.account_ledger.equity,
                cash=self.account_ledger.cash,
                gross_exposure=self.account_ledger.gross_exposure,
                open_positions=self.account_ledger.open_positions,
            )
        )
        if len(self.equity_curve) > 20_000:
            del self.equity_curve[:-20_000]
        return fills

    def _track_round_trip(self, fill: Fill) -> None:
        """Fold a fill into the open round trip, closing it when the position flattens.

        Kept here rather than derived from the position later, because a position that
        opens and closes several times inside a session collapses into one row if you
        only look at the end state.
        """
        signed = fill.signed_qty
        book = self._open.get(fill.symbol)

        if book is None:
            self._open[fill.symbol] = {
                "side": 1 if signed > 0 else -1,
                "opened_at": fill.ts,
                "entry": fill.price,
                "qty": abs(signed),
                "pnl": 0.0,
                "fees": fill.commission,
                "reason": fill.tag,
            }
            return

        book["fees"] += fill.commission
        same_way = (book["side"] > 0) == (signed > 0)
        if same_way:
            total = book["qty"] + abs(signed)
            book["entry"] = (
                book["entry"] * book["qty"] + fill.price * abs(signed)
            ) / total
            book["qty"] = total
            return

        closing = min(book["qty"], abs(signed))
        book["pnl"] += (fill.price - book["entry"]) * closing * book["side"]
        book["qty"] -= closing
        if book["qty"] > 1e-12:
            return

        self.round_trips.append(
            RoundTrip(
                symbol=fill.symbol,
                side=book["side"],
                opened_at=book["opened_at"],
                closed_at=fill.ts,
                entry=book["entry"],
                exit=fill.price,
                qty=closing,
                pnl=book["pnl"] - book["fees"],
                fees=book["fees"],
                reason=book["reason"],
            )
        )
        del self._open[fill.symbol]

        residual = abs(signed) - closing
        if residual > 1e-12:
            self._open[fill.symbol] = {
                "side": 1 if signed > 0 else -1,
                "opened_at": fill.ts,
                "entry": fill.price,
                "qty": residual,
                "pnl": 0.0,
                "fees": 0.0,
                "reason": fill.tag,
            }

    def mark(self, symbol: str, price: float) -> None:
        """Update a mark without advancing a bar, for between-bar price ticks."""
        self.account_ledger.mark(symbol, price)

    # ------------------------------------------------------------------- state
    async def positions(self) -> dict[str, Position]:
        return dict(self.account_ledger.positions)

    async def account(self) -> AccountSnapshot | None:
        return self.account_ledger.snapshot(self._last_ts or utcnow())

    @property
    def equity(self) -> float:
        return self.account_ledger.equity

    @property
    def pending_orders(self) -> int:
        return self._matcher.pending_count

    # ----------------------------------------------------------------- results
    def results(self) -> "PaperResults":
        return PaperResults.of(self)


@dataclass
class PaperResults:
    """Session outcome. Deliberately the numbers that decide whether to keep going."""

    starting_equity: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    peak_equity: float
    max_drawdown: float
    trades: int
    wins: int
    losses: int
    open_positions: int
    fills: int
    best: float
    worst: float

    @property
    def total_pnl(self) -> float:
        return self.equity - self.starting_equity

    @property
    def total_return(self) -> float:
        return self.total_pnl / self.starting_equity if self.starting_equity else 0.0

    @property
    def hit_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def net_of_fees(self) -> float:
        """P&L before costs. Shown because fees often are the whole result."""
        return self.total_pnl + self.fees_paid

    @classmethod
    def of(cls, broker: PaperBroker) -> "PaperResults":
        ledger = broker.account_ledger
        trips = broker.round_trips
        equities = [t.equity for t in broker.equity_curve] or [ledger.equity]
        peak = equities[0]
        worst_dd = 0.0
        for value in equities:
            peak = max(peak, value)
            if peak > 0:
                worst_dd = max(worst_dd, (peak - value) / peak)
        return cls(
            starting_equity=broker.starting_equity,
            equity=ledger.equity,
            realized_pnl=ledger.realized_pnl,
            unrealized_pnl=ledger.unrealized_pnl,
            fees_paid=ledger.fees_paid,
            peak_equity=max(equities),
            max_drawdown=worst_dd,
            trades=len(trips),
            wins=sum(1 for t in trips if t.won),
            losses=sum(1 for t in trips if not t.won),
            open_positions=ledger.open_positions,
            fills=len(broker.all_fills),
            best=max((t.pnl for t in trips), default=0.0),
            worst=min((t.pnl for t in trips), default=0.0),
        )

    def summary(self) -> list[str]:
        return [
            f"equity       {self.starting_equity:,.2f} -> {self.equity:,.2f}",
            f"P&L          {self.total_pnl:+,.2f}  ({self.total_return:+.2%})",
            f"realised     {self.realized_pnl:+,.2f}   "
            f"unrealised {self.unrealized_pnl:+,.2f}",
            f"fees         {self.fees_paid:,.2f}  "
            f"(P&L before costs {self.net_of_fees:+,.2f})",
            f"drawdown     {self.max_drawdown:.2%} from a peak of {self.peak_equity:,.2f}",
            f"trades       {self.trades}  hit rate {self.hit_rate:.1%}  "
            f"({self.wins}W / {self.losses}L)",
            f"best {self.best:+,.2f}  worst {self.worst:+,.2f}  "
            f"fills {self.fills}  open {self.open_positions}",
        ]
