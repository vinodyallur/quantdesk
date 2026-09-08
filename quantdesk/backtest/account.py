"""Simulated account ledger.

``SimBroker`` is a matching engine: it decides whether and at what price an order
fills. It deliberately does not track cash or equity, because in live trading those
numbers come from the broker and inventing a second source of truth is how books
drift apart.

For a backtest something has to keep them, and that is this class. Kept separate from
the matching engine so the same ledger logic can mirror a live account for display
without ever being treated as authoritative.

Equity is cash plus the mark-to-market value of open positions. Positions use
:meth:`~quantdesk.core.types.Position.apply_fill`, which already handles adding,
reducing and flipping through zero, so P&L accounting lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from quantdesk.core.types import AccountSnapshot, Fill, Position


@dataclass
class Account:
    """Cash, positions and equity for a simulated run."""

    starting_equity: float = 100_000.0
    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    peak_equity: float = 0.0
    day_start_equity: float = 0.0

    def __post_init__(self) -> None:
        if self.cash == 0.0:
            self.cash = self.starting_equity
        self.peak_equity = self.peak_equity or self.starting_equity
        self.day_start_equity = self.day_start_equity or self.starting_equity

    # ------------------------------------------------------------------ fills
    def apply(self, fill: Fill) -> float:
        """Fold a fill into the account. Returns realised P&L from it.

        Cash moves by the signed notional plus commission: buying costs cash, selling
        raises it, and a short sale credits cash while creating a negative position.
        Equity is unaffected at the moment of the fill except for costs, which is the
        arithmetic check that this is right.
        """
        pos = self.positions.get(fill.symbol)
        if pos is None:
            pos = Position(symbol=fill.symbol)
            self.positions[fill.symbol] = pos

        realized = pos.apply_fill(fill)
        self.realized_pnl += realized
        self.fees_paid += fill.commission
        self.cash -= fill.signed_qty * fill.price
        self.cash -= fill.commission
        self.marks[fill.symbol] = fill.price
        return realized

    # ------------------------------------------------------------------ marks
    def mark(self, symbol: str, price: float) -> None:
        if price <= 0:
            return
        self.marks[symbol] = price
        pos = self.positions.get(symbol)
        if pos is not None:
            pos.last_price = price

    def last_price(self, symbol: str) -> float:
        return self.marks.get(symbol, 0.0)

    # ----------------------------------------------------------------- equity
    @property
    def position_value(self) -> float:
        """Signed mark-to-market value of all open positions."""
        total = 0.0
        for symbol, pos in self.positions.items():
            if pos.is_flat:
                continue
            price = self.marks.get(symbol, pos.avg_price)
            total += pos.qty * price
        return total

    @property
    def equity(self) -> float:
        return self.cash + self.position_value

    @property
    def gross_exposure(self) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            if pos.is_flat:
                continue
            total += abs(pos.qty * self.marks.get(symbol, pos.avg_price))
        return total

    @property
    def net_exposure(self) -> float:
        return self.position_value

    @property
    def unrealized_pnl(self) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            if pos.is_flat:
                continue
            total += pos.unrealized_pnl(self.marks.get(symbol, pos.avg_price))
        return total

    @property
    def open_positions(self) -> int:
        return sum(1 for p in self.positions.values() if not p.is_flat)

    def touch_peak(self) -> None:
        self.peak_equity = max(self.peak_equity, self.equity)

    def live_positions(self) -> dict[str, Position]:
        """Only the non-flat positions, which is what risk and sizing care about."""
        return {s: p for s, p in self.positions.items() if not p.is_flat}

    def snapshot(self, ts: datetime) -> AccountSnapshot:
        self.touch_peak()
        return AccountSnapshot(
            ts=ts,
            equity=self.equity,
            cash=self.cash,
            gross_exposure=self.gross_exposure,
            net_exposure=self.net_exposure,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            day_start_equity=self.day_start_equity,
            peak_equity=self.peak_equity,
        )

    def reset(self) -> None:
        self.cash = self.starting_equity
        self.positions.clear()
        self.marks.clear()
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.peak_equity = self.starting_equity
        self.day_start_equity = self.starting_equity
