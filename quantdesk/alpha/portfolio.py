"""Portfolio accounting.

Single source of truth for what the desk owns, what it is worth, and how it got
there. Both the backtester and the live desk use this class, so P&L is computed
one way only.

Accounting model: cash plus mark-to-market position value. Buying moves cash into
position value; selling short raises cash and creates a negative position value.
Equity is the sum. This is a cash-account model without margin interest or
borrow costs, which is the right level of detail for paper trading but understates
the true cost of holding shorts.
"""

from __future__ import annotations

from datetime import datetime, timezone

from quantdesk.core.types import AccountSnapshot, Fill, Position


class Portfolio:
    """Positions, cash and P&L history."""

    def __init__(self, starting_equity: float = 100_000.0) -> None:
        self.starting_equity = starting_equity
        self.cash = starting_equity
        self.positions: dict[str, Position] = {}
        self.fees_paid = 0.0
        self.realized_pnl = 0.0
        self.fill_count = 0
        self.turnover = 0.0

        self._prices: dict[str, float] = {}
        self._peak_equity = starting_equity
        self._day_start_equity = starting_equity
        self._day: datetime | None = None
        self.equity_curve: list[tuple[datetime, float]] = []
        #: Realised P&L attributed to the agents that motivated each trade.
        self.attribution: dict[str, float] = {}

    # ------------------------------------------------------------------ marks
    def mark(self, prices: dict[str, float]) -> None:
        """Update marks. Only non-positive prices are ignored."""
        for sym, px in prices.items():
            if px and px > 0:
                self._prices[sym] = px
                pos = self.positions.get(sym)
                if pos is not None:
                    pos.last_price = px

    def price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)

    # ------------------------------------------------------------------ fills
    def apply_fill(self, fill: Fill) -> float:
        """Book a fill. Returns realised P&L from it."""
        pos = self.positions.get(fill.symbol)
        if pos is None:
            pos = Position(symbol=fill.symbol)
            self.positions[fill.symbol] = pos

        # Buying spends cash, selling raises it. Commission always costs.
        self.cash -= fill.signed_qty * fill.price
        self.cash -= fill.commission
        self.fees_paid += fill.commission
        self.turnover += abs(fill.notional)
        self.fill_count += 1

        realized = pos.apply_fill(fill)
        self.realized_pnl += realized
        self._prices[fill.symbol] = fill.price

        if fill.tag:
            # Attribute to the dominant agent recorded on the order tag.
            self.attribution[fill.tag] = self.attribution.get(fill.tag, 0.0) + realized
        return realized

    # -------------------------------------------------------------- valuation
    @property
    def position_value(self) -> float:
        """Signed mark-to-market value of all positions."""
        return sum(
            p.market_value(self._prices.get(s, p.last_price))
            for s, p in self.positions.items()
        )

    @property
    def equity(self) -> float:
        return self.cash + self.position_value

    @property
    def unrealized_pnl(self) -> float:
        return sum(
            p.unrealized_pnl(self._prices.get(s, p.last_price))
            for s, p in self.positions.items()
        )

    @property
    def gross_exposure(self) -> float:
        """Sum of |position value| over equity."""
        eq = self.equity
        if eq <= 0:
            return 0.0
        gross = sum(
            abs(p.market_value(self._prices.get(s, p.last_price)))
            for s, p in self.positions.items()
        )
        return gross / eq

    @property
    def net_exposure(self) -> float:
        """Signed net position value over equity."""
        eq = self.equity
        if eq <= 0:
            return 0.0
        return self.position_value / eq

    def weight(self, symbol: str) -> float:
        """Current signed weight of a symbol as a fraction of equity."""
        pos = self.positions.get(symbol)
        eq = self.equity
        if pos is None or eq <= 0:
            return 0.0
        return pos.market_value(self._prices.get(symbol, pos.last_price)) / eq

    def weights(self) -> dict[str, float]:
        return {s: self.weight(s) for s in self.positions if not self.positions[s].is_flat}

    def open_positions(self) -> dict[str, Position]:
        return {s: p for s, p in self.positions.items() if not p.is_flat}

    # -------------------------------------------------------------- snapshots
    def record(self, ts: datetime) -> AccountSnapshot:
        """Append to the equity curve and return a snapshot."""
        eq = self.equity

        # Roll the daily baseline when the UTC date changes. Day-loss limits are
        # meaningless without this and would otherwise measure since inception.
        day = ts.astimezone(timezone.utc).date()
        if self._day is None or day != self._day:
            self._day = day
            self._day_start_equity = eq

        self._peak_equity = max(self._peak_equity, eq)
        self.equity_curve.append((ts, eq))

        return AccountSnapshot(
            ts=ts,
            equity=eq,
            cash=self.cash,
            gross_exposure=self.gross_exposure,
            net_exposure=self.net_exposure,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            day_start_equity=self._day_start_equity,
            peak_equity=self._peak_equity,
        )

    def snapshot(self, ts: datetime) -> AccountSnapshot:
        """Snapshot without mutating the equity curve."""
        eq = self.equity
        return AccountSnapshot(
            ts=ts,
            equity=eq,
            cash=self.cash,
            gross_exposure=self.gross_exposure,
            net_exposure=self.net_exposure,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            day_start_equity=self._day_start_equity,
            peak_equity=max(self._peak_equity, eq),
        )

    @property
    def total_return(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return (self.equity - self.starting_equity) / self.starting_equity

    def reset(self) -> None:
        self.__init__(self.starting_equity)  # type: ignore[misc]
