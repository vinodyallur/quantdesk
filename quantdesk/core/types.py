"""Domain types.

Plain dataclasses (not pydantic) because these live on the hot path and get
created thousands of times per backtest bar. Validation happens at the edges.

All timestamps are timezone-aware UTC. This is enforced by convention rather
than by runtime checks to keep the hot path cheap; feeds are responsible for
normalising incoming data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


def from_epoch(seconds: float) -> datetime:
    """UTC datetime from epoch seconds, correct for dates before 1970.

    ``datetime.fromtimestamp`` raises ``OSError: [Errno 22]`` for negative values on
    Windows, because it delegates to a C runtime that rejects them. Every bar before
    1970 has a negative epoch, so a century of index history is exactly the case that
    breaks. Adding a timedelta to the epoch has no platform dependency and round-trips
    to the same value.
    """
    return EPOCH + timedelta(seconds=seconds)


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @classmethod
    def from_sign(cls, value: float) -> "Side":
        return cls.BUY if value >= 0 else cls.SELL


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"


class OrderStatus(str, Enum):
    NEW = "new"
    SUBMITTED = "submitted"
    PARTIAL = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)


# --------------------------------------------------------------------- market


@dataclass(slots=True)
class Bar:
    """One OHLCV candle for a symbol."""

    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trade_count: int | None = None

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def range_pct(self) -> float:
        """High-low range as a fraction of close. Guards against zero close."""
        if self.close <= 0:
            return 0.0
        return (self.high - self.low) / self.close


@dataclass(slots=True)
class Quote:
    """Top-of-book snapshot."""

    symbol: str
    ts: datetime
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.ask or self.bid

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        if mid <= 0:
            return 0.0
        return (self.ask - self.bid) / mid * 10_000.0


# --------------------------------------------------------------------- alpha


@dataclass(slots=True)
class Signal:
    """A single agent's opinion on a single symbol at a point in time.

    ``score`` is the directional view in [-1, 1]: negative is short, positive is
    long, zero is no opinion. ``confidence`` in [0, 1] scales how much the
    blender trusts this score. Keeping direction and conviction separate lets an
    agent say "strongly flat" versus "weakly long", which matters when blending.
    """

    agent: str
    symbol: str
    ts: datetime
    score: float
    confidence: float = 1.0
    horizon_bars: int = 12
    rationale: str = ""
    features: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Clamp rather than raise: a misbehaving agent should not halt the desk.
        self.score = _clamp(self.score, -1.0, 1.0)
        self.confidence = _clamp(self.confidence, 0.0, 1.0)

    @property
    def weighted_score(self) -> float:
        return self.score * self.confidence


@dataclass(slots=True)
class TargetPosition:
    """Desired exposure to a symbol as a signed fraction of equity."""

    symbol: str
    weight: float
    score: float = 0.0
    reason: str = ""
    contributors: dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------- execution


@dataclass(slots=True)
class Order:
    symbol: str
    side: Side
    qty: float
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    tif: TimeInForce = TimeInForce.DAY
    id: str = ""
    ts: datetime = field(default_factory=utcnow)
    status: OrderStatus = OrderStatus.NEW
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    broker_order_id: str | None = None
    tag: str = ""
    reject_reason: str = ""
    planned_risk: float = 0.0
    """Currency this order puts at risk if its protective stop is hit.

    Set at decision time, when the stop and the approved size are both known, and
    carried through to the fill. The alternative - reconstructing it downstream from a
    fill plus whatever the trade manager currently holds - silently under-counts on
    partial fills and pyramid layers, and reads the wrong trade entirely once an exit
    has already been booked. R is only as trustworthy as this number, so it travels
    with the order rather than being inferred.

    Zero for exits and for any order that reduces exposure: closing a position puts
    no new risk on.
    """

    @property
    def remaining(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    @property
    def signed_qty(self) -> float:
        return self.qty * self.side.sign


@dataclass(slots=True)
class Fill:
    order_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    ts: datetime = field(default_factory=utcnow)
    commission: float = 0.0
    slippage_bps: float = 0.0
    tag: str = ""

    @property
    def notional(self) -> float:
        return self.qty * self.price

    @property
    def signed_qty(self) -> float:
        return self.qty * self.side.sign


@dataclass(slots=True)
class Position:
    """A signed position with running average cost and realised P&L."""

    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    last_price: float = 0.0
    opened_at: datetime | None = None

    @property
    def is_flat(self) -> bool:
        return abs(self.qty) < 1e-12

    @property
    def direction(self) -> int:
        if self.is_flat:
            return 0
        return 1 if self.qty > 0 else -1

    def market_value(self, price: float | None = None) -> float:
        """Signed mark-to-market notional."""
        px = price if price is not None else self.last_price
        return self.qty * px

    def unrealized_pnl(self, price: float | None = None) -> float:
        px = price if price is not None else self.last_price
        if self.is_flat or px <= 0:
            return 0.0
        return (px - self.avg_price) * self.qty

    def total_pnl(self, price: float | None = None) -> float:
        return self.realized_pnl + self.unrealized_pnl(price)

    def apply_fill(self, fill: Fill) -> float:
        """Fold a fill into this position. Returns realised P&L from this fill.

        Handles the three distinct cases explicitly: opening/adding in the same
        direction (recompute weighted average cost), reducing (bank realised
        P&L, keep cost basis), and flipping through zero (bank P&L on the closed
        portion, then reset cost basis to the fill price for the new leg).
        """
        signed = fill.signed_qty
        self.fees_paid += fill.commission
        self.last_price = fill.price
        realized_now = 0.0

        if self.is_flat:
            self.qty = signed
            self.avg_price = fill.price
            self.opened_at = fill.ts
        elif (self.qty > 0) == (signed > 0):
            # Adding to the existing side: weighted average cost.
            total = self.qty + signed
            self.avg_price = (
                self.avg_price * self.qty + fill.price * signed
            ) / total
            self.qty = total
        else:
            closing_qty = min(abs(signed), abs(self.qty))
            # Sign of the position being closed determines P&L direction.
            realized_now = (fill.price - self.avg_price) * closing_qty * self.direction
            self.realized_pnl += realized_now
            new_qty = self.qty + signed
            if abs(new_qty) < 1e-12:
                self.qty = 0.0
                self.avg_price = 0.0
                self.opened_at = None
            elif (new_qty > 0) != (self.qty > 0):
                # Flipped through zero: the residual is a brand new position.
                self.qty = new_qty
                self.avg_price = fill.price
                self.opened_at = fill.ts
            else:
                self.qty = new_qty
        return realized_now


@dataclass(slots=True)
class AccountSnapshot:
    """Point-in-time account state, used by risk and the UI."""

    ts: datetime
    equity: float
    cash: float
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    day_start_equity: float = 0.0
    peak_equity: float = 0.0

    @property
    def day_pnl(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return self.equity - self.day_start_equity

    @property
    def day_pnl_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return self.day_pnl / self.day_start_equity

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)


@dataclass(slots=True)
class RiskDecision:
    """Outcome of a risk check. ``approved=False`` means do not send."""

    approved: bool
    reason: str = ""
    adjusted_qty: float | None = None

    @classmethod
    def ok(cls, adjusted_qty: float | None = None) -> "RiskDecision":
        return cls(approved=True, adjusted_qty=adjusted_qty)

    @classmethod
    def veto(cls, reason: str) -> "RiskDecision":
        return cls(approved=False, reason=reason)


# --------------------------------------------------------------------- utils


def _clamp(value: float, low: float, high: float) -> float:
    if value is None or math.isnan(value):
        return 0.0
    return max(low, min(high, value))


def as_dict(obj: Any) -> dict[str, Any]:
    """Shallow dict view of a slotted dataclass, for logging and the UI."""
    return {
        slot: getattr(obj, slot)
        for slot in getattr(obj, "__slots__", ())
        if hasattr(obj, slot)
    }
