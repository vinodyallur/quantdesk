"""Pre-trade risk management and the kill switch.

This is the last thing between a misbehaving agent and the order book, so it is
written to be boring and readable rather than clever.

The central distinction, and the thing most naive risk layers get wrong, is
**risk-reducing versus risk-increasing** orders. A blanket "stop trading when we
hit the loss limit" also blocks the orders that close losing positions, which
means the desk sits frozen holding exactly the exposure that triggered the halt.
Every check here asks whether an order increases exposure, and only constrains
those. Closing and flattening are always permitted.

Three escalating states:

* ``NORMAL`` - full limits apply.
* ``REDUCE_ONLY`` - daily loss limit breached. No new or increased positions, but
  existing ones can be trimmed or closed. Resets when the trading day rolls.
* ``HALTED`` - max drawdown breached. Targets are forced flat and the desk stops
  taking risk. Requires an explicit ``resume()`` to clear, because an automatic
  reset after the worst drawdown on record is not a decision software should make
  on its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

from quantdesk.core.types import (
    AccountSnapshot,
    Fill,
    Order,
    RiskDecision,
    Side,
    TargetPosition,
)

log = logging.getLogger(__name__)


class RiskState(str, Enum):
    NORMAL = "normal"
    REDUCE_ONLY = "reduce_only"
    HALTED = "halted"


@dataclass(slots=True)
class RiskLimits:
    max_position_weight: float = 0.20
    max_gross_exposure: float = 1.00
    max_net_exposure: float = 0.60
    daily_loss_limit_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    cooldown_bars: int = 3
    #: Orders below this notional are not worth the fees and spread.
    min_notional: float = 25.0
    #: Refuse any single order larger than this fraction of equity, as a
    #: backstop against a sizing bug producing an absurd quantity.
    max_order_weight: float = 0.35
    #: Tolerance on weight caps so floating point noise doesn't cause churn.
    tolerance: float = 1e-4


@dataclass(slots=True)
class RiskEvent:
    ts: datetime
    kind: str
    detail: str
    state: str = ""


class RiskManager:
    """Pre-trade checks, exposure limits and the kill switch."""

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.state = RiskState.NORMAL
        self.halt_reason = ""
        self.events: list[RiskEvent] = []
        self.vetoes = 0
        self.adjustments = 0
        self._cooldown: dict[str, int] = {}
        self._bar_index = 0
        self._day: date | None = None

    # ------------------------------------------------------------ bar / clock
    def on_bar(self) -> None:
        """Advance the internal bar counter and expire cooldowns."""
        self._bar_index += 1
        expired = [s for s, until in self._cooldown.items() if until <= self._bar_index]
        for s in expired:
            del self._cooldown[s]

    def in_cooldown(self, symbol: str) -> bool:
        return self._cooldown.get(symbol, 0) > self._bar_index

    def on_fill(self, fill: Fill) -> None:
        """Start a cooldown so the desk doesn't thrash a symbol."""
        if self.limits.cooldown_bars > 0:
            self._cooldown[fill.symbol] = self._bar_index + self.limits.cooldown_bars

    # -------------------------------------------------------- account checks
    def evaluate_account(self, snap: AccountSnapshot) -> RiskState:
        """Escalate or de-escalate risk state based on account health."""
        lim = self.limits

        # Roll REDUCE_ONLY off at the start of a new trading day.
        day = snap.ts.date()
        if self._day is None:
            self._day = day
        elif day != self._day:
            self._day = day
            if self.state is RiskState.REDUCE_ONLY:
                self._transition(
                    RiskState.NORMAL, snap.ts, "new trading day, daily loss limit reset"
                )

        # Max drawdown is terminal: flatten and stop.
        if lim.max_drawdown_pct > 0 and snap.drawdown_pct >= lim.max_drawdown_pct:
            if self.state is not RiskState.HALTED:
                self.halt_reason = (
                    f"drawdown {snap.drawdown_pct:.2%} breached limit "
                    f"{lim.max_drawdown_pct:.2%}"
                )
                self._transition(RiskState.HALTED, snap.ts, self.halt_reason)
            return self.state

        if self.state is RiskState.HALTED:
            return self.state

        # Daily loss limit: stop adding risk for the rest of the day.
        if lim.daily_loss_limit_pct > 0 and snap.day_pnl_pct <= -lim.daily_loss_limit_pct:
            if self.state is RiskState.NORMAL:
                self._transition(
                    RiskState.REDUCE_ONLY,
                    snap.ts,
                    f"day P&L {snap.day_pnl_pct:.2%} breached limit "
                    f"-{lim.daily_loss_limit_pct:.2%}",
                )
        return self.state

    def _transition(self, new_state: RiskState, ts: datetime, detail: str) -> None:
        old = self.state
        self.state = new_state
        self.events.append(
            RiskEvent(ts=ts, kind="state_change", detail=detail, state=new_state.value)
        )
        log.warning("risk state %s -> %s: %s", old.value, new_state.value, detail)

    # ----------------------------------------------------------------- targets
    def filter_targets(
        self, targets: list[TargetPosition], current_weights: dict[str, float]
    ) -> list[TargetPosition]:
        """Adjust desired targets for the current risk state.

        In ``HALTED`` every target becomes flat, including symbols the blender had
        no opinion on, so nothing is left stranded on the book. In ``REDUCE_ONLY``
        targets may not exceed the current position in magnitude or flip its sign.
        """
        if self.state is RiskState.HALTED:
            out = [
                TargetPosition(symbol=t.symbol, weight=0.0, score=t.score,
                               reason=f"HALTED: {self.halt_reason}")
                for t in targets
            ]
            # Include anything held but not mentioned in this round of targets.
            mentioned = {t.symbol for t in targets}
            for sym, w in current_weights.items():
                if sym not in mentioned and abs(w) > 1e-9:
                    out.append(
                        TargetPosition(symbol=sym, weight=0.0,
                                       reason=f"HALTED: {self.halt_reason}")
                    )
            return out

        if self.state is RiskState.REDUCE_ONLY:
            out = []
            for t in targets:
                cur = current_weights.get(t.symbol, 0.0)
                new_w = t.weight
                if abs(cur) < 1e-9:
                    # No position: cannot open a new one.
                    new_w = 0.0
                elif (new_w > 0) != (cur > 0) and abs(new_w) > 1e-9:
                    # Would flip direction, which is opening fresh risk.
                    new_w = 0.0
                elif abs(new_w) > abs(cur):
                    # Cap at the existing size: trimming allowed, adding is not.
                    new_w = cur
                out.append(
                    TargetPosition(
                        symbol=t.symbol,
                        weight=new_w,
                        score=t.score,
                        reason=(t.reason if new_w == t.weight
                                else f"REDUCE_ONLY: capped from {t.weight:+.3f}"),
                        contributors=t.contributors,
                    )
                )
            return out

        return targets

    # ------------------------------------------------------------------ orders
    def check_order(
        self,
        order: Order,
        price: float,
        equity: float,
        current_qty: float,
        gross_exposure: float,
        net_exposure: float,
    ) -> RiskDecision:
        """Approve, resize, or veto a single order.

        ``current_qty`` is the signed existing position in this symbol.
        """
        lim = self.limits

        if price <= 0:
            return self._veto("no valid price")
        if equity <= 0:
            return self._veto("non-positive equity")
        if order.qty <= 0:
            return self._veto("non-positive quantity")

        signed = order.signed_qty
        new_qty = current_qty + signed

        # Does this order add exposure, or reduce it? Everything hinges on this.
        reducing = abs(new_qty) < abs(current_qty) - 1e-12
        closing_only = reducing or abs(new_qty) < 1e-12

        # Risk-reducing orders bypass the exposure limits entirely. They must:
        # a stuck desk that cannot exit is more dangerous than one over its cap.
        if closing_only:
            notional = order.qty * price
            if notional < lim.min_notional and abs(new_qty) > 1e-12:
                # Allow dust-clearing full closes, skip trivial partial trims.
                return self._veto(
                    f"trim notional {notional:.2f} below minimum {lim.min_notional:.2f}"
                )
            return RiskDecision.ok()

        # ---- from here on the order increases exposure ----
        if self.state is RiskState.HALTED:
            return self._veto(f"desk halted: {self.halt_reason}")
        if self.state is RiskState.REDUCE_ONLY:
            return self._veto("reduce-only mode: daily loss limit reached")
        if self.in_cooldown(order.symbol):
            return self._veto(f"{order.symbol} in cooldown")

        notional = order.qty * price
        if notional < lim.min_notional:
            return self._veto(
                f"notional {notional:.2f} below minimum {lim.min_notional:.2f}"
            )

        # Backstop against a sizing bug producing an absurd single order.
        if notional / equity > lim.max_order_weight:
            capped_qty = (lim.max_order_weight * equity) / price
            if capped_qty * price < lim.min_notional:
                return self._veto("order exceeds max order weight and cannot be resized")
            self.adjustments += 1
            return RiskDecision.ok(adjusted_qty=capped_qty)

        # Per-symbol weight cap.
        new_weight = abs(new_qty * price / equity)
        if new_weight > lim.max_position_weight + lim.tolerance:
            allowed_qty = (lim.max_position_weight * equity) / price
            # How much more can we add in this direction before hitting the cap?
            room = allowed_qty - abs(current_qty)
            if room <= 0:
                return self._veto(
                    f"{order.symbol} already at position cap "
                    f"{lim.max_position_weight:.0%}"
                )
            if room * price < lim.min_notional:
                return self._veto(
                    f"{order.symbol} remaining room below minimum notional"
                )
            self.adjustments += 1
            return RiskDecision.ok(adjusted_qty=room)

        # Gross exposure cap, measured on the incremental exposure added.
        added_gross = (abs(new_qty) - abs(current_qty)) * price / equity
        if gross_exposure + added_gross > lim.max_gross_exposure + lim.tolerance:
            room_weight = lim.max_gross_exposure - gross_exposure
            if room_weight <= lim.tolerance:
                return self._veto(
                    f"gross exposure {gross_exposure:.2f} at cap "
                    f"{lim.max_gross_exposure:.2f}"
                )
            capped_qty = (room_weight * equity) / price
            if capped_qty * price < lim.min_notional:
                return self._veto("gross exposure room below minimum notional")
            self.adjustments += 1
            return RiskDecision.ok(adjusted_qty=capped_qty)

        # Net exposure cap: only binding if this order pushes net further out.
        added_net = signed * price / equity
        projected_net = net_exposure + added_net
        if abs(projected_net) > lim.max_net_exposure + lim.tolerance and (
            abs(projected_net) > abs(net_exposure)
        ):
            room_weight = max(0.0, lim.max_net_exposure - abs(net_exposure))
            if room_weight <= lim.tolerance:
                return self._veto(
                    f"net exposure {net_exposure:+.2f} at cap "
                    f"±{lim.max_net_exposure:.2f}"
                )
            capped_qty = (room_weight * equity) / price
            if capped_qty * price < lim.min_notional:
                return self._veto("net exposure room below minimum notional")
            self.adjustments += 1
            return RiskDecision.ok(adjusted_qty=capped_qty)

        return RiskDecision.ok()

    def _veto(self, reason: str) -> RiskDecision:
        self.vetoes += 1
        return RiskDecision.veto(reason)

    # ------------------------------------------------------------------ manual
    def kill(self, reason: str, ts: datetime) -> None:
        """Manual kill switch, wired to the terminal UI."""
        self.halt_reason = reason
        self._transition(RiskState.HALTED, ts, f"manual kill: {reason}")

    def resume(self, ts: datetime) -> None:
        """Clear a halt. Deliberately manual."""
        self.halt_reason = ""
        self._transition(RiskState.NORMAL, ts, "manually resumed")

    @property
    def is_halted(self) -> bool:
        return self.state is RiskState.HALTED

    @property
    def can_open(self) -> bool:
        return self.state is RiskState.NORMAL

    def status(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "halt_reason": self.halt_reason,
            "vetoes": self.vetoes,
            "adjustments": self.adjustments,
            "cooldowns": sorted(self._cooldown),
        }

    def reset(self) -> None:
        self.state = RiskState.NORMAL
        self.halt_reason = ""
        self.events.clear()
        self.vetoes = 0
        self.adjustments = 0
        self._cooldown.clear()
        self._bar_index = 0
        self._day = None
