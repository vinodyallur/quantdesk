"""Trade management: entries, stops, targets, pyramiding, exits.

This exists because of a mismatch that a first backtest made obvious. A
continuously-rebalancing portfolio system and Murphy's method are different
animals. Rebalancing adjusts a target weight every bar; his method takes a
position, defines where it is wrong, and then leaves it alone until either the stop
or the objective is reached. Running the second idea through the first machinery
produces nine fills per trade, pays the spread on every one of them, and cuts
winners short - which is the exact opposite of "letting profits run".

Worse, sizing a trade against a stop and then never enforcing that stop means the
5% risk cap is fiction. The stop is what makes the risk number true.

So this module holds the trade, and the answer to "what now?" is one of a small set
of discrete actions rather than a new target weight:

* **OPEN** - flat, and there is a case.
* **HOLD** - in a position and nothing has changed. The default, and by far the most
  common answer. Doing nothing is a decision.
* **ADD** - Murphy's pyramiding: only into a winner, each layer smaller, stop moved
  to breakeven.
* **EXIT_STOP** - price traded through the protective stop. Non-negotiable.
* **EXIT_TARGET** - the measured objective was reached.
* **EXIT_SIGNAL** - the reason for the trade has gone.

Stop checks come first, before anything else, because a stop that can be
out-voted by a fresh signal is not a stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from quantdesk.analysis.patterns import Bias
from quantdesk.core.types import Bar


class TradeAction(str, Enum):
    NONE = "none"
    OPEN = "open"
    ADD = "add"
    HOLD = "hold"
    EXIT_STOP = "exit_stop"
    EXIT_TARGET = "exit_target"
    EXIT_SIGNAL = "exit_signal"

    @property
    def is_exit(self) -> bool:
        return self in (
            TradeAction.EXIT_STOP,
            TradeAction.EXIT_TARGET,
            TradeAction.EXIT_SIGNAL,
        )

    @property
    def is_entry(self) -> bool:
        return self in (TradeAction.OPEN, TradeAction.ADD)


@dataclass
class ActiveTrade:
    """A position the desk is managing, with the levels that govern it."""

    symbol: str
    side: int
    """+1 long, -1 short."""
    entry: float
    stop: float
    objective: float
    qty: float
    opened_index: int
    opened_ts: datetime
    layer: int = 1
    #: Best price reached since entry, for trailing and for breakeven logic.
    high_water: float = 0.0
    initial_risk: float = 0.0
    reason: str = ""
    stop_moved_to_breakeven: bool = False

    def __post_init__(self) -> None:
        self.high_water = self.entry
        self.initial_risk = abs(self.entry - self.stop)

    @property
    def long_side(self) -> bool:
        return self.side > 0

    def unrealized(self, price: float) -> float:
        return (price - self.entry) * self.qty * self.side

    def r_multiple(self, price: float) -> float:
        """Profit in units of initial risk. The only scale-free way to compare trades."""
        if self.initial_risk <= 0:
            return 0.0
        return (price - self.entry) * self.side / self.initial_risk

    def stop_hit(self, bar: Bar) -> bool:
        """Did the bar trade through the stop?

        Uses the bar's extreme, not its close. A bar whose low pierced the stop and
        closed back above it still took the loss in reality, and pretending otherwise
        is how a backtest quietly removes its own worst trades.
        """
        return bar.low <= self.stop if self.long_side else bar.high >= self.stop

    def target_hit(self, bar: Bar) -> bool:
        if self.objective <= 0:
            return False
        return bar.high >= self.objective if self.long_side else bar.low <= self.objective

    def update_high_water(self, bar: Bar) -> None:
        self.high_water = (
            max(self.high_water, bar.high) if self.long_side else min(self.high_water, bar.low)
        )


@dataclass
class TradeDecision:
    """What to do about one symbol on this bar."""

    action: TradeAction
    symbol: str
    reason: str = ""
    qty: float = 0.0
    """Quantity to trade. For exits this is the whole position."""
    trade: ActiveTrade | None = None
    exit_price: float = 0.0
    """Price the exit is assumed to occur at, for stop and target fills."""


class TradeManager:
    """Holds the active trades and decides entries, adds and exits."""

    def __init__(
        self,
        max_layers: int = 3,
        breakeven_at_r: float = 1.5,
        add_at_r: float = 1.0,
        trail_after_r: float = 2.0,
        trail_distance_r: float = 1.5,
        exit_on_reversal: bool = True,
        max_bars_held: int = 0,
    ) -> None:
        self.max_layers = max_layers
        #: Profit in R before the stop moves to breakeven. Set too low this destroys
        #: the strategy: with a 3R target, moving to breakeven at 1R turns most
        #: would-be winners into scratches, and the surviving wins are small. A
        #: year-long backtest showed a 1:1 payoff against a 3:1 entry gate for exactly
        #: this reason. Murphy's maxim is to let profits run, and a premature
        #: breakeven stop is the most common way desks violate it while believing they
        #: are being prudent.
        self.breakeven_at_r = breakeven_at_r
        #: Profit in R before a pyramid layer is considered. Murphy says add only to
        #: winners; this quantifies "winner" so a trade a tick onside does not qualify.
        self.add_at_r = add_at_r
        #: Once this far onside, trail the stop behind the high water mark.
        self.trail_after_r = trail_after_r
        #: How far behind the high water mark to trail, in units of initial risk. Wide
        #: enough that ordinary retracement does not end the trade.
        self.trail_distance_r = trail_distance_r
        self.exit_on_reversal = exit_on_reversal
        #: Optional time stop. 0 disables it.
        self.max_bars_held = max_bars_held
        self.trades: dict[str, ActiveTrade] = {}
        self.closed: list[ActiveTrade] = []

    # ---------------------------------------------------------------- decide
    def decide(
        self,
        bar: Bar,
        index: int,
        signal_bias: Bias,
        conviction: float,
    ) -> TradeDecision:
        """Decide what to do about this symbol, stops first."""
        trade = self.trades.get(bar.symbol)

        if trade is None:
            if signal_bias is Bias.NEUTRAL or conviction <= 0:
                return TradeDecision(TradeAction.NONE, bar.symbol, "no case to open")
            return TradeDecision(
                TradeAction.OPEN, bar.symbol, f"{signal_bias.value} case", 0.0
            )

        # --- exits, in priority order. The stop outranks everything, including a
        # fresh signal in the same direction: a level that can be talked out of is
        # not a protective stop.
        if trade.stop_hit(bar):
            return TradeDecision(
                TradeAction.EXIT_STOP,
                bar.symbol,
                f"stop {trade.stop:.6g} hit",
                qty=trade.qty,
                trade=trade,
                # Assume the stop price, not the close. A gap through the stop fills
                # worse, and the open is the honest estimate in that case.
                exit_price=self._stop_fill_price(trade, bar),
            )

        if trade.target_hit(bar):
            return TradeDecision(
                TradeAction.EXIT_TARGET,
                bar.symbol,
                f"objective {trade.objective:.6g} reached",
                qty=trade.qty,
                trade=trade,
                exit_price=trade.objective,
            )

        trade_bias = Bias.BULLISH if trade.long_side else Bias.BEARISH
        if self.exit_on_reversal and signal_bias is trade_bias.opposite:
            return TradeDecision(
                TradeAction.EXIT_SIGNAL,
                bar.symbol,
                f"signal flipped to {signal_bias.value}",
                qty=trade.qty,
                trade=trade,
                exit_price=bar.close,
            )

        if self.max_bars_held and index - trade.opened_index >= self.max_bars_held:
            return TradeDecision(
                TradeAction.EXIT_SIGNAL,
                bar.symbol,
                f"time stop after {self.max_bars_held} bars",
                qty=trade.qty,
                trade=trade,
                exit_price=bar.close,
            )

        # --- still in. Ratchet the stop, then consider adding.
        trade.update_high_water(bar)
        self._ratchet_stop(trade, bar)

        r = trade.r_multiple(bar.close)
        if (
            signal_bias is trade_bias
            and trade.layer < self.max_layers
            and r >= self.add_at_r
        ):
            return TradeDecision(
                TradeAction.ADD,
                bar.symbol,
                f"pyramiding into a winner at {r:.1f}R",
                trade=trade,
            )

        return TradeDecision(
            TradeAction.HOLD, bar.symbol, f"holding at {r:+.1f}R", trade=trade
        )

    def _stop_fill_price(self, trade: ActiveTrade, bar: Bar) -> float:
        """Where a stop realistically fills.

        If the bar opened beyond the stop, the market gapped and the fill is the open,
        not the stop level. Assuming the stop price on a gap flatters the result on
        exactly the trades that hurt most.
        """
        if trade.long_side:
            return min(trade.stop, bar.open) if bar.open < trade.stop else trade.stop
        return max(trade.stop, bar.open) if bar.open > trade.stop else trade.stop

    def _ratchet_stop(self, trade: ActiveTrade, bar: Bar) -> None:
        """Move the stop up (never down) as the trade goes right.

        Two stages: breakeven once a full R is banked, then trailing behind the high
        water mark. Only ever tightened - a stop that can widen is not a risk limit.
        """
        # Measured on the high water mark, not the close: a trade that reached 2R and
        # pulled back has still earned its ratchet, and re-testing on the close makes
        # the stop oscillate.
        r = trade.r_multiple(trade.high_water)
        if r >= self.breakeven_at_r and not trade.stop_moved_to_breakeven:
            if self._is_tighter(trade, trade.entry):
                trade.stop = trade.entry
                trade.stop_moved_to_breakeven = True

        if r >= self.trail_after_r and trade.initial_risk > 0:
            offset = self.trail_distance_r * trade.initial_risk
            trail = (
                trade.high_water - offset if trade.long_side else trade.high_water + offset
            )
            if self._is_tighter(trade, trail):
                trade.stop = trail

    def _is_tighter(self, trade: ActiveTrade, candidate: float) -> bool:
        return candidate > trade.stop if trade.long_side else candidate < trade.stop

    # ----------------------------------------------------------------- record
    def open_trade(
        self,
        symbol: str,
        side: int,
        entry: float,
        stop: float,
        objective: float,
        qty: float,
        index: int,
        ts: datetime,
        reason: str = "",
    ) -> ActiveTrade:
        trade = ActiveTrade(
            symbol=symbol,
            side=side,
            entry=entry,
            stop=stop,
            objective=objective,
            qty=qty,
            opened_index=index,
            opened_ts=ts,
            reason=reason,
        )
        self.trades[symbol] = trade
        return trade

    def add_layer(self, symbol: str, qty: float, price: float) -> ActiveTrade | None:
        """Record a pyramid layer and move the stop to breakeven, per Murphy."""
        trade = self.trades.get(symbol)
        if trade is None:
            return None
        total = trade.qty + qty
        if total <= 0:
            return trade
        trade.entry = (trade.entry * trade.qty + price * qty) / total
        trade.qty = total
        trade.layer += 1
        # "Adjust protective stops to the breakeven point."
        if self._is_tighter(trade, trade.entry):
            trade.stop = trade.entry
            trade.stop_moved_to_breakeven = True
        return trade

    def close_trade(self, symbol: str) -> ActiveTrade | None:
        trade = self.trades.pop(symbol, None)
        if trade is not None:
            self.closed.append(trade)
            if len(self.closed) > 2000:
                del self.closed[:-2000]
        return trade

    # -------------------------------------------------------------- accessors
    def active(self, symbol: str) -> ActiveTrade | None:
        return self.trades.get(symbol)

    @property
    def open_count(self) -> int:
        return len(self.trades)

    def summary(self) -> list[str]:
        return [
            f"{t.symbol:<10} {'long' if t.long_side else 'short':<5} "
            f"{t.qty:>12,.6g} @ {t.entry:.6g} stop {t.stop:.6g} "
            f"target {t.objective:.6g} layer {t.layer}"
            for t in self.trades.values()
        ]
