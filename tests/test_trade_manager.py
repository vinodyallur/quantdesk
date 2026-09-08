"""Trade management: stops must be inviolable, and winners must be allowed to run."""

from __future__ import annotations

import pytest
from conftest import make_bar

from quantdesk.analysis.patterns import Bias
from quantdesk.trade_manager import TradeAction, TradeManager


#: Must match the symbol ``make_bar`` produces, or the manager looks up nothing.
SYMBOL = "TEST"


def opened(tm: TradeManager, side: int = 1, entry: float = 100.0, stop: float = 90.0,
           objective: float = 130.0, qty: float = 10.0):
    return tm.open_trade(
        SYMBOL, side=side, entry=entry, stop=stop, objective=objective,
        qty=qty, index=0, ts=make_bar(0, 100, 100, 100, 100).ts,
    )


def test_stop_is_honoured_even_when_the_signal_still_agrees():
    """A protective stop a fresh signal can override is not a stop."""
    tm = TradeManager()
    opened(tm)
    # Bar pierces 90 but closes back above it. The loss happened.
    bar = make_bar(1, 95.0, 96.0, 89.0, 95.0)
    decision = tm.decide(bar, 1, Bias.BULLISH, 1.0)
    assert decision.action is TradeAction.EXIT_STOP


def test_stop_uses_the_bar_low_not_the_close():
    """Closing back above a pierced stop must not rescue the trade."""
    tm = TradeManager()
    trade = opened(tm)
    assert trade.stop_hit(make_bar(1, 95.0, 96.0, 89.9, 95.0)) is True
    assert trade.stop_hit(make_bar(1, 95.0, 96.0, 90.1, 95.0)) is False


def test_gap_through_the_stop_fills_at_the_open_not_the_stop():
    """Assuming the stop price on a gap flatters exactly the worst trades."""
    tm = TradeManager()
    opened(tm)
    gapped = make_bar(1, 80.0, 81.0, 79.0, 80.5)
    decision = tm.decide(gapped, 1, Bias.BULLISH, 1.0)
    assert decision.action is TradeAction.EXIT_STOP
    assert decision.exit_price == pytest.approx(80.0), "should fill at the gapped open"


def test_target_exit_when_objective_is_reached():
    tm = TradeManager()
    opened(tm)
    decision = tm.decide(make_bar(1, 125.0, 131.0, 124.0, 130.0), 1, Bias.BULLISH, 1.0)
    assert decision.action is TradeAction.EXIT_TARGET
    assert decision.exit_price == pytest.approx(130.0)


def test_signal_reversal_closes_the_trade():
    tm = TradeManager()
    opened(tm)
    decision = tm.decide(make_bar(1, 101.0, 102.0, 100.0, 101.0), 1, Bias.BEARISH, 1.0)
    assert decision.action is TradeAction.EXIT_SIGNAL


def test_holding_is_the_default_and_is_a_real_decision():
    tm = TradeManager()
    opened(tm)
    decision = tm.decide(make_bar(1, 101.0, 102.0, 100.0, 101.0), 1, Bias.BULLISH, 1.0)
    assert decision.action is TradeAction.HOLD


def test_breakeven_ratchet_waits_long_enough_to_let_profits_run():
    """Moving to breakeven at 1R against a 3R target scratches most winners.

    Initial risk here is 10 (100 -> 90). At +1R (110) the stop must NOT have moved;
    only at the configured 1.5R (115) should it.
    """
    tm = TradeManager(breakeven_at_r=1.5)
    trade = opened(tm)

    tm.decide(make_bar(1, 109.0, 110.0, 108.0, 110.0), 1, Bias.BULLISH, 1.0)
    assert trade.stop == pytest.approx(90.0), "1R is too early to protect"

    tm.decide(make_bar(2, 114.0, 115.0, 113.0, 115.0), 1, Bias.BULLISH, 1.0)
    assert trade.stop == pytest.approx(100.0), "1.5R should move the stop to breakeven"
    assert trade.stop_moved_to_breakeven


def test_stops_only_ever_tighten():
    """A stop that can widen is not a risk limit."""
    tm = TradeManager(breakeven_at_r=1.5, trail_after_r=2.0, trail_distance_r=1.5)
    trade = opened(tm)
    tm.decide(make_bar(1, 129.0, 129.0, 128.0, 129.0), 1, Bias.BULLISH, 1.0)
    tightened = trade.stop
    assert tightened > 90.0
    # Price falls back; the stop must not follow it down.
    tm.decide(make_bar(2, 112.0, 113.0, 111.0, 112.0), 1, Bias.BULLISH, 1.0)
    assert trade.stop == pytest.approx(tightened)


def test_trailing_uses_the_high_water_mark():
    tm = TradeManager(breakeven_at_r=1.5, trail_after_r=2.0, trail_distance_r=1.5)
    trade = opened(tm)
    # Reaches 125 = 2.5R. Trail sits 1.5R (15) behind the high water mark.
    tm.decide(make_bar(1, 120.0, 125.0, 119.0, 124.0), 1, Bias.BULLISH, 1.0)
    assert trade.high_water == pytest.approx(125.0)
    assert trade.stop == pytest.approx(110.0)


def test_pyramiding_only_into_a_winner():
    tm = TradeManager(add_at_r=1.0, breakeven_at_r=99.0)
    opened(tm)
    # +1R: eligible to add.
    add = tm.decide(make_bar(1, 109.0, 111.0, 108.0, 110.0), 1, Bias.BULLISH, 1.0)
    assert add.action is TradeAction.ADD
    # Slightly offside: not eligible.
    tm2 = TradeManager(add_at_r=1.0)
    opened(tm2)
    hold = tm2.decide(make_bar(1, 99.0, 100.0, 98.0, 99.0), 1, Bias.BULLISH, 1.0)
    assert hold.action is TradeAction.HOLD


def test_pyramid_layer_moves_stop_to_breakeven():
    """Murphy: "Adjust protective stops to the breakeven point."""
    tm = TradeManager()
    opened(tm)
    trade = tm.add_layer(SYMBOL, qty=5.0, price=112.0)
    assert trade.layer == 2
    assert trade.qty == pytest.approx(15.0)
    # Weighted average entry of 10 @ 100 and 5 @ 112.
    assert trade.entry == pytest.approx(104.0)
    assert trade.stop == pytest.approx(104.0)


def test_layers_are_capped():
    tm = TradeManager(max_layers=2, add_at_r=1.0, breakeven_at_r=99.0)
    trade = opened(tm)
    trade.layer = 2
    decision = tm.decide(make_bar(1, 109.0, 111.0, 108.0, 110.0), 1, Bias.BULLISH, 1.0)
    assert decision.action is TradeAction.HOLD, "should not add beyond the cap"


def test_short_side_is_symmetric():
    tm = TradeManager()
    opened(tm, side=-1, entry=100.0, stop=110.0, objective=70.0)
    stopped = tm.decide(make_bar(1, 105.0, 111.0, 104.0, 105.0), 1, Bias.BEARISH, 1.0)
    assert stopped.action is TradeAction.EXIT_STOP

    tm2 = TradeManager()
    opened(tm2, side=-1, entry=100.0, stop=110.0, objective=70.0)
    won = tm2.decide(make_bar(1, 75.0, 76.0, 69.0, 70.0), 1, Bias.BEARISH, 1.0)
    assert won.action is TradeAction.EXIT_TARGET


def test_r_multiple_is_scale_free():
    tm = TradeManager()
    trade = opened(tm, entry=100.0, stop=90.0)
    assert trade.r_multiple(110.0) == pytest.approx(1.0)
    assert trade.r_multiple(130.0) == pytest.approx(3.0)
    assert trade.r_multiple(95.0) == pytest.approx(-0.5)


def test_no_trade_and_no_signal_does_nothing():
    tm = TradeManager()
    decision = tm.decide(make_bar(1, 100, 101, 99, 100), 1, Bias.NEUTRAL, 0.0)
    assert decision.action is TradeAction.NONE


def test_time_stop_when_configured():
    tm = TradeManager(max_bars_held=5)
    opened(tm)
    decision = tm.decide(make_bar(6, 101, 102, 100, 101), 6, Bias.BULLISH, 1.0)
    assert decision.action is TradeAction.EXIT_SIGNAL
    assert "time stop" in decision.reason
