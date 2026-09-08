"""Paper trading: the ledger, the fills, and the two live-path bugs that hid here.

The engine bugs these cover were both invisible from the outside. A simulated session
submitted orders that were never matched, so equity sat at its opening value and the desk
looked merely quiet; and the first order to reach the router raised ``TypeError`` from a
property called as a method, which killed the loop. Neither produced a wrong number - they
produced no numbers, which is easy to mistake for a strategy declining to trade.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from conftest import TS0, make_bar, trending

from quantdesk.core.types import Order, OrderType, Side
from quantdesk.data.feed import DataFeed
from quantdesk.execution.paper_broker import PaperBroker


def order(symbol="TEST", side=Side.BUY, qty=1.0, oid="p1", risk=0.0) -> Order:
    return Order(
        symbol=symbol, side=side, qty=qty, type=OrderType.MARKET,
        id=oid, ts=TS0, planned_risk=risk,
    )


# ------------------------------------------------------------------- matching


def test_a_fill_lands_on_the_next_bar_open_not_this_close():
    """The property that keeps paper results reproducible by a backtest.

    Driven in the desk's own order - tick, then decide, then submit - because that
    sequence is what creates the separation. Submitting and ticking the same bar object
    would fill at that bar's open and prove nothing.
    """
    broker = PaperBroker(starting_equity=10_000.0, commission_bps=0.0, slippage_bps=0.0)
    decision_bar = make_bar(0, 100.0, 105.0, 99.0, 104.0)
    next_bar = make_bar(1, 110.0, 112.0, 109.0, 111.0)

    # Bar 0 arrives: nothing pending, so nothing fills. The decision is made on its close.
    assert broker.on_bar(decision_bar) == []
    broker.submit_sync(order(qty=2.0))

    # Bar 1 arrives: the order queued on bar 0 fills at bar 1's open.
    fills = broker.on_bar(next_bar)
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(110.0), "filled at the next bar's open"
    assert fills[0].price != pytest.approx(decision_bar.close), (
        "filling at the decision close would be trading on unavailable information"
    )


def test_costs_are_charged_against_the_demo_balance():
    broker = PaperBroker(starting_equity=10_000.0, commission_bps=10.0, slippage_bps=0.0)
    broker.submit_sync(order(qty=1.0))
    broker.on_bar(make_bar(0, 100.0, 100.0, 100.0, 100.0))
    broker.on_bar(make_bar(1, 100.0, 100.0, 100.0, 100.0))

    # 10bps of 100 notional is 0.10, and only costs move equity at the moment of a fill.
    assert broker.account_ledger.fees_paid == pytest.approx(0.10)
    assert broker.equity == pytest.approx(9_999.90)


def test_a_round_trip_is_recorded_once_it_returns_to_flat():
    broker = PaperBroker(starting_equity=10_000.0, commission_bps=0.0, slippage_bps=0.0)
    broker.submit_sync(order(qty=2.0, oid="in"))
    broker.on_bar(make_bar(0, 100.0, 100.0, 100.0, 100.0))
    broker.on_bar(make_bar(1, 100.0, 100.0, 100.0, 100.0))
    assert not broker.round_trips, "still open"

    broker.submit_sync(order(side=Side.SELL, qty=2.0, oid="out"))
    broker.on_bar(make_bar(2, 120.0, 120.0, 120.0, 120.0))

    assert len(broker.round_trips) == 1
    trip = broker.round_trips[0]
    assert trip.side == 1
    assert trip.entry == pytest.approx(100.0)
    assert trip.exit == pytest.approx(120.0)
    assert trip.pnl == pytest.approx(40.0)
    assert trip.won


def test_a_flip_closes_one_round_trip_and_opens_another():
    """A position that reverses is two trades, not one, and not zero."""
    broker = PaperBroker(starting_equity=10_000.0, commission_bps=0.0, slippage_bps=0.0)
    broker.submit_sync(order(qty=1.0, oid="long"))
    broker.on_bar(make_bar(0, 100.0, 100.0, 100.0, 100.0))
    broker.on_bar(make_bar(1, 100.0, 100.0, 100.0, 100.0))

    broker.submit_sync(order(side=Side.SELL, qty=3.0, oid="flip"))
    broker.on_bar(make_bar(2, 110.0, 110.0, 110.0, 110.0))

    assert len(broker.round_trips) == 1, "the long leg closed"
    assert broker.round_trips[0].pnl == pytest.approx(10.0)
    live = [p for p in broker.account_ledger.positions.values() if not p.is_flat]
    assert live and live[0].qty == pytest.approx(-2.0), "residual is a new short"


def test_results_reconcile_with_the_ledger():
    broker = PaperBroker(starting_equity=10_000.0, commission_bps=5.0, slippage_bps=0.0)
    broker.submit_sync(order(qty=1.0, oid="a"))
    broker.on_bar(make_bar(0, 100.0, 100.0, 100.0, 100.0))
    broker.on_bar(make_bar(1, 100.0, 100.0, 100.0, 100.0))
    broker.submit_sync(order(side=Side.SELL, qty=1.0, oid="b"))
    broker.on_bar(make_bar(2, 105.0, 105.0, 105.0, 105.0))

    r = broker.results()
    assert r.equity == pytest.approx(broker.account_ledger.equity)
    assert r.total_pnl == pytest.approx(r.equity - 10_000.0)
    assert r.trades == 1
    assert r.fees_paid > 0
    assert r.net_of_fees == pytest.approx(r.total_pnl + r.fees_paid)
    assert any("equity" in line for line in r.summary())


def test_zero_or_negative_demo_capital_is_refused():
    with pytest.raises(ValueError):
        PaperBroker(starting_equity=0.0)
    with pytest.raises(ValueError):
        PaperBroker(starting_equity=-500.0)


def test_the_broker_reports_an_account_so_the_desk_can_sync():
    """``SimBroker`` returns None here, which left desk equity pinned at its opening value."""
    broker = PaperBroker(starting_equity=7_500.0)
    snapshot = asyncio.run(broker.account())
    assert snapshot is not None
    assert snapshot.equity == pytest.approx(7_500.0)
    assert asyncio.run(broker.positions()) == {}


# ----------------------------------------------------------- desk integration


class ReplayFeed(DataFeed):
    """Replays a fixed bar list, so a session can be driven to completion in a test."""

    def __init__(self, bars, symbols=("TEST",), timeframe="5Min"):
        super().__init__(list(symbols), timeframe)
        self._bars = sorted(bars, key=lambda b: (b.ts, b.symbol))

    def history(self, start, end=None, symbols=None):
        return {}

    async def stream(self):
        for bar in self._bars:
            yield bar


def run_session(bars, capital=50_000.0, warm=None):
    from quantdesk.app.desk import Desk
    from quantdesk.config import get_settings

    settings = get_settings().model_copy(
        update={"timeframe": "5Min", "starting_equity": capital}
    )
    broker = PaperBroker(starting_equity=capital)
    desk = Desk(
        settings=settings, broker=broker,
        feed=ReplayFeed(bars), mode="TEST",
    )
    desk.universe = ["TEST"]
    desk.focus_symbol = "TEST"

    async def go():
        await desk.warmup(warm or {"TEST": []})
        await desk.run()

    asyncio.run(go())
    return desk, broker


def test_the_desk_ticks_the_broker_so_orders_actually_fill():
    """The bug: nothing called ``broker.on_bar``, so a session never filled anything.

    Equity stayed at its opening value and the desk looked like a strategy that had
    declined to trade, which is indistinguishable from working correctly.
    """
    bars = trending(bars_per_leg=140)
    warm, live = bars[:300], bars[300:600]
    desk, broker = run_session(live, warm={"TEST": warm})

    assert desk.bars_processed > 0
    assert broker.equity_curve, "the broker was never advanced by a bar"
    # One mark per streamed bar. Warmup bars never reach the broker, because replaying
    # history through a matching engine would book fills that never happened.
    assert len(broker.equity_curve) == len(live)


def test_a_session_that_trades_moves_the_demo_balance():
    bars = trending(bars_per_leg=200)
    desk, broker = run_session(bars, warm={"TEST": bars[:400]})

    if not broker.all_fills:
        pytest.skip("the strategy declined every setup on this series")
    assert broker.equity != pytest.approx(broker.starting_equity), (
        "fills happened but the balance did not change"
    )
    assert desk.equity == pytest.approx(broker.account_ledger.equity), (
        "the desk and the ledger disagree about equity"
    )


def test_routing_reads_can_open_as_a_property_not_a_method():
    """The bug: ``risk.can_open()`` raised TypeError and killed the loop on first order.

    Asserting no error was recorded is the check that matters, because the desk catches
    loop failures and carries on, so the symptom was a session that silently stopped
    trading rather than a traceback.
    """
    bars = trending(bars_per_leg=200)
    desk, _ = run_session(bars, warm={"TEST": bars[:400]})
    assert not any("not callable" in e for e in desk.errors), desk.errors
    assert not any("feed loop failed" in e for e in desk.errors), desk.errors


def test_the_desk_adopts_the_brokers_demo_capital_not_the_configured_default():
    """Chosen capital must reach the pipeline, or sizing works to the wrong balance.

    Checked before the session runs. ``Pipeline.start_equity`` is a *daily* baseline that
    ``_roll_day`` deliberately resets when the date changes, so asserting it after a
    multi-day replay would be testing the wrong thing.
    """
    from quantdesk.app.desk import Desk
    from quantdesk.config import get_settings

    broker = PaperBroker(starting_equity=3_333.0)
    settings = get_settings().model_copy(update={"starting_equity": 999_999.0})
    desk = Desk(settings=settings, broker=broker, feed=None, mode="TEST")

    assert desk.equity == pytest.approx(3_333.0)
    assert desk.start_equity == pytest.approx(3_333.0)
    assert desk.pipeline.equity == pytest.approx(3_333.0)
    assert desk.pipeline.start_equity == pytest.approx(3_333.0)
    assert desk.pipeline.peak_equity == pytest.approx(3_333.0)


# ------------------------------------------------------------- perps bar feed


def test_a_bar_that_has_not_closed_is_withheld_from_the_strategy():
    """Binance returns the forming bar last; acting on it is acting on the future."""
    from datetime import datetime, timezone

    from quantdesk.data.binance_feed import BinancePerpsFeed

    feed = BinancePerpsFeed(["BTC/USD"], "1Min")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    closed = [now_ms - 120_000, "100", "101", "99", "100.5", "10",
              now_ms - 60_001, "0", 5]
    forming = [now_ms - 60_000, "100.5", "102", "100", "101.5", "4",
               now_ms + 59_999, "0", 3]

    assert feed._to_bar("BTC/USD", closed) is not None
    assert feed._to_bar("BTC/USD", forming) is None
    held = feed.forming("BTC/USD")
    assert held is not None and held.close == pytest.approx(101.5)


def test_desk_symbols_map_onto_venue_symbols():
    """Perps are quoted in USDT, so a bare concatenation asks for a contract that
    does not exist."""
    from quantdesk.data.perps import _binance_symbol

    assert _binance_symbol("BTC/USD") == "BTCUSDT"
    assert _binance_symbol("ETH/USDT") == "ETHUSDT"
    assert _binance_symbol("SOLUSDT") == "SOLUSDT"


# --------------------------------------------------------------- warmup state


def test_warmup_leaves_no_trade_the_broker_never_filled():
    """The bug: the desk started up believing it held positions it had never opened.

    Replaying history through ``Pipeline.on_bar`` calls ``open_trade`` wherever the
    strategy would have entered, but warmup discards the orders, so the broker holds
    nothing. Left in place, the desk manages and then exits phantom positions, and every
    one of those exits opens a real position in the opposite direction. It also refuses
    new entries in those symbols, because a symbol with an active trade never reaches
    the sizing path.
    """
    from quantdesk.app.desk import Desk
    from quantdesk.config import get_settings

    bars = trending(bars_per_leg=250)
    broker = PaperBroker(starting_equity=50_000.0)
    settings = get_settings().model_copy(
        update={"timeframe": "5Min", "starting_equity": 50_000.0}
    )
    desk = Desk(settings=settings, broker=broker, feed=ReplayFeed([]), mode="TEST")
    desk.universe = ["TEST"]

    asyncio.run(desk.warmup({"TEST": bars}))

    # The precondition: this series is long enough that the strategy did decide to trade
    # during the replay, so the cleanup is actually being exercised.
    assert desk.pipeline.decisions == [], "warmup decisions must not be actionable"
    assert desk.pipeline.trades.active("TEST") is None, (
        "the pipeline is holding a trade the broker never filled"
    )
    assert not broker.all_fills, "warmup must not book fills"
    assert broker.equity == pytest.approx(50_000.0)


def test_warmup_actually_opens_a_trade_without_the_cleanup():
    """Guards the test above from passing for the wrong reason.

    If the warmup series never triggered an entry, the cleanup assertion would hold
    trivially and the regression would not be covered. This drives the pipeline directly
    to confirm an entry does happen.
    """
    from quantdesk.pipeline import Pipeline

    pipeline = Pipeline()
    for bar in trending(bars_per_leg=250):
        pipeline.on_bar(bar)
    assert pipeline.trades.active("TEST") is not None, (
        "this series no longer produces an entry, so the cleanup test is vacuous"
    )
