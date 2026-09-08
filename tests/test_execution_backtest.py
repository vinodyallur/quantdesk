"""Account ledger, order router gating, and backtest lookahead safety."""

from __future__ import annotations

import asyncio

import pytest
from conftest import TS0, make_bar, trending, walk

from quantdesk.backtest import Account, Backtester
from quantdesk.core.types import (
    Fill,
    Order,
    OrderStatus,
    Position,
    Side,
)
from quantdesk.execution.broker import Broker
from quantdesk.execution.router import OrderRouter

# --------------------------------------------------------------------- account


def fill(symbol="BTC/USD", side=Side.BUY, qty=1.0, price=100.0, commission=0.0):
    return Fill(
        order_id="o1", symbol=symbol, side=side, qty=qty, price=price,
        ts=TS0, commission=commission,
    )


def test_buying_moves_cash_not_equity():
    """A fill changes what you own, not what you are worth - only costs do that."""
    acct = Account(starting_equity=100_000.0)
    acct.apply(fill(qty=10.0, price=100.0))
    acct.mark("BTC/USD", 100.0)
    assert acct.cash == pytest.approx(99_000.0)
    assert acct.position_value == pytest.approx(1_000.0)
    assert acct.equity == pytest.approx(100_000.0)


def test_commission_is_the_only_immediate_equity_drag():
    acct = Account(starting_equity=100_000.0)
    acct.apply(fill(qty=10.0, price=100.0, commission=7.5))
    acct.mark("BTC/USD", 100.0)
    assert acct.equity == pytest.approx(99_992.5)
    assert acct.fees_paid == pytest.approx(7.5)


def test_short_sale_credits_cash_and_creates_negative_position():
    acct = Account(starting_equity=100_000.0)
    acct.apply(fill(side=Side.SELL, qty=5.0, price=200.0))
    acct.mark("BTC/USD", 200.0)
    assert acct.cash == pytest.approx(101_000.0)
    assert acct.positions["BTC/USD"].qty == pytest.approx(-5.0)
    assert acct.equity == pytest.approx(100_000.0)
    assert acct.gross_exposure == pytest.approx(1_000.0)
    assert acct.net_exposure == pytest.approx(-1_000.0)


def test_round_trip_realises_pnl_into_equity():
    acct = Account(starting_equity=100_000.0)
    acct.apply(fill(qty=10.0, price=100.0))
    acct.apply(fill(side=Side.SELL, qty=10.0, price=110.0))
    assert acct.realized_pnl == pytest.approx(100.0)
    assert acct.equity == pytest.approx(100_100.0)
    assert acct.open_positions == 0


def test_mark_to_market_moves_equity():
    acct = Account(starting_equity=100_000.0)
    acct.apply(fill(qty=10.0, price=100.0))
    acct.mark("BTC/USD", 130.0)
    assert acct.equity == pytest.approx(100_300.0)
    assert acct.unrealized_pnl == pytest.approx(300.0)


# ---------------------------------------------------------------------- router


class FakeBroker(Broker):
    is_simulated = True

    def __init__(self, reject: bool = False):
        self.reject = reject
        self.submitted: list[Order] = []
        self.cancelled = 0

    async def submit(self, order: Order) -> Order:
        self.submitted.append(order)
        order.status = OrderStatus.REJECTED if self.reject else OrderStatus.SUBMITTED
        if self.reject:
            order.reject_reason = "nope"
        return order

    async def cancel_all(self) -> int:
        self.cancelled += 1
        return 2

    def drain_fills(self):
        return []


def order(symbol="BTC/USD", side=Side.BUY, qty=1.0, oid="o1") -> Order:
    return Order(symbol=symbol, side=side, qty=qty, id=oid, ts=TS0)


def test_router_allows_only_one_working_order_per_symbol():
    """Without this a persistent signal stacks duplicate positions every bar."""
    broker = FakeBroker()
    router = OrderRouter(broker)
    first = asyncio.run(router.route([order(oid="a")], now=TS0))
    second = asyncio.run(router.route([order(oid="b")], now=TS0))

    assert len(first.sent) == 1
    assert not second.sent
    assert "already working" in second.blocked[0][1]
    assert len(broker.submitted) == 1


def test_closing_orders_pass_even_when_opening_is_disallowed():
    """A halted desk must still be able to get out."""
    broker = FakeBroker()
    router = OrderRouter(broker)
    positions = {"BTC/USD": Position(symbol="BTC/USD", qty=5.0, avg_price=100.0)}

    blocked = asyncio.run(
        router.route([order(side=Side.BUY, qty=1.0)], positions, allow_opening=False, now=TS0)
    )
    assert not blocked.sent

    router2 = OrderRouter(FakeBroker())
    allowed = asyncio.run(
        router2.route([order(side=Side.SELL, qty=5.0)], positions, allow_opening=False, now=TS0)
    )
    assert allowed.sent, "reducing exposure must never be blocked"


def test_repeated_rejections_put_a_symbol_in_timeout():
    broker = FakeBroker(reject=True)
    router = OrderRouter(broker, max_rejects=2)
    for i in range(2):
        asyncio.run(router.route([order(oid=f"o{i}")], now=TS0))
    assert "BTC/USD" in router.blocked_symbols

    after = asyncio.run(router.route([order(oid="o9")], now=TS0))
    assert "timeout" in after.blocked[0][1]


def test_stale_working_order_is_released():
    """A missed fill must not silence a symbol permanently."""
    from datetime import timedelta

    broker = FakeBroker()
    router = OrderRouter(broker, stale_after_seconds=60.0)
    asyncio.run(router.route([order(oid="a")], now=TS0))
    assert router.working

    later = asyncio.run(router.route([order(oid="b")], now=TS0 + timedelta(seconds=120)))
    assert later.sent, "the stale order should have been released"


def test_kill_switch_cancels_and_clears():
    broker = FakeBroker()
    router = OrderRouter(broker)
    asyncio.run(router.route([order()], now=TS0))
    detail = asyncio.run(router.kill())
    assert broker.cancelled == 1
    assert not router.working
    assert "cancelled" in detail


# -------------------------------------------------------------------- backtest


def test_backtest_fills_on_the_next_bar_not_the_signal_bar():
    """The decision bar's close must never be the fill price.

    Filling on the same bar you decided means trading on information you did not
    have. Every fill should match some later bar's open, adjusted for slippage.
    """
    bars = trending(bars_per_leg=200)
    bt = Backtester()
    bt.run(bars)
    assert bt.fills, "no fills to check"

    by_ts = {}
    for bar in bars:
        by_ts.setdefault(bar.ts, bar)

    for f in bt.fills:
        bar = by_ts.get(f.ts)
        assert bar is not None, "fill timestamp does not match any bar"
        # Slippage moves the price off the open, so allow a small band, but the fill
        # must be anchored to the open rather than to that bar's close.
        assert abs(f.price - bar.open) / bar.open < 0.01


def test_backtest_equity_curve_covers_every_bar():
    bars = trending(bars_per_leg=150)
    bt = Backtester()
    result = bt.run(bars)
    assert result.bars == len(bars)
    assert len(result.equity_curve) == len(bars)


def test_backtest_records_why_trades_were_declined():
    """Refusals are results. A strategy that declines everything must say so."""
    result = Backtester().run(trending(bars_per_leg=150))
    assert result.rejections
    assert any("reward-to-risk" in r for r in result.rejections)


def test_backtest_is_deterministic():
    bars = trending(bars_per_leg=120)
    first = Backtester().run(bars)
    second = Backtester().run(bars)
    assert first.final_equity == pytest.approx(second.final_equity)
    assert first.trades == second.trades


def test_multi_symbol_bars_are_replayed_in_timestamp_order():
    """Symbols must advance together or the cross-sectional agents can peek ahead."""
    a = walk([100, 140, 120, 170], 60, symbol="AAA")
    b = walk([50, 40, 55, 45], 60, symbol="BBB")
    bt = Backtester()
    bt.run({"AAA": a, "BBB": b})
    stamps = [p.ts for p in bt.equity_curve]
    assert stamps == sorted(stamps)


def test_empty_backtest_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError):
        Backtester().run([])


# ------------------------------------------------------------ risk accounting


def test_planned_risk_travels_from_order_to_trade_record():
    """Every unit of risk an order carried must land on exactly one trade record.

    Reconstructing risk downstream from a fill under-counts partial fills and pyramid
    layers, and reads the wrong trade once an exit has been booked. The invariant that
    catches all of those at once is conservation: what the orders planned is what the
    trades recorded.
    """
    bt = Backtester()
    result = bt.run(trending(bars_per_leg=200))
    assert result.trade_list, "no trades to check"

    planned = sum(o.planned_risk for o in bt.broker.orders)
    recorded = sum(t.initial_risk for t in result.trade_list)
    assert planned > 0, "orders carried no risk at all"
    assert recorded == pytest.approx(planned, rel=1e-9)


def test_exits_carry_no_planned_risk():
    """Closing a position puts no new risk on, so an exit must not add any."""
    bt = Backtester()
    bt.run(trending(bars_per_leg=200))
    exits = [o for o in bt.broker.orders if o.tag.startswith("exit|")]
    assert exits, "no exit orders were produced"
    assert all(o.planned_risk == 0.0 for o in exits)


def test_closed_trades_remember_how_big_they_were():
    """``qty`` is driven to zero as a position closes, so size needs its own field."""
    result = Backtester().run(trending(bars_per_leg=200))
    assert result.trade_list
    assert all(t.peak_qty > 0 for t in result.trade_list)


def test_no_trade_is_left_without_a_risk_denominator():
    """A trade with no recorded risk reports 0.0R, indistinguishable from a scratch."""
    result = Backtester().run(trending(bars_per_leg=200))
    assert result.trades > 0
    assert result.trades_without_risk == 0


def test_risk_weighted_expectancy_agrees_in_sign_with_the_money():
    """The invariant that makes the two reported expectancies impossible to contradict.

    A plain average of R over trades that risked wildly different amounts can be
    positive while the account loses money; that is arithmetic, not an edge. Weighting
    each trade by the risk it took is the only aggregate that cannot disagree with the
    P&L, because it *is* total P&L over total risk.
    """
    result = Backtester().run(trending(bars_per_leg=200))
    measurable = [t for t in result.trade_list if t.risk_known]
    assert measurable, "no measurable trades"

    net = sum(t.pnl for t in measurable)
    total_risk = sum(t.initial_risk for t in measurable)
    assert result.risk_weighted_r == pytest.approx(net / total_risk)
    assert (result.risk_weighted_r > 0) == (net > 0)


def test_uneven_risk_is_reported_rather_than_hidden():
    """Dispersion is what tells a reader whether per-trade R means anything."""
    result = Backtester().run(trending(bars_per_leg=200))
    assert result.min_risk > 0
    assert result.max_risk >= result.min_risk
    assert result.risk_dispersion == pytest.approx(result.max_risk / result.min_risk)
    if result.risk_dispersion > 10.0:
        assert not result.r_is_trustworthy
        assert any("WARNING" in line for line in result.summary())


def test_risk_weighted_and_mean_r_coincide_when_risk_is_even():
    """Sanity check on the weighting: equal denominators make the two identical."""
    from quantdesk.backtest.engine import TradeRecord
    from quantdesk.backtest.metrics import compute_metrics

    trades = [
        TradeRecord(symbol="X", opened_at=TS0, closed_at=TS0, pnl=pnl, initial_risk=100.0)
        for pnl in (300.0, -100.0, -100.0, 250.0, -100.0)
    ]
    result = compute_metrics(
        equity_curve=[], trades=trades, fills=[], decisions=[],
        bars_per_year=105_120.0, starting_equity=100_000.0,
    )
    assert result.mean_r == pytest.approx(result.risk_weighted_r)
    assert result.risk_dispersion == pytest.approx(1.0)
    assert result.r_is_trustworthy


def test_a_tiny_denominator_cannot_dominate_the_risk_weighted_figure():
    """The exact artefact this accounting exists to prevent.

    One trade risking almost nothing and winning gives a huge R. The unweighted mean
    turns positive on the strength of that single trade while the account is down; the
    risk-weighted figure stays negative, matching the money.
    """
    from quantdesk.backtest.engine import TradeRecord
    from quantdesk.backtest.metrics import compute_metrics

    trades = [
        TradeRecord(symbol="X", opened_at=TS0, closed_at=TS0, pnl=30.0, initial_risk=0.1),
        TradeRecord(symbol="X", opened_at=TS0, closed_at=TS0, pnl=-500.0, initial_risk=500.0),
        TradeRecord(symbol="X", opened_at=TS0, closed_at=TS0, pnl=-500.0, initial_risk=500.0),
    ]
    result = compute_metrics(
        equity_curve=[], trades=trades, fills=[], decisions=[],
        bars_per_year=105_120.0, starting_equity=100_000.0,
    )
    assert result.mean_r > 0, "the unweighted mean is fooled, as expected"
    assert result.risk_weighted_r < 0, "the risk-weighted figure must follow the money"
    assert not result.r_is_trustworthy


# ------------------------------------------------------- annualising guardrail


def test_annualising_a_tiny_sample_does_not_crash():
    """``**`` raises OverflowError rather than returning infinity.

    Annualising a two-bar sample of 5-minute crypto data raises the growth factor to the
    power of ~100,000. The direct form crashed metrics entirely for short runs, which is a
    worse outcome than reporting an absurd number as absurd.
    """
    from quantdesk.backtest.metrics import EquityPoint, compute_metrics

    curve = [
        EquityPoint(ts=TS0, equity=100_000.0),
        EquityPoint(ts=TS0, equity=101_000.0),
    ]
    result = compute_metrics(
        equity_curve=curve, trades=[], fills=[], decisions=[],
        bars_per_year=105_120.0, starting_equity=100_000.0,
    )
    assert result.total_return == pytest.approx(0.01)
    assert result.annualised_return == float("inf")


def test_annualising_is_correct_over_a_real_window():
    from quantdesk.backtest.metrics import _annualise

    # Doubling over exactly one year is a 100% annual return.
    assert _annualise(2.0, 1.0) == pytest.approx(1.0)
    # Doubling over two years compounds at about 41.4%.
    assert _annualise(2.0, 2.0) == pytest.approx(2 ** 0.5 - 1.0)
    # Degenerate inputs return zero rather than raising.
    assert _annualise(0.0, 1.0) == 0.0
    assert _annualise(1.5, 0.0) == 0.0


def test_a_total_loss_annualises_to_minus_one_not_an_exception():
    from quantdesk.backtest.metrics import _annualise

    assert _annualise(1e-300, 1e-6) == -1.0
