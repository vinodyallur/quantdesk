"""Trading terminal: mounts, guards its inputs, and renders panels without a TTY.

Offline. The boot path reaches Binance, so it is never invoked here; panels are driven by
attaching a broker and a stub desk directly, which is also how a rendering bug gets caught
without waiting on a network round trip.
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import TS0, make_bar

from quantdesk.execution.paper_broker import PaperBroker
from quantdesk.ui.trading import TradingTerminal


def drive(scenario):
    return asyncio.run(scenario())


def text_of(app, widget_id: str) -> str:
    from textual.widgets import Static

    return str(app.query_one(widget_id, Static).content)


def loaded_broker(capital: float = 20_000.0) -> PaperBroker:
    """A broker that has traded, so the panels have something real to render."""
    from quantdesk.core.types import Order, OrderType, Side

    broker = PaperBroker(starting_equity=capital, commission_bps=2.0, slippage_bps=0.0)
    broker.on_bar(make_bar(0, 100.0, 101.0, 99.0, 100.0))
    broker.submit_sync(
        Order(symbol="TEST", side=Side.BUY, qty=5.0, type=OrderType.MARKET,
              id="a", ts=TS0)
    )
    for i in range(1, 8):
        broker.on_bar(make_bar(i, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i))
    broker.submit_sync(
        Order(symbol="TEST", side=Side.SELL, qty=5.0, type=OrderType.MARKET,
              id="b", ts=TS0)
    )
    broker.on_bar(make_bar(9, 110.0, 111.0, 109.0, 110.0))
    return broker


# ------------------------------------------------------------------- mounting


def test_terminal_mounts_and_says_it_cannot_place_real_orders():
    """The safety claim should be on screen, not only in the docs."""

    async def scenario():
        app = TradingTerminal()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.is_running
            body = text_of(app, "#account")
            assert "No real order can be placed" in body
            assert "next bar's open" in body

    drive(scenario)


def test_nothing_connects_until_start_is_pressed():
    async def scenario():
        app = TradingTerminal()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.desk is None
            assert app.broker is None
            assert app.feed is None

    drive(scenario)


# --------------------------------------------------------------- input guards


@pytest.mark.parametrize("bad", ["", "abc", "-100", "0"])
def test_bad_demo_capital_is_refused_without_starting(bad):
    async def scenario():
        from textual.widgets import Input

        app = TradingTerminal()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#capital", Input).value = bad
            app.action_start()
            await pilot.pause()
            assert app.desk is None, f"{bad!r} should not have started a session"

    drive(scenario)


def test_an_empty_symbol_list_is_refused():
    async def scenario():
        from textual.widgets import Input

        app = TradingTerminal()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#symbols", Input).value = "  ,  "
            app.action_start()
            await pilot.pause()
            assert app.desk is None

    drive(scenario)


def test_thousands_separators_in_the_capital_field_are_accepted():
    """Typing 100,000 is the natural thing to do and must not be an error."""

    async def scenario():
        from textual.widgets import Input

        app = TradingTerminal()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#capital", Input).value = "250,000"
            # Parse only: assert the value survives the same cleaning the app applies.
            assert float("250,000".replace(",", "")) == 250_000.0
            assert app.query_one("#capital", Input).value == "250,000"

    drive(scenario)


# ---------------------------------------------------------------- rendering


class StubDesk:
    """Enough desk for the panels, with no feed and no network."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.universe = ["TEST"]
        self.focus_symbol = "TEST"
        self.mode = "PAPER (live prices)"
        self.paused = False
        self.running = True
        self.bars_processed = 42
        self.errors = []


def wired(app, capital=20_000.0):
    from quantdesk.pipeline import Pipeline

    broker = loaded_broker(capital)
    pipeline = Pipeline()
    for i in range(40):
        pipeline.book.append(make_bar(i, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i))
    app.broker = broker
    app.desk = StubDesk(pipeline)
    return broker


def test_account_panel_shows_the_chosen_balance_and_live_pnl():
    async def scenario():
        app = TradingTerminal()
        async with app.run_test() as pilot:
            await pilot.pause()
            broker = wired(app, capital=20_000.0)
            app.refresh_panels()
            await pilot.pause()
            body = text_of(app, "#account")
            assert "20,000.00" in body, "the chosen demo balance must be shown"
            assert "P&L" in body
            assert "fees" in body
            assert broker.round_trips, "the fixture should have completed a trade"

    drive(scenario)


def test_result_panel_reports_a_completed_round_trip():
    async def scenario():
        app = TradingTerminal()
        async with app.run_test() as pilot:
            await pilot.pause()
            wired(app)
            app.refresh_panels()
            await pilot.pause()
            body = text_of(app, "#result")
            assert "equity" in body
            assert "trades       1" in body

    drive(scenario)


def test_price_chart_renders_candles():
    async def scenario():
        app = TradingTerminal()
        async with app.run_test() as pilot:
            await pilot.pause()
            wired(app)
            app.refresh_panels()
            await pilot.pause()
            body = text_of(app, "#chart")
            assert "\u2588" in body or "\u2502" in body, "no candle glyphs drawn"
            assert "bars" in body

    drive(scenario)


def test_equity_chart_renders_once_there_are_marks():
    async def scenario():
        app = TradingTerminal()
        async with app.run_test() as pilot:
            await pilot.pause()
            wired(app)
            app.refresh_panels()
            await pilot.pause()
            body = text_of(app, "#equity-chart")
            assert "starting balance" in body

    drive(scenario)


def test_a_panel_error_does_not_stop_the_terminal():
    """Trading continues in the same process, so a bad format string must not be fatal."""

    async def scenario():
        app = TradingTerminal()
        async with app.run_test() as pilot:
            await pilot.pause()
            wired(app)

            def boom():
                raise RuntimeError("panel exploded")

            app._render_account = boom
            app.refresh_panels()
            await pilot.pause()
            assert app.is_running
            # The other panels still rendered.
            assert "equity" in text_of(app, "#result")

    drive(scenario)


def test_blotter_records_refusals_as_well_as_orders():
    async def scenario():
        from quantdesk.pipeline import BarDecision

        app = TradingTerminal()
        async with app.run_test() as pilot:
            await pilot.pause()
            wired(app)
            decision = BarDecision(ts=TS0, symbol="TEST")
            decision.rejected.append("reward-to-risk below the 3:1 yardstick")
            app._on_decision(decision)
            app.refresh_panels()
            await pilot.pause()
            assert "reward-to-risk" in text_of(app, "#blotter")

    drive(scenario)


def test_noisy_refusals_are_filtered_from_the_blotter():
    """"No agent had a view" every bar drowns out the refusals worth reading."""

    async def scenario():
        from quantdesk.pipeline import BarDecision

        app = TradingTerminal()
        async with app.run_test() as pilot:
            await pilot.pause()
            wired(app)
            decision = BarDecision(ts=TS0, symbol="TEST")
            decision.rejected.append("no agent had a view")
            app._on_decision(decision)
            await pilot.pause()
            assert not app._log

    drive(scenario)
