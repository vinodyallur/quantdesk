"""Research terminal: mounts, renders, and reports failures without a TTY.

Textual's ``run_test()`` drives the app with a headless driver, which is the only way this
UI can be verified in an environment with no terminal. Every test here is offline: the
worker methods that reach the network are never invoked, and the render methods are called
directly with synthetic results so the assertions are deterministic.
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import TS0

from quantdesk.backtest.engine import TradeRecord
from quantdesk.backtest.metrics import EquityPoint, compute_metrics
from quantdesk.ui.research import ResearchApp


def drive(scenario):
    """Run an async pilot scenario. Avoids a pytest-asyncio dependency."""
    return asyncio.run(scenario())


def trade(pnl: float, risk: float, side: int = 1, qty: float = 1.0) -> TradeRecord:
    return TradeRecord(
        symbol="X", opened_at=TS0, closed_at=TS0, side=side,
        pnl=pnl, initial_risk=risk, peak_qty=qty,
    )


def results(trades, equity_end: float = 100_000.0):
    """Build a real BacktestResult so the panel is tested against the true shape."""
    steps = 500
    curve = [
        EquityPoint(
            ts=TS0,
            equity=100_000.0 + (equity_end - 100_000.0) * i / (steps - 1),
        )
        for i in range(steps)
    ]
    return compute_metrics(
        equity_curve=curve, trades=trades, fills=[], decisions=[],
        bars_per_year=105_120.0, starting_equity=100_000.0,
    )


def text_of(app, widget_id: str) -> str:
    """Read what a Static was last updated with.

    ``content`` rather than ``renderable``: Textual 8 replaced the old attribute, and
    reaching for the wrong one fails identically for every panel, which looks like a
    rendering bug rather than a test bug.
    """
    from textual.widgets import Static

    return str(app.query_one(widget_id, Static).content)


# ------------------------------------------------------------------- mounting


def test_terminal_mounts_without_a_tty():
    """The whole point: verifiable in an environment with no terminal."""

    async def scenario():
        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.is_running

    drive(scenario)


def test_every_tab_is_present_and_reachable():
    async def scenario():
        from textual.widgets import TabbedContent

        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one("#tabs", TabbedContent)
            assert tabs.active == "tab-risk"
            for target in ("tab-history", "tab-perps", "tab-risk"):
                tabs.active = target
                await pilot.pause()
                assert tabs.active == target

    drive(scenario)


def test_no_work_starts_until_asked():
    """Mounting must not fetch anything, or the terminal cannot open offline."""

    async def scenario():
        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "Press Load" in text_of(app, "#hist-summary")
            assert "Press Load" in text_of(app, "#perp-summary")
            assert "Run a backtest" in text_of(app, "#risk-summary")

    drive(scenario)


# ----------------------------------------------------------------- risk panel


def test_risk_panel_reports_both_expectancies_and_the_spread():
    losing = [trade(-500.0, 500.0), trade(-500.0, 500.0), trade(300.0, 100.0)]

    async def scenario():
        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._render_risk(results(losing, equity_end=99_300.0))
            await pilot.pause()
            body = text_of(app, "#risk-summary")
            assert "risk-weighted" in body
            assert "unweighted" in body
            assert "spread" in body
            assert "5x" in body, "dispersion of 500/100 should be reported"

    drive(scenario)


def test_risk_panel_flags_a_sign_disagreement_as_a_defect():
    """The invariant. Risk-weighted R is P&L over risk, so it cannot disagree in sign.

    Equity ends up while the trades net down, which is impossible in a real run and is
    exactly the corruption this panel exists to catch.
    """
    trades = [trade(-500.0, 500.0), trade(-500.0, 500.0)]

    async def scenario():
        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._render_risk(results(trades, equity_end=101_000.0))
            await pilot.pause()
            assert "DEFECT" in text_of(app, "#risk-summary")

    drive(scenario)


def test_risk_panel_is_quiet_when_the_figures_agree():
    trades = [trade(-500.0, 500.0), trade(-500.0, 500.0)]

    async def scenario():
        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._render_risk(results(trades, equity_end=99_000.0))
            await pilot.pause()
            body = text_of(app, "#risk-summary")
            assert "DEFECT" not in body
            assert "agrees in sign" in body

    drive(scenario)


def test_risk_panel_warns_when_denominators_are_incomparable():
    """One near-zero denominator is what fooled the unweighted mean originally."""
    trades = [trade(30.0, 0.1), trade(-500.0, 500.0), trade(-500.0, 500.0)]

    async def scenario():
        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            result = results(trades, equity_end=99_030.0)
            assert result.mean_r > 0, "unweighted mean should be fooled"
            assert result.risk_weighted_r < 0, "risk-weighted should follow the money"
            app._render_risk(result)
            await pilot.pause()
            assert "WARNING" in text_of(app, "#risk-summary")

    drive(scenario)


def test_trade_table_ranks_by_absolute_r_and_shows_the_denominator():
    from textual.widgets import DataTable

    trades = [trade(30.0, 0.1), trade(-500.0, 500.0), trade(50.0, 500.0)]

    async def scenario():
        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._render_risk(results(trades, equity_end=99_580.0))
            await pilot.pause()
            table = app.query_one("#risk-trades", DataTable)
            assert table.row_count == 3
            first = table.get_row_at(0)
            # +300R from a 0.1 denominator must sort to the top.
            assert "+300" in str(first[5])
            assert "0.1000" in str(first[3])

    drive(scenario)


def test_a_trade_without_risk_is_marked_not_silently_zero():
    from textual.widgets import DataTable

    trades = [trade(-500.0, 500.0), trade(25.0, 0.0)]

    async def scenario():
        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            result = results(trades, equity_end=99_475.0)
            assert result.trades_without_risk == 1
            app._render_risk(result)
            await pilot.pause()
            table = app.query_one("#risk-trades", DataTable)
            flags = [str(table.get_row_at(i)[7]) for i in range(table.row_count)]
            assert "NO" in flags
            assert "no denominator" in text_of(app, "#risk-summary")

    drive(scenario)


# --------------------------------------------------------------- history panel


def test_history_panel_states_the_shiller_caveat_and_pre_1970_handling():
    from datetime import datetime, timezone

    from quantdesk.core.types import Bar

    bars = [
        Bar(symbol="SP500", ts=datetime(y, 1, 1, tzinfo=timezone.utc),
            open=p, high=p, low=p, close=p, volume=0.0)
        for y, p in ((1871, 4.44), (1900, 6.1), (1950, 17.0), (2000, 1425.0))
    ]

    async def scenario():
        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._render_history("shiller", "SP500", bars, bars[:-1], "CAPE for 3 months")
            await pilot.pause()
            body = text_of(app, "#hist-summary")
            assert "monthly AVERAGES" in body
            assert "no volume" in body
            assert "Pre-1970" in body
            assert "1871" in body

    drive(scenario)


def test_history_panel_buckets_by_decade():
    from datetime import datetime, timezone

    from quantdesk.core.types import Bar

    bars = [
        Bar(symbol="^GSPC", ts=datetime(y, 6, 1, tzinfo=timezone.utc),
            open=p, high=p, low=p, close=p, volume=100.0)
        for y, p in ((1930, 20.0), (1935, 10.0), (1940, 12.0), (1945, 15.0))
    ]

    async def scenario():
        from textual.widgets import DataTable

        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._render_history("yahoo", "^GSPC", bars, bars, "")
            await pilot.pause()
            table = app.query_one("#hist-table", DataTable)
            assert table.row_count == 2, "1930s and 1940s"
            assert "1930s" in str(table.get_row_at(0)[0])

    drive(scenario)


# ----------------------------------------------------------------- perps panel


def test_perps_panel_shows_carry_and_the_retention_caveat():
    from datetime import datetime, timedelta, timezone

    from quantdesk.data.perps import FundingRate, OpenInterest, summarise_funding

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rates = [
        FundingRate(symbol="BTC/USD", ts=base + timedelta(hours=8 * i),
                    rate=0.0001, mark_price=90_000.0 + i)
        for i in range(30)
    ]
    interest = [
        OpenInterest(symbol="BTC/USD", ts=base + timedelta(days=i), contracts=100.0 + i)
        for i in range(10)
    ]

    async def scenario():
        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            summary = summarise_funding(rates, "BTC/USD")
            app._render_perps("binance", summary, rates, interest,
                             ("new money long", "price and open interest both rising"))
            await pilot.pause()
            body = text_of(app, "#perp-summary")
            assert "crowded long" in body
            assert "a long pays" in body
            assert "Retention is short" in body
            assert "new money long" in body

    drive(scenario)


def test_perps_panel_says_so_when_the_venue_gives_no_mark_price():
    from datetime import datetime, timedelta, timezone

    from quantdesk.data.perps import FundingRate, OpenInterest, summarise_funding

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rates = [
        FundingRate(symbol="BTC/USD", ts=base + timedelta(hours=8 * i), rate=0.0001)
        for i in range(10)
    ]
    interest = [
        OpenInterest(symbol="BTC/USD", ts=base + timedelta(days=i), contracts=100.0)
        for i in range(4)
    ]

    async def scenario():
        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._render_perps("bybit", summarise_funding(rates, "BTC/USD"),
                             rates, interest, None)
            await pilot.pause()
            assert "cannot be read together" in text_of(app, "#perp-summary")

    drive(scenario)


# --------------------------------------------------------------------- errors


def test_a_failure_is_reported_and_the_terminal_survives():
    async def scenario():
        app = ResearchApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._fail("#perp-summary", RuntimeError("endpoint unreachable"))
            await pilot.pause()
            body = text_of(app, "#perp-summary")
            assert "FAILED" in body
            assert "endpoint unreachable" in body
            assert app.is_running, "a failed panel must not take the app down"

    drive(scenario)
