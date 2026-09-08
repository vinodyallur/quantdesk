"""Research terminal: inspect what the engine computed, without a live desk.

The live terminal in ``app.py`` polls a running :class:`~quantdesk.app.desk.Desk`. That is
the wrong shape for research: long-run history and perps funding are one-shot pulls with
no desk behind them, and a backtest is a batch job rather than a stream.

The risk tab is the reason this exists. Reported R-multiple expectancy was able to be
positive while the account lost money, because risk per trade varied across three orders
of magnitude and a plain average of R is dominated by whichever trades risked least. A
summary line saying "fixed" is not evidence, so this tab puts the per-trade denominators
on screen next to both aggregates and lets you see the disagreement directly.

Nothing here computes strategy state. Panels render what the engine already decided, for
the same reason the live panels do: two implementations of the same number eventually
disagree, and then there is no way to tell which one the desk acted on.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

log = logging.getLogger(__name__)

UP = "bold #33dd77"
DOWN = "bold #ff5566"
FLAT = "#8899aa"
KEY = "bold #ffb000"
DIM = "#667788"
WARN = "bold #ffb000"


def _signed(value: float, fmt: str = "+.3f", suffix: str = "") -> Text:
    style = UP if value > 0 else DOWN if value < 0 else FLAT
    return Text(f"{value:{fmt}}{suffix}", style=style)


def _row(label: str, value: str, style: str = FLAT) -> Text:
    out = Text()
    out.append(f"{label:<14}", style=DIM)
    out.append(f"{value}\n", style=style)
    return out


class ResearchApp(App):
    """Three tabs over the three things that had no interactive surface."""

    CSS_PATH = "research.tcss"
    TITLE = "QUANTDESK RESEARCH"
    SUB_TITLE = "read-only: no desk, no keys, no orders"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("1", "show('tab-risk')", "Risk"),
        Binding("2", "show('tab-history')", "History"),
        Binding("3", "show('tab-perps')", "Perps"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="tab-risk", id="tabs"):
            with TabPane("RISK", id="tab-risk"):
                with Horizontal(classes="controls"):
                    yield Static("days", classes="lbl")
                    yield Input(value="20", id="risk-days", classes="tiny")
                    yield Static("symbols", classes="lbl")
                    yield Input(value="BTC/USD,ETH/USD", id="risk-symbols")
                    yield Select(
                        [("offline (synthetic)", "offline"), ("live feed", "live")],
                        value="offline",
                        allow_blank=False,
                        id="risk-feed",
                        classes="pick",
                    )
                    yield Button("Run backtest", id="risk-run", variant="primary")
                with VerticalScroll():
                    yield Static(self._risk_placeholder(), id="risk-summary", classes="report")
                    yield Static("TRADES BY ABSOLUTE R", classes="title")
                    yield DataTable(id="risk-trades")
            with TabPane("HISTORY", id="tab-history"):
                with Horizontal(classes="controls"):
                    yield Select(
                        [("yahoo: daily from 1928", "yahoo"),
                         ("shiller: monthly from 1871", "shiller")],
                        value="yahoo",
                        allow_blank=False,
                        id="hist-source",
                        classes="pick",
                    )
                    yield Static("symbol", classes="lbl")
                    yield Input(value="^GSPC", id="hist-symbol", classes="tiny")
                    yield Select(
                        [("use cache", "cache"), ("force refresh", "refresh")],
                        value="cache",
                        allow_blank=False,
                        id="hist-cache",
                        classes="pick",
                    )
                    yield Button("Load", id="hist-run", variant="primary")
                with VerticalScroll():
                    yield Static(
                        Text("Press Load. Yahoo is cached locally after the first pull.",
                             style=DIM),
                        id="hist-summary", classes="report",
                    )
                    yield Static("DECADE SUMMARY", classes="title")
                    yield DataTable(id="hist-table")
            with TabPane("PERPS", id="tab-perps"):
                with Horizontal(classes="controls"):
                    yield Static("symbol", classes="lbl")
                    yield Input(value="BTC/USD", id="perp-symbol", classes="tiny")
                    yield Select(
                        [("binance", "binance"), ("bybit", "bybit")],
                        value="binance",
                        allow_blank=False,
                        id="perp-venue",
                        classes="pick",
                    )
                    yield Static("days", classes="lbl")
                    yield Input(value="45", id="perp-days", classes="tiny")
                    yield Button("Load", id="perp-run", variant="primary")
                with VerticalScroll():
                    yield Static(
                        Text("Press Load. Needs network; no API key.", style=DIM),
                        id="perp-summary", classes="report",
                    )
                    yield Static("LARGEST SETTLEMENTS", classes="title")
                    yield DataTable(id="perp-table")
        yield Footer()

    def on_mount(self) -> None:
        trades = self.query_one("#risk-trades", DataTable)
        for label, width in (
            ("#", 4), ("SYMBOL", 11), ("SIDE", 6), ("RISK", 12),
            ("PNL", 12), ("R", 10), ("PEAK QTY", 12), ("R VALID", 9),
        ):
            trades.add_column(label, width=width)
        trades.zebra_stripes = True

        hist = self.query_one("#hist-table", DataTable)
        for label, width in (
            ("DECADE", 9), ("BARS", 8), ("FIRST", 12), ("LAST", 12),
            ("CHANGE", 10), ("NO VOLUME", 11),
        ):
            hist.add_column(label, width=width)
        hist.zebra_stripes = True

        perp = self.query_one("#perp-table", DataTable)
        for label, width in (
            ("WHEN", 18), ("RATE", 11), ("ANNUALISED", 12), ("WHO PAID", 12),
        ):
            perp.add_column(label, width=width)
        perp.zebra_stripes = True

    # ------------------------------------------------------------------ events
    def action_show(self, tab: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "risk-run":
            self._busy("#risk-summary", "running backtest")
            self.run_risk()
        elif event.button.id == "hist-run":
            self._busy("#hist-summary", "loading history")
            self.run_history()
        elif event.button.id == "perp-run":
            self._busy("#perp-summary", "loading funding")
            self.run_perps()

    def _busy(self, target: str, what: str) -> None:
        self.query_one(target, Static).update(
            Text(f"{what}... this runs off the UI thread, the interface stays live.",
                 style=WARN)
        )

    def _value(self, widget_id: str, default: str = "") -> str:
        try:
            widget = self.query_one(widget_id)
        except Exception:
            return default
        return str(getattr(widget, "value", default) or default)

    # ------------------------------------------------------------------- risk
    def _risk_placeholder(self) -> Text:
        out = Text()
        out.append("Run a backtest to see risk accounting.\n\n", style=DIM)
        out.append(
            "What to look for: if risk per trade is uneven, the unweighted expectancy\n"
            "is an average over incomparable denominators and can be positive while the\n"
            "account loses money. The risk-weighted figure is total P&L over total risk,\n"
            "so it cannot disagree in sign with the money.\n",
            style=DIM,
        )
        return out

    @work(thread=True, exclusive=True, group="risk")
    def run_risk(self) -> None:
        try:
            days = max(1, int(float(self._value("#risk-days", "20"))))
        except ValueError:
            days = 20
        raw = self._value("#risk-symbols", "BTC/USD")
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        offline = self._value("#risk-feed", "offline") == "offline"

        try:
            from quantdesk.backtest import Backtester
            from quantdesk.config import get_settings
            from quantdesk.data import build_feed

            settings = get_settings()
            feed = build_feed(settings, offline=offline)
            end = datetime.now(timezone.utc)
            history = feed.history(start=end - timedelta(days=days), end=end,
                                  symbols=symbols or settings.universe)
            if not any(history.values()):
                raise RuntimeError("no bars returned; try the offline feed")
            result = Backtester(settings).run(history)
        except Exception as exc:  # noqa: BLE001 - report, never crash the terminal
            log.exception("risk tab failed")
            self.call_from_thread(self._fail, "#risk-summary", exc)
            return
        self.call_from_thread(self._render_risk, result)

    def _render_risk(self, result) -> None:
        out = Text()
        out.append("BACKTEST\n", style=KEY)
        out.append(_row("return", f"{result.total_return:+.2%}",
                        UP if result.total_return > 0 else DOWN))
        out.append(_row("equity", f"{result.starting_equity:,.0f} -> {result.final_equity:,.0f}"))
        out.append(_row("trades", f"{result.trades}  hit rate {result.hit_rate:.1%}"))
        out.append(_row("expectancy", f"{result.expectancy:+,.2f} per trade in currency",
                        UP if result.expectancy > 0 else DOWN))

        out.append("\nRISK PER TRADE\n", style=KEY)
        out.append(_row("median", f"{result.median_risk:,.2f}"))
        out.append(_row("range", f"{result.min_risk:,.2f} to {result.max_risk:,.2f}"))
        spread_style = DOWN if result.risk_dispersion > 10 else UP
        out.append(_row("spread", f"{result.risk_dispersion:,.0f}x", spread_style))
        if result.trades_without_risk:
            out.append(_row("no denominator",
                            f"{result.trades_without_risk} trade(s), excluded from R", DOWN))

        out.append("\nEXPECTANCY IN R\n", style=KEY)
        out.append(Text("  risk-weighted ", style=DIM))
        out.append(_signed(result.risk_weighted_r, "+.4f", "R"))
        out.append(Text("   <- compare this one with the money\n", style=DIM))
        out.append(Text("  unweighted    ", style=DIM))
        out.append(_signed(result.mean_r, "+.4f", "R"))
        weight_note = "   <- average over uneven denominators\n" if not result.r_is_trustworthy else "\n"
        out.append(Text(weight_note, style=DIM))

        money_positive = result.final_equity > result.starting_equity
        rwr_positive = result.risk_weighted_r > 0
        out.append("\n")
        if money_positive != rwr_positive:
            out.append(
                "DEFECT: risk-weighted R disagrees in sign with the P&L. It is defined as\n"
                "total P&L over total risk, so this is arithmetically impossible and means\n"
                "the risk accounting is broken again.\n",
                style=DOWN,
            )
        else:
            out.append("Risk-weighted R agrees in sign with the P&L, as it must.\n", style=UP)

        if not result.r_is_trustworthy:
            out.append(
                f"\nWARNING: risk per trade spans {result.risk_dispersion:,.0f}x, so the\n"
                "unweighted figure describes the sizing rather than the edge. This is\n"
                "expected on an unleveraged account: the 10% commitment ceiling binds\n"
                "before the risk target, so stop distance sets the risk.\n",
                style=WARN,
            )
        self.query_one("#risk-summary", Static).update(out)

        table = self.query_one("#risk-trades", DataTable)
        table.clear()
        ranked = sorted(result.trade_list, key=lambda t: abs(t.r_multiple), reverse=True)
        for i, t in enumerate(ranked[:25], start=1):
            table.add_row(
                str(i),
                Text(t.symbol, style=KEY),
                Text("long" if t.side > 0 else "short",
                     style=UP if t.side > 0 else DOWN),
                f"{t.initial_risk:,.4f}",
                _signed(t.pnl, "+,.2f"),
                _signed(t.r_multiple, "+.2f"),
                f"{t.peak_qty:,.6g}",
                Text("yes", style=UP) if t.risk_known else Text("NO", style=DOWN),
            )
        if not ranked:
            table.add_row("-", Text("no closed trades", style=DIM), "", "", "", "", "", "")

    # ---------------------------------------------------------------- history
    @work(thread=True, exclusive=True, group="history")
    def run_history(self) -> None:
        which = self._value("#hist-source", "yahoo")
        symbol = self._value("#hist-symbol", "^GSPC")
        refresh = self._value("#hist-cache", "cache") == "refresh"
        try:
            from quantdesk.config import get_settings
            from quantdesk.data import BarCache, HistoryStore
            from quantdesk.data.history import (
                ShillerSource,
                YahooChartSource,
                resample_monthly,
            )

            settings = get_settings()
            store = HistoryStore(BarCache(settings.db_path))
            if which == "shiller":
                source = ShillerSource()
                symbol = symbol if symbol and symbol != "^GSPC" else "SP500"
            else:
                source = YahooChartSource()
            bars = store.load(source, symbol, refresh=refresh)
            if not bars:
                raise RuntimeError(f"no bars returned for {symbol}")
            monthly = resample_monthly(bars)
            extra = ""
            if which == "shiller":
                capes = [r for r in source.records() if r.cape is not None]
                if capes:
                    extra = (f"CAPE for {len(capes):,} months from "
                             f"{capes[0].ts:%Y-%m}, latest {capes[-1].cape:.1f} "
                             f"at {capes[-1].ts:%Y-%m}")
        except Exception as exc:  # noqa: BLE001
            log.exception("history tab failed")
            self.call_from_thread(self._fail, "#hist-summary", exc)
            return
        self.call_from_thread(self._render_history, which, symbol, bars, monthly, extra)

    def _render_history(self, which, symbol, bars, monthly, extra) -> None:
        years = (bars[-1].ts - bars[0].ts).days / 365.25
        no_vol = sum(1 for b in bars if b.volume <= 0)
        out = Text()
        out.append(f"{symbol}  via {which}\n", style=KEY)
        out.append(_row("bars", f"{len(bars):,}"))
        out.append(_row("span", f"{bars[0].ts:%Y-%m-%d} to {bars[-1].ts:%Y-%m-%d}"
                                f"  ({years:.1f} years)"))
        out.append(_row("close", f"{bars[0].close:,.2f} -> {bars[-1].close:,.2f}"))
        growth = (bars[-1].close / bars[0].close) if bars[0].close else 0.0
        out.append(_row("growth", f"{growth:,.0f}x nominal, price only, no dividends"))
        out.append(_row("months", f"{len(monthly):,} complete "
                                  f"(the unfinished month is excluded, to avoid lookahead)"))
        if no_vol:
            out.append(_row("no volume",
                            f"{no_vol:,} bars ({no_vol / len(bars):.0%}) - volume filters "
                            f"cannot work on those", WARN))
        if extra:
            out.append("\n")
            out.append(extra + "\n", style=FLAT)
        if which == "shiller":
            out.append(
                "\nNOTE: Shiller prices are monthly AVERAGES of daily closes. All four\n"
                "OHLC values are deliberately equal, so there is no intramonth range and\n"
                "these must not be used to simulate stops or fills.\n",
                style=WARN,
            )
        if bars[0].ts.year < 1970:
            out.append(
                f"\nPre-1970 dates present ({bars[0].ts:%Y}). Those have negative epoch\n"
                "timestamps, which datetime.fromtimestamp rejects on Windows; the cache\n"
                "reads them via an epoch offset instead.\n",
                style=DIM,
            )
        self.query_one("#hist-summary", Static).update(out)

        table = self.query_one("#hist-table", DataTable)
        table.clear()
        buckets: dict[int, list] = {}
        for b in bars:
            buckets.setdefault(b.ts.year // 10 * 10, []).append(b)
        for decade in sorted(buckets):
            group = buckets[decade]
            first, last = group[0].close, group[-1].close
            change = (last / first - 1.0) if first else 0.0
            flat = sum(1 for b in group if b.volume <= 0)
            table.add_row(
                Text(f"{decade}s", style=KEY),
                f"{len(group):,}",
                f"{first:,.2f}",
                f"{last:,.2f}",
                _signed(change * 100.0, "+.1f", "%"),
                f"{flat:,}" if flat else "-",
            )

    # ------------------------------------------------------------------ perps
    @work(thread=True, exclusive=True, group="perps")
    def run_perps(self) -> None:
        symbol = self._value("#perp-symbol", "BTC/USD")
        venue = self._value("#perp-venue", "binance")
        try:
            days = max(1, int(float(self._value("#perp-days", "45"))))
        except ValueError:
            days = 45
        try:
            from quantdesk.data.perps import (
                default_source,
                open_interest_signal,
                summarise_funding,
            )

            source = default_source(venue)
            end = datetime.now(timezone.utc)
            rates = source.funding(symbol, start=end - timedelta(days=days), end=end)
            if not rates:
                raise RuntimeError("no funding history returned")
            interest = source.open_interest(symbol, period="1d")
            summary = summarise_funding(rates, symbol)
            marks = [r.mark_price for r in rates if r.mark_price is not None]
            reading = None
            if marks and len(interest) >= 2:
                reading = open_interest_signal(marks, [o.contracts for o in interest])
        except Exception as exc:  # noqa: BLE001
            log.exception("perps tab failed")
            self.call_from_thread(self._fail, "#perp-summary", exc)
            return
        self.call_from_thread(self._render_perps, venue, summary, rates, interest, reading)

    def _render_perps(self, venue, summary, rates, interest, reading) -> None:
        out = Text()
        out.append(f"{summary.symbol}  via {venue}\n", style=KEY)
        for line in summary.summary():
            out.append(f"  {line}\n", style=FLAT)
        carry_style = DOWN if summary.annualised_cost > 0 else UP
        out.append("\n")
        out.append("  a long pays ", style=DIM)
        out.append(f"{summary.annualised_cost:+.2%}", style=carry_style)
        out.append(" a year in funding alone, before any price move\n", style=DIM)

        out.append("\nOPEN INTEREST\n", style=KEY)
        if len(interest) >= 2:
            out.append(_row("points", f"{len(interest)}  {interest[0].ts:%Y-%m-%d} to "
                                      f"{interest[-1].ts:%Y-%m-%d}"))
            out.append(_row("contracts",
                            f"{interest[0].contracts:,.0f} -> {interest[-1].contracts:,.0f}"))
            if reading:
                out.append(_row("reading", reading[0], KEY))
                out.append(Text(f"  {reading[1]}\n", style=DIM))
            else:
                out.append(Text(
                    "  this venue does not return a mark price with funding, so price and\n"
                    "  open interest cannot be read together. Try binance.\n", style=WARN))
            span = (interest[-1].ts - interest[0].ts).days
            out.append(Text(
                f"\n  Retention is short: {span} days returned here. Perps themselves only\n"
                "  start around 2019, so a long series has to be accumulated forward\n"
                "  rather than backfilled.\n", style=DIM))
        else:
            out.append(Text("  not enough history returned to read a trend\n", style=WARN))
        self.query_one("#perp-summary", Static).update(out)

        table = self.query_one("#perp-table", DataTable)
        table.clear()
        for r in sorted(rates, key=lambda r: abs(r.rate), reverse=True)[:20]:
            table.add_row(
                f"{r.ts:%Y-%m-%d %H:%M}",
                _signed(r.rate * 100.0, "+.4f", "%"),
                _signed(r.annualised * 100.0, "+.1f", "%"),
                Text("longs paid" if r.longs_pay else "shorts paid",
                     style=DOWN if r.longs_pay else UP),
            )

    # ----------------------------------------------------------------- errors
    def _fail(self, target: str, exc: Exception) -> None:
        out = Text()
        out.append("FAILED\n", style=DOWN)
        out.append(f"{type(exc).__name__}: {exc}\n", style=FLAT)
        out.append("\nThe terminal stays usable; other tabs are unaffected.\n", style=DIM)
        self.query_one(target, Static).update(out)


def main() -> int:
    ResearchApp().run()
    return 0
