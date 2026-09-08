"""Paper trading terminal: live perps prices, simulated money.

What this is, precisely, because the distinction matters more than anything else here:
prices, funding and open interest are real and live from Binance USD-M futures; orders are
matched by :class:`~quantdesk.execution.paper_broker.PaperBroker` against those prices and
settle against a demo balance you choose at the start. Nothing in this file can place a
real order. There is no credential path and no venue write call.

Fills land at the next bar's open, never the close the decision was made on. That is the
single most important property of the simulation: a paper session that fills at the
decision price shows an edge that evaporates the moment real money is involved, and nothing
about the equity curve looks wrong while it is happening.

The forming bar is drawn in amber and excluded from the strategy's view. It is the bar most
likely to make a session look profitable, because its close has not settled yet.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Footer, Header, Input, Select, Static

from quantdesk.ui.charts import candlestick, line_chart

log = logging.getLogger(__name__)

UP = "bold #33dd77"
DOWN = "bold #ff5566"
FLAT = "#8899aa"
KEY = "bold #ffb000"
DIM = "#667788"
WARN = "bold #ffb000"


def _signed(value: float, fmt: str = "+,.2f", suffix: str = "") -> Text:
    style = UP if value > 0 else DOWN if value < 0 else FLAT
    return Text(f"{value:{fmt}}{suffix}", style=style)


def _line(label: str, value: str, style: str = FLAT) -> Text:
    out = Text()
    out.append(f"{label:<12}", style=DIM)
    out.append(f"{value}\n", style=style)
    return out


class TradingTerminal(App):
    """Live-price paper trading with charts, funding, and a running result."""

    CSS_PATH = "trading.tcss"
    TITLE = "QUANTDESK PAPER TRADING"
    SUB_TITLE = "live prices - simulated money - no real orders"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "start", "Start"),
        Binding("p", "pause", "Pause entries"),
        Binding("k", "kill", "KILL", key_display="K"),
        Binding("tab", "next_symbol", "Next symbol"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.desk = None
        self.broker = None
        self.feed = None
        self._desk_task: asyncio.Task | None = None
        self._funding: dict[str, object] = {}
        self._marks: dict[str, float] = {}
        self._booting = False
        self._log: list[Text] = []

    # ----------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="setup"):
            yield Static("demo capital", classes="lbl")
            yield Input(value="100000", id="capital", classes="tiny")
            yield Static("symbols", classes="lbl")
            yield Input(value="BTC/USD,ETH/USD", id="symbols")
            yield Select(
                [("1 minute bars", "1Min"), ("5 minute bars", "5Min"),
                 ("15 minute bars", "15Min"), ("1 hour bars", "1Hour")],
                value="5Min", allow_blank=False, id="timeframe", classes="pick",
            )
            yield Button("START SESSION", id="start", variant="success")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static("PRICE  (live perps)", classes="title", id="chart-title")
                yield Static(
                    Text("press START SESSION to connect", style=DIM),
                    id="chart", classes="chart",
                )
                yield Static("EQUITY", classes="title")
                yield Static(Text("", style=DIM), id="equity-chart", classes="chart")
                yield Static("ACTIVITY", classes="title")
                yield Static(Text("", style=DIM), id="blotter", classes="pane")
            with Vertical(id="right"):
                yield Static("ACCOUNT", classes="title")
                yield Static(self._idle_account(), id="account", classes="pane")
                yield Static("POSITIONS", classes="title")
                yield DataTable(id="positions")
                yield Static("FUNDING  (live)", classes="title")
                yield Static(Text("", style=DIM), id="funding", classes="pane")
                yield Static("RESULT", classes="title")
                with VerticalScroll():
                    yield Static(Text("", style=DIM), id="result", classes="pane")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#positions", DataTable)
        for label, width in (
            ("SYMBOL", 10), ("QTY", 12), ("AVG", 11), ("LAST", 11), ("UNREAL", 11),
        ):
            table.add_column(label, width=width)
        table.zebra_stripes = True
        self.set_interval(1.0, self.refresh_panels)

    def _idle_account(self) -> Text:
        out = Text()
        out.append("not started\n\n", style=DIM)
        out.append(
            "Set your demo balance, pick symbols, then START.\n\n"
            "Prices, funding and open interest are real and live.\n"
            "Orders are simulated and fill at the next bar's open.\n"
            "No real order can be placed from this screen.\n",
            style=DIM,
        )
        return out

    # ------------------------------------------------------------------ start
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.action_start()

    def action_start(self) -> None:
        if self.desk is not None or self._booting:
            self.notify("session already running", severity="warning")
            return
        try:
            capital = float(self.query_one("#capital", Input).value.replace(",", ""))
        except ValueError:
            self.notify("demo capital must be a number", severity="error")
            return
        if capital <= 0:
            self.notify("demo capital must be positive", severity="error")
            return
        raw = self.query_one("#symbols", Input).value
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        if not symbols:
            self.notify("pick at least one symbol", severity="error")
            return
        timeframe = str(self.query_one("#timeframe", Select).value)

        self._booting = True
        self.query_one("#start", Button).disabled = True
        self.query_one("#account", Static).update(
            Text("connecting to Binance and warming up indicators...", style=WARN)
        )
        self.boot(capital, symbols, timeframe)

    @work(exclusive=True, group="boot")
    async def boot(self, capital: float, symbols: list[str], timeframe: str) -> None:
        """Build the session, warm it up, and start the loop.

        History is fetched in a thread and handed to ``warmup`` rather than letting the
        desk fetch it itself, because the feed's ``history`` is synchronous and would
        otherwise stall the event loop and freeze the interface for several seconds.
        """
        try:
            from quantdesk.app.desk import Desk
            from quantdesk.config import get_settings
            from quantdesk.data.binance_feed import BinancePerpsFeed
            from quantdesk.execution.paper_broker import PaperBroker

            settings = get_settings().model_copy(
                update={"timeframe": timeframe, "starting_equity": capital}
            )
            feed = BinancePerpsFeed(symbols, timeframe, poll_seconds=10)
            broker = PaperBroker(
                starting_equity=capital,
                commission_bps=settings.commission_bps,
                slippage_bps=settings.slippage_bps,
            )
            desk = Desk(settings=settings, broker=broker, feed=feed, mode="PAPER (live prices)")
            desk.universe = list(symbols)
            desk.focus_symbol = symbols[0]
            desk.on_decision = self._on_decision

            minutes = settings.timeframe_minutes * settings.warmup_bars
            start = datetime.now(timezone.utc) - timedelta(minutes=minutes * 1.5)
            history = await asyncio.to_thread(feed.history, start, None, symbols)
            total = sum(len(v) for v in history.values())
            if total == 0:
                raise RuntimeError(
                    "Binance returned no bars. Check the symbols: perps are quoted in "
                    "USDT, so BTC/USD maps to BTCUSDT."
                )
            await desk.warmup(history)

            self.feed, self.broker, self.desk = feed, broker, desk
            self._desk_task = asyncio.create_task(desk.run())
            self.poll_funding()
            self.notify(
                f"session live: {capital:,.0f} demo, {total:,} warmup bars, "
                f"{timeframe} on {', '.join(symbols)}"
            )
        except Exception as exc:  # noqa: BLE001 - report, never crash the terminal
            log.exception("boot failed")
            self.query_one("#account", Static).update(
                Text(f"FAILED TO START\n{type(exc).__name__}: {exc}\n", style=DOWN)
            )
            self.query_one("#start", Button).disabled = False
            self.notify(f"could not start: {exc}", severity="error", timeout=12)
        finally:
            self._booting = False

    # ---------------------------------------------------------------- funding
    @work(thread=True, exclusive=True, group="funding")
    def poll_funding(self) -> None:
        """Refresh funding, open interest and marks on a slow loop.

        Separate from the trading loop on purpose: funding settles every eight hours, so
        polling it at bar frequency would be wasted requests, and a funding outage must
        not interrupt trading.
        """
        from quantdesk.data.perps import (
            BinancePerps,
            open_interest_signal,
            summarise_funding,
        )

        source = BinancePerps()
        while self.desk is not None:
            try:
                end = datetime.now(timezone.utc)
                snapshot: dict[str, object] = {}
                for symbol in list(self.desk.universe):
                    rates = source.funding(symbol, start=end - timedelta(days=7), end=end)
                    if not rates:
                        continue
                    interest = source.open_interest(symbol, period="1d", limit=30)
                    marks = [r.mark_price for r in rates if r.mark_price is not None]
                    reading = (
                        open_interest_signal(marks, [o.contracts for o in interest])
                        if marks and len(interest) >= 2
                        else None
                    )
                    snapshot[symbol] = (
                        summarise_funding(rates, symbol), rates[-1], interest, reading
                    )
                marks_now = self.feed.mark_prices() if self.feed else {}
                self._funding = snapshot
                self._marks = marks_now
            except Exception as exc:  # noqa: BLE001
                log.debug("funding poll failed: %s", exc)
            for _ in range(60):
                if self.desk is None:
                    return
                import time

                time.sleep(1)

    # ---------------------------------------------------------------- refresh
    def refresh_panels(self) -> None:
        if self.desk is None:
            return
        for name, render in (
            ("chart", self._render_chart),
            ("account", self._render_account),
            ("equity-chart", self._render_equity),
            ("funding", self._render_funding),
            ("result", self._render_result),
            ("blotter", self._render_blotter),
        ):
            try:
                self.query_one(f"#{name}", Static).update(render())
            except Exception as exc:  # noqa: BLE001 - a panel must not stop trading
                log.debug("panel %s failed: %s", name, exc)
        try:
            self._render_positions()
        except Exception as exc:  # noqa: BLE001
            log.debug("positions panel failed: %s", exc)

    def _chart_size(self) -> tuple[int, int]:
        widget = self.query_one("#chart", Static)
        width = max(40, widget.size.width - 2)
        height = max(8, widget.size.height - 1)
        return width, height

    def _render_chart(self) -> Text:
        symbol = self.desk.focus_symbol
        bars = self.desk.pipeline.book.series(symbol).bars()
        if not bars:
            return Text("waiting for bars...", style=DIM)
        width, height = self._chart_size()
        forming = self.feed.forming(symbol) if self.feed else None
        self.query_one("#chart-title", Static).update(
            Text(f"{symbol}  {bars[-1].close:,.2f}   live perps   "
                 f"tab switches symbol", style=KEY)
        )
        return candlestick(bars, width=width, height=height, forming=forming)

    def _render_equity(self) -> Text:
        curve = [t.equity for t in self.broker.equity_curve]
        if len(curve) < 2:
            return Text("equity curve builds as bars close", style=DIM)
        width, _ = self._chart_size()
        return line_chart(
            curve, width=width, height=7,
            label=f"{len(curve)} marks, dashed row = your starting balance",
            baseline=self.broker.starting_equity,
        )

    def _render_account(self) -> Text:
        ledger = self.broker.account_ledger
        equity = ledger.equity
        start = self.broker.starting_equity
        pnl = equity - start
        out = Text()
        out.append(_line("mode", self.desk.mode, KEY))
        out.append(_line("state", "PAUSED" if self.desk.paused else
                         ("running" if self.desk.running else "stopped"),
                         WARN if self.desk.paused else
                         (UP if self.desk.running else DOWN)))
        out.append(_line("demo start", f"{start:,.2f}"))
        out.append("equity      ", style=DIM)
        out.append(f"{equity:,.2f}\n", style=KEY)
        out.append("P&L         ", style=DIM)
        out.append(f"{pnl:+,.2f}", style=UP if pnl >= 0 else DOWN)
        out.append(f"  ({pnl / start:+.2%})\n", style=UP if pnl >= 0 else DOWN)
        out.append(_line("cash", f"{ledger.cash:,.2f}"))
        out.append("unrealised  ", style=DIM)
        out.append(f"{ledger.unrealized_pnl:+,.2f}\n",
                   style=UP if ledger.unrealized_pnl >= 0 else DOWN)
        out.append("realised    ", style=DIM)
        out.append(f"{ledger.realized_pnl:+,.2f}\n",
                   style=UP if ledger.realized_pnl >= 0 else DOWN)
        out.append(_line("fees", f"{ledger.fees_paid:,.2f}", DOWN if ledger.fees_paid else FLAT))
        out.append(_line("bars", f"{self.desk.bars_processed:,}"))
        out.append(_line("pending", f"{self.broker.pending_orders} order(s)"))
        risk = self.desk.pipeline.risk.state.value
        out.append(_line("risk", risk.upper(), UP if risk == "normal" else DOWN))
        if self.desk.errors:
            out.append(f"\n{self.desk.errors[-1][:44]}\n", style=DOWN)
        return out

    def _render_positions(self) -> None:
        table = self.query_one("#positions", DataTable)
        table.clear()
        live = [p for p in self.broker.account_ledger.positions.values() if not p.is_flat]
        if not live:
            table.add_row(Text("flat", style=DIM), "", "", "", "")
            return
        for pos in live:
            last = self._marks.get(pos.symbol) or pos.last_price or pos.avg_price
            table.add_row(
                Text(pos.symbol, style=KEY),
                _signed(pos.qty, "+,.6g"),
                f"{pos.avg_price:,.2f}",
                f"{last:,.2f}",
                _signed(pos.unrealized_pnl(last)),
            )

    def _render_funding(self) -> Text:
        if not self._funding:
            return Text("loading funding...", style=DIM)
        out = Text()
        for symbol, payload in self._funding.items():
            summary, latest, interest, reading = payload
            out.append(f"{symbol}\n", style=KEY)
            out.append("  rate now  ", style=DIM)
            out.append(f"{latest.rate * 100:+.4f}%", style=DOWN if latest.rate > 0 else UP)
            out.append(f"  ({latest.annualised:+.1%}/yr)\n", style=DIM)
            out.append("  a long pays ", style=DIM)
            out.append(f"{summary.annualised_cost:+.2%}",
                       style=DOWN if summary.annualised_cost > 0 else UP)
            out.append(" a year\n", style=DIM)
            out.append("  longs paid ", style=DIM)
            out.append(f"{summary.positive_share:.0%} of settlements\n", style=FLAT)
            lean = ("crowded long" if summary.crowded_long else
                    "crowded short" if summary.crowded_short else "no persistent lean")
            out.append(f"  {lean}\n", style=WARN if "crowded" in lean else FLAT)
            if reading:
                out.append(f"  OI: {reading[0]}\n", style=FLAT)
            out.append("\n")
        return out

    def _render_result(self) -> Text:
        results = self.broker.results()
        out = Text()
        for line in results.summary():
            out.append(f"{line}\n", style=FLAT)
        if results.trades == 0:
            out.append(
                "\nNo round trip has completed yet. The 3:1 reward-to-risk gate declines\n"
                "most setups by design, so quiet stretches are the strategy working.\n",
                style=DIM,
            )
        elif results.fees_paid > abs(results.total_pnl) * 0.5:
            out.append(
                "\nFees are over half the size of the P&L. At this trade frequency costs\n"
                "are the dominant term, not the signal.\n",
                style=WARN,
            )
        return out

    def _render_blotter(self) -> Text:
        if not self._log:
            return Text("orders and refusals appear here", style=DIM)
        out = Text()
        for entry in self._log[-14:]:
            out.append_text(entry)
        return out

    def _on_decision(self, decision) -> None:
        """Record what the desk decided. Refusals matter as much as orders."""
        stamp = f"{decision.ts:%H:%M:%S}"
        if decision.orders:
            for order in decision.orders:
                entry = Text()
                entry.append(f"{stamp} ", style=DIM)
                entry.append(f"{decision.symbol:<9}", style=KEY)
                entry.append(f"{order.side.value.upper():<5}",
                             style=UP if order.side.value == "buy" else DOWN)
                entry.append(f"{order.qty:>12,.6g}  ", style=FLAT)
                entry.append(f"risk {order.planned_risk:,.0f}  ", style=DIM)
                entry.append(f"{order.tag[:30]}\n", style=DIM)
                self._log.append(entry)
        else:
            for reason in decision.rejected:
                if reason.startswith("no agent") or "rebalance threshold" in reason:
                    continue
                entry = Text()
                entry.append(f"{stamp} ", style=DIM)
                entry.append(f"{decision.symbol:<9}", style=KEY)
                entry.append("skip  ", style=WARN)
                entry.append(f"{reason[:46]}\n", style=DIM)
                self._log.append(entry)
        if len(self._log) > 400:
            del self._log[:-400]

    # --------------------------------------------------------------- controls
    def action_next_symbol(self) -> None:
        if self.desk is not None:
            self.desk.cycle_focus()
            self.refresh_panels()

    def action_pause(self) -> None:
        if self.desk is None:
            return
        paused = self.desk.toggle_pause()
        self.notify(
            "new entries paused, exits still allowed" if paused else "entries resumed",
            severity="warning" if paused else "information",
        )

    async def action_kill(self) -> None:
        if self.desk is None:
            return
        detail = await self.desk.kill("terminal kill switch")
        self.notify(f"KILLED: {detail}", severity="error", timeout=10)

    async def action_quit(self) -> None:
        desk = self.desk
        self.desk = None
        if desk is not None:
            try:
                await desk.stop()
            except Exception:  # noqa: BLE001 - shutting down anyway
                pass
        if self._desk_task is not None:
            self._desk_task.cancel()
        self.exit()


def main() -> int:
    TradingTerminal().run()
    return 0
