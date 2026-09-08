"""The terminal.

A Bloomberg-style monitor over the running desk: market monitor, positions, signal
blotter, Murphy's checklist for the focused symbol, risk state, and a decision log.

The UI owns no trading state. It holds a reference to the :class:`~quantdesk.app.desk.Desk`
and reads from it on a timer. That separation is deliberate - the desk must keep
trading correctly whether or not anything is watching, and closing the terminal must
never change what the strategy does.

Keys are bound for the things you actually need in a hurry: cycling the focused symbol,
pausing new entries, and the kill switch. The kill switch is the one control that has
to work when everything else is going wrong, so it does not depend on the render loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static

from quantdesk.ui.panels import (
    ChecklistPanel,
    DecisionLog,
    MarketMonitor,
    PositionsPanel,
    RiskPanel,
    SignalBlotter,
)

if TYPE_CHECKING:  # pragma: no cover
    from quantdesk.app.desk import Desk


class QuantDeskApp(App):
    """Terminal UI for a live or replayed desk session."""

    CSS_PATH = "theme.tcss"
    TITLE = "QUANTDESK"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("k", "kill", "KILL", key_display="K"),
        Binding("p", "pause", "Pause entries"),
        Binding("tab", "next_symbol", "Next symbol"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, desk: "Desk") -> None:
        super().__init__()
        self.desk = desk
        self.sub_title = f"{desk.mode} | {len(desk.universe)} symbols"

    # ---------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static("MARKET MONITOR", classes="title")
                yield MarketMonitor(id="market")
                yield Static("POSITIONS", classes="title")
                yield PositionsPanel(id="positions")
                yield Static("SIGNALS", classes="title")
                yield SignalBlotter(id="signals")
                yield Static("DECISIONS", classes="title")
                yield DecisionLog(id="decisions")
            with Vertical(id="right"):
                yield Static("RISK", classes="title")
                yield RiskPanel(id="risk")
                yield Static("TECHNICAL CHECKLIST", classes="title")
                with VerticalScroll():
                    yield ChecklistPanel(id="checklist")
        yield Footer()

    def on_mount(self) -> None:
        # The desk pushes decisions to the log as they happen; everything else is
        # polled, because redrawing a table on every bar of a fast feed is wasted work.
        self.desk.on_decision = self._on_decision
        self.set_interval(1.0, self.refresh_panels)
        self.refresh_panels()

    # --------------------------------------------------------------- refresh
    def refresh_panels(self) -> None:
        """Pull current state into every panel.

        Wrapped so a rendering error cannot take the desk down with it. The trading
        loop runs in the same process, and a bad format string in a panel must not
        stop orders being managed.
        """
        for widget_id, panel_type in (
            ("#market", MarketMonitor),
            ("#positions", PositionsPanel),
            ("#signals", SignalBlotter),
            ("#risk", RiskPanel),
            ("#checklist", ChecklistPanel),
        ):
            try:
                self.query_one(widget_id, panel_type).refresh_from(self.desk)
            except Exception as exc:  # noqa: BLE001 - UI must not kill the desk
                self.log(f"panel {widget_id} failed: {exc}")

    def _on_decision(self, decision) -> None:
        try:
            self.query_one("#decisions", DecisionLog).log_decision(decision)
        except Exception:  # noqa: BLE001 - pragma: no cover
            pass

    # --------------------------------------------------------------- actions
    def action_refresh(self) -> None:
        self.refresh_panels()

    def action_next_symbol(self) -> None:
        self.desk.cycle_focus()
        self.refresh_panels()

    def action_pause(self) -> None:
        paused = self.desk.toggle_pause()
        self.notify(
            "new entries paused (exits still allowed)" if paused else "entries resumed",
            severity="warning" if paused else "information",
        )
        self.refresh_panels()

    async def action_kill(self) -> None:
        """Cancel working orders and halt. Exits remain possible afterwards."""
        detail = await self.desk.kill("terminal kill switch")
        self.notify(f"KILLED: {detail}", severity="error", timeout=10)
        self.refresh_panels()

    async def action_quit(self) -> None:
        await self.desk.stop()
        self.exit()
