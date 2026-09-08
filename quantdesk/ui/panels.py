"""Terminal panels.

Each panel is a self-contained widget with a ``refresh_from(desk)`` method, so the
app's job is only to lay them out and tick them. Panels never compute anything - they
read state that the pipeline already decided. A UI that recalculates is a UI that can
disagree with the engine, and then you have two answers and no way to tell which one
the desk acted on.

The layout follows what you actually need to see, in the order you need it: what the
market is doing, what you own, what the agents think, and why the desk did or did not
act. The last of those is the panel most systems omit and the one that matters most
when a strategy is quiet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.widgets import DataTable, RichLog, Static

from quantdesk.analysis.patterns import Bias

if TYPE_CHECKING:  # pragma: no cover
    from quantdesk.app.desk import Desk

# Bloomberg-ish palette: amber on black, green up, red down.
UP = "bold #33dd77"
DOWN = "bold #ff5566"
FLAT = "#8899aa"
KEY = "bold #ffb000"
DIM = "#667788"


def _signed(value: float, fmt: str = "+.2f", suffix: str = "") -> Text:
    style = UP if value > 0 else DOWN if value < 0 else FLAT
    return Text(f"{value:{fmt}}{suffix}", style=style)


def _bias_text(bias: Bias) -> Text:
    return Text(
        bias.value.upper()[:4],
        style=UP if bias is Bias.BULLISH else DOWN if bias is Bias.BEARISH else FLAT,
    )


class MarketMonitor(DataTable):
    """Price, trend, and the structural read for every symbol in the universe."""

    def on_mount(self) -> None:
        self.cursor_type = "row"
        self.zebra_stripes = True
        for label, width in (
            ("SYMBOL", 11),
            ("LAST", 12),
            ("CHG%", 8),
            ("TREND", 7),
            ("DEG", 6),
            ("ADX", 6),
            ("RSI", 6),
            ("TF", 6),
            ("ATR%", 7),
            ("PATTERNS", 22),
        ):
            self.add_column(label, width=width)

    def refresh_from(self, desk: "Desk") -> None:
        self.clear()
        for symbol in desk.universe:
            sa = desk.pipeline.analysis.get(symbol)
            if sa is None or sa.last_bar is None:
                self.add_row(symbol, "-", "-", "-", "-", "-", "-", "-", "-", "waiting for data")
                continue

            price = sa.price
            change = desk.session_change(symbol)
            patterns = sa.patterns.actionable()
            pattern_text = (
                ", ".join(p.kind.label for p in patterns[:2]) if patterns else "-"
            )
            atr_pct = (sa.atr.value / price * 100.0) if price > 0 else 0.0

            self.add_row(
                Text(symbol, style=KEY),
                f"{price:,.4g}",
                _signed(change * 100.0, "+.2f", "%"),
                _bias_text(sa.trend_bias),
                sa.degree[:5],
                f"{sa.dmi.adx:.0f}" if sa.dmi.ready else "-",
                f"{sa.rsi.value:.0f}" if sa.rsi.ready else "-",
                _bias_text(sa.timeframes.permitted_direction()),
                f"{atr_pct:.2f}",
                Text(pattern_text, style=DIM),
            )


class PositionsPanel(DataTable):
    """Open positions with mark-to-market P&L."""

    def on_mount(self) -> None:
        self.zebra_stripes = True
        for label, width in (
            ("SYMBOL", 11),
            ("QTY", 14),
            ("AVG", 12),
            ("LAST", 12),
            ("VALUE", 13),
            ("UNREAL", 12),
            ("REAL", 12),
        ):
            self.add_column(label, width=width)

    def refresh_from(self, desk: "Desk") -> None:
        self.clear()
        positions = [p for p in desk.positions.values() if not p.is_flat]
        if not positions:
            self.add_row(Text("flat", style=DIM), "", "", "", "", "", "")
            return
        for pos in positions:
            last = pos.last_price or pos.avg_price
            self.add_row(
                Text(pos.symbol, style=KEY),
                _signed(pos.qty, "+,.6g"),
                f"{pos.avg_price:,.4g}",
                f"{last:,.4g}",
                f"{pos.market_value(last):,.0f}",
                _signed(pos.unrealized_pnl(last), "+,.0f"),
                _signed(pos.realized_pnl, "+,.0f"),
            )


class SignalBlotter(DataTable):
    """What each agent currently thinks, and why."""

    def on_mount(self) -> None:
        self.zebra_stripes = True
        for label, width in (
            ("SYMBOL", 11),
            ("AGENT", 20),
            ("SCORE", 8),
            ("CONF", 7),
            ("RATIONALE", 60),
        ):
            self.add_column(label, width=width)

    def refresh_from(self, desk: "Desk") -> None:
        self.clear()
        rows = 0
        for symbol in desk.universe:
            for sig in desk.pipeline.latest_signals(symbol):
                self.add_row(
                    Text(symbol, style=KEY),
                    sig.agent,
                    _signed(sig.score, "+.2f"),
                    f"{sig.confidence:.2f}",
                    Text(sig.rationale[:60], style=DIM),
                )
                rows += 1
        if rows == 0:
            self.add_row(Text("no agent has a view", style=DIM), "", "", "", "")


class ChecklistPanel(Static):
    """Murphy's chapter 19 checklist for the focused symbol."""

    def refresh_from(self, desk: "Desk") -> None:
        symbol = desk.focus_symbol
        verdict = desk.pipeline.verdicts.get(symbol)
        if verdict is None:
            self.update(Text("waiting for analysis...", style=DIM))
            return

        out = Text()
        head_style = (
            UP if verdict.bias is Bias.BULLISH
            else DOWN if verdict.bias is Bias.BEARISH
            else FLAT
        )
        out.append(f"{symbol}  ", style=KEY)
        out.append(f"{verdict.bias.value.upper()}\n", style=head_style)
        out.append(
            f"score {verdict.score:+.2f}   conviction {verdict.conviction:.2f}   "
            f"agreement {verdict.agreement:.0%}   answered {verdict.coverage:.0%}\n",
            style=DIM,
        )
        if verdict.blockers:
            for blocker in verdict.blockers:
                out.append(f"BLOCKED  {blocker}\n", style=DOWN)
        out.append("\n")

        for item in verdict.items:
            if item.status.value != "answered":
                out.append(f"  ?  {item.question[:52]}\n", style=DIM)
                continue
            mark, style = (
                ("+", UP) if item.bias is Bias.BULLISH
                else ("-", DOWN) if item.bias is Bias.BEARISH
                else ("=", FLAT)
            )
            out.append(f"  {mark}  ", style=style)
            out.append(f"{item.question[:52]}\n", style="#bbccdd")
            out.append(f"       {item.answer[:70]}\n", style=DIM)
        self.update(out)


class RiskPanel(Static):
    """Account state, Murphy's allocation limits, and the risk manager's mode."""

    def refresh_from(self, desk: "Desk") -> None:
        out = Text()
        equity = desk.equity
        start = desk.start_equity
        pnl = equity - start
        day_pct = (pnl / start) if start > 0 else 0.0

        out.append("EQUITY   ", style=DIM)
        out.append(f"{equity:>14,.2f}\n", style=KEY)
        out.append("SESSION  ", style=DIM)
        out.append(f"{pnl:>+14,.2f}  ({day_pct:+.2%})\n", style=UP if pnl >= 0 else DOWN)

        state = desk.pipeline.risk.state
        state_style = UP if state.value == "normal" else DOWN
        out.append("RISK     ", style=DIM)
        out.append(f"{state.value.upper()}\n", style=state_style)

        out.append("APPETITE ", style=DIM)
        out.append(f"{desk.pipeline.risk_appetite:.2f}\n", style=FLAT)

        gross = sum(abs(p.market_value(p.last_price)) for p in desk.positions.values())
        invested = (gross / equity) if equity > 0 else 0.0
        rules = desk.pipeline.money.rules
        style = DOWN if invested > rules.max_invested_pct else FLAT
        out.append("INVESTED ", style=DIM)
        out.append(
            f"{invested:.1%} of {rules.max_invested_pct:.0%} ceiling\n", style=style
        )
        out.append("RISK/MKT ", style=DIM)
        out.append(f"{rules.max_market_risk_pct:.0%} max\n", style=FLAT)
        out.append("R:R GATE ", style=DIM)
        out.append(f"{rules.min_reward_risk:.1f}:1\n", style=FLAT)

        out.append("\nROUTER\n", style=DIM)
        status = desk.router.status() if desk.router else {}
        out.append(f"  broker    {status.get('broker', '-')}\n", style=FLAT)
        out.append(f"  working   {status.get('working', 0)}\n", style=FLAT)
        out.append(f"  sent      {status.get('orders_sent', 0)}\n", style=FLAT)
        timeout = status.get("in_timeout") or []
        if timeout:
            out.append(f"  timeout   {', '.join(timeout)}\n", style=DOWN)
        self.update(out)


class DecisionLog(RichLog):
    """Orders sent and, just as importantly, why orders were not sent."""

    def on_mount(self) -> None:
        self.wrap = False
        self.markup = True
        self.max_lines = 500

    def log_decision(self, decision) -> None:
        stamp = decision.ts.strftime("%H:%M:%S")
        if decision.orders:
            for order in decision.orders:
                colour = "#33dd77" if order.side.value == "buy" else "#ff5566"
                self.write(
                    f"[{DIM}]{stamp}[/] [{KEY}]{decision.symbol:<10}[/] "
                    f"[{colour}]{order.side.value.upper():<4}[/] "
                    f"{order.qty:>12,.6g}  [{DIM}]{order.tag}[/]"
                )
            return
        # Only surface refusals that say something. "No agent had a view" every bar
        # is noise; a risk veto or a failed reward-to-risk test is worth reading.
        for reason in decision.rejected:
            if reason.startswith("no agent") or "rebalance threshold" in reason:
                continue
            self.write(
                f"[{DIM}]{stamp}[/] [{KEY}]{decision.symbol:<10}[/] "
                f"[#ffb000]skip[/] [{DIM}]{reason}[/]"
            )
