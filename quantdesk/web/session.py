"""Session manager for the browser terminal.

Wraps a paper session so an HTTP handler can start it, stop it, and read a
JSON-serialisable snapshot of it, without knowing anything about asyncio.

The desk loop is asyncio and the HTTP server is threads, so the two are kept apart: the
session owns a background thread running its own event loop, and the only thing crossing
the boundary is :meth:`PaperSession.snapshot`, which reads state and never mutates it.
Letting a request handler touch the desk directly would mutate trading state from a
network thread mid-decision.

Snapshot building is defensive throughout. It runs while the desk is mutating the same
objects, and a display read must never raise into the HTTP layer or, worse, interrupt the
loop that is managing open positions.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

MAX_CHART_BARS = 240
MAX_LOG = 300


class PaperSession:
    """One paper trading session: live prices, simulated money."""

    def __init__(self) -> None:
        self.desk = None
        self.broker = None
        self.feed = None
        self.symbols: list[str] = []
        self.timeframe = "1Min"
        self.capital = 0.0
        self.status = "idle"
        self.error = ""
        self.started_at: datetime | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._funding: dict[str, dict] = {}
        self._marks: dict[str, float] = {}
        self._log: list[dict] = []
        self.max_leverage = 1.0
        self.confidence_leverage = False
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ start
    def start(
        self,
        capital: float,
        symbols: list[str],
        timeframe: str,
        max_leverage: float = 1.0,
        confidence_leverage: bool = False,
    ) -> None:
        """Validate, then boot the session on its own thread."""
        if self.status in ("starting", "running"):
            raise RuntimeError("a session is already running; stop it first")
        if capital <= 0:
            raise ValueError("demo capital must be positive")
        if not symbols:
            raise ValueError("choose at least one symbol")

        from quantdesk.data.binance_feed import _INTERVALS

        if timeframe not in _INTERVALS:
            raise ValueError(f"timeframe must be one of {sorted(_INTERVALS)}")

        if not 1.0 <= max_leverage <= 125.0:
            raise ValueError("max leverage must be between 1 and 125")
        self.capital = float(capital)
        self.symbols = list(symbols)
        self.timeframe = timeframe
        self.max_leverage = float(max_leverage)
        self.confidence_leverage = bool(confidence_leverage)
        self.status = "starting"
        self.error = ""
        self._stop_flag.clear()
        with self._lock:
            self._log = []

        self._thread = threading.Thread(target=self._run_thread, daemon=True,
                                        name="quantdesk-session")
        self._thread.start()

    def _run_thread(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._boot_and_run())
        except Exception as exc:  # noqa: BLE001
            log.exception("session thread failed")
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "failed"
        finally:
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass
            if self.status == "running":
                self.status = "stopped"

    async def _boot_and_run(self) -> None:
        from quantdesk.app.desk import Desk
        from quantdesk.config import get_settings
        from quantdesk.data.binance_feed import BinancePerpsFeed
        from quantdesk.execution.paper_broker import PaperBroker

        base = get_settings()
        settings = base.model_copy(update={
            "timeframe": self.timeframe,
            "starting_equity": self.capital,
            "max_leverage": self.max_leverage,
            "confidence_leverage": self.confidence_leverage,
            # With confidence sizing on, the flat figure is unused; kept at 1x so a
            # misconfiguration cannot silently leverage everything.
            "leverage": 1.0 if self.confidence_leverage else self.max_leverage,
        })
        feed = BinancePerpsFeed(self.symbols, self.timeframe, poll_seconds=10)
        broker = PaperBroker(
            starting_equity=self.capital,
            commission_bps=settings.commission_bps,
            slippage_bps=settings.slippage_bps,
        )
        desk = Desk(settings=settings, broker=broker, feed=feed,
                    mode="PAPER (live Binance perps)")
        desk.universe = list(self.symbols)
        desk.focus_symbol = self.symbols[0]
        desk.on_decision = self._record_decision

        # History is fetched in a thread: the feed's history() is synchronous and would
        # otherwise block this loop for several seconds before it starts trading.
        minutes = settings.timeframe_minutes * settings.warmup_bars
        start = datetime.now(timezone.utc) - timedelta(minutes=minutes * 1.5)
        history = await asyncio.to_thread(feed.history, start, None, self.symbols)
        total = sum(len(v) for v in history.values())
        if total == 0:
            raise RuntimeError(
                "Binance returned no bars. Perps are quoted in USDT, so BTC/USD maps to "
                "BTCUSDT; check the symbols exist as USD-M contracts."
            )
        await desk.warmup(history)

        # Per-contract maintenance margin drives the liquidation-distance cap. Without it
        # every symbol falls back to the tier-1 default, which under-margins the exotics.
        try:
            from quantdesk.data.symbols import UNIVERSE

            for symbol in self.symbols:
                contract = UNIVERSE.get(symbol)
                if contract is not None and contract.maint_margin_pct > 0:
                    desk.pipeline.maint_margin[symbol] = contract.maint_margin_pct
        except Exception as exc:  # noqa: BLE001 - fall back to the default, do not fail
            log.warning("could not load contract margins: %s", exc)

        self.feed, self.broker, self.desk = feed, broker, desk
        self.started_at = datetime.now(timezone.utc)
        self.status = "running"
        log.info("session live: %.2f demo, %d warmup bars, %s on %s",
                 self.capital, total, self.timeframe, ", ".join(self.symbols))

        threading.Thread(target=self._poll_market, daemon=True,
                         name="quantdesk-market").start()
        await desk.run()
        self.status = "stopped"

    # ----------------------------------------------------------- market poller
    def _poll_market(self) -> None:
        """Refresh funding, open interest and marks on a slow loop.

        Separate from the trading loop because funding settles every eight hours, so
        polling it per bar is wasted requests, and a funding outage must not interrupt
        trading.
        """
        from quantdesk.data.perps import (
            BinancePerps,
            open_interest_signal,
            summarise_funding,
        )

        source = BinancePerps()
        while not self._stop_flag.is_set() and self.status == "running":
            try:
                end = datetime.now(timezone.utc)
                snapshot: dict[str, dict] = {}
                for symbol in list(self.symbols):
                    rates = source.funding(symbol, start=end - timedelta(days=7), end=end)
                    if not rates:
                        continue
                    interest = source.open_interest(symbol, period="1d", limit=30)
                    marks = [r.mark_price for r in rates if r.mark_price is not None]
                    reading = (
                        open_interest_signal(marks, [o.contracts for o in interest])
                        if marks and len(interest) >= 2 else None
                    )
                    summary = summarise_funding(rates, symbol)
                    latest = rates[-1]
                    snapshot[symbol] = {
                        "rate": latest.rate,
                        "rate_annualised": latest.annualised,
                        "carry": summary.annualised_cost,
                        "positive_share": summary.positive_share,
                        "settlements": summary.count,
                        "lean": (
                            "crowded long" if summary.crowded_long
                            else "crowded short" if summary.crowded_short
                            else "no persistent lean"
                        ),
                        "oi_reading": reading[0] if reading else "",
                        "oi_why": reading[1] if reading else "",
                        "oi_points": len(interest),
                        "oi_first": interest[0].contracts if interest else 0.0,
                        "oi_last": interest[-1].contracts if interest else 0.0,
                    }
                marks_now = self.feed.mark_prices() if self.feed else {}
                # Confidence scores funding as a real, published cost of holding a
                # direction, unlike every other input, which is an estimate.
                if self.desk is not None:
                    for symbol, payload in snapshot.items():
                        self.desk.pipeline.funding_carry[symbol] = payload["carry"]
                self._funding = snapshot
                if marks_now:
                    self._marks = marks_now
            except Exception as exc:  # noqa: BLE001
                log.debug("market poll failed: %s", exc)
            for _ in range(45):
                if self._stop_flag.is_set() or self.status != "running":
                    return
                time.sleep(1)

    # ------------------------------------------------------------- decisions
    def _record_decision(self, decision) -> None:
        entries: list[dict] = []
        stamp = decision.ts.strftime("%H:%M:%S")
        if decision.orders:
            for order in decision.orders:
                entries.append({
                    "ts": stamp,
                    "symbol": decision.symbol,
                    "kind": "order",
                    "side": order.side.value,
                    "qty": order.qty,
                    "risk": order.planned_risk,
                    "text": order.tag[:60],
                })
        else:
            for reason in decision.rejected:
                # "No agent had a view" every bar buries the refusals worth reading.
                if reason.startswith("no agent") or "rebalance threshold" in reason:
                    continue
                entries.append({
                    "ts": stamp, "symbol": decision.symbol, "kind": "skip",
                    "side": "", "qty": 0.0, "risk": 0.0, "text": reason[:70],
                })
        if not entries:
            return
        with self._lock:
            self._log.extend(entries)
            if len(self._log) > MAX_LOG:
                del self._log[:-MAX_LOG]

    # --------------------------------------------------------------- controls
    def pause(self) -> bool:
        if self.desk is None:
            raise RuntimeError("no session running")
        return self.desk.toggle_pause()

    def kill(self) -> str:
        if self.desk is None or self._loop is None:
            raise RuntimeError("no session running")
        future = asyncio.run_coroutine_threadsafe(
            self.desk.kill("browser kill switch"), self._loop
        )
        return future.result(timeout=15)

    def stop(self) -> str:
        desk, loop = self.desk, self._loop
        self._stop_flag.set()
        if desk is None or loop is None:
            self.status = "idle"
            return "nothing was running"
        desk.running = False
        try:
            asyncio.run_coroutine_threadsafe(desk.stop(), loop).result(timeout=15)
        except Exception as exc:  # noqa: BLE001
            log.debug("stop raised: %s", exc)
        self.status = "stopped"
        return "session stopped"

    def reset(self) -> None:
        """Clear a finished session so a new one can start with fresh capital."""
        if self.status == "running":
            self.stop()
        self.desk = self.broker = self.feed = None
        self.status = "idle"
        self.error = ""
        self.started_at = None
        self._funding = {}
        self._marks = {}
        with self._lock:
            self._log = []

    # --------------------------------------------------------------- snapshot
    def snapshot(self, focus: str | None = None) -> dict:
        """Everything the browser needs, as plain JSON types."""
        out: dict = {
            "status": self.status,
            "error": self.error,
            "capital": self.capital,
            "symbols": list(self.symbols),
            "timeframe": self.timeframe,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "funding": self._funding,
            "marks": self._marks,
            "max_leverage": self.max_leverage,
            "confidence_leverage": self.confidence_leverage,
        }
        if self.broker is None or self.desk is None:
            out["log"] = []
            return out

        try:
            out.update(self._account_block())
            out["positions"] = self._positions_block()
            out["chart"] = self._chart_block(focus or self.desk.focus_symbol)
            out["equity_curve"] = self._equity_block()
            out["results"] = self._results_block()
            out["trades"] = self._trades_block()
            out["conviction"] = self._conviction_block()
        except Exception as exc:  # noqa: BLE001 - a display read must never raise
            log.debug("snapshot failed: %s", exc)
            out["snapshot_error"] = str(exc)
        with self._lock:
            out["log"] = list(self._log[-60:])
        return out

    def _conviction_block(self) -> list[dict]:
        """Latest confidence and leverage per symbol, with the breakdown kept.

        The breakdown is the point: when a trade goes wrong the only useful question is
        which piece of evidence was wrong, and a bare score cannot answer it.
        """
        pipeline = self.desk.pipeline
        out = []
        for symbol in self.symbols:
            conf = pipeline.confidences.get(symbol)
            lev = pipeline.leverages.get(symbol)
            if conf is None and lev is None:
                continue
            row: dict = {"symbol": symbol}
            if conf is not None:
                row.update({
                    "score": conf.score,
                    "band": conf.band,
                    "tradable": conf.tradable,
                    "components": [
                        {"name": k, "value": v, "weight": conf.weights.get(k, 0.0)}
                        for k, v in sorted(
                            conf.components.items(),
                            key=lambda kv: -conf.weights.get(kv[0], 0.0),
                        )
                    ],
                    "notes": list(conf.notes),
                })
            if lev is not None:
                row.update({
                    "leverage": lev.leverage,
                    "requested": lev.requested,
                    "binding": lev.binding,
                    "capped": lev.capped,
                    "stop_distance_pct": lev.stop_distance_pct,
                    "liquidation_move_pct": lev.liquidation_move_pct,
                    "liquidation_cap": lev.liquidation_cap,
                })
            out.append(row)
        return out
    def _account_block(self) -> dict:
        ledger = self.broker.account_ledger
        equity = ledger.equity
        start = self.broker.starting_equity
        return {
            "equity": equity,
            "start_equity": start,
            "pnl": equity - start,
            "pnl_pct": (equity - start) / start if start else 0.0,
            "cash": ledger.cash,
            "realized": ledger.realized_pnl,
            "unrealized": ledger.unrealized_pnl,
            "fees": ledger.fees_paid,
            "bars": self.desk.bars_processed,
            "pending": self.broker.pending_orders,
            "paused": self.desk.paused,
            "running": self.desk.running,
            "risk_state": self.desk.pipeline.risk.state.value,
            "mode": self.desk.mode,
            "last_error": self.desk.errors[-1] if self.desk.errors else "",
        }

    def _positions_block(self) -> list[dict]:
        out = []
        for pos in self.broker.account_ledger.positions.values():
            if pos.is_flat:
                continue
            last = self._marks.get(pos.symbol) or pos.last_price or pos.avg_price
            out.append({
                "symbol": pos.symbol,
                "qty": pos.qty,
                "avg": pos.avg_price,
                "last": last,
                "value": pos.market_value(last),
                "unrealized": pos.unrealized_pnl(last),
            })
        return out

    def _chart_block(self, symbol: str) -> dict:
        bars = self.desk.pipeline.book.series(symbol).bars()[-MAX_CHART_BARS:]
        forming = self.feed.forming(symbol) if self.feed else None
        return {
            "symbol": symbol,
            "bars": [
                {
                    "t": b.ts.isoformat(),
                    "o": b.open, "h": b.high, "l": b.low, "c": b.close,
                    "v": b.volume,
                }
                for b in bars
            ],
            "forming": (
                {"t": forming.ts.isoformat(), "o": forming.open, "h": forming.high,
                 "l": forming.low, "c": forming.close, "v": forming.volume}
                if forming is not None else None
            ),
        }

    def _equity_block(self) -> list[dict]:
        curve = self.broker.equity_curve
        # Thin to a fixed budget so a long session does not grow the payload without
        # bound; the browser cannot draw more points than it has pixels anyway.
        step = max(1, len(curve) // 600)
        return [
            {"t": tick.ts.isoformat(), "e": tick.equity}
            for tick in curve[::step]
        ]

    def _results_block(self) -> dict:
        r = self.broker.results()
        return {
            "trades": r.trades, "wins": r.wins, "losses": r.losses,
            "hit_rate": r.hit_rate, "best": r.best, "worst": r.worst,
            "fills": r.fills, "open_positions": r.open_positions,
            "max_drawdown": r.max_drawdown, "peak_equity": r.peak_equity,
            "total_pnl": r.total_pnl, "total_return": r.total_return,
            "fees_paid": r.fees_paid, "net_of_fees": r.net_of_fees,
            "summary": r.summary(),
        }

    def _trades_block(self) -> list[dict]:
        return [
            {
                "symbol": t.symbol,
                "side": "long" if t.side > 0 else "short",
                "opened": t.opened_at.strftime("%m-%d %H:%M"),
                "closed": t.closed_at.strftime("%m-%d %H:%M"),
                "entry": t.entry, "exit": t.exit, "qty": t.qty,
                "pnl": t.pnl, "fees": t.fees,
                "return_pct": t.return_pct,
            }
            for t in self.broker.round_trips[-40:][::-1]
        ]
