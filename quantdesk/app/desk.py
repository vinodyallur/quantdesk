"""The desk: the running system.

Owns the feed, the pipeline, the router and the account, and runs the loop that
connects them. Everything above this is presentation and everything below is a
component; this is where the session lives.

Three decisions worth explaining.

**Warmup is replayed through the pipeline, not skipped.** Murphy's structural
techniques need history - a swing pivot needs bars either side, a trend needs two
peaks and two troughs, a 50-period average needs fifty bars. Starting cold means the
desk is blind for hours. So historical bars are pushed through the same
:meth:`Pipeline.on_bar` path with trading suppressed, which warms every indicator and
detector to exactly the state it would have reached live.

**The broker is the authority on positions.** Every cycle reconciles before deciding.
Local bookkeeping drifts on a partial fill or a missed rejection, and a wrong position
size makes every risk figure downstream wrong too.

**Pausing and killing are different things.** Pause stops new entries and leaves exits
working. Kill cancels everything and halts. A control that blocks exits as well as
entries turns a bad session into a trapped one.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timedelta

from quantdesk.config import Settings, get_settings
from quantdesk.core.clock import LiveClock
from quantdesk.core.types import Bar, Position
from quantdesk.execution.broker import Broker
from quantdesk.execution.router import OrderRouter
from quantdesk.pipeline import BarDecision, Pipeline

log = logging.getLogger(__name__)


class Desk:
    """A live or simulated trading session."""

    def __init__(
        self,
        settings: Settings | None = None,
        pipeline: Pipeline | None = None,
        broker: Broker | None = None,
        feed=None,
        mode: str = "paper",
    ) -> None:
        self.settings = settings or get_settings()
        self.pipeline = pipeline or Pipeline(self.settings)
        self.mode = mode
        self.clock = LiveClock()
        self.feed = feed
        self.broker = broker
        self.router = OrderRouter(broker) if broker is not None else None

        self.universe: list[str] = list(self.settings.universe)
        self.focus_symbol = self.universe[0] if self.universe else ""
        # A paper broker carries its own starting capital, chosen per session. Taking it
        # from settings instead would silently ignore what the operator asked for.
        opening = getattr(broker, "starting_equity", None) or self.settings.starting_equity
        self.equity = opening
        self.start_equity = opening
        self.pipeline.equity = opening
        self.pipeline.start_equity = opening
        self.pipeline.peak_equity = opening
        self.positions: dict[str, Position] = {}
        self.paused = False
        self.running = False
        self.warmed = False
        self.bars_processed = 0
        self.errors: list[str] = []

        #: Set by the UI to receive decisions as they happen.
        self.on_decision: Callable[[BarDecision], None] | None = None
        self._session_open: dict[str, float] = {}
        self._task: asyncio.Task | None = None

    # ----------------------------------------------------------------- warmup
    async def warmup(self, history: dict[str, list[Bar]] | None = None) -> int:
        """Replay history through the pipeline with trading suppressed.

        Returns how many bars were absorbed. The pipeline's own decisions are computed
        during warmup (which is unavoidable, since the analysis and the decision share
        one call) but the resulting orders are discarded, so no trade is ever placed
        from historical data.
        """
        if history is None:
            if self.feed is None:
                return 0
            history = await self._fetch_history()

        merged: list[Bar] = []
        for bars in history.values():
            merged.extend(bars)
        merged.sort(key=lambda b: (b.ts, b.symbol))

        for bar in merged:
            self.pipeline.on_bar(bar)
            self.bars_processed += 1
            self._session_open.setdefault(bar.symbol, bar.close)
        # Decisions made during warmup are historical and must not be acted on.
        self.pipeline.decisions.clear()
        # The same applies to trades. Replaying history through ``Pipeline.on_bar`` calls
        # ``open_trade`` whenever the strategy would have entered, but the resulting orders
        # were discarded, so the broker holds nothing. Left in place, the desk starts up
        # believing it is in the market: it manages and then *exits* positions it never
        # opened, and each of those exits opens a real position in the opposite direction.
        # It also refuses new entries in those symbols, because a symbol with an active
        # trade never reaches the sizing path.
        phantom = list(self.pipeline.trades.trades)
        for symbol in phantom:
            self.pipeline.trades.close_trade(symbol)
        if phantom:
            log.info(
                "cleared %d trade(s) opened during warmup replay: %s",
                len(phantom), ", ".join(phantom),
            )
        self.warmed = True
        log.info("warmed up on %d bars across %d symbols", len(merged), len(history))
        return len(merged)

    async def _fetch_history(self) -> dict[str, list[Bar]]:
        """Pull enough history to warm the analysis stack.

        The window is derived from the configured bar size so ``warmup_bars`` means
        the same thing on any timeframe: asking for 300 bars should fetch 300 bars'
        worth of wall-clock time, not a fixed number of days.
        """
        fetch = getattr(self.feed, "history", None)
        if fetch is None:
            return {}
        minutes = self.settings.timeframe_minutes * self.settings.warmup_bars
        # Ask for extra: exchanges have gaps, and a short fetch leaves the desk blind.
        start = self.clock.now() - timedelta(minutes=minutes * 1.5)
        result = fetch(start=start, symbols=self.universe)
        if asyncio.iscoroutine(result):
            result = await result
        return result or {}

    # -------------------------------------------------------------------- run
    async def run(self) -> None:
        """Main loop: reconcile, take bars, decide, route."""
        if self.feed is None:
            raise RuntimeError("the desk needs a feed to run")
        if not self.warmed:
            await self.warmup()

        self.running = True
        log.info("desk running in %s mode on %s", self.mode, ", ".join(self.universe))
        try:
            async for bar in self.feed.stream():
                if not self.running:
                    break
                await self._handle_bar(bar)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        except Exception as exc:  # noqa: BLE001
            self._record_error(f"feed loop failed: {exc}")
            log.exception("desk loop failed")
        finally:
            self.running = False

    async def _handle_bar(self, bar: Bar) -> None:
        """One bar: sync with the broker, decide, then route."""
        self.bars_processed += 1
        self._session_open.setdefault(bar.symbol, bar.close)

        # Match pending orders first, at this bar's open. An order decided on the previous
        # bar must fill before this bar's decision is made, exactly as in the backtester -
        # and without this call a simulated session submits orders that never fill at all,
        # so equity never moves and the session looks inert rather than broken.
        self._tick_broker(bar)
        await self.sync()
        decision = self.pipeline.on_bar(bar)

        if self.on_decision is not None:
            try:
                self.on_decision(decision)
            except Exception as exc:  # noqa: BLE001 - display must not stop trading
                self._record_error(f"decision callback failed: {exc}")

        if not decision.orders or self.router is None:
            return

        # Paused or halted: exits still go, entries do not.
        # ``can_open`` is a property. Calling it raised TypeError on the first bar that
        # produced an order, which killed the desk loop the moment it tried to trade.
        allow_opening = not self.paused and self.pipeline.risk.can_open
        result = await self.router.route(
            decision.orders,
            positions=self.positions,
            allow_opening=allow_opening,
            now=self.clock.now(),
        )
        for order, reason in result.blocked:
            decision.rejected.append(f"router blocked {order.symbol}: {reason}")

    def _tick_broker(self, bar: Bar) -> None:
        """Advance a simulating broker by one bar, if it matches its own fills.

        A real broker fills on its own venue and has no such method, so this is a no-op
        against Alpaca. Guarded rather than assumed, because the desk must run against
        either without knowing which it has.
        """
        tick = getattr(self.broker, "on_bar", None)
        if tick is None:
            return
        try:
            tick(bar)
        except Exception as exc:  # noqa: BLE001 - a match failure must not stop the loop
            self._record_error(f"broker tick failed: {exc}")

    async def sync(self) -> None:
        """Adopt the broker's view of positions and equity."""
        if self.router is None or self.broker is None:
            return
        try:
            positions = await self.router.reconcile()
            self.positions = positions
            account = await self.broker.account()
            if account is not None:
                self.equity = account.equity
                if self.start_equity <= 0:
                    self.start_equity = account.day_start_equity or account.equity
            self.pipeline.sync_account(self.equity, self.positions)
        except Exception as exc:  # noqa: BLE001
            self._record_error(f"sync failed: {exc}")

    # ------------------------------------------------------------- controls
    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        log.warning("entries %s", "paused" if self.paused else "resumed")
        return self.paused

    async def kill(self, reason: str = "manual") -> str:
        """Cancel working orders and halt the desk. Exits remain possible."""
        self.paused = True
        self.pipeline.risk.kill(reason, self.clock.now())
        if self.router is None:
            return "halted (no broker attached)"
        detail = await self.router.kill(flatten=False)
        log.warning("desk killed: %s (%s)", reason, detail)
        return detail

    async def stop(self) -> None:
        self.running = False
        if self._task is not None:
            self._task.cancel()
        if self.feed is not None:
            closer = getattr(self.feed, "close", None)
            if closer is not None:
                result = closer()
                if asyncio.iscoroutine(result):
                    await result
        if self.broker is not None:
            await self.broker.close()

    def cycle_focus(self) -> str:
        if not self.universe:
            return ""
        index = self.universe.index(self.focus_symbol) if self.focus_symbol in self.universe else -1
        self.focus_symbol = self.universe[(index + 1) % len(self.universe)]
        return self.focus_symbol

    # -------------------------------------------------------------- readouts
    def session_change(self, symbol: str) -> float:
        """Fractional move since the desk started watching this symbol."""
        opened = self._session_open.get(symbol, 0.0)
        sa = self.pipeline.analysis.get(symbol)
        if opened <= 0 or sa is None or sa.price <= 0:
            return 0.0
        return sa.price / opened - 1.0

    def _record_error(self, message: str) -> None:
        self.errors.append(f"{datetime.now():%H:%M:%S} {message}")
        if len(self.errors) > 50:
            del self.errors[:-50]

    def status(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "running": self.running,
            "paused": self.paused,
            "warmed": self.warmed,
            "bars": self.bars_processed,
            "equity": self.equity,
            "positions": len([p for p in self.positions.values() if not p.is_flat]),
            "risk_state": self.pipeline.risk.state.value,
            "errors": len(self.errors),
        }
