"""Live perpetual-futures bars from Binance USD-M futures.

Real prices for the contract the desk is actually reasoning about. The Alpaca feed serves
*spot* crypto, and perps do not trade at spot: the basis moves, funding is charged, and a
strategy sized against spot volatility is sized against the wrong series.

No API key. Binance serves futures market data publicly, verified against
``/fapi/v1/klines`` reaching back to September 2019 for BTCUSDT.

**Only closed bars are emitted.** The last kline Binance returns is the one still forming,
and its high, low and close all move. Acting on it means acting on a bar that has not
happened, which produces a live session that cannot be reproduced by a backtest. The
forming bar is filtered by comparing its close time against the clock, and it is available
separately through :meth:`forming` for display only.

Streaming is REST polling rather than the websocket. The desk trades on closed bars, so the
only thing a socket would improve is the latency of learning a bar has closed - worth
little at minute-or-longer bars, against a reconnect-and-resubscribe path that is a real
source of silent gaps.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

from quantdesk.core.types import Bar, from_epoch
from quantdesk.data.feed import DataFeed
from quantdesk.data.perps import PerpsError, _binance_symbol

log = logging.getLogger(__name__)

BASE = "https://fapi.binance.com"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

#: Desk timeframe label -> Binance interval string.
_INTERVALS = {
    "1Min": "1m",
    "5Min": "5m",
    "15Min": "15m",
    "1Hour": "1h",
    "1Day": "1d",
}

_MINUTES = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60, "1Day": 1440}


def _get(url: str, timeout: float = 20.0):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise PerpsError(f"{url} returned HTTP {exc.code} ({exc.reason})") from exc
    except OSError as exc:
        raise PerpsError(f"{url} unreachable: {exc}") from exc


class BinancePerpsFeed(DataFeed):
    """Closed perpetual-futures bars, polled."""

    def __init__(
        self,
        symbols: list[str],
        timeframe: str = "1Min",
        poll_seconds: int = 15,
        timeout: float = 20.0,
    ) -> None:
        if timeframe not in _INTERVALS:
            raise ValueError(f"timeframe must be one of {sorted(_INTERVALS)}")
        super().__init__(symbols, timeframe)
        self.interval = _INTERVALS[timeframe]
        self.bar_minutes = _MINUTES[timeframe]
        self.poll_seconds = max(2, poll_seconds)
        self.timeout = timeout
        self._last_emitted: dict[str, datetime] = {}
        self._forming: dict[str, Bar] = {}
        self._closed = False

    # ---------------------------------------------------------------- history
    def history(
        self,
        start: datetime,
        end: datetime | None = None,
        symbols: list[str] | None = None,
    ) -> dict[str, list[Bar]]:
        out: dict[str, list[Bar]] = {}
        for symbol in symbols or self.symbols:
            try:
                out[symbol] = self._klines(symbol, start, end)
            except PerpsError as exc:
                log.error("history failed for %s: %s", symbol, exc)
                out[symbol] = []
        return out

    def _klines(
        self, symbol: str, start: datetime, end: datetime | None
    ) -> list[Bar]:
        """Page through klines. Binance caps a response at 1500 rows."""
        venue = _binance_symbol(symbol)
        cursor = int(start.timestamp() * 1000)
        ceiling = int((end or datetime.now(timezone.utc)).timestamp() * 1000)
        bars: list[Bar] = []
        seen: set[int] = set()

        while cursor < ceiling and len(bars) < 20_000:
            url = (
                f"{BASE}/fapi/v1/klines?symbol={urllib.parse.quote(venue)}"
                f"&interval={self.interval}&startTime={cursor}"
                f"&endTime={ceiling}&limit=1500"
            )
            rows = _get(url, self.timeout)
            if not isinstance(rows, list) or not rows:
                break
            for row in rows:
                bar = self._to_bar(symbol, row)
                if bar is None or bar.ts.timestamp() * 1000 in seen:
                    continue
                seen.add(bar.ts.timestamp() * 1000)
                bars.append(bar)
            newest = max(int(r[0]) for r in rows)
            if newest <= cursor:
                break
            cursor = newest + 1
            if len(rows) < 1500:
                break

        bars.sort(key=lambda b: b.ts)
        return bars

    def _to_bar(self, symbol: str, row: list) -> Bar | None:
        """Convert one kline, discarding it if it has not closed yet.

        Binance's array is [openTime, o, h, l, c, volume, closeTime, quoteVolume,
        trades, ...]. The close time is what decides whether this bar is final.
        """
        close_ms = int(row[6])
        now_ms = datetime.now(timezone.utc).timestamp() * 1000
        bar = Bar(
            symbol=symbol,
            ts=from_epoch(int(row[0]) / 1000.0),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            trade_count=int(row[8]) if len(row) > 8 else None,
        )
        if close_ms > now_ms:
            # Still forming. Kept for display, never returned as a closed bar.
            self._forming[symbol] = bar
            return None
        return bar

    def forming(self, symbol: str) -> Bar | None:
        """The in-progress bar, for the chart only. Never trade on this."""
        return self._forming.get(symbol)

    def mark_prices(self) -> dict[str, float]:
        """Current mark price per symbol, for display between bar closes."""
        out: dict[str, float] = {}
        for symbol in self.symbols:
            try:
                venue = _binance_symbol(symbol)
                data = _get(
                    f"{BASE}/fapi/v1/premiumIndex?symbol={urllib.parse.quote(venue)}",
                    self.timeout,
                )
                out[symbol] = float(data["markPrice"])
            except (PerpsError, KeyError, TypeError, ValueError) as exc:
                log.debug("mark price failed for %s: %s", symbol, exc)
        return out

    # ----------------------------------------------------------------- stream
    async def stream(self) -> AsyncIterator[Bar]:
        """Poll for newly closed bars and yield them in timestamp order."""
        lookback = timedelta(minutes=self.bar_minutes * 3)
        seed = await asyncio.to_thread(
            self.history, datetime.now(timezone.utc) - lookback
        )
        for symbol, bars in seed.items():
            if bars:
                self._last_emitted[symbol] = bars[-1].ts

        while not self._closed:
            try:
                await asyncio.sleep(self.poll_seconds)
                if self._closed:
                    break
                fresh = await asyncio.to_thread(
                    self.history, datetime.now(timezone.utc) - lookback
                )
                new: list[Bar] = []
                for symbol, bars in fresh.items():
                    high_water = self._last_emitted.get(symbol)
                    for bar in bars:
                        if high_water is None or bar.ts > high_water:
                            new.append(bar)
                # Chronological across symbols, so the pipeline's cross-sectional
                # agents never see one symbol ahead of another.
                new.sort(key=lambda b: (b.ts, b.symbol))
                for bar in new:
                    self._last_emitted[bar.symbol] = bar.ts
                    yield bar
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a bad poll must not kill the desk
                log.warning("perps poll failed, retrying: %s", exc)

    async def close(self) -> None:
        self._closed = True
