"""Alpaca market data feed.

Crypto bars are served without authentication, so the desk works end-to-end
before you have any keys. US equities need paper keys and use the free IEX feed
by default.

Streaming is implemented by polling the REST bars endpoint rather than the
websocket. Reasons:

* Keyless crypto has no websocket, so polling is the only path that works with
  zero setup.
* We only ever emit *closed* bars, which is what the backtester replays. Acting
  on a partially formed bar would make live results diverge from backtests in a
  way that is very hard to notice.

At 5-minute bars a 20-second poll is far inside any latency that matters for this
class of strategy. If you move to sub-minute trading, swap this for the websocket
stream.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

from quantdesk.core.types import Bar
from quantdesk.data.feed import DataFeed

log = logging.getLogger(__name__)


_UNIT_MAP = {
    "1Min": (1, "Minute"),
    "5Min": (5, "Minute"),
    "15Min": (15, "Minute"),
    "1Hour": (1, "Hour"),
    "1Day": (1, "Day"),
}


def timeframe_delta(timeframe: str) -> timedelta:
    """Bar duration as a timedelta."""
    amount, unit = _UNIT_MAP[timeframe]
    if unit == "Minute":
        return timedelta(minutes=amount)
    if unit == "Hour":
        return timedelta(hours=amount)
    return timedelta(days=amount)


def _build_timeframe(timeframe: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    amount, unit = _UNIT_MAP[timeframe]
    return TimeFrame(amount, getattr(TimeFrameUnit, unit))


def _to_bar(symbol: str, raw: Any) -> Bar:
    """Convert an alpaca-py Bar model into our domain Bar."""
    ts: datetime = raw.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    tc = getattr(raw, "trade_count", None)
    return Bar(
        symbol=symbol,
        ts=ts,
        open=float(raw.open),
        high=float(raw.high),
        low=float(raw.low),
        close=float(raw.close),
        volume=float(raw.volume or 0.0),
        vwap=float(raw.vwap) if getattr(raw, "vwap", None) is not None else None,
        trade_count=int(tc) if tc is not None else None,
    )


class AlpacaFeed(DataFeed):
    """Historical + polled-streaming bars from Alpaca."""

    def __init__(
        self,
        symbols: list[str],
        timeframe: str = "5Min",
        asset_class: str = "crypto",
        api_key: str | None = None,
        secret_key: str | None = None,
        stock_feed: str = "iex",
        poll_seconds: int = 20,
    ) -> None:
        super().__init__(symbols, timeframe)
        self.asset_class = asset_class
        self.stock_feed = stock_feed
        self.poll_seconds = max(2, poll_seconds)
        self._tf = _build_timeframe(timeframe)
        self._bar_delta = timeframe_delta(timeframe)
        self._last_emitted: dict[str, datetime] = {}
        self._closed = False

        if asset_class == "crypto":
            from alpaca.data.historical import CryptoHistoricalDataClient

            # Keys are optional for crypto data and improve rate limits if present.
            self._client = CryptoHistoricalDataClient(api_key, secret_key)
        else:
            if not (api_key and secret_key):
                raise ValueError(
                    "US equity data requires Alpaca API keys. Set QD_ALPACA_API_KEY "
                    "and QD_ALPACA_SECRET_KEY, or use QD_ASSET_CLASS=crypto which "
                    "needs no keys."
                )
            from alpaca.data.historical import StockHistoricalDataClient

            self._client = StockHistoricalDataClient(api_key, secret_key)

    # ---------------------------------------------------------------- history
    def history(
        self,
        start: datetime,
        end: datetime | None = None,
        symbols: list[str] | None = None,
    ) -> dict[str, list[Bar]]:
        syms = symbols or self.symbols
        # Alpaca rejects an `end` in the future; also back off one bar so we never
        # hand a still-forming bar to the caller.
        ceiling = datetime.now(timezone.utc) - self._bar_delta
        end = min(end or ceiling, ceiling)
        if end <= start:
            return {s: [] for s in syms}

        try:
            barset = self._fetch(syms, start, end)
        except Exception as exc:  # noqa: BLE001 - surface as empty, caller decides
            log.error("Alpaca history request failed: %s", exc)
            return {s: [] for s in syms}

        out: dict[str, list[Bar]] = {s: [] for s in syms}
        for sym, raws in barset.data.items():
            out[sym] = [_to_bar(sym, r) for r in raws]
        return out

    def _fetch(self, syms: list[str], start: datetime, end: datetime):
        if self.asset_class == "crypto":
            from alpaca.data.requests import CryptoBarsRequest

            req = CryptoBarsRequest(
                symbol_or_symbols=syms,
                timeframe=self._tf,
                start=start,
                end=end,
            )
            return self._client.get_crypto_bars(req)

        from alpaca.data.requests import StockBarsRequest

        req = StockBarsRequest(
            symbol_or_symbols=syms,
            timeframe=self._tf,
            start=start,
            end=end,
            feed=self.stock_feed,
        )
        return self._client.get_stock_bars(req)

    # ----------------------------------------------------------------- stream
    async def stream(self) -> AsyncIterator[Bar]:
        """Poll for newly closed bars and yield them in timestamp order."""
        # Seed high-water marks so we don't replay history as if it were live.
        lookback = self._bar_delta * 5
        seed = self.history(datetime.now(timezone.utc) - lookback)
        for sym, bars in seed.items():
            if bars:
                self._last_emitted[sym] = bars[-1].ts

        while not self._closed:
            try:
                await asyncio.sleep(self.poll_seconds)
                if self._closed:
                    break
                fresh = await asyncio.to_thread(
                    self.history, datetime.now(timezone.utc) - lookback
                )
                new_bars: list[Bar] = []
                for sym, bars in fresh.items():
                    hwm = self._last_emitted.get(sym)
                    for bar in bars:
                        if hwm is None or bar.ts > hwm:
                            new_bars.append(bar)
                # Chronological across symbols keeps the pipeline's view of time sane.
                new_bars.sort(key=lambda b: b.ts)
                for bar in new_bars:
                    self._last_emitted[bar.symbol] = bar.ts
                    yield bar
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a bad poll must not kill the desk
                log.warning("stream poll failed, retrying: %s", exc)

    async def close(self) -> None:
        self._closed = True
