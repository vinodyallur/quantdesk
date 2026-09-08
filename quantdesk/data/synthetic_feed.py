"""Offline synthetic feed.

Generates regime-switching price paths with no network access. Two uses:

* Deterministic development and testing. Same seed, same bars, same trades.
* Demoing the live desk quickly - ``stream()`` can emit bars far faster than real
  time so you can watch the agents work without waiting for real 5-minute bars.

The generator alternates between trending and mean-reverting regimes with
stochastic volatility. That matters: a pure random walk would make every strategy
look equally worthless, so it would tell you nothing about whether the plumbing
works. This is a test fixture, not a market simulator. Never read anything about
edge into results produced here.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import numpy as np

from quantdesk.core.types import Bar
from quantdesk.data.feed import DataFeed
from quantdesk.data.alpaca_feed import timeframe_delta


class SyntheticFeed(DataFeed):
    """Regime-switching random price paths."""

    def __init__(
        self,
        symbols: list[str],
        timeframe: str = "5Min",
        seed: int = 7,
        start_price: float = 100.0,
        bar_vol: float = 0.004,
        emit_interval: float = 0.25,
    ) -> None:
        super().__init__(symbols, timeframe)
        self.seed = seed
        self.start_price = start_price
        self.bar_vol = bar_vol
        self.emit_interval = emit_interval
        self._bar_delta = timeframe_delta(timeframe)
        self._closed = False
        # Independent stream per symbol so symbols are not perfectly correlated.
        self._rngs = {
            sym: np.random.default_rng(seed + i) for i, sym in enumerate(symbols)
        }
        self._live_state: dict[str, tuple[float, float, int]] = {}

    # ---------------------------------------------------------------- history
    def history(
        self,
        start: datetime,
        end: datetime | None = None,
        symbols: list[str] | None = None,
    ) -> dict[str, list[Bar]]:
        end = end or datetime.now(timezone.utc)
        syms = symbols or self.symbols
        n = max(0, int((end - start) / self._bar_delta))
        return {s: self._path(s, start, n) for s in syms}

    def _path(self, symbol: str, start: datetime, n: int) -> list[Bar]:
        if n <= 0:
            return []
        rng = np.random.default_rng(self.seed + hash(symbol) % 10_000)
        # Spread starting prices out so symbols look different in the UI.
        price = self.start_price * (1.0 + (hash(symbol) % 500) / 100.0)

        bars: list[Bar] = []
        drift = 0.0
        vol = self.bar_vol
        regime_left = 0
        for i in range(n):
            if regime_left <= 0:
                # New regime: trend up, trend down, or chop.
                regime_left = int(rng.integers(30, 120))
                kind = rng.integers(0, 3)
                if kind == 0:
                    drift = rng.uniform(0.0002, 0.0008)
                elif kind == 1:
                    drift = -rng.uniform(0.0002, 0.0008)
                else:
                    drift = 0.0
                vol = self.bar_vol * rng.uniform(0.6, 1.8)
            regime_left -= 1

            ret = drift + rng.normal(0.0, vol)
            open_px = price
            close_px = max(0.01, price * (1.0 + ret))
            # Wicks scale with the bar's own move, which keeps ATR sensible.
            wick = abs(ret) * price * rng.uniform(0.3, 1.2) + price * vol * 0.3
            high = max(open_px, close_px) + wick * rng.uniform(0.0, 1.0)
            low = min(open_px, close_px) - wick * rng.uniform(0.0, 1.0)
            volume = float(rng.uniform(500, 5000) * (1.0 + abs(ret) * 50))

            bars.append(
                Bar(
                    symbol=symbol,
                    ts=start + self._bar_delta * i,
                    open=round(open_px, 4),
                    high=round(max(high, close_px, open_px), 4),
                    low=round(max(0.01, min(low, close_px, open_px)), 4),
                    close=round(close_px, 4),
                    volume=round(volume, 2),
                    vwap=round((open_px + close_px + high + low) / 4.0, 4),
                    trade_count=int(rng.integers(20, 400)),
                )
            )
            price = close_px
        return bars

    # ----------------------------------------------------------------- stream
    async def stream(self) -> AsyncIterator[Bar]:
        """Emit synthetic bars every ``emit_interval`` seconds (accelerated time)."""
        now = datetime.now(timezone.utc)
        for sym in self.symbols:
            seeded = self._path(sym, now - self._bar_delta * 2, 2)
            last = seeded[-1].close if seeded else self.start_price
            self._live_state[sym] = (last, self.bar_vol, 0)

        ts = now
        while not self._closed:
            await asyncio.sleep(self.emit_interval)
            if self._closed:
                break
            ts = ts + self._bar_delta
            for sym in self.symbols:
                price, vol, regime_left = self._live_state[sym]
                rng = self._rngs[sym]
                if regime_left <= 0:
                    regime_left = int(rng.integers(20, 80))
                    vol = self.bar_vol * rng.uniform(0.6, 1.8)
                drift = rng.normal(0.0, 0.0004)
                ret = drift + rng.normal(0.0, vol)
                open_px = price
                close_px = max(0.01, price * (1.0 + ret))
                wick = abs(ret) * price * rng.uniform(0.3, 1.2)
                yield Bar(
                    symbol=sym,
                    ts=ts,
                    open=round(open_px, 4),
                    high=round(max(open_px, close_px) + wick, 4),
                    low=round(max(0.01, min(open_px, close_px) - wick), 4),
                    close=round(close_px, 4),
                    volume=round(float(rng.uniform(500, 5000)), 2),
                    vwap=round((open_px + close_px) / 2.0, 4),
                    trade_count=int(rng.integers(20, 400)),
                )
                self._live_state[sym] = (close_px, vol, regime_left - 1)

    async def close(self) -> None:
        self._closed = True
