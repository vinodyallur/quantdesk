"""Rolling per-symbol bar history.

A bounded ring buffer of bars plus cached numpy views. Agents that need a window
(z-scores, Donchian channels, correlations) read from here; agents that only need
streaming state use the feature engine instead.

The numpy cache is invalidated on append and rebuilt lazily, so N agents reading
the same window in one bar cycle pay the conversion cost once.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Deque, Iterator

import numpy as np

from quantdesk.core.types import Bar


class BarSeries:
    """Fixed-capacity rolling history for one symbol."""

    __slots__ = ("symbol", "maxlen", "_bars", "_cache_len", "_closes", "_highs", "_lows", "_volumes")

    def __init__(self, symbol: str, maxlen: int = 1000) -> None:
        self.symbol = symbol
        self.maxlen = maxlen
        self._bars: Deque[Bar] = deque(maxlen=maxlen)
        self._cache_len = -1
        self._closes: np.ndarray = np.empty(0, dtype=np.float64)
        self._highs: np.ndarray = np.empty(0, dtype=np.float64)
        self._lows: np.ndarray = np.empty(0, dtype=np.float64)
        self._volumes: np.ndarray = np.empty(0, dtype=np.float64)

    # ------------------------------------------------------------------ write
    def append(self, bar: Bar) -> None:
        """Add a bar. Replaces the last bar if the timestamp repeats.

        Repeat timestamps happen when polling a still-forming bar: we want the
        latest snapshot of that bar, not two copies of it.
        """
        if self._bars and bar.ts == self._bars[-1].ts:
            self._bars[-1] = bar
        else:
            self._bars.append(bar)
        self._cache_len = -1

    # ------------------------------------------------------------------- read
    def __len__(self) -> int:
        return len(self._bars)

    def __iter__(self) -> Iterator[Bar]:
        return iter(self._bars)

    def __getitem__(self, idx: int) -> Bar:
        return self._bars[idx]

    @property
    def last(self) -> Bar | None:
        return self._bars[-1] if self._bars else None

    @property
    def last_price(self) -> float:
        return self._bars[-1].close if self._bars else 0.0

    @property
    def last_ts(self) -> datetime | None:
        return self._bars[-1].ts if self._bars else None

    def bars(self, n: int | None = None) -> list[Bar]:
        if n is None:
            return list(self._bars)
        return list(self._bars)[-n:]

    # --------------------------------------------------------- numpy accessors
    def _rebuild(self) -> None:
        if self._cache_len == len(self._bars):
            return
        self._closes = np.fromiter((b.close for b in self._bars), np.float64, len(self._bars))
        self._highs = np.fromiter((b.high for b in self._bars), np.float64, len(self._bars))
        self._lows = np.fromiter((b.low for b in self._bars), np.float64, len(self._bars))
        self._volumes = np.fromiter((b.volume for b in self._bars), np.float64, len(self._bars))
        self._cache_len = len(self._bars)

    def closes(self, n: int | None = None) -> np.ndarray:
        self._rebuild()
        return self._closes if n is None else self._closes[-n:]

    def highs(self, n: int | None = None) -> np.ndarray:
        self._rebuild()
        return self._highs if n is None else self._highs[-n:]

    def lows(self, n: int | None = None) -> np.ndarray:
        self._rebuild()
        return self._lows if n is None else self._lows[-n:]

    def volumes(self, n: int | None = None) -> np.ndarray:
        self._rebuild()
        return self._volumes if n is None else self._volumes[-n:]

    def returns(self, n: int | None = None) -> np.ndarray:
        """Simple close-to-close returns. Length is len(closes) - 1."""
        closes = self.closes()
        if closes.size < 2:
            return np.empty(0, dtype=np.float64)
        rets = np.diff(closes) / np.where(closes[:-1] == 0, np.nan, closes[:-1])
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
        return rets if n is None else rets[-n:]


class MarketBook:
    """All symbols' rolling history in one place."""

    def __init__(self, symbols: list[str] | None = None, maxlen: int = 1000) -> None:
        self.maxlen = maxlen
        self._series: dict[str, BarSeries] = {}
        for sym in symbols or []:
            self._series[sym] = BarSeries(sym, maxlen)

    def series(self, symbol: str) -> BarSeries:
        s = self._series.get(symbol)
        if s is None:
            s = BarSeries(symbol, self.maxlen)
            self._series[symbol] = s
        return s

    def append(self, bar: Bar) -> None:
        self.series(bar.symbol).append(bar)

    @property
    def symbols(self) -> list[str]:
        return list(self._series)

    def last_prices(self) -> dict[str, float]:
        return {sym: s.last_price for sym, s in self._series.items() if s.last_price > 0}

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._series
