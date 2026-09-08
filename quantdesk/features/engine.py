"""Feature engine.

Turns a stream of bars into a per-symbol ``FeatureSnapshot`` that agents read.
Centralising this means every agent sees exactly the same numbers on a given bar,
and each indicator is computed once rather than once per agent.

``ready`` is the important field. Indicators need warm-up before their output
means anything, and acting on a half-warm RSI is a classic way to generate
convincing nonsense. Agents must check it, and the pipeline enforces it too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from quantdesk.core.types import Bar
from quantdesk.features.indicators import (
    ATR,
    EMA,
    ROC,
    RSI,
    RollingExtreme,
    RollingWindow,
)


@dataclass(slots=True)
class FeatureSnapshot:
    """All derived numbers for one symbol at one bar."""

    symbol: str
    ts: datetime
    price: float
    bars_seen: int = 0
    ready: bool = False

    # trend
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    ema_spread: float = 0.0      # (fast - slow) / price
    trend_slope: float = 0.0     # normalised slope of the slow EMA
    roc_fast: float = 0.0
    roc_slow: float = 0.0

    # mean reversion
    rsi: float = 50.0
    zscore: float = 0.0          # price vs its rolling mean, in stdevs

    # volatility
    atr: float = 0.0
    atr_pct: float = 0.0         # ATR / price
    realized_vol: float = 0.0    # annualised stdev of bar returns
    vol_percentile: float = 0.5  # where current vol sits in its own history

    # structure
    donchian_high: float = 0.0
    donchian_low: float = 0.0
    channel_pos: float = 0.5     # 0 = at channel low, 1 = at channel high

    # flow
    volume_z: float = 0.0

    extras: dict[str, float] = field(default_factory=dict)

    def as_features(self) -> dict[str, float]:
        """Flat dict for logging and for attaching to signals."""
        return {
            "price": self.price,
            "ema_spread": self.ema_spread,
            "trend_slope": self.trend_slope,
            "roc_fast": self.roc_fast,
            "roc_slow": self.roc_slow,
            "rsi": self.rsi,
            "zscore": self.zscore,
            "atr_pct": self.atr_pct,
            "realized_vol": self.realized_vol,
            "vol_percentile": self.vol_percentile,
            "channel_pos": self.channel_pos,
            "volume_z": self.volume_z,
        }


class SymbolFeatures:
    """Indicator bundle for a single symbol."""

    __slots__ = (
        "symbol", "bars_per_year", "_n", "_prev_close", "_prev_ema_slow",
        "ema_fast", "ema_slow", "rsi", "atr", "roc_fast", "roc_slow",
        "price_win", "ret_win", "vol_hist", "vol_win", "dc_high", "dc_low",
    )

    def __init__(
        self,
        symbol: str,
        bars_per_year: float,
        fast: int = 12,
        slow: int = 26,
        rsi_period: int = 14,
        atr_period: int = 14,
        window: int = 50,
        channel: int = 40,
    ) -> None:
        self.symbol = symbol
        self.bars_per_year = bars_per_year
        self._n = 0
        self._prev_close: float | None = None
        self._prev_ema_slow: float | None = None

        self.ema_fast = EMA(fast)
        self.ema_slow = EMA(slow)
        self.rsi = RSI(rsi_period)
        self.atr = ATR(atr_period)
        self.roc_fast = ROC(fast)
        self.roc_slow = ROC(slow * 2)
        self.price_win = RollingWindow(window)
        self.ret_win = RollingWindow(window)
        self.vol_hist = RollingWindow(window * 4)
        self.vol_win = RollingWindow(window)
        self.dc_high = RollingExtreme(channel, "max")
        self.dc_low = RollingExtreme(channel, "min")

    @property
    def warmup_needed(self) -> int:
        """Bars required before this bundle reports ready."""
        return max(self.ema_slow.period, self.price_win.period, self.dc_high.period) + 2

    def update(self, bar: Bar) -> FeatureSnapshot:
        self._n += 1
        px = bar.close

        ema_f = self.ema_fast.update(px)
        ema_s = self.ema_slow.update(px)
        rsi = self.rsi.update(px)
        atr = self.atr.update(bar.high, bar.low, px)
        roc_f = self.roc_fast.update(px)
        roc_s = self.roc_slow.update(px)

        self.price_win.update(px)
        self.dc_high.update(bar.high)
        self.dc_low.update(bar.low)
        self.vol_win.update(bar.volume)

        # Bar return, guarded against a zero/absent previous close.
        if self._prev_close and self._prev_close > 0:
            ret = (px - self._prev_close) / self._prev_close
        else:
            ret = 0.0
        self._prev_close = px
        self.ret_win.update(ret)

        # Annualise the per-bar stdev.
        realized_vol = self.ret_win.std * math.sqrt(self.bars_per_year)
        self.vol_hist.update(realized_vol)

        # Slope of the slow EMA, normalised by price so it's comparable across symbols.
        if self._prev_ema_slow and px > 0:
            slope = (ema_s - self._prev_ema_slow) / px
        else:
            slope = 0.0
        self._prev_ema_slow = ema_s

        dc_hi = self.dc_high.value
        dc_lo = self.dc_low.value
        span = dc_hi - dc_lo
        channel_pos = (px - dc_lo) / span if span > 1e-12 else 0.5

        snap = FeatureSnapshot(
            symbol=self.symbol,
            ts=bar.ts,
            price=px,
            bars_seen=self._n,
            ready=self._n >= self.warmup_needed,
            ema_fast=ema_f,
            ema_slow=ema_s,
            ema_spread=(ema_f - ema_s) / px if px > 0 else 0.0,
            trend_slope=slope,
            roc_fast=roc_f,
            roc_slow=roc_s,
            rsi=rsi,
            zscore=self.price_win.zscore(px),
            atr=atr,
            atr_pct=atr / px if px > 0 else 0.0,
            realized_vol=realized_vol,
            vol_percentile=self.vol_hist.percentile_rank(realized_vol),
            donchian_high=dc_hi,
            donchian_low=dc_lo,
            channel_pos=min(1.0, max(0.0, channel_pos)),
            volume_z=self.vol_win.zscore(bar.volume),
        )
        return snap


class FeatureEngine:
    """Feature bundles for the whole universe."""

    def __init__(self, bars_per_year: float, **kwargs) -> None:
        self.bars_per_year = bars_per_year
        self._kwargs = kwargs
        self._symbols: dict[str, SymbolFeatures] = {}
        self._latest: dict[str, FeatureSnapshot] = {}

    def _bundle(self, symbol: str) -> SymbolFeatures:
        sf = self._symbols.get(symbol)
        if sf is None:
            sf = SymbolFeatures(symbol, self.bars_per_year, **self._kwargs)
            self._symbols[symbol] = sf
        return sf

    def update(self, bar: Bar) -> FeatureSnapshot:
        snap = self._bundle(bar.symbol).update(bar)
        self._latest[bar.symbol] = snap
        return snap

    def latest(self, symbol: str) -> FeatureSnapshot | None:
        return self._latest.get(symbol)

    def snapshots(self) -> dict[str, FeatureSnapshot]:
        return dict(self._latest)

    @property
    def warmup_needed(self) -> int:
        probe = SymbolFeatures("__probe__", self.bars_per_year, **self._kwargs)
        return probe.warmup_needed
