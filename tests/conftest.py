"""Shared fixtures and bar builders for the test suite."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quantdesk.core.types import Bar

TS0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_bar(
    index: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1000.0,
    symbol: str = "TEST",
    minutes: int = 5,
) -> Bar:
    return Bar(
        symbol=symbol,
        ts=TS0 + timedelta(minutes=minutes * index),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def walk(
    waypoints: list[float],
    bars_per_leg: int,
    symbol: str = "TEST",
    minutes: int = 5,
) -> list[Bar]:
    """Smooth linear bars through price waypoints.

    Volume scales with the size of the move, so breakout bars naturally carry
    volume expansion, which is what the pattern layer tests against.
    """
    bars: list[Bar] = []
    prev = waypoints[0]
    i = 0
    for a, b in zip(waypoints, waypoints[1:]):
        for k in range(bars_per_leg):
            close = a + (b - a) * (k + 1) / bars_per_leg
            open_ = prev
            pad = abs(close - open_) * 0.15 + 0.05
            bars.append(
                make_bar(
                    i,
                    open_,
                    max(open_, close) + pad,
                    min(open_, close) - pad,
                    close,
                    500.0 + 200.0 * abs(close - open_),
                    symbol,
                    minutes,
                )
            )
            prev = close
            i += 1
    return bars


def flat(level: float = 100.0, n: int = 30, symbol: str = "TEST") -> list[Bar]:
    """Quiet bars so ATR and rolling scales warm up before a shape starts."""
    out: list[Bar] = []
    for i in range(n):
        close = level + (0.6 if i % 2 else -0.6)
        open_ = level - (0.6 if i % 2 else -0.6)
        out.append(
            make_bar(
                i,
                open_,
                max(open_, close) + 0.2,
                min(open_, close) - 0.2,
                close,
                600.0,
                symbol,
            )
        )
    return out


def trending(
    bars_per_leg: int = 300,
    symbol: str = "TEST",
    minutes: int = 5,
    up: bool = True,
) -> list[Bar]:
    """A zigzag trend: rising peaks *and* rising troughs, which is what Dow requires.

    A monotonic ramp is not a usable uptrend for testing, because a swing detector
    never sees a retracement and so produces no pivots at all - and without pivots
    there is no trend to classify. Legs are long enough that the 4h aggregation also
    accumulates real structure.
    """
    ups = [1000, 1100, 1060, 1180, 1140, 1280, 1240, 1400]
    waypoints = ups if up else [2000 - p for p in ups]
    return walk(waypoints, bars_per_leg, symbol, minutes)


@pytest.fixture
def hs_top_bars() -> list[Bar]:
    """Head and shoulders top followed by a neckline break."""
    return flat() + walk([100, 120, 108, 135, 110, 121, 88], 12)


@pytest.fixture
def double_top_bars() -> list[Bar]:
    return flat() + walk([100, 130, 112, 129, 95], 12)


@pytest.fixture
def triangle_bars() -> list[Bar]:
    return flat() + walk([100, 150, 112, 144, 120, 138, 126, 185], 5)


@pytest.fixture
def flag_bars() -> list[Bar]:
    return (
        flat()
        + walk([100, 160], 10)
        + walk([160, 152, 158, 153, 157], 3)
        + walk([157, 220], 10)
    )
