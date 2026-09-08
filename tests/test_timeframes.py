"""Aggregation correctness and Murphy's top-down hierarchy."""

from __future__ import annotations

from datetime import timedelta

from conftest import TS0, make_bar, trending

from quantdesk.analysis.patterns.base import Bias
from quantdesk.analysis.timeframes import BarAggregator, Timeframe, TimeframeStack


def test_aggregation_is_exact():
    """A closed hourly bar must be the true OHLCV of its twelve 5m bars."""
    agg = BarAggregator(Timeframe.H1)
    emitted = []
    for i in range(30):
        bar = make_bar(i, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0)
        done = agg.update(bar)
        if done is not None:
            emitted.append((i, done))

    assert len(emitted) == 2, "2.5 hours of 5m bars should close two hourly bars"
    first_index, h1 = emitted[0]
    assert h1.open == 100.0
    assert h1.high == 112.0
    assert h1.low == 99.0
    assert h1.close == 111.5
    assert h1.volume == 120.0
    assert h1.ts == TS0


def test_aggregation_lags_one_bar_so_it_cannot_leak():
    """The hour closes on base bar 11 but is only emitted on bar 12.

    At the instant the last minute of an hour closes you do not yet know the hour
    is over. Emitting on bar 11 would hand a strategy the future.
    """
    agg = BarAggregator(Timeframe.H1)
    emitted_at = None
    for i in range(15):
        done = agg.update(make_bar(i, 100.0, 101.0, 99.0, 100.5))
        if done is not None and emitted_at is None:
            emitted_at = i
    assert emitted_at == 12


def test_forming_bar_is_separate_from_closed_bars():
    agg = BarAggregator(Timeframe.H1)
    for i in range(5):
        assert agg.update(make_bar(i, 100.0, 101.0 + i, 99.0, 100.5)) is None
    forming = agg.forming
    assert forming is not None
    assert agg.bars_in_period == 5
    assert forming.high == 105.0


def test_out_of_order_bar_does_not_rewrite_a_closed_period():
    agg = BarAggregator(Timeframe.H1)
    for i in range(14):
        agg.update(make_bar(i, 100.0, 101.0, 99.0, 100.5))
    stale = make_bar(0, 500.0, 500.0, 500.0, 500.0)
    assert agg.update(stale) is None
    assert agg.forming is not None
    assert agg.forming.high == 101.0


def test_week_floor_anchors_to_monday():
    # 2024-01-01 was a Monday; 2024-01-04 floors back to it.
    thursday = TS0 + timedelta(days=3)
    assert Timeframe.W1.floor(thursday) == TS0


def test_higher_timeframe_gates_direction():
    """The slow frames decide what is permitted; the fast one only times it."""
    stack = TimeframeStack(
        "TEST",
        [Timeframe.M5, Timeframe.H1, Timeframe.H4],
        base=Timeframe.M5,
        timing_count=1,
    )
    for bar in trending(bars_per_leg=300):
        stack.update(bar)

    assert stack.permitted_direction() is Bias.BULLISH
    assert stack.agrees(Bias.BULLISH) is True
    assert stack.agrees(Bias.BEARISH) is False, "counter-trend must be vetoed"
    assert stack.alignment() > 0
    assert stack.direction_frames == [Timeframe.H1, Timeframe.H4]
    assert stack.timing_frames == [Timeframe.M5]


def test_no_structure_permits_nothing():
    """With too little history to have a view, the answer is stand aside."""
    stack = TimeframeStack("TEST", [Timeframe.M5, Timeframe.H1, Timeframe.H4])
    for i in range(5):
        stack.update(make_bar(i, 100.0, 101.0, 99.0, 100.0))
    assert stack.permitted_direction() is Bias.NEUTRAL
    assert stack.agrees(Bias.BULLISH) is False
    assert stack.agrees(Bias.BEARISH) is False
