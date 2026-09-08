"""Candlestick trend gating, and Elliott's inviolable rules."""

from __future__ import annotations

from datetime import timedelta

import pytest
from conftest import TS0, make_bar

from quantdesk.analysis.candles import CandleReader, CandleKind
from quantdesk.analysis.elliott import ElliottCounter, WaveKind, WaveLabel
from quantdesk.analysis.patterns.base import Bias
from quantdesk.analysis.swings import SwingKind, SwingPoint

#: Small body at the top of the range with a long lower shadow. This single shape is
#: a bullish hammer after a decline and a bearish hanging man after an advance.
UMBRELLA = (30, 100.0, 100.3, 94.0, 99.0)


def warmed_reader() -> CandleReader:
    """Reader with its rolling body/range scale established (average body 2.0)."""
    reader = CandleReader("TEST")
    for i in range(25):
        open_ = 100.0
        close = 102.0 if i % 2 else 98.0
        reader.update(
            make_bar(i, open_, max(open_, close) + 0.5, min(open_, close) - 0.5, close),
            Bias.NEUTRAL,
        )
    return reader


def kinds(signals) -> set:
    return {s.kind for s in signals}


def test_same_shape_reads_opposite_ways_by_trend():
    """Murphy: "You cannot have a bullish reversal pattern in an uptrend."""
    after_decline = warmed_reader().update(make_bar(*UMBRELLA), Bias.BEARISH)
    after_advance = warmed_reader().update(make_bar(*UMBRELLA), Bias.BULLISH)

    assert CandleKind.HAMMER in kinds(after_decline)
    assert CandleKind.HAMMER not in kinds(after_advance)
    assert CandleKind.HANGING_MAN in kinds(after_advance)
    assert CandleKind.HANGING_MAN not in kinds(after_decline)


def test_reversal_shape_without_a_trend_is_not_a_pattern():
    """There is nothing to reverse, so neither reading applies."""
    neutral = warmed_reader().update(make_bar(*UMBRELLA), Bias.NEUTRAL)
    assert CandleKind.HAMMER not in kinds(neutral)
    assert CandleKind.HANGING_MAN not in kinds(neutral)


def test_evening_star_requires_an_uptrend_to_reverse():
    def run(trend: Bias):
        reader = warmed_reader()
        reader.update(make_bar(30, 100.0, 106.2, 99.8, 106.0), trend)   # long white
        reader.update(make_bar(31, 107.5, 108.2, 107.3, 107.6), trend)  # star, gapped up
        return reader.update(make_bar(32, 106.5, 106.7, 100.2, 100.5), trend)

    assert CandleKind.EVENING_STAR in kinds(run(Bias.BULLISH))
    assert CandleKind.EVENING_STAR not in kinds(run(Bias.BEARISH))


def test_dark_cloud_needs_to_open_above_the_prior_high():
    """Murphy specifies the open is above the previous *high*, not its close."""
    reader = warmed_reader()
    reader.update(make_bar(30, 100.0, 106.2, 99.8, 106.0), Bias.BULLISH)
    # Opens at 106.1: inside the prior high of 106.2, so not a dark cloud.
    weak = reader.update(make_bar(31, 106.1, 106.2, 101.0, 102.0), Bias.BULLISH)
    assert CandleKind.DARK_CLOUD_COVER not in kinds(weak)

    reader2 = warmed_reader()
    reader2.update(make_bar(30, 100.0, 106.2, 99.8, 106.0), Bias.BULLISH)
    # Opens at 106.5, above the prior high, closing below the body midpoint of 103.
    real = reader2.update(make_bar(31, 106.5, 106.7, 101.5, 102.0), Bias.BULLISH)
    assert CandleKind.DARK_CLOUD_COVER in kinds(real)


# ------------------------------------------------------------------- Elliott


def pivot(index: int, price: float, kind: SwingKind) -> SwingPoint:
    return SwingPoint(
        index=index,
        ts=TS0 + timedelta(minutes=index),
        price=price,
        kind=kind,
        confirmed_index=index + 2,
    )


T, P = SwingKind.TROUGH, SwingKind.PEAK

#: Textbook five-wave advance. Wave 3 (200) is the longest, wave 4 (200) stays well
#: above wave 1's top (100), and wave 2 retraces only 40% of wave 1.
GOOD_IMPULSE = [
    pivot(0, 0, T),
    pivot(10, 100, P),
    pivot(20, 60, T),
    pivot(40, 260, P),
    pivot(50, 200, T),
    pivot(65, 320, P),
]


def five_wave(counter: ElliottCounter):
    return [
        c for c in counter.counts if c.kind is WaveKind.IMPULSE and len(c.waves) == 5
    ]


def test_valid_impulse_is_accepted_with_high_confidence():
    counter = ElliottCounter("TEST")
    counter.update(70, GOOD_IMPULSE)
    best = counter.best
    assert best is not None
    assert best.kind is WaveKind.IMPULSE
    assert best.valid, best.violations
    assert best.complete
    assert best.confidence > 0.8


def test_wave3_target_matches_murphys_formula():
    """"multiplying the length of wave 1 by 1.618 and adding that to the bottom of 2"."""
    counter = ElliottCounter("TEST")
    counter.update(70, GOOD_IMPULSE)
    targets = counter.targets()
    assert targets["wave3_min"] == pytest.approx(60 + 100 * 1.618)
    # "wave 1 by 3.236 ... added to the top or bottom of wave 1"
    assert targets["wave5_min"] == pytest.approx(0 + 100 * 3.236)
    assert targets["wave5_max"] == pytest.approx(100 + 100 * 3.236)


def test_wave4_overlapping_wave1_is_caught():
    overlapping = list(GOOD_IMPULSE)
    overlapping[4] = pivot(50, 90, T)  # dips below wave 1's top of 100
    counter = ElliottCounter("TEST", strict_overlap=True)
    counter.update(70, overlapping)
    counts = five_wave(counter)
    assert counts
    assert any("overlap" in v for v in counts[0].violations)


def test_wave3_shortest_is_caught():
    short = [
        pivot(0, 0, T),
        pivot(10, 100, P),
        pivot(20, 60, T),
        pivot(40, 100, P),   # wave 3 length 40
        pivot(50, 80, T),
        pivot(65, 280, P),   # wave 5 length 200
    ]
    counter = ElliottCounter("TEST")
    counter.update(70, short)
    counts = five_wave(counter)
    assert counts
    assert any("shortest" in v for v in counts[0].violations)


def test_completed_fifth_wave_flips_the_bias():
    """A finished five-wave advance is a warning, not a reason to keep buying."""
    counter = ElliottCounter("TEST")
    counter.update(70, GOOD_IMPULSE)
    assert counter.best.current_wave is WaveLabel.FIVE
    assert counter.bias() is Bias.BEARISH
