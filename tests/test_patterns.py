"""Chart pattern detection: measurements, Murphy's filters, and lookahead safety."""

from __future__ import annotations

import pytest
from conftest import flat, walk

from quantdesk.analysis.patterns import (
    Bias,
    PatternKind,
    PatternScanner,
    PatternStage,
    PenetrationFilter,
)


def replay(bars, **kw):
    """Run a scanner over bars, collecting every pattern whose state changed."""
    kw.setdefault("strict", True)
    scanner = PatternScanner("TEST", **kw)
    seen = {}
    for bar in bars:
        index = scanner.bar_index + 1
        for pattern in scanner.update(bar):
            seen.setdefault(pattern.key, (index, pattern))
    return scanner, [v for v in seen.values()]


def test_head_and_shoulders_measures_head_to_neckline(hs_top_bars):
    scanner, found = replay(hs_top_bars)
    hs = [p for _, p in found if p.kind is PatternKind.HEAD_SHOULDERS_TOP]
    assert hs, "no head and shoulders top detected"
    pat = hs[0]

    left, _, head, _, right = pat.pivots
    assert head.price > left.price and head.price > right.price
    assert pat.bias is Bias.BEARISH
    assert pat.stop == pytest.approx(right.price), "stop belongs above the right shoulder"

    # Height is head-to-neckline measured at the head's own bar.
    neck_at_head = pat.boundary_at(head.index)
    assert pat.height == pytest.approx(abs(head.price - neck_at_head))

    assert pat.stage.is_actionable, "should have completed on the neckline break"
    # Objective projects the height from where the neckline was pierced.
    origin = pat.boundary_at(pat.breakout_index)
    assert pat.objective == pytest.approx(origin - pat.height)
    assert pat.objective < pat.breakout_price


def test_double_top_breaks_the_middle_trough(double_top_bars):
    _, found = replay(double_top_bars)
    dt = [p for _, p in found if p.kind is PatternKind.DOUBLE_TOP]
    assert dt
    pat = dt[0]
    first, trough, second = pat.pivots

    assert pat.boundary == pytest.approx(trough.price), "the signal is the trough giving way"
    assert pat.height == pytest.approx(max(first.price, second.price) - trough.price)
    assert pat.bias is Bias.BEARISH
    assert pat.objective == pytest.approx(trough.price - pat.height)


def test_double_top_needs_murphy_time_separation():
    """"Most valid double tops should have at least a month between the two peaks."

    Same shape, half the duration, must be rejected.
    """
    tight = flat() + walk([100, 130, 112, 129, 95], 6)
    _, found = replay(tight)
    assert not [p for _, p in found if p.kind is PatternKind.DOUBLE_TOP]


def test_triangle_apex_and_objective(triangle_bars):
    scanner, found = replay(triangle_bars)
    tris = [p for _, p in found if "triangle" in p.kind.value]
    assert tris
    pat = tris[0]

    assert len(pat.pivots) >= 4, "Murphy's minimum is four reversal points"
    assert pat.apex_index is not None
    assert pat.apex_index > pat.detected_index, "the apex is a future deadline"

    upside = [p for p in tris if p.bias is Bias.BULLISH and p.stage.is_actionable]
    assert upside
    done = upside[0]
    origin = done.boundary_at(done.breakout_index)
    assert done.objective == pytest.approx(origin + done.height)


def test_flag_objective_is_the_flagpole_not_the_pattern(flag_bars):
    """"Flags and pennants fly at half-mast" - the target duplicates the pole."""
    _, found = replay(flag_bars)
    flags = [
        p for _, p in found if p.kind in (PatternKind.FLAG, PatternKind.PENNANT)
    ]
    assert flags
    pat = flags[0]
    assert pat.objective == pytest.approx(pat.boundary + pat.height)


@pytest.mark.parametrize(
    "fixture", ["hs_top_bars", "double_top_bars", "triangle_bars", "flag_bars"]
)
def test_no_pattern_uses_an_unconfirmed_pivot(fixture, request):
    """A formation cannot be built from a turning point that had not happened yet."""
    bars = request.getfixturevalue(fixture)
    scanner = PatternScanner("TEST", strict=True)
    seen = set()
    checked = 0
    for bar in bars:
        index = scanner.bar_index + 1
        for pattern in scanner.update(bar):
            if pattern.key in seen:
                continue
            seen.add(pattern.key)
            checked += 1
            assert pattern.detected_index <= index
            for pivot in pattern.pivots:
                assert pivot.confirmed_index <= index
                assert pivot.confirmed_index <= pattern.detected_index
    assert checked > 0


def test_penetration_filter_rejects_a_marginal_poke():
    """Murphy's filters: close beyond, past a price filter, for two sessions."""
    filt = PenetrationFilter(pct=0.01, confirm_bars=2)
    level = 100.0
    assert filt.check("k", 100.5, level, upside=True) is False, "0.5% is inside the filter"
    assert filt.check("k", 99.0, level, upside=True) is False
    assert filt.check("k", 101.5, level, upside=True) is False, "one close is not two"
    assert filt.check("k", 101.6, level, upside=True) is True


def test_scanner_records_detector_errors_instead_of_hiding_them():
    class Broken:
        kinds = ()

        def update(self, ctx):
            raise ValueError("boom")

        @property
        def active(self):
            return []

    scanner = PatternScanner("TEST", detectors=[Broken()], strict=False)
    for bar in flat(n=5):
        scanner.update(bar)
    assert scanner.errors, "a failing detector must be visible, not silent"
    assert "ValueError" in scanner.errors[0]

    strict = PatternScanner("TEST", detectors=[Broken()], strict=True)
    with pytest.raises(ValueError):
        strict.update(flat(n=1)[0])
