"""Douglas's principles as invariants, and sample-size honesty about edges."""

from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from quantdesk.psychology import (
    FUNDAMENTAL_TRUTHS,
    DisciplineMonitor,
    Principle,
    Severity,
    TradeOutcome,
    expected_worst_streak,
    is_streak_alarming,
    measure,
    streak_probability,
)

TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_the_five_truths_are_present_and_quoted():
    assert len(FUNDAMENTAL_TRUTHS) == 5
    assert FUNDAMENTAL_TRUTHS[0] == "Anything can happen."
    assert "random distribution" in FUNDAMENTAL_TRUTHS[2]


def test_seven_principles_each_have_a_mechanical_check():
    assert len(list(Principle)) == 7
    for principle in Principle:
        assert principle.mechanical_check, f"P{principle.number} has no check"


# ------------------------------------------------------------------ statistics


def test_expectancy_and_payoff_in_r():
    outcomes = [TradeOutcome("X", 2.0)] * 4 + [TradeOutcome("X", -1.0)] * 6
    stats = measure(outcomes, min_sample=5)
    assert stats.count == 10
    assert stats.win_rate == pytest.approx(0.4)
    # 4 wins x 2R = 8R, 6 losses x -1R = -6R, net +2R over 10 trades.
    assert stats.expectancy_r == pytest.approx(0.2)
    assert stats.payoff_ratio == pytest.approx(2.0)
    assert stats.profitable


def test_breakeven_win_rate_shows_why_hit_rate_alone_misleads():
    """35% wins is excellent at 3:1 and fatal at 1:1."""
    good = measure([TradeOutcome("X", 3.0)] * 35 + [TradeOutcome("X", -1.0)] * 65)
    bad = measure([TradeOutcome("X", 1.0)] * 35 + [TradeOutcome("X", -1.0)] * 65)
    assert good.win_rate == pytest.approx(bad.win_rate)
    assert good.breakeven_win_rate() == pytest.approx(0.25)
    assert bad.breakeven_win_rate() == pytest.approx(0.5)
    assert good.edge_margin > 0 and bad.edge_margin < 0
    assert good.profitable and not bad.profitable


def test_small_samples_are_flagged_unreliable():
    small = measure([TradeOutcome("X", 1.0)] * 9, min_sample=30)
    assert not small.reliable
    lo, hi = small.win_rate_interval()
    assert hi - lo > 0.0
    big = measure([TradeOutcome("X", 1.0)] * 40, min_sample=30)
    assert big.reliable


def test_confidence_interval_narrows_with_sample_size():
    def width(n: int) -> float:
        wins = int(n * 0.6)
        outs = [TradeOutcome("X", 1.0)] * wins + [TradeOutcome("X", -1.0)] * (n - wins)
        lo, hi = measure(outs).win_rate_interval()
        return hi - lo

    assert width(400) < width(40) < width(10)


def test_streaks_are_tracked():
    pattern = [1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0]
    stats = measure([TradeOutcome("X", r) for r in pattern], min_sample=1)
    assert stats.longest_win_streak == 2
    assert stats.longest_loss_streak == 3
    assert stats.current_streak == -2


# ---------------------------------------------------------------- streak sanity


def test_a_normal_losing_run_is_not_alarming():
    """Douglas's third truth: wins and losses are randomly distributed.

    At a 45% win rate over 200 trades, a run of five or six losses is ordinary, and
    treating it as evidence the edge has broken is the expensive mistake.
    """
    alarming, explanation = is_streak_alarming(5, win_rate=0.45, trades=200)
    assert not alarming
    assert "45%" in explanation or "expected" in explanation


def test_a_genuinely_extreme_run_does_alarm():
    alarming, _ = is_streak_alarming(25, win_rate=0.45, trades=200)
    assert alarming


def test_streak_probability_behaves_sensibly():
    # Longer runs are less likely...
    assert streak_probability(3, 0.5, 100) > streak_probability(10, 0.5, 100)
    # ...and more likely with more chances to occur.
    assert streak_probability(6, 0.5, 1000) > streak_probability(6, 0.5, 50)
    # A perfect win rate can never produce a losing run.
    assert streak_probability(3, 1.0, 100) == pytest.approx(0.0)


def test_expected_worst_streak_grows_with_sample():
    assert expected_worst_streak(0.5, 1000) > expected_worst_streak(0.5, 50)
    # A worse win rate produces longer expected runs of losses.
    assert expected_worst_streak(0.3, 200) > expected_worst_streak(0.7, 200)


# ---------------------------------------------------------------- discipline


def test_entry_without_a_stop_violates_predefined_risk():
    mon = DisciplineMonitor()
    mon.on_entry("BTC/USD", stop=0.0, side=1, agent="murphy_patterns")
    kinds = {e.principle for e in mon.report.violations}
    assert Principle.PREDEFINE_RISK in kinds
    # P7 is violated whenever any other is.
    assert Principle.NEVER_VIOLATE in kinds


def test_entry_without_an_attributable_edge_violates_identify_edges():
    mon = DisciplineMonitor()
    mon.on_entry("BTC/USD", stop=90.0, side=1, agent="", rationale="")
    assert Principle.IDENTIFY_EDGES in {e.principle for e in mon.report.violations}


def test_a_clean_entry_produces_no_violations():
    mon = DisciplineMonitor()
    mon.on_entry("BTC/USD", stop=90.0, side=1, agent="murphy_structure", rationale="uptrend")
    assert mon.report.clean


def test_widening_a_stop_violates_accepting_the_risk():
    """Moving a stop away from entry converts a defined loss into an open-ended one."""
    mon = DisciplineMonitor()
    mon.on_entry("BTC/USD", stop=90.0, side=1, agent="a", rationale="r")
    mon.on_stop_change("BTC/USD", 85.0)
    assert Principle.ACCEPT_RISK in {e.principle for e in mon.report.violations}


def test_tightening_a_stop_is_fine():
    mon = DisciplineMonitor()
    mon.on_entry("BTC/USD", stop=90.0, side=1, agent="a", rationale="r")
    mon.on_stop_change("BTC/USD", 95.0)
    assert mon.report.clean


def test_widening_is_direction_aware_for_shorts():
    mon = DisciplineMonitor()
    mon.on_entry("BTC/USD", stop=110.0, side=-1, agent="a", rationale="r")
    mon.on_stop_change("BTC/USD", 105.0)   # tighter for a short
    assert mon.report.clean
    mon.on_stop_change("BTC/USD", 120.0)   # wider for a short
    assert Principle.ACCEPT_RISK in {e.principle for e in mon.report.violations}


def test_skipping_a_qualifying_edge_is_recorded():
    """Douglas: do not "pick and choose the edges they think are going to work"."""
    mon = DisciplineMonitor()
    mon.on_signal(qualified=True, acted=True)
    mon.on_signal(qualified=True, acted=False, symbol="ETH/USD")
    mon.on_signal(qualified=False, acted=False)
    assert mon.report.qualifying_signals == 2
    assert mon.report.signals_acted_on == 1
    assert mon.report.action_rate == pytest.approx(0.5)
    assert Principle.ACT_WITHOUT_HESITATION in {e.principle for e in mon.report.warnings}


def test_intervention_during_a_drawdown_is_flagged_louder():
    mon = DisciplineMonitor()
    calm = mon.on_intervention("pause", TS, "routine", equity=100_000, drawdown=0.0)
    assert not calm.during_drawdown

    stressed = mon.on_intervention("kill", TS, "losing streak", equity=90_000, drawdown=0.10)
    assert stressed.during_drawdown
    warnings = [e for e in mon.report.warnings if e.principle is Principle.MONITOR_ERRORS]
    assert warnings, "a kill during a drawdown should be flagged"
    assert mon.report.interventions == 2


def test_giving_back_a_large_open_profit_is_flagged():
    mon = DisciplineMonitor()
    mon.on_profit_given_back("BTC/USD", peak_r=3.0, final_r=-0.1)
    assert Principle.PAY_MYSELF in {e.principle for e in mon.report.warnings}


def test_summary_warns_when_the_sample_is_too_small_to_judge():
    mon = DisciplineMonitor(min_sample=30)
    for _ in range(5):
        mon.on_exit("BTC/USD", r_multiple=-1.0)
    text = "\n".join(mon.summary())
    assert "too few" in text


def test_realistic_edge_measures_as_profitable():
    """45% win rate at 2:1 should read as a real edge over a large sample."""
    rng = random.Random(3)
    mon = DisciplineMonitor()
    for _ in range(300):
        mon.on_exit("BTC/USD", r_multiple=2.0 if rng.random() < 0.45 else -1.0)
    stats = mon.stats
    assert stats.reliable
    assert stats.profitable
    assert stats.edge_margin > 0
    alarming, _ = mon.streak_verdict()
    assert not alarming
