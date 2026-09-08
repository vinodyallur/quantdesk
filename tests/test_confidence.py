"""Confidence scoring and the leverage it is allowed to buy.

The tests that matter here are the ones about the liquidation cap. Confidence sizing is a
preference; the cap is arithmetic, and getting it wrong turns every protective stop in the
system into a decoration.
"""

from __future__ import annotations

import pytest

from quantdesk.alpha.confidence import (
    DEFAULT_MAINT_MARGIN,
    MIN_TRADABLE_CONFIDENCE,
    Evidence,
    decide_leverage,
    score_confidence,
)


def strong(**over) -> Evidence:
    base = dict(
        checklist_conviction=0.90, checklist_agreement=0.88, checklist_coverage=0.95,
        agent_consensus=0.85, agent_count=8, timeframe_aligned=True,
        pattern_reward_risk=5.0, has_pattern_stop=True, regime_appetite=1.0,
        funding_against=0.0,
    )
    base.update(over)
    return Evidence(**base)


# ------------------------------------------------------------------- scoring


def test_a_textbook_setup_scores_high_and_a_marginal_one_does_not():
    high = score_confidence(strong())
    low = score_confidence(Evidence(
        checklist_conviction=0.2, checklist_agreement=0.1, checklist_coverage=0.5,
        agent_consensus=0.1, agent_count=1, timeframe_aligned=False,
        regime_appetite=0.4,
    ))
    assert high.score > 0.80 and high.tradable
    assert low.score < MIN_TRADABLE_CONFIDENCE and not low.tradable


def test_the_score_is_bounded_and_every_component_is_reported():
    """An opaque scalar cannot tell you which piece of evidence was wrong."""
    conf = score_confidence(strong())
    assert 0.0 <= conf.score <= 1.0
    for name in ("checklist", "agreement", "consensus", "timeframe", "structure",
                 "regime", "funding"):
        assert name in conf.components, f"{name} missing from the breakdown"
        assert name in conf.weights
    assert sum(conf.weights.values()) == pytest.approx(1.0, abs=0.01)


def test_a_checklist_blocker_collapses_the_score():
    """A hard veto that only nudges the score is not a veto."""
    clear = score_confidence(strong())
    blocked = score_confidence(strong(blockers=1))
    assert blocked.score == pytest.approx(clear.score * 0.25, rel=1e-6)
    assert any("blocker" in n for n in blocked.notes)


def test_trading_against_the_higher_timeframe_costs_the_whole_component():
    with_tf = score_confidence(strong())
    against = score_confidence(strong(timeframe_aligned=False))
    assert against.score < with_tf.score
    assert against.components["timeframe"] == 0.0
    assert any("higher timeframe" in n for n in against.notes)


def test_consensus_is_discounted_when_only_a_couple_of_agents_spoke():
    """Two agents agreeing is an opinion; eight agreeing is evidence."""
    few = score_confidence(strong(agent_count=2))
    many = score_confidence(strong(agent_count=8))
    assert few.components["consensus"] < many.components["consensus"]
    assert any("agent(s) had a view" in n for n in few.notes)


def test_agents_agreeing_on_the_other_side_is_not_conviction():
    opposed = score_confidence(strong(agent_consensus=-0.9))
    assert opposed.components["consensus"] == 0.0


def test_expensive_funding_reduces_confidence_in_that_direction():
    cheap = score_confidence(strong(funding_against=0.0))
    dear = score_confidence(strong(funding_against=0.25))
    assert dear.score < cheap.score
    assert dear.components["funding"] == 0.0
    assert any("funding costs" in n for n in dear.notes)


def test_a_mostly_unanswerable_checklist_is_discounted():
    """During warmup most questions cannot be answered, and that is not conviction."""
    full = score_confidence(strong(checklist_coverage=1.0))
    thin = score_confidence(strong(checklist_coverage=0.2))
    assert thin.score < full.score
    assert any("checklist could be answered" in n for n in thin.notes)


# ------------------------------------------------------------------ leverage


def test_leverage_rises_with_confidence():
    low = decide_leverage(0.40, 0.004, user_cap=100.0)
    high = decide_leverage(0.75, 0.004, user_cap=100.0)
    assert high.requested > low.requested


def test_the_response_is_concave_so_weak_evidence_buys_little():
    """Linear scaling hands out real leverage at the first hint of a signal."""
    half = decide_leverage(0.5, 0.001, user_cap=100.0)
    assert half.requested < 0.5 * 100.0, "half confidence must buy less than half the cap"


def test_a_wide_stop_cannot_support_high_leverage():
    """The constraint that makes this design defensible rather than reckless."""
    tight = decide_leverage(1.0, 0.002, maint_margin_pct=0.005, user_cap=100.0)
    wide = decide_leverage(1.0, 0.05, maint_margin_pct=0.005, user_cap=100.0)
    assert tight.leverage > wide.leverage
    assert wide.binding == "liquidation distance"
    assert wide.capped


def test_the_stop_is_always_nearer_than_liquidation():
    """The invariant. If liquidation comes first the stop is fiction and the loss is
    the whole margin rather than the intended risk."""
    for stop in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25):
        d = decide_leverage(1.0, stop, maint_margin_pct=0.005, user_cap=100.0)
        assert d.liquidation_move_pct > stop, (
            f"at {d.leverage:.1f}x a {stop:.1%} stop is behind liquidation at "
            f"{d.liquidation_move_pct:.2%}"
        )


def test_a_hundred_x_request_on_a_two_percent_stop_is_cut_hard():
    d = decide_leverage(1.0, 0.02, maint_margin_pct=0.005, user_cap=100.0)
    assert d.requested == pytest.approx(100.0)
    assert d.leverage < 25.0
    assert d.binding == "liquidation distance"
    assert any("would not survive" in line for line in d.summary())


def test_no_stop_means_no_leverage():
    """Without a stop there is no distance to protect, so there is no basis for size."""
    d = decide_leverage(1.0, 0.0, user_cap=100.0)
    assert d.leverage == pytest.approx(1.0)


def test_leverage_never_drops_below_one_or_exceeds_the_request():
    for conf in (0.0, 0.3, 0.7, 1.0):
        for stop in (0.0005, 0.01, 0.4):
            d = decide_leverage(conf, stop, user_cap=50.0)
            assert 1.0 <= d.leverage <= max(1.0, d.requested) + 1e-9
            assert d.leverage <= 50.0 + 1e-9


def test_the_user_cap_is_respected_even_with_a_very_tight_stop():
    d = decide_leverage(1.0, 0.0001, maint_margin_pct=0.0001, user_cap=5.0)
    assert d.leverage <= 5.0 + 1e-9


def test_the_binding_limit_is_named_honestly():
    tight = decide_leverage(0.5, 0.0005, maint_margin_pct=0.0005, user_cap=100.0)
    assert tight.binding == "confidence", "nothing else should be binding here"
    wide = decide_leverage(1.0, 0.30, user_cap=100.0)
    assert wide.binding == "liquidation distance"


def test_the_default_maintenance_margin_is_documented_as_tier_one():
    """A reminder in test form: the published exchangeInfo figure is the top tier."""
    assert DEFAULT_MAINT_MARGIN == pytest.approx(0.005)
