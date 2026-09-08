"""Murphy's chapter 16 allocation limits, checked against his stated numbers."""

from __future__ import annotations

import pytest

from quantdesk.risk.money import (
    MoneyManager,
    MoneyRules,
    PortfolioState,
    SizingRefusal,
)

EQUITY = 100_000.0


def portfolio(**kw) -> PortfolioState:
    kw.setdefault("equity", EQUITY)
    kw.setdefault("start_equity", EQUITY)
    return PortfolioState(**kw)


def test_ceiling_binds_when_no_risk_target_is_set():
    """"A $100,000 account should not risk more than $5,000 on a single trade."

    With the risk target disabled, Murphy's 5% ceiling is what sizes the trade.
    """
    mm = MoneyManager(MoneyRules(risk_target_pct=0.0))
    # Entry 100, stop 90: 10 of risk per unit. 5% of 100k is 5,000 -> 500 units.
    plan = mm.plan(
        "BTC/USD", entry=100.0, stop=90.0, objective=140.0,
        portfolio=portfolio(), leverage=10.0,
    )
    assert plan.approved
    assert plan.risk_amount == pytest.approx(5_000.0)
    assert plan.risk_pct == pytest.approx(0.05)
    assert "risk cap" in plan.binding_limit
    assert plan.qty == pytest.approx(500.0)


def test_risk_target_equalises_risk_across_different_stop_distances():
    """The invariant that makes an edge compound.

    A wide stop and a tight stop must risk the *same* amount of money, or a +3R win on a
    small position earns less than a -1R loss on a large one and a positive expectancy in
    R never becomes a positive return. Sizing against a ceiling alone does not do this,
    because whichever limit happens to bind decides the risk.
    """
    mm = MoneyManager(MoneyRules(risk_target_pct=0.005))
    tight = mm.plan("BTC/USD", 100.0, 99.0, 110.0, portfolio(), leverage=10.0)
    wide = mm.plan("BTC/USD", 100.0, 80.0, 200.0, portfolio(), leverage=10.0)

    assert tight.approved and wide.approved
    assert tight.risk_pct == pytest.approx(0.005)
    assert wide.risk_pct == pytest.approx(0.005)
    assert tight.risk_amount == pytest.approx(wide.risk_amount)
    # The tight stop necessarily takes the larger position to risk the same money.
    assert tight.qty > wide.qty


def test_risk_target_never_exceeds_murphys_ceiling():
    mm = MoneyManager(MoneyRules(risk_target_pct=0.50, max_market_risk_pct=0.05))
    plan = mm.plan("BTC/USD", 100.0, 90.0, 140.0, portfolio(), leverage=10.0)
    assert plan.approved
    assert plan.risk_pct <= 0.05 + 1e-9


def test_unleveraged_account_still_respects_the_commitment_ceiling():
    """Cash trading ties up full notional, so 10% of equity remains a hard limit."""
    mm = MoneyManager()
    plan = mm.plan("BTC/USD", entry=100.0, stop=90.0, objective=140.0, portfolio=portfolio())
    assert plan.approved
    assert plan.commitment_pct <= 0.10 + 1e-9
    assert plan.risk_pct <= 0.05 + 1e-9


def test_tight_stop_never_breaches_the_commitment_cap():
    mm = MoneyManager()
    plan = mm.plan("BTC/USD", entry=100.0, stop=99.0, objective=110.0, portfolio=portfolio())
    assert plan.approved
    assert plan.commitment_pct <= 0.10 + 1e-9
    assert plan.risk_pct <= 0.05 + 1e-9


def test_reward_to_risk_gate_refuses_thin_trades():
    """"The profit potential must be at least three times the possible loss."""
    mm = MoneyManager()
    # Risk 10, reward 20 -> 2:1, below the yardstick.
    plan = mm.plan("BTC/USD", entry=100.0, stop=90.0, objective=120.0, portfolio=portfolio())
    assert not plan.approved
    assert plan.refusal is SizingRefusal.REWARD_TOO_SMALL

    ok = mm.plan("BTC/USD", entry=100.0, stop=90.0, objective=130.0, portfolio=portfolio())
    assert ok.approved and ok.reward_risk == pytest.approx(3.0)


def test_fifty_percent_invested_ceiling():
    """"Total invested funds should be limited to 50% of total capital."""
    mm = MoneyManager()
    full = portfolio(committed={"A": 50_000.0})
    plan = mm.plan("BTC/USD", entry=100.0, stop=90.0, objective=140.0, portfolio=full)
    assert not plan.approved
    assert plan.refusal is SizingRefusal.PORTFOLIO_FULL


def test_group_ceiling_treats_correlated_markets_as_one():
    """"Markets within groups tend to move together" - gold and silver share a cap."""
    mm = MoneyManager()
    state = portfolio(
        committed={"GOLD": 20_000.0},
        groups={"GOLD": "metals", "SILVER": "metals"},
    )
    plan = mm.plan("SILVER", entry=100.0, stop=90.0, objective=140.0, portfolio=state)
    assert not plan.approved
    assert plan.refusal is SizingRefusal.GROUP_FULL

    # An unrelated market is unaffected by the metals group being full.
    other = mm.plan("BTC/USD", entry=100.0, stop=90.0, objective=140.0, portfolio=state)
    assert other.approved


def test_short_side_is_sized_identically():
    """Risk is symmetric: the stop distance decides the loss on either side."""
    mm = MoneyManager()
    short = mm.plan(
        "BTC/USD", entry=100.0, stop=110.0, objective=60.0,
        portfolio=portfolio(), leverage=10.0,
    )
    long = mm.plan(
        "BTC/USD", entry=100.0, stop=90.0, objective=140.0,
        portfolio=portfolio(), leverage=10.0,
    )
    assert short.approved and not short.long_side
    assert short.risk_pct == pytest.approx(long.risk_pct)
    assert short.qty == pytest.approx(long.qty)


def test_objective_on_the_wrong_side_is_refused():
    mm = MoneyManager()
    # Long setup but the target sits below entry.
    plan = mm.plan("BTC/USD", entry=100.0, stop=90.0, objective=95.0, portfolio=portfolio())
    assert not plan.approved
    assert plan.refusal is SizingRefusal.STOP_WRONG_SIDE


def test_conviction_scales_down_but_never_past_a_limit():
    mm = MoneyManager()
    full = mm.plan("BTC/USD", 100.0, 90.0, 140.0, portfolio(), conviction=1.0)
    half = mm.plan("BTC/USD", 100.0, 90.0, 140.0, portfolio(), conviction=0.5)
    over = mm.plan("BTC/USD", 100.0, 90.0, 140.0, portfolio(), conviction=5.0)
    assert half.qty == pytest.approx(full.qty * 0.5)
    assert over.qty == pytest.approx(full.qty), "conviction must not scale past the cap"


def test_pyramid_layers_shrink_and_are_capped():
    """"Each successive layer should be smaller than before."""
    mm = MoneyManager()
    sizes = [
        mm.plan("BTC/USD", 100.0, 90.0, 140.0, portfolio(), layer=n).qty
        for n in (1, 2, 3)
    ]
    assert sizes[0] > sizes[1] > sizes[2] > 0
    beyond = mm.plan("BTC/USD", 100.0, 90.0, 140.0, portfolio(), layer=4)
    assert not beyond.approved


def test_never_add_to_a_loser():
    mm = MoneyManager()
    allowed, _ = mm.may_add(unrealized_pnl=500.0, layer=2)
    refused, reason = mm.may_add(unrealized_pnl=-500.0, layer=2)
    assert allowed
    assert not refused and "losing" in reason


def test_qty_rounding_only_ever_reduces_risk():
    mm = MoneyManager()
    plan = mm.plan(
        "BTC/USD", 100.0, 90.0, 140.0, portfolio(), qty_step=7.0
    )
    assert plan.approved
    assert plan.qty % 7.0 == pytest.approx(0.0)
    assert plan.risk_pct <= 0.05 + 1e-9


def test_stop_too_wide_for_minimum_size_is_refused_distinctly():
    mm = MoneyManager()
    # Risk per unit 50,000 against a 5,000 budget: even one unit is too much.
    plan = mm.plan(
        "BTC/USD", entry=100_000.0, stop=50_000.0, objective=260_000.0,
        portfolio=portfolio(), min_qty=1.0,
    )
    assert not plan.approved
    assert plan.refusal is SizingRefusal.RISK_TOO_WIDE


def test_winning_streak_size_growth_is_capped():
    """Murphy's one firm answer on streaks: do not double up after a hot run."""
    mm = MoneyManager(MoneyRules(streak_growth_cap=1.5))
    tripled = portfolio(equity=300_000.0, start_equity=100_000.0)
    assert mm.equity_scalar(tripled) == pytest.approx(1.5)
    # Drawdowns are not overridden: proportional sizing already cuts exposure.
    halved = portfolio(equity=50_000.0, start_equity=100_000.0)
    assert mm.equity_scalar(halved) == pytest.approx(0.5)


def test_incoherent_rules_are_rejected_at_construction():
    with pytest.raises(ValueError):
        MoneyRules(max_market_risk_pct=0.20, max_market_commitment_pct=0.10)


# ------------------------------------------------- intended vs delivered risk


def test_plan_records_the_risk_it_intended_not_only_what_it_got():
    """The gap between intended and delivered risk is the whole R-multiple story.

    On an unleveraged account the 10% commitment ceiling allows far less size than the
    risk target asks for, so actual risk per trade ends up set by stop distance. Without
    both numbers on the plan that shortfall is invisible, and a plain average of R across
    such trades silently becomes an average over incomparable denominators.
    """
    mm = MoneyManager(MoneyRules(risk_target_pct=0.005))
    plan = mm.plan("BTC/USD", entry=300.0, stop=298.0, objective=310.0,
                   portfolio=portfolio(), leverage=1.0)

    assert plan.approved
    assert plan.risk_target == pytest.approx(500.0), "0.5% of 100k was intended"
    # 10% of 100k at 300 a unit is 33.33 units; 2 of risk each is ~66.7, not 500.
    assert plan.risk_amount == pytest.approx(66.67, abs=0.01)
    assert plan.risk_delivered == pytest.approx(0.1333, abs=0.001)
    assert "commitment" in plan.binding_limit


def test_risk_delivered_is_one_when_the_target_actually_binds():
    mm = MoneyManager(MoneyRules(risk_target_pct=0.005))
    plan = mm.plan("BTC/USD", 100.0, 90.0, 140.0, portfolio(), leverage=10.0)
    assert plan.approved
    assert plan.risk_amount == pytest.approx(plan.risk_target)
    assert plan.risk_delivered == pytest.approx(1.0)


def test_binding_limit_names_conviction_when_conviction_is_what_cut_the_size():
    """Reporting the ceiling when conviction decided the size names the wrong cause."""
    mm = MoneyManager(MoneyRules(risk_target_pct=0.005))
    full = mm.plan("BTC/USD", 100.0, 90.0, 140.0, portfolio(), leverage=10.0)
    half = mm.plan("BTC/USD", 100.0, 90.0, 140.0, portfolio(), conviction=0.5,
                   leverage=10.0)

    assert half.qty == pytest.approx(full.qty * 0.5)
    assert "conviction" in half.binding_limit
    assert "conviction" not in full.binding_limit
    # The target is what was intended regardless of the cut, so the shortfall shows.
    assert half.risk_target == pytest.approx(full.risk_target)
    assert half.risk_delivered == pytest.approx(0.5)


def test_binding_limit_stays_low_cardinality_for_grouping():
    """These strings get counted and grouped, so they must not embed the percentage."""
    mm = MoneyManager(MoneyRules(risk_target_pct=0.005))
    labels = {
        mm.plan("BTC/USD", 100.0, 90.0, 140.0, portfolio(), conviction=c,
                leverage=10.0).binding_limit
        for c in (0.31, 0.47, 0.62, 0.88)
    }
    assert len(labels) == 1, f"expected one shared label, got {labels}"


def test_risk_per_unit_is_the_stop_distance():
    mm = MoneyManager()
    plan = mm.plan("BTC/USD", 100.0, 92.5, 140.0, portfolio(), leverage=10.0)
    assert plan.risk_per_unit == pytest.approx(7.5)
