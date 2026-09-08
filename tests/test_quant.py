"""Distribution shape, edge validation, regime fitting and Kelly sizing."""

from __future__ import annotations

import numpy as np
import pytest

from quantdesk.psychology.edge import TradeOutcome, measure
from quantdesk.quant import (
    analyse,
    bootstrap_sharpe,
    classify_trend_regime,
    deflated_sharpe,
    drawdown_series,
    fit_vol_regime,
    from_edge_stats,
    growth_rate,
    hurst_exponent,
    kelly_fraction,
    kelly_position,
    permutation_test,
    sharpe,
    tail_ratio,
    walk_forward_splits,
)

RNG = np.random.default_rng(11)
HOURS_PER_YEAR = 24 * 365


# ----------------------------------------------------------------- distributions


def test_normal_returns_are_recognised_as_normal():
    data = RNG.normal(0.0, 0.01, 4000)
    stats = analyse(data)
    assert stats.is_normal
    assert abs(stats.excess_kurtosis) < 0.5
    assert not stats.fat_tailed


def test_fat_tails_are_detected():
    """Student-t with low degrees of freedom is the classic fat-tailed case."""
    data = RNG.standard_t(3, 4000) * 0.01
    stats = analyse(data)
    assert stats.fat_tailed
    assert not stats.is_normal
    # A normal model understates the extreme tail, which is the whole point. Measured
    # at 99%: at 95% the inflated fitted std makes the normal fit look conservative.
    assert stats.tail_understatement > 1.0


def test_cvar_is_always_worse_than_var():
    data = RNG.standard_t(4, 3000) * 0.01
    stats = analyse(data)
    assert stats.cvar_95 < stats.var_95
    assert stats.cvar_99 < stats.var_99


def test_var_reads_the_empirical_quantile():
    data = np.linspace(-0.10, 0.10, 1001)
    stats = analyse(data)
    assert stats.var_95 == pytest.approx(-0.09, abs=1e-3)


def ornstein_uhlenbeck(n: int = 3000, theta: float = 0.3, seed: int = 5) -> np.ndarray:
    """A genuinely mean-reverting series.

    A sine wave is the wrong test case: it is smooth, so locally it looks strongly
    trending and Hurst correctly reports it that way. Mean reversion means each step
    pulls *back toward a level*, which is what an OU process does.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = out[i - 1] - theta * out[i - 1] + rng.normal(0, 1)
    return out


def persistent_series(n: int = 3000, phi: float = 0.6, seed: int = 4) -> np.ndarray:
    """A series whose increments are positively autocorrelated.

    This is what "trending" means to the Hurst exponent: moves tend to continue. Note
    that a random walk plus a constant drift does *not* qualify, because adding a
    constant changes the mean of the increments but not their variance, and Hurst reads
    variance scaling.
    """
    rng = np.random.default_rng(seed)
    increments = np.zeros(n)
    for i in range(1, n):
        increments[i] = phi * increments[i - 1] + rng.normal(0, 1)
    return np.cumsum(increments)


def test_hurst_separates_persistence_from_mean_reversion():
    random_walk = np.cumsum(RNG.normal(0, 1, 3000))
    persistent = persistent_series()
    reverting = ornstein_uhlenbeck()

    assert hurst_exponent(persistent) > hurst_exponent(random_walk)
    assert hurst_exponent(reverting) < hurst_exponent(random_walk)


def test_hurst_is_drift_invariant():
    """Documented behaviour worth pinning: drift is not persistence."""
    steps = RNG.normal(0, 1, 3000)
    plain = hurst_exponent(np.cumsum(steps))
    drifting = hurst_exponent(np.cumsum(steps + 0.6))
    assert plain == pytest.approx(drifting, abs=1e-6)


def test_too_little_data_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        analyse([0.01, -0.01, 0.02])


def test_drawdown_series_is_never_positive():
    equity = np.array([100.0, 110.0, 105.0, 120.0, 90.0])
    dd = drawdown_series(equity)
    assert (dd <= 1e-12).all()
    assert dd[-1] == pytest.approx((90.0 - 120.0) / 120.0)


def test_tail_ratio_above_one_means_bigger_wins_than_losses():
    good = np.concatenate([RNG.normal(0, 0.01, 900), RNG.normal(0.10, 0.01, 100)])
    assert tail_ratio(good) > 1.0


# ------------------------------------------------------------------- validation


def test_pure_noise_shows_no_significant_sharpe():
    """The most important negative test: random returns must not look like an edge."""
    noise = RNG.normal(0.0, 0.01, 800)
    result = bootstrap_sharpe(noise, HOURS_PER_YEAR, resamples=400, seed=5)
    assert not result.significant, "noise must not read as a demonstrated edge"
    assert result.lower < 0 < result.upper


def test_a_genuine_edge_is_detected():
    edged = RNG.normal(0.002, 0.01, 800)
    result = bootstrap_sharpe(edged, HOURS_PER_YEAR, resamples=400, seed=5)
    assert result.significant
    assert result.lower > 0
    assert result.p_value < 0.05


def test_bootstrap_interval_brackets_the_point_estimate():
    data = RNG.normal(0.001, 0.01, 500)
    result = bootstrap_sharpe(data, HOURS_PER_YEAR, resamples=400, seed=3)
    assert result.lower <= result.statistic <= result.upper


def test_tiny_samples_return_nothing_rather_than_a_false_positive():
    result = bootstrap_sharpe([0.01, -0.02, 0.03], HOURS_PER_YEAR)
    assert result.resamples == 0
    assert not result.significant


def test_deflated_sharpe_punishes_many_trials():
    """Reporting the best of fifty variants inflates the apparent result."""
    one = deflated_sharpe(1.5, trials=1, sample_size=500)
    many = deflated_sharpe(1.5, trials=200, sample_size=500)
    assert one > many
    assert 0.0 <= many <= 1.0


def test_deflated_sharpe_rewards_a_bigger_sample():
    """More data makes a genuinely good Sharpe more credible.

    The observed Sharpe has to clear what 10 trials would produce by chance (about 1.57)
    for extra data to help. Below that threshold more data makes the desk *more*
    confident the result is selection noise, which is correct and is a separate case.
    """
    small = deflated_sharpe(2.5, trials=10, sample_size=60)
    large = deflated_sharpe(2.5, trials=10, sample_size=2000)
    assert large > small


def test_more_data_increases_confidence_that_a_weak_sharpe_is_noise():
    weak_small = deflated_sharpe(0.8, trials=50, sample_size=60)
    weak_large = deflated_sharpe(0.8, trials=50, sample_size=2000)
    assert weak_large < weak_small
    assert weak_large < 0.5


def test_permutation_test_rejects_random_timing():
    market = RNG.normal(0.0, 0.01, 600)
    # Exposure that knows the future: perfectly timed.
    perfect = np.sign(market) * market
    result = permutation_test(perfect, market, resamples=300, seed=2)
    assert result.p_value < 0.05, "perfect timing should beat shuffled timing"


def test_permutation_test_accepts_that_random_is_random():
    market = RNG.normal(0.0, 0.01, 600)
    exposure = RNG.choice([-1.0, 1.0], 600)
    result = permutation_test(exposure * market, market, resamples=300, seed=2)
    assert result.p_value > 0.05, "random timing must not look informative"


def test_walk_forward_splits_never_overlap():
    splits = walk_forward_splits(1000, train_size=400, test_size=100)
    assert splits
    for split in splits:
        assert split.train_end <= split.test_start
        assert split.test_end <= 1000
    for a, b in zip(splits, splits[1:]):
        assert a.test_end <= b.test_start or a.test_start < b.test_start


def test_anchored_splits_grow_the_training_window():
    rolling = walk_forward_splits(1000, 300, 100, anchored=False)
    anchored = walk_forward_splits(1000, 300, 100, anchored=True)
    assert rolling[-1].train_start > 0
    assert anchored[-1].train_start == 0
    assert anchored[-1].train_size > anchored[0].train_size


def test_gap_prevents_information_bleeding_between_windows():
    splits = walk_forward_splits(1000, 300, 100, gap=20)
    for split in splits:
        assert split.test_start - split.train_end == 20


def test_invalid_split_sizes_are_rejected():
    with pytest.raises(ValueError):
        walk_forward_splits(1000, 0, 100)


# ----------------------------------------------------------------------- regime


def test_vol_regime_separates_calm_from_turbulent():
    calm = RNG.normal(0.0, 0.005, 600)
    turbulent = RNG.normal(0.0, 0.030, 600)
    data = np.concatenate([calm, turbulent])
    regime = fit_vol_regime(data)

    assert regime.turbulent_std > regime.calm_std
    assert regime.vol_ratio > 2.0
    # The series ends in the turbulent block.
    assert regime.label == "turbulent"
    assert regime.risk_scalar() < 1.0


def test_calm_regime_takes_full_size():
    data = np.concatenate([RNG.normal(0, 0.03, 600), RNG.normal(0, 0.005, 600)])
    regime = fit_vol_regime(data)
    assert regime.label == "calm"
    assert regime.risk_scalar() == pytest.approx(1.0)


def test_states_are_ordered_so_labels_never_flip():
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        data = np.concatenate([rng.normal(0, 0.004, 400), rng.normal(0, 0.02, 400)])
        regime = fit_vol_regime(data)
        assert regime.calm_std <= regime.turbulent_std


def test_regime_fit_degrades_gracefully_on_short_series():
    regime = fit_vol_regime([0.01, -0.01, 0.02])
    assert not regime.converged
    assert regime.risk_scalar() == pytest.approx(1.0)


def test_trend_regime_picks_the_right_toolkit():
    trending = np.cumsum(RNG.normal(0.5, 1.0, 2000))
    result = classify_trend_regime(trending, adx=35.0)
    assert result.trending
    assert "trend following" in result.toolkit

    ranging = ornstein_uhlenbeck(seed=9) + 100.0
    quiet = classify_trend_regime(ranging, adx=12.0)
    assert not quiet.trending
    assert "range" in quiet.toolkit


def test_disagreement_between_hurst_and_adx_lowers_confidence():
    """ADX says trend, structure says range: report low confidence, not a coin flip."""
    ranging = ornstein_uhlenbeck(seed=9) + 100.0
    conflicted = classify_trend_regime(ranging, adx=35.0)
    assert conflicted.confidence < 0.5


# ------------------------------------------------------------------------ kelly


def test_kelly_formula_matches_the_textbook():
    # p=0.6, b=2  ->  0.6 - 0.4/2 = 0.4
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.4)
    # A losing proposition returns zero, never a negative size.
    assert kelly_fraction(0.3, 1.0) == 0.0


def test_kelly_uses_the_lower_confidence_bound_not_the_point_estimate():
    """Sizing off the optimistic end of an interval is how accounts die."""
    outcomes = [TradeOutcome("X", 2.0)] * 30 + [TradeOutcome("X", -1.0)] * 20
    stats = measure(outcomes)
    conservative = from_edge_stats(stats, use_lower_bound=True)
    optimistic = from_edge_stats(stats, use_lower_bound=False)
    assert conservative.win_rate_used < optimistic.win_rate_used
    assert conservative.full_kelly < optimistic.full_kelly


def test_fractional_kelly_is_a_fraction_of_full():
    stats = measure([TradeOutcome("X", 2.0)] * 40 + [TradeOutcome("X", -1.0)] * 30)
    quarter = from_edge_stats(stats, fraction=0.25)
    assert quarter.recommended == pytest.approx(quarter.full_kelly * 0.25)


def test_no_edge_means_no_position():
    stats = measure([TradeOutcome("X", 1.0)] * 20 + [TradeOutcome("X", -1.0)] * 60)
    sizing = from_edge_stats(stats)
    assert not sizing.positive_edge
    assert sizing.recommended == 0.0
    assert "does not justify" in sizing.note


def test_murphy_cap_binds_when_kelly_is_aggressive():
    stats = measure([TradeOutcome("X", 5.0)] * 80 + [TradeOutcome("X", -1.0)] * 20)
    qty, binding = kelly_position(stats, equity=100_000.0, risk_per_unit=10.0)
    # Kelly would happily bet far more than 5% here.
    assert "risk cap" in binding
    assert qty == pytest.approx(100_000.0 * 0.05 / 10.0)


def test_small_sample_falls_back_to_the_fixed_cap():
    stats = measure([TradeOutcome("X", 2.0)] * 5, min_sample=30)
    qty, binding = kelly_position(stats, equity=100_000.0, risk_per_unit=10.0)
    assert "too small" in binding
    assert qty > 0


def test_growth_rate_peaks_at_full_kelly():
    p, b = 0.6, 2.0
    optimum = kelly_fraction(p, b)
    best = growth_rate(p, b, optimum)
    assert best > growth_rate(p, b, optimum * 0.5)
    assert best > growth_rate(p, b, min(0.99, optimum * 1.8))


def test_overbetting_past_full_capital_is_ruinous():
    assert growth_rate(0.6, 2.0, 1.0) == float("-inf")


def test_half_kelly_keeps_most_of_the_growth():
    """The practical argument for betting under Kelly."""
    p, b = 0.55, 2.0
    full = kelly_fraction(p, b)
    assert growth_rate(p, b, full * 0.5) > 0.7 * growth_rate(p, b, full)
