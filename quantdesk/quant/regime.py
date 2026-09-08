"""Regime detection.

Murphy quotes Wilder's observation that markets trend strongly only about 30% of the
time, and his whole framework behaves differently depending on which state you are in:
trendlines and pattern breakouts work in a trending market, oscillators and mean
reversion work in a range. Getting that classification wrong is worse than having no
signal, because it applies the right technique to the wrong conditions.

Two independent classifiers, because they answer different questions and their
disagreement is informative:

**A two-state Gaussian mixture on returns**, fitted by expectation-maximisation. Splits
history into a calm, low-volatility state and a turbulent, high-volatility one, and
reports the probability of being in each right now. This is a volatility regime and says
nothing about direction.

**A directional classifier** from ADX and the Hurst exponent, which says whether the
market is trending or ranging - which is what decides which of Murphy's toolkits
applies.

Fitted with EM by hand rather than pulled from a library, because the model is small
enough that the assumptions should be visible: two states, Gaussian emissions,
independent observations. That last assumption is wrong for financial returns - volatility
clusters - so this is a mixture model, not a full hidden Markov model. The transition
matrix is estimated after the fact from the assigned states, which is a reasonable
approximation and is labelled as one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from quantdesk.quant.distributions import hurst_exponent


@dataclass(slots=True)
class VolRegime:
    """Two-state volatility classification."""

    calm_mean: float
    calm_std: float
    turbulent_mean: float
    turbulent_std: float
    prob_turbulent: float
    """Probability the most recent observation belongs to the turbulent state."""
    state_sequence: np.ndarray = field(default_factory=lambda: np.array([]))
    transition: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    converged: bool = False
    iterations: int = 0

    @property
    def label(self) -> str:
        return "turbulent" if self.prob_turbulent > 0.5 else "calm"

    #: Ceiling on the reported volatility ratio. A fit that separates states by more
    #: than this has almost certainly found a degenerate solution rather than two real
    #: regimes, and reporting "296 million times more volatile" is worse than useless.
    MAX_VOL_RATIO: float = 50.0

    @property
    def degenerate(self) -> bool:
        """Did the fit collapse onto a near-zero-variance state?"""
        if self.calm_std <= 0 or self.turbulent_std <= 0:
            return True
        return self.turbulent_std / self.calm_std > self.MAX_VOL_RATIO

    @property
    def vol_ratio(self) -> float:
        """How much wider the turbulent state is than the calm one, capped."""
        if self.calm_std <= 0:
            return 1.0
        return min(self.MAX_VOL_RATIO, self.turbulent_std / self.calm_std)

    @property
    def persistence(self) -> float:
        """Probability the current state continues into the next bar."""
        index = 1 if self.prob_turbulent > 0.5 else 0
        return float(self.transition[index, index])

    def risk_scalar(self, floor: float = 0.35) -> float:
        """How much size to take given the regime.

        A turbulent state means the same position carries more risk, so exposure should
        fall. Scaling inversely with the volatility ratio keeps risk roughly constant
        rather than keeping position size constant, which is the entire point of
        detecting the regime.
        """
        if self.prob_turbulent <= 0.5:
            return 1.0
        scaled = 1.0 / max(1.0, self.vol_ratio)
        return max(floor, scaled)

    def summary(self) -> list[str]:
        return [
            f"regime       {self.label.upper()} "
            f"(p={self.prob_turbulent:.2f}, persistence {self.persistence:.2f})",
            f"calm         mean {self.calm_mean:+.5f}  std {self.calm_std:.5f}",
            f"turbulent    mean {self.turbulent_mean:+.5f}  std {self.turbulent_std:.5f}",
            f"vol ratio    {self.vol_ratio:.2f}x  -> risk scalar {self.risk_scalar():.2f}"
            + ("  (capped)" if self.degenerate else ""),
            f"fit          {'converged' if self.converged else 'did not converge'} "
            f"in {self.iterations} iterations"
            + ("  DEGENERATE: states are not distinguishable" if self.degenerate else ""),
        ]


def fit_vol_regime(
    returns: np.ndarray | list[float],
    max_iter: int = 200,
    tol: float = 1e-6,
    drop_zeros: bool = True,
) -> VolRegime:
    """Fit a two-state Gaussian mixture to returns by expectation-maximisation.

    ``drop_zeros`` matters when fitting *strategy* returns rather than market returns.
    A flat desk produces exactly-zero returns, and a large spike of identical zeros makes
    the mixture converge on a degenerate solution: one state becomes "exactly zero" with
    near-zero variance and the volatility ratio explodes to millions. Being flat is not a
    volatility regime, so those observations are removed by default. Pass False when
    analysing market returns, where a zero is a genuine quiet bar.
    """
    data = np.asarray(returns, dtype=float)
    data = data[np.isfinite(data)]
    if drop_zeros:
        data = data[np.abs(data) > 0.0]
    n = data.size
    if n < 30:
        std = float(data.std(ddof=1)) if n > 1 else 0.0
        return VolRegime(0.0, std, 0.0, std, 0.0, np.zeros(n), np.eye(2), False, 0)

    # Initialise by splitting on absolute size: a crude but stable starting point that
    # matches the structure we expect to find.
    threshold = np.quantile(np.abs(data), 0.7)
    calm_mask = np.abs(data) <= threshold
    mu = np.array([data[calm_mask].mean(), data[~calm_mask].mean()])
    sigma = np.array(
        [max(data[calm_mask].std(ddof=1), 1e-9), max(data[~calm_mask].std(ddof=1), 1e-9)]
    )
    weights = np.array([calm_mask.mean(), 1.0 - calm_mask.mean()])

    responsibility = np.zeros((n, 2))
    log_likelihood = -np.inf
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        # E step: responsibility of each state for each observation.
        for k in range(2):
            responsibility[:, k] = weights[k] * _normal_pdf(data, mu[k], sigma[k])
        total = responsibility.sum(axis=1, keepdims=True)
        total[total <= 0] = 1e-300
        responsibility /= total

        # M step.
        counts = responsibility.sum(axis=0)
        counts[counts <= 0] = 1e-300
        weights = counts / n
        mu = (responsibility * data[:, None]).sum(axis=0) / counts
        for k in range(2):
            var = (responsibility[:, k] * (data - mu[k]) ** 2).sum() / counts[k]
            sigma[k] = max(math.sqrt(max(var, 0.0)), 1e-12)

        new_ll = float(np.log(total).sum())
        if abs(new_ll - log_likelihood) < tol:
            converged = True
            break
        log_likelihood = new_ll

    # Order the states so index 0 is always the calm one. Without this the labels flip
    # between runs and downstream logic silently inverts.
    order = np.argsort(sigma)
    mu, sigma = mu[order], sigma[order]
    responsibility = responsibility[:, order]

    states = responsibility.argmax(axis=1)
    return VolRegime(
        calm_mean=float(mu[0]),
        calm_std=float(sigma[0]),
        turbulent_mean=float(mu[1]),
        turbulent_std=float(sigma[1]),
        prob_turbulent=float(responsibility[-1, 1]),
        state_sequence=states,
        transition=_transition_matrix(states),
        converged=converged,
        iterations=iteration,
    )


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.zeros_like(x)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _transition_matrix(states: np.ndarray) -> np.ndarray:
    """Empirical transitions between assigned states.

    Estimated after assignment rather than jointly with the emissions, which a full
    hidden Markov model would do. Good enough for reading persistence, and stated as an
    approximation rather than presented as a fitted HMM.
    """
    matrix = np.zeros((2, 2))
    for a, b in zip(states[:-1], states[1:]):
        matrix[a, b] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


@dataclass(slots=True)
class TrendRegime:
    """Is this market trending or ranging?"""

    hurst: float
    adx: float
    label: str
    confidence: float

    @property
    def trending(self) -> bool:
        return self.label == "trending"

    @property
    def toolkit(self) -> str:
        """Which of Murphy's toolkits applies here."""
        if self.trending:
            return "trend following: trendlines, breakouts, moving averages"
        return "range trading: oscillators, support and resistance, mean reversion"

    def summary(self) -> list[str]:
        return [
            f"trend        {self.label.upper()} (confidence {self.confidence:.2f})",
            f"hurst {self.hurst:.3f}  adx {self.adx:.1f}",
            f"use          {self.toolkit}",
        ]


def classify_trend_regime(
    prices: np.ndarray | list[float], adx: float
) -> TrendRegime:
    """Combine the Hurst exponent with ADX to decide which toolkit applies.

    Two independent measures deliberately: ADX is a short-window indicator that reacts
    quickly, Hurst is a structural property measured over the whole sample. When they
    agree the classification is solid; when they disagree the confidence drops, which is
    the honest answer rather than picking a winner.

    Murphy's threshold for ADX is 20 - below it there is no trend worth following.
    """
    hurst = hurst_exponent(np.asarray(prices, dtype=float))
    hurst_says_trend = hurst > 0.55
    adx_says_trend = adx >= 20.0

    if hurst_says_trend and adx_says_trend:
        return TrendRegime(hurst, adx, "trending", 0.85)
    if not hurst_says_trend and not adx_says_trend:
        return TrendRegime(hurst, adx, "ranging", 0.75)
    # Disagreement: defer to ADX because it is the more responsive of the two, but say
    # so with low confidence.
    label = "trending" if adx_says_trend else "ranging"
    return TrendRegime(hurst, adx, label, 0.4)
