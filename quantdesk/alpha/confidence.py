"""Confidence scoring, and the leverage that confidence is allowed to buy.

Two jobs, deliberately separated, because conflating them is how leveraged accounts die.

**Scoring** turns the evidence the pipeline already produced into one number in [0, 1],
with the contribution of each component kept alongside it. A score with no breakdown is
untraceable: when a trade goes wrong the only useful question is which piece of evidence
was wrong, and an opaque scalar cannot answer it.

**Leverage** converts that score into a multiplier - and then throws most of it away,
because confidence is not the binding constraint. The binding constraint is arithmetic:

    at leverage L, the venue closes the position at roughly (1/L - maintenance) adverse

So a stop further away than that distance is not a stop. The position is gone before price
reaches it, the loss is the entire margin rather than the intended risk, and every risk
figure computed against that stop was fiction. Rearranged, the *only* leverage a stop can
survive is::

    L < 1 / (stop_distance + maintenance_margin)

A 2% stop on a contract with 2.5% maintenance margin therefore tolerates about 22x, and no
amount of conviction changes that. :func:`decide_leverage` computes the requested leverage
from confidence and then caps it here, reporting both so the gap is visible.

**What leverage does and does not do in this system.** Risk per trade is targeted at a
fixed fraction of equity, and the stop distance sets the loss if wrong, so leverage does
not increase the intended loss. What it does is relax Murphy's commitment ceiling, which
frees the risk target to actually bind. Two consequences follow, both real:

* Fees scale with notional, so a higher multiplier pays proportionally more commission.
  Measured on ten days of real perps bars, the loss tracked the fee bill almost exactly:
  -1.33% return against 0.51% fees at 1x, -7.38% against 2.86% at 20x, with hit rate and
  payoff unchanged throughout.
* A gap through the stop is no longer bounded by the stop. Leverage converts an
  unlikely-but-survivable event into an account-ending one.

Neither is a reason to refuse leverage. Both are reasons to state that **leverage is a
multiplier on expectancy, not a source of it.** Multiplying a negative expectancy by 100
does not make it positive; it makes it arrive faster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Fraction of the liquidation-safe leverage actually used. The stop needs room to be hit
#: on a wick without the venue closing first, and the fill itself slips.
LIQUIDATION_SAFETY = 0.55

#: Confidence below this is not traded at all. A weak signal sized small is still a weak
#: signal paying full commission both ways.
MIN_TRADABLE_CONFIDENCE = 0.35

#: Curvature of the confidence-to-leverage map. Above 1 the response is concave: leverage
#: stays low until the evidence is genuinely strong, rather than rising linearly from the
#: first hint of a signal.
LEVERAGE_GAMMA = 2.2

#: Maintenance margin assumed when the contract does not supply a usable one.
#:
#: Binance's requirement is *tiered by notional*, and the single ``maintMarginPercent`` in
#: ``exchangeInfo`` is the highest tier - 2.5% for BTCUSDT, against roughly 0.4% for a
#: small position. Using the published figure makes the liquidation cap about six times
#: tighter than reality, which sounds safe but in practice pins leverage to the cap at every
#: confidence level and throws the scoring away. The real brackets need an authenticated
#: ``leverageBracket`` call, so this is the tier-1 figure for majors, and positions large
#: enough to climb the tiers will be under-margined by it. That is a genuine limitation,
#: not a rounding choice.
DEFAULT_MAINT_MARGIN = 0.005


@dataclass(slots=True)
class Evidence:
    """Everything the score is computed from, as plain numbers.

    A flat record rather than the pipeline's objects, so the scorer is testable in
    isolation and so a change to an analysis class cannot silently alter scoring.
    """

    checklist_conviction: float = 0.0
    """Murphy chapter 19 weight-of-evidence conviction, [0, 1]."""
    checklist_agreement: float = 0.0
    """How much of the checklist agrees with itself, [0, 1]."""
    checklist_coverage: float = 0.0
    """Fraction of checklist questions that could be answered at all, [0, 1]."""
    agent_consensus: float = 0.0
    """Net agent view in the trade's direction, [0, 1] after normalising."""
    agent_count: int = 0
    """How many agents had a view. One agent agreeing with itself is not consensus."""
    timeframe_aligned: bool = False
    """Whether the higher timeframe permits this direction."""
    pattern_reward_risk: float = 0.0
    """Reward-to-risk of a completed pattern, 0 when the stop is generic."""
    has_pattern_stop: bool = False
    """True when the stop comes from a pattern rather than an ATR multiple."""
    regime_appetite: float = 1.0
    """Volatility-regime risk appetite, [0, 1]."""
    funding_against: float = 0.0
    """Annualised funding cost of holding this direction, as a fraction. Negative pays you."""
    blockers: int = 0
    """Hard checklist blockers. Any at all should collapse the score."""


@dataclass
class Confidence:
    """A score in [0, 1] and the components that produced it."""

    score: float
    components: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def tradable(self) -> bool:
        return self.score >= MIN_TRADABLE_CONFIDENCE

    @property
    def band(self) -> str:
        if self.score >= 0.80:
            return "very strong"
        if self.score >= 0.65:
            return "strong"
        if self.score >= 0.50:
            return "moderate"
        if self.score >= MIN_TRADABLE_CONFIDENCE:
            return "weak"
        return "not tradable"

    def summary(self) -> list[str]:
        lines = [f"confidence {self.score:.3f} ({self.band})"]
        for name in sorted(self.components, key=lambda k: -self.weights.get(k, 0)):
            value = self.components[name]
            weight = self.weights.get(name, 0.0)
            lines.append(f"  {name:<18} {value:>6.3f} x weight {weight:.2f}")
        lines.extend(f"  note: {n}" for n in self.notes)
        return lines


def score_confidence(evidence: Evidence) -> Confidence:
    """Combine the evidence into a single confidence in [0, 1].

    Weighted mean rather than a product: a product lets any single zero veto the trade,
    which sounds prudent but in practice means a missing indicator during warmup silences
    the desk entirely. Hard vetoes are expressed as blockers instead, which is what they
    are.
    """
    components: dict[str, float] = {}
    weights: dict[str, float] = {}
    notes: list[str] = []

    def add(name: str, value: float, weight: float) -> None:
        components[name] = _clamp(value)
        weights[name] = weight

    # Murphy's checklist is the heaviest input: it is the whole weight-of-evidence idea,
    # and it already aggregates trend, structure and momentum.
    add("checklist", evidence.checklist_conviction, 0.28)
    add("agreement", evidence.checklist_agreement, 0.12)

    # Consensus is discounted when few agents spoke. Two agents agreeing is an opinion;
    # eight agreeing is evidence, and the square-root taper says so without a hard cutoff.
    breadth = min(1.0, math.sqrt(max(0, evidence.agent_count) / 6.0))
    add("consensus", evidence.agent_consensus * breadth, 0.20)
    if evidence.agent_count and evidence.agent_count < 3:
        notes.append(f"only {evidence.agent_count} agent(s) had a view")

    # "Work from the long term to the short term." Trading against the higher timeframe is
    # the single most reliable way to be right about direction and lose anyway.
    add("timeframe", 1.0 if evidence.timeframe_aligned else 0.0, 0.16)
    if not evidence.timeframe_aligned:
        notes.append("higher timeframe does not permit this direction")

    # A pattern carries its own protective level and measured objective, which is strictly
    # better information than a volatility stop. Reward-to-risk above the 3:1 yardstick
    # adds nothing further - the gate already required it.
    quality = 0.0
    if evidence.has_pattern_stop:
        quality = 0.6 + 0.4 * _clamp((evidence.pattern_reward_risk - 3.0) / 3.0)
    add("structure", quality, 0.12)

    add("regime", evidence.regime_appetite, 0.07)

    # Perps only: funding is a published, certain cost of holding this direction, unlike
    # every other input here, which is an estimate. A 20%/yr carry against the trade is
    # material on a position held for hours.
    carry = _clamp(1.0 - max(0.0, evidence.funding_against) / 0.20)
    add("funding", carry, 0.05)
    if evidence.funding_against > 0.10:
        notes.append(f"funding costs {evidence.funding_against:.1%}/yr in this direction")

    total_weight = sum(weights.values())
    raw = sum(components[k] * weights[k] for k in components) / total_weight

    if evidence.blockers > 0:
        # A blocker is a hard refusal in the checklist. Scoring around it would make the
        # veto advisory, which is the same as not having one.
        raw *= 0.25
        notes.append(f"{evidence.blockers} checklist blocker(s), score cut to a quarter")

    if evidence.checklist_coverage < 0.5:
        raw *= 0.5 + evidence.checklist_coverage
        notes.append(
            f"only {evidence.checklist_coverage:.0%} of the checklist could be answered"
        )

    return Confidence(score=_clamp(raw), components=components, weights=weights, notes=notes)


@dataclass
class LeverageDecision:
    """What leverage was asked for, what was allowed, and why."""

    leverage: float
    """The multiplier to actually use."""
    requested: float
    """What confidence alone asked for."""
    liquidation_cap: float
    """Largest multiplier at which the stop is still reached before liquidation."""
    venue_cap: float
    user_cap: float
    binding: str
    """Which limit decided it."""
    stop_distance_pct: float = 0.0
    maint_margin_pct: float = 0.0
    liquidation_move_pct: float = 0.0
    """Adverse move that liquidates at the chosen leverage."""

    @property
    def capped(self) -> bool:
        return self.leverage < self.requested - 1e-9

    def summary(self) -> list[str]:
        out = [
            f"leverage {self.leverage:.1f}x  (confidence asked for "
            f"{self.requested:.1f}x, {self.binding} decided it)",
            f"  stop is {self.stop_distance_pct:.2%} away; liquidation at "
            f"{self.liquidation_move_pct:.2%} adverse",
            f"  caps: liquidation {self.liquidation_cap:.1f}x, venue "
            f"{self.venue_cap:.1f}x, requested max {self.user_cap:.1f}x",
        ]
        if self.capped:
            out.append(
                "  the stop would not survive the requested multiplier, so it was reduced"
            )
        return out


def decide_leverage(
    confidence: float,
    stop_distance_pct: float,
    maint_margin_pct: float = DEFAULT_MAINT_MARGIN,
    user_cap: float = 100.0,
    venue_cap: float = 125.0,
    safety: float = LIQUIDATION_SAFETY,
    gamma: float = LEVERAGE_GAMMA,
) -> LeverageDecision:
    """Turn confidence into a leverage multiplier the stop can actually survive.

    ``stop_distance_pct`` is the distance from entry to the protective stop as a fraction
    of price. It is the input that matters: a tight stop tolerates high leverage and a wide
    one does not, regardless of how good the setup looks.
    """
    confidence = _clamp(confidence)
    user_cap = max(1.0, user_cap)
    requested = 1.0 + (confidence ** gamma) * (user_cap - 1.0)

    # The arithmetic constraint. Without a stop there is no distance to protect and no
    # basis for leverage at all, so it collapses to 1x rather than to infinity.
    if stop_distance_pct <= 0:
        liquidation_cap = 1.0
    else:
        liquidation_cap = safety / (stop_distance_pct + max(0.0, maint_margin_pct))
    liquidation_cap = max(1.0, liquidation_cap)
    venue_cap = max(1.0, venue_cap)

    options = {
        "confidence": requested,
        "liquidation distance": liquidation_cap,
        "venue ceiling": venue_cap,
        "requested max": user_cap,
    }
    binding, leverage = min(options.items(), key=lambda kv: kv[1])
    leverage = max(1.0, leverage)

    return LeverageDecision(
        leverage=leverage,
        requested=requested,
        liquidation_cap=liquidation_cap,
        venue_cap=venue_cap,
        user_cap=user_cap,
        binding=binding,
        stop_distance_pct=stop_distance_pct,
        maint_margin_pct=maint_margin_pct,
        liquidation_move_pct=max(0.0, 1.0 / leverage - maint_margin_pct),
    )


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return low
    return max(low, min(high, value))
