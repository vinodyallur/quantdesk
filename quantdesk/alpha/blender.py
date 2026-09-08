"""Signal blending and position sizing.

Turns a pile of per-agent opinions into a single target weight per symbol. Three
distinct steps, kept separate because they answer different questions:

1. **Consensus** - what does the roster collectively think? A confidence-weighted
   mean of agent scores, so an agent that says "weakly long" pulls the blend less
   than one that says "strongly long". Crucially the denominator uses confidence
   too, so five silent agents and one loud one produce a strong blend rather than
   a diluted one.

2. **Conviction** - how much of the roster actually has a view? Computed
   separately from consensus. Two agents agreeing at full confidence is a very
   different situation from five agents where four are near-silent, even though
   the consensus score can be identical. Conviction scales size.

3. **Sizing** - volatility targeting. A raw score of +0.5 should mean a much
   smaller position in something with 80% annualised vol than in something with
   20%. Without this step the book's risk is dominated by whatever is most
   volatile, and position weights stop meaning anything comparable.

Exposure caps are applied last, and applied proportionally so the *shape* of the
book survives scaling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from quantdesk.agents.base import RegimeView
from quantdesk.core.types import Signal, TargetPosition
from quantdesk.features.engine import FeatureSnapshot

log = logging.getLogger(__name__)


@dataclass(slots=True)
class BlendConfig:
    min_signal_score: float = 0.15
    max_position_weight: float = 0.20
    max_gross_exposure: float = 1.00
    max_net_exposure: float = 0.60
    target_volatility: float = 0.20
    #: Ceiling on the vol-targeting multiplier. Without this, a very quiet asset
    #: would be levered up enormously on the assumption that quiet stays quiet.
    max_vol_scalar: float = 2.5
    min_vol_scalar: float = 0.25
    #: Agent trust weights by name. Missing agents default to 1.0.
    agent_weights: dict[str, float] = field(default_factory=dict)


class SignalBlender:
    """Combines agent signals into target portfolio weights."""

    def __init__(self, config: BlendConfig | None = None) -> None:
        self.config = config or BlendConfig()
        self.last_diagnostics: dict[str, dict[str, float]] = {}

    def blend(
        self,
        signals: dict[str, list[Signal]],
        snapshots: dict[str, FeatureSnapshot],
        regime_views: list[RegimeView] | None = None,
        agent_weights: dict[str, float] | None = None,
    ) -> list[TargetPosition]:
        """Produce target weights for every symbol with a usable view."""
        cfg = self.config
        weights = {**cfg.agent_weights, **(agent_weights or {})}

        # Risk appetite is the minimum across regime agents: any one of them saying
        # "stand down" should be respected rather than averaged away.
        appetite = 1.0
        regime_label = ""
        if regime_views:
            worst = min(regime_views, key=lambda r: r.risk_appetite)
            appetite = worst.risk_appetite
            regime_label = worst.label

        targets: list[TargetPosition] = []
        diagnostics: dict[str, dict[str, float]] = {}

        for symbol, sigs in signals.items():
            snap = snapshots.get(symbol)
            if snap is None or not snap.ready or not sigs:
                continue

            num = 0.0
            den = 0.0
            weight_total = 0.0
            contributors: dict[str, float] = {}
            for sig in sigs:
                w = weights.get(sig.agent, 1.0)
                if w <= 0:
                    continue
                effective = w * sig.confidence
                num += effective * sig.score
                den += effective
                weight_total += w
                contributors[sig.agent] = sig.weighted_score

            if den <= 1e-9 or weight_total <= 1e-9:
                continue

            consensus = num / den                 # confidence-weighted mean score
            conviction = min(1.0, den / weight_total)  # share of roster engaged

            if abs(consensus) < cfg.min_signal_score:
                # Explicit zero target: this is how positions get closed when the
                # edge fades, rather than being silently left on the book.
                targets.append(
                    TargetPosition(
                        symbol=symbol,
                        weight=0.0,
                        score=consensus,
                        reason=f"consensus {consensus:+.2f} below threshold",
                        contributors=contributors,
                    )
                )
                continue

            vol_scalar = self._vol_scalar(snap)
            raw = consensus * conviction * vol_scalar * appetite
            capped = max(-cfg.max_position_weight, min(cfg.max_position_weight, raw))

            reason = (
                f"consensus {consensus:+.2f} x conviction {conviction:.2f} "
                f"x vol {vol_scalar:.2f} x regime {appetite:.2f}"
            )
            if regime_label and appetite < 1.0:
                reason += f" [{regime_label}]"

            targets.append(
                TargetPosition(
                    symbol=symbol,
                    weight=capped,
                    score=consensus,
                    reason=reason,
                    contributors=contributors,
                )
            )
            diagnostics[symbol] = {
                "consensus": consensus,
                "conviction": conviction,
                "vol_scalar": vol_scalar,
                "appetite": appetite,
                "raw_weight": raw,
                "weight": capped,
            }

        targets = self._apply_exposure_caps(targets)
        self.last_diagnostics = diagnostics
        return targets

    # ---------------------------------------------------------------- sizing
    def _vol_scalar(self, snap: FeatureSnapshot) -> float:
        """Multiplier that equalises risk contribution across symbols."""
        cfg = self.config
        vol = snap.realized_vol
        if vol <= 1e-6:
            # No usable volatility estimate: refuse to lever up on ignorance.
            return cfg.min_vol_scalar
        scalar = cfg.target_volatility / vol
        return max(cfg.min_vol_scalar, min(cfg.max_vol_scalar, scalar))

    def _apply_exposure_caps(self, targets: list[TargetPosition]) -> list[TargetPosition]:
        """Scale the whole book to respect gross and net exposure limits.

        Scaling is proportional rather than per-symbol truncation, which preserves
        the relative sizing the blender chose. Truncating the largest positions
        instead would quietly reshape the portfolio into something nobody decided.
        """
        cfg = self.config
        active = [t for t in targets if abs(t.weight) > 1e-9]
        if not active:
            return targets

        gross = sum(abs(t.weight) for t in active)
        if gross > cfg.max_gross_exposure > 0:
            scale = cfg.max_gross_exposure / gross
            for t in active:
                t.weight *= scale
                t.reason += f" | gross-scaled x{scale:.2f}"

        net = sum(t.weight for t in active)
        if abs(net) > cfg.max_net_exposure > 0:
            # Shrink only the side causing the imbalance, so we reduce the
            # directional tilt without flattening the offsetting positions.
            excess_side = 1.0 if net > 0 else -1.0
            same_side = [t for t in active if (t.weight > 0) == (excess_side > 0)]
            same_total = sum(abs(t.weight) for t in same_side)
            reduction = abs(net) - cfg.max_net_exposure
            if same_total > 1e-9:
                factor = max(0.0, 1.0 - reduction / same_total)
                for t in same_side:
                    t.weight *= factor
                    t.reason += f" | net-scaled x{factor:.2f}"

        return targets
