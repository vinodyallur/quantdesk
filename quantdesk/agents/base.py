"""Agent framework.

Two kinds of agent, because "what should I own" and "how much risk should the
desk be taking" are different questions and conflating them produces muddled
strategies:

* :class:`SignalAgent` - directional. Emits a :class:`Signal` scoring a symbol
  from -1 (short) to +1 (long) with a separate confidence.
* :class:`RegimeAgent` - non-directional. Emits a :class:`RegimeView` scaling the
  desk's overall risk appetite. A vol-spike detector belongs here: it has no view
  on direction, only on whether now is a good time to be sized up.

Agents receive an :class:`AgentContext` rather than reaching for global state, so
they are trivially testable and can see the cross-section when they need it.

Contract for implementors:

* Be pure with respect to the context. Mutating shared state is a bug.
* Never look at ``series`` beyond the current bar. The context only ever contains
  closed bars up to and including now, but sloppy indexing can still cheat.
* Return ``None`` when you have no view. A flat score of 0.0 and "no opinion" are
  treated the same by the blender, but ``None`` is cheaper and clearer.
* Never raise. The pipeline catches exceptions and disables repeat offenders, but
  a quiet degradation is better than a disabled agent.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from quantdesk.core.types import Bar, Position, Signal
from quantdesk.data.series import BarSeries, MarketBook
from quantdesk.features.engine import FeatureSnapshot

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from quantdesk.analysis.checklist import Verdict
    from quantdesk.analysis.symbol import SymbolAnalysis

log = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentContext:
    """Everything an agent is allowed to see on a given bar."""

    bar: Bar
    snapshot: FeatureSnapshot
    series: BarSeries
    all_snapshots: dict[str, FeatureSnapshot]
    book: MarketBook
    now: datetime
    positions: dict[str, Position] = field(default_factory=dict)
    equity: float = 0.0
    #: Murphy analysis stack for this symbol. Computed once per bar by the pipeline
    #: and shared, because running it per agent would be both wasteful and a source
    #: of disagreement between agents that should be seeing the same chart.
    analysis: "SymbolAnalysis | None" = None
    all_analysis: dict[str, "SymbolAnalysis"] = field(default_factory=dict)
    #: Chapter 19 checklist verdict for this symbol, also computed once and shared.
    verdict: "Verdict | None" = None

    @property
    def symbol(self) -> str:
        return self.bar.symbol

    @property
    def price(self) -> float:
        return self.bar.close

    def position(self, symbol: str | None = None) -> Position | None:
        return self.positions.get(symbol or self.symbol)

    def current_weight(self, symbol: str | None = None) -> float:
        """Signed position weight as a fraction of equity."""
        pos = self.position(symbol)
        if pos is None or self.equity <= 0:
            return 0.0
        return pos.market_value(self.snapshot.price) / self.equity


@dataclass(slots=True)
class RegimeView:
    """A non-directional opinion on how much risk the desk should carry."""

    agent: str
    ts: datetime
    risk_appetite: float          # 0 = stand down, 1 = full size
    label: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        self.risk_appetite = max(0.0, min(1.0, self.risk_appetite))


class BaseAgent(ABC):
    """Shared plumbing for all agents."""

    #: Human-readable identifier, used in the UI and in attribution.
    name: str = "agent"
    #: Blender trust weight. Relative, not absolute; the blender normalises.
    weight: float = 1.0
    #: Bars this agent needs before its output is meaningful.
    warmup: int = 0

    def __init__(self, name: str | None = None, weight: float | None = None) -> None:
        if name:
            self.name = name
        if weight is not None:
            self.weight = weight
        self.enabled = True
        self.error_count = 0
        self.signal_count = 0

    def reset(self) -> None:
        """Clear per-run state. Called before each backtest."""
        self.error_count = 0
        self.signal_count = 0

    def describe(self) -> str:
        return self.__doc__.strip().splitlines()[0] if self.__doc__ else self.name

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} weight={self.weight}>"


class SignalAgent(BaseAgent):
    """An agent with a directional view on individual symbols."""

    @abstractmethod
    def evaluate(self, ctx: AgentContext) -> Signal | None:
        """Return a directional view, or None for no opinion."""

    def on_bar(self, ctx: AgentContext) -> Signal | None:
        """Guarded entry point used by the pipeline."""
        if not self.enabled:
            return None
        if not ctx.snapshot.ready or ctx.snapshot.bars_seen < self.warmup:
            return None
        try:
            sig = self.evaluate(ctx)
        except Exception:  # noqa: BLE001 - one bad agent must not stop the desk
            self.error_count += 1
            log.exception("agent %s failed on %s", self.name, ctx.symbol)
            if self.error_count >= 10:
                self.enabled = False
                log.error("agent %s disabled after repeated failures", self.name)
            return None
        if sig is not None:
            self.signal_count += 1
        return sig

    # ------------------------------------------------------------- helpers
    def signal(
        self,
        ctx: AgentContext,
        score: float,
        confidence: float = 1.0,
        rationale: str = "",
        horizon_bars: int = 12,
        **features: float,
    ) -> Signal:
        """Build a Signal stamped with this agent's identity."""
        return Signal(
            agent=self.name,
            symbol=ctx.symbol,
            ts=ctx.bar.ts,
            score=score,
            confidence=confidence,
            horizon_bars=horizon_bars,
            rationale=rationale,
            features=features,
        )


class RegimeAgent(BaseAgent):
    """An agent that scales risk appetite instead of picking a direction."""

    @abstractmethod
    def evaluate(self, ctx: AgentContext) -> RegimeView | None:
        ...

    def on_bar(self, ctx: AgentContext) -> RegimeView | None:
        if not self.enabled or not ctx.snapshot.ready:
            return None
        try:
            return self.evaluate(ctx)
        except Exception:  # noqa: BLE001
            self.error_count += 1
            log.exception("regime agent %s failed on %s", self.name, ctx.symbol)
            return None


# ---------------------------------------------------------------- math helpers


def squash(x: float, scale: float = 1.0) -> float:
    """Map an unbounded value into (-1, 1) via tanh.

    Used so every agent's raw statistic lands on the same [-1, 1] axis and the
    blender can compare them. ``scale`` is the value that maps to roughly 0.76,
    so pick it to represent "a strong reading" for your statistic.
    """
    if scale <= 0 or not math.isfinite(x):
        return 0.0
    return math.tanh(x / scale)


def linear_map(x: float, lo: float, hi: float, out_lo: float = 0.0, out_hi: float = 1.0) -> float:
    """Rescale x from [lo, hi] onto [out_lo, out_hi], clamped at the ends."""
    if hi - lo <= 1e-12:
        return out_lo
    t = (x - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    return out_lo + t * (out_hi - out_lo)
