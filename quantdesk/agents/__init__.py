"""The agent roster.

Deliberately a mix of styles that disagree with each other. A desk of five
momentum agents wearing different names is one bet with false diversification;
mean reversion and trend following genuinely offset, and the cross-sectional
agent trades relative value rather than direction.
"""

from __future__ import annotations

from quantdesk.agents.base import (
    AgentContext,
    BaseAgent,
    RegimeAgent,
    RegimeView,
    SignalAgent,
)
from quantdesk.agents.breakout import BreakoutAgent
from quantdesk.agents.formations import (
    CandleAgent,
    ElliottAgent,
    PatternAgent,
    murphy_agents,
)
from quantdesk.agents.mean_reversion import MeanReversionAgent
from quantdesk.agents.momentum import MomentumAgent
from quantdesk.agents.murphy import (
    DivergenceAgent,
    MurphyChecklistAgent,
    StructureAgent,
)
from quantdesk.agents.relative_strength import RelativeStrengthAgent
from quantdesk.agents.vol_regime import VolRegimeAgent

__all__ = [
    "AgentContext",
    "BaseAgent",
    "BreakoutAgent",
    "CandleAgent",
    "DivergenceAgent",
    "ElliottAgent",
    "MeanReversionAgent",
    "MomentumAgent",
    "MurphyChecklistAgent",
    "PatternAgent",
    "RegimeAgent",
    "RegimeView",
    "RelativeStrengthAgent",
    "SignalAgent",
    "StructureAgent",
    "VolRegimeAgent",
    "build_roster",
    "murphy_agents",
]


def build_roster(settings=None) -> tuple[list[SignalAgent], list[RegimeAgent]]:
    """Construct the default agent roster from configuration.

    Returns ``(signal_agents, regime_agents)``. The LLM agent is only included
    when explicitly enabled and its dependencies resolve.
    """
    from quantdesk.config import get_settings

    settings = settings or get_settings()

    # The statistical agents and the Murphy agents genuinely disagree with each
    # other, which is the point. Momentum and mean reversion offset; the Murphy set
    # reads structure the statistical agents are blind to, and refuses trades they
    # would happily take against a higher-timeframe trend.
    signal_agents: list[SignalAgent] = [
        MomentumAgent(),
        MeanReversionAgent(),
        BreakoutAgent(),
        RelativeStrengthAgent(),
    ]
    if getattr(settings, "enable_murphy_agents", True):
        signal_agents.extend(murphy_agents())

    if settings.enable_llm_agent:
        from quantdesk.agents.llm_research import LLMResearchAgent

        llm = LLMResearchAgent(
            model=settings.llm_model,
            api_key=settings.openai_key(),
            interval_seconds=settings.llm_interval_seconds,
        )
        # Only add it if it actually initialised; otherwise it's dead weight.
        if llm.enabled:
            signal_agents.append(llm)

    regime_agents: list[RegimeAgent] = [VolRegimeAgent()]
    return signal_agents, regime_agents
