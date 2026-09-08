"""The decision pipeline: bars in, risk-checked orders out.

This is the single path from market data to an order, and both the backtester and the
live desk run *this exact object*. That is the design's most important property. The
usual way a backtest lies is that live trading and simulation run different code and
diverge somewhere subtle; here the only difference is which clock and which broker get
injected.

The per-bar sequence, and why it is in this order:

1. **Analysis.** :class:`~quantdesk.analysis.symbol.SymbolAnalysis` per symbol, and
   the chapter 19 checklist verdict. Computed once and shared with every agent, so
   agents cannot disagree about what the chart says - only about what it means.
2. **Agents.** Each emits a :class:`~quantdesk.core.types.Signal` or abstains.
3. **Blend.** Signals combine into target weights, scaled by the regime agents'
   risk appetite.
4. **Money management.** Murphy's chapter 16 limits convert a target into a size,
   with a real stop and objective. A target that cannot clear the 3:1 gate is
   dropped here rather than sent.
5. **Risk.** The account-level manager gets the last word and can veto or shrink.
6. **Orders.** Emitted, never executed here - execution belongs to the broker.

Nothing in this file touches wall-clock time or the network, which is what makes it
testable and identical across backtest and live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from quantdesk.agents.base import AgentContext, RegimeAgent, RegimeView, SignalAgent
from quantdesk.alpha.blender import SignalBlender
from quantdesk.alpha.confidence import (
    Confidence,
    Evidence,
    LeverageDecision,
    decide_leverage,
    score_confidence,
)
from quantdesk.analysis.checklist import TechnicalChecklist, Verdict
from quantdesk.analysis.patterns import Bias
from quantdesk.analysis.symbol import SymbolAnalysis
from quantdesk.analysis.timeframes import Timeframe
from quantdesk.config import Settings, get_settings
from quantdesk.core.types import (
    AccountSnapshot,
    Bar,
    Order,
    OrderType,
    Position,
    Side,
    Signal,
    TargetPosition,
)
from quantdesk.data.series import MarketBook
from quantdesk.execution.broker import next_order_id
from quantdesk.features.engine import FeatureEngine
from quantdesk.risk.manager import RiskManager
from quantdesk.risk.money import MoneyManager, MoneyRules, PortfolioState, PositionPlan
from quantdesk.trade_manager import TradeAction, TradeManager

log = logging.getLogger(__name__)


@dataclass
class BarDecision:
    """Everything the pipeline concluded on one bar, for the UI and for auditing."""

    ts: datetime
    symbol: str
    signals: list[Signal] = field(default_factory=list)
    target: TargetPosition | None = None
    plan: PositionPlan | None = None
    orders: list[Order] = field(default_factory=list)
    verdict: Verdict | None = None
    rejected: list[str] = field(default_factory=list)
    """Why nothing was traded, when nothing was traded. As useful as the trades."""
    trade_action: "TradeAction | None" = None
    """What trade management decided: open, hold, add, or which kind of exit."""
    confidence: "Confidence | None" = None
    """How strong the evidence was, with the contribution of each component kept."""
    leverage: "LeverageDecision | None" = None
    """What leverage confidence asked for, what was allowed, and which limit bound."""

    @property
    def acted(self) -> bool:
        return bool(self.orders)


class Pipeline:
    """Owns the analysis, agents, blending, sizing and risk for one desk."""

    def __init__(
        self,
        settings: Settings | None = None,
        signal_agents: list[SignalAgent] | None = None,
        regime_agents: list[RegimeAgent] | None = None,
        risk: RiskManager | None = None,
        money: MoneyManager | None = None,
        checklist: TechnicalChecklist | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if signal_agents is None or regime_agents is None:
            from quantdesk.agents import build_roster

            built_signals, built_regimes = build_roster(self.settings)
            signal_agents = signal_agents if signal_agents is not None else built_signals
            regime_agents = regime_agents if regime_agents is not None else built_regimes
        self.signal_agents = signal_agents
        self.regime_agents = regime_agents

        self.features = FeatureEngine(bars_per_year=self.settings.bars_per_year)
        self.book = MarketBook(maxlen=max(600, self.settings.warmup_bars))
        self.blender = SignalBlender()
        self.risk = risk or RiskManager()
        self.money = money or MoneyManager(
            MoneyRules(
                max_market_risk_pct=self.settings.max_risk_per_trade_pct,
                risk_target_pct=self.settings.risk_target_pct,
                min_reward_risk=self.settings.min_reward_risk,
            )
        )
        self.checklist = checklist or TechnicalChecklist(
            require_timeframe_agreement=self.settings.require_timeframe_agreement,
        )

        # Discrete trade management rather than continuous rebalancing. See
        # trade_manager.py for why: rebalancing a Murphy-style position every bar pays
        # the spread nine times per trade and cuts winners short.
        self.trades = TradeManager(
            max_layers=self.money.rules.max_pyramid_layers,
            exit_on_reversal=True,
        )
        self.analysis: dict[str, SymbolAnalysis] = {}
        self.verdicts: dict[str, Verdict] = {}
        #: Annualised funding carry per symbol, signed for a *long*. Populated from
        #: outside because funding is a venue fact rather than something the analysis can
        #: derive, and it is a real cost the confidence score should see.
        self.funding_carry: dict[str, float] = {}
        #: Maintenance margin per symbol, for the liquidation-distance cap. Falls back to
        #: the module default when the contract is unknown.
        self.maint_margin: dict[str, float] = {}
        self.confidences: dict[str, Confidence] = {}
        self.leverages: dict[str, LeverageDecision] = {}
        self.positions: dict[str, Position] = {}
        self.equity = self.settings.starting_equity
        self.start_equity = self.settings.starting_equity
        self.peak_equity = self.settings.starting_equity
        self.decisions: list[BarDecision] = []
        self._latest_signals: dict[str, list[Signal]] = {}
        self._regime_views: list[RegimeView] = []
        self._bars_seen = 0
        self._current_day = None

    # ------------------------------------------------------------------ setup
    def _analysis_for(self, symbol: str) -> SymbolAnalysis:
        sa = self.analysis.get(symbol)
        if sa is None:
            base = _base_timeframe(self.settings.timeframe)
            sa = SymbolAnalysis(
                symbol,
                timeframes=[base, Timeframe.H1, Timeframe.H4],
                base_timeframe=base,
            )
            self.analysis[symbol] = sa
        return sa

    # ------------------------------------------------------------------- bars
    def on_bar(self, bar: Bar) -> BarDecision:
        """Process one closed bar for one symbol and return what was decided."""
        self._bars_seen += 1
        decision = BarDecision(ts=bar.ts, symbol=bar.symbol)

        self.book.append(bar)
        snapshot = self.features.update(bar)
        snapshots = self.features.snapshots()
        sa = self._analysis_for(bar.symbol)
        sa.update(bar)

        verdict = self.checklist.evaluate(sa)
        self.verdicts[bar.symbol] = verdict
        decision.verdict = verdict

        # Mark the position before anything else, so risk and sizing see the truth.
        pos = self.positions.get(bar.symbol)
        if pos is not None:
            pos.last_price = bar.close

        ctx = AgentContext(
            bar=bar,
            snapshot=snapshot,
            series=self.book.series(bar.symbol),
            all_snapshots=snapshots,
            book=self.book,
            now=bar.ts,
            positions=self.positions,
            equity=self.equity,
            analysis=sa,
            all_analysis=self.analysis,
            verdict=verdict,
        )

        # --- agents
        signals = [s for a in self.signal_agents if (s := a.on_bar(ctx)) is not None]
        self._latest_signals[bar.symbol] = signals
        decision.signals = signals

        views = [v for a in self.regime_agents if (v := a.on_bar(ctx)) is not None]
        if views:
            self._regime_views = views

        # --- trade management runs before any new decision, because an open trade's
        # protective stop must be honoured whatever the agents now think. A stop a
        # fresh signal can override is not a stop, and the 5% risk cap that was sized
        # against it would be fiction.
        signal_bias = self._signal_bias(signals)
        if self.trades.active(bar.symbol) is not None:
            handled = self._manage_open_trade(bar, sa, decision, signal_bias)
            if handled:
                self.decisions.append(decision)
                return decision

        if not signals:
            decision.rejected.append("no agent had a view")
            self.decisions.append(decision)
            return decision

        # --- blend. The blender wants the whole cross-section, so accumulated
        # signals for other symbols are passed alongside this bar's.
        pending = dict(self._latest_signals)
        targets = self.blender.blend(
            pending, snapshots, regime_views=self._regime_views or None
        )
        target = next((t for t in targets if t.symbol == bar.symbol), None)
        if target is None or abs(target.weight) < self.settings.rebalance_threshold:
            decision.rejected.append("blended target below the rebalance threshold")
            self.decisions.append(decision)
            return decision
        decision.target = target

        # --- money management
        plan = self._size(bar, sa, target, decision)
        decision.plan = plan
        if (
            self.settings.confidence_leverage
            and decision.confidence is not None
            and not decision.confidence.tradable
        ):
            # A weak signal sized small is still a weak signal paying commission both
            # ways, so it is refused rather than shrunk.
            decision.rejected.append(
                f"confidence {decision.confidence.score:.2f} below the tradable floor"
            )
            self.decisions.append(decision)
            return decision
        if plan is None or not plan.approved:
            decision.rejected.append(
                plan.refusal.value if plan is not None else "no stop available to size against"
            )
            self.decisions.append(decision)
            return decision

        # --- orders, then the account-level risk veto
        orders = self._orders_for(bar, plan, target)
        snap = self._snapshot(bar.ts)
        for order in orders:
            current = self.positions.get(bar.symbol)
            # The risk manager works in *weights*, not currency: its caps are
            # fractions of equity. Passing absolute notional here silently compares
            # thousands of dollars against a cap of 1.0 and vetoes everything.
            check = self.risk.check_order(
                order,
                price=bar.close,
                equity=self.equity,
                current_qty=current.qty if current else 0.0,
                gross_exposure=snap.gross_exposure / self.equity if self.equity > 0 else 0.0,
                net_exposure=snap.net_exposure / self.equity if self.equity > 0 else 0.0,
            )
            if not check.approved:
                decision.rejected.append(f"risk veto: {check.reason}")
                continue
            if check.adjusted_qty is not None:
                if check.adjusted_qty <= 0:
                    decision.rejected.append("risk resized the order to nothing")
                    continue
                order.qty = check.adjusted_qty
                # Risk must follow the resize. Leaving the original figure on a
                # shrunken order overstates risk and quietly deflates every R.
                order.planned_risk = plan.risk_per_unit * order.qty
            decision.orders.append(order)

        # Record the trade so its stop and objective are enforced from the next bar.
        # Only after risk has approved, so the recorded size matches what was sent.
        if decision.orders:
            sent = sum(o.signed_qty for o in decision.orders)
            if abs(sent) > 0:
                self.trades.open_trade(
                    symbol=bar.symbol,
                    side=1 if sent > 0 else -1,
                    entry=bar.close,
                    stop=plan.stop,
                    objective=plan.objective,
                    qty=abs(sent),
                    index=sa.index,
                    ts=bar.ts,
                    reason=plan.notes,
                )
                decision.trade_action = TradeAction.OPEN

        self.decisions.append(decision)
        if len(self.decisions) > 5000:
            del self.decisions[:-5000]
        return decision

    # ------------------------------------------------------- trade management
    def _signal_bias(self, signals: list[Signal]) -> Bias:
        """Net direction of the agents' views, weighted by their trust and conviction."""
        if not signals:
            return Bias.NEUTRAL
        weights = {a.name: a.weight for a in self.signal_agents}
        total = sum(s.weighted_score * weights.get(s.agent, 1.0) for s in signals)
        return Bias.from_sign(total) if abs(total) > 1e-9 else Bias.NEUTRAL

    def _manage_open_trade(
        self,
        bar: Bar,
        sa: SymbolAnalysis,
        decision: BarDecision,
        signal_bias: Bias,
    ) -> bool:
        """Handle an existing position. Returns True when the bar is fully handled.

        Holding returns True as well: doing nothing is the decision, and falling
        through to the sizing path would re-open the continuous-rebalance behaviour
        this replaces.
        """
        action = self.trades.decide(bar, sa.index, signal_bias, 1.0)
        decision.trade_action = action.action
        decision.rejected.append(f"{action.action.value}: {action.reason}")

        if action.action.is_exit:
            trade = action.trade
            if trade is None:
                return True
            side = Side.SELL if trade.long_side else Side.BUY
            decision.orders.append(
                Order(
                    symbol=bar.symbol,
                    side=side,
                    qty=trade.qty,
                    type=OrderType.MARKET,
                    id=next_order_id(),
                    ts=bar.ts,
                    tag=f"exit|{action.action.value}|{action.reason}",
                    # An exit puts no new risk on, so it carries none. The risk it
                    # closes was already booked when the position was opened.
                    planned_risk=0.0,
                )
            )
            self.trades.close_trade(bar.symbol)
            return True

        if action.action is TradeAction.ADD:
            trade = action.trade
            if trade is None:
                return True
            allowed, why = self.money.may_add(trade.unrealized(bar.close), trade.layer + 1)
            if not allowed:
                decision.rejected.append(f"no pyramid: {why}")
                return True
            plan = self._plan_for_side(bar, sa, trade.side, layer=trade.layer + 1)
            if plan is None or not plan.approved or plan.qty <= 0:
                decision.rejected.append("pyramid layer would not size")
                return True
            side = Side.BUY if trade.long_side else Side.SELL
            decision.orders.append(
                Order(
                    symbol=bar.symbol,
                    side=side,
                    qty=plan.qty,
                    type=OrderType.MARKET,
                    id=next_order_id(),
                    ts=bar.ts,
                    tag=f"pyramid|layer {trade.layer + 1}",
                    planned_risk=plan.risk_per_unit * plan.qty,
                )
            )
            self.trades.add_layer(bar.symbol, plan.qty, bar.close)
            decision.plan = plan
            return True

        # HOLD or NONE: nothing to do, and deliberately so.
        return True

    def _plan_for_side(
        self, bar: Bar, sa: SymbolAnalysis, side: int, layer: int = 1
    ) -> PositionPlan | None:
        """Size a position on a given side, using the chart for stop and target."""
        bias = Bias.BULLISH if side > 0 else Bias.BEARISH
        entry = bar.close
        stop, objective, source = self._stop_and_target(sa, bias, entry)
        if stop <= 0:
            return None
        portfolio = self._portfolio_state()
        leverage = self._sizing_leverage(bar, sa, side, stop, objective, source)
        plan = self.money.plan(
            symbol=bar.symbol,
            entry=entry,
            stop=stop,
            objective=objective,
            portfolio=portfolio,
            layer=layer,
            leverage=leverage,
        )
        plan.notes = f"stop from {source}"
        return plan

    # ------------------------------------------------------- confidence sizing
    def _evidence_for(
        self,
        bar: Bar,
        sa: SymbolAnalysis,
        side: int,
        stop: float,
        objective: float,
        source: str,
    ) -> Evidence:
        """Collect what is already known into a flat record for scoring.

        Everything here was computed earlier in the bar. Nothing is recalculated, so the
        score cannot disagree with the analysis the desk acted on.
        """
        verdict = self.verdicts.get(bar.symbol)
        signals = self._latest_signals.get(bar.symbol, [])
        weights = {a.name: a.weight for a in self.signal_agents}
        total_weight = sum(weights.get(s.agent, 1.0) for s in signals) or 1.0
        net = sum(
            s.weighted_score * weights.get(s.agent, 1.0) for s in signals
        ) / total_weight
        # Consensus is measured *in the trade's direction*: agents agreeing on the
        # opposite side is disagreement, not conviction.
        directional = max(0.0, net * side)

        permitted = sa.timeframes.permitted_direction()
        wanted = Bias.BULLISH if side > 0 else Bias.BEARISH
        aligned = permitted is wanted or permitted is Bias.NEUTRAL

        risk_per_unit = abs(bar.close - stop)
        reward_risk = (
            abs(objective - bar.close) / risk_per_unit if risk_per_unit > 0 else 0.0
        )
        # Funding is quoted for a long; a short earns it, so the sign flips.
        carry = self.funding_carry.get(bar.symbol, 0.0) * side

        return Evidence(
            checklist_conviction=verdict.conviction if verdict else 0.0,
            checklist_agreement=getattr(verdict, "agreement", 0.0) if verdict else 0.0,
            checklist_coverage=getattr(verdict, "coverage", 1.0) if verdict else 0.0,
            agent_consensus=directional,
            agent_count=len(signals),
            timeframe_aligned=aligned,
            pattern_reward_risk=reward_risk,
            has_pattern_stop="pattern" in source,
            regime_appetite=self._risk_appetite(),
            funding_against=carry,
            blockers=len(verdict.blockers) if verdict else 0,
        )

    def _leverage_for(
        self, bar: Bar, confidence: Confidence, stop: float
    ) -> LeverageDecision:
        """Convert confidence into a multiplier the protective stop can survive."""
        from quantdesk.alpha.confidence import DEFAULT_MAINT_MARGIN

        distance = abs(bar.close - stop) / bar.close if bar.close > 0 else 0.0
        return decide_leverage(
            confidence.score,
            stop_distance_pct=distance,
            maint_margin_pct=self.maint_margin.get(bar.symbol, DEFAULT_MAINT_MARGIN),
            user_cap=self.settings.max_leverage,
        )

    def _sizing_leverage(
        self,
        bar: Bar,
        sa: SymbolAnalysis,
        side: int,
        stop: float,
        objective: float,
        source: str,
        decision: BarDecision | None = None,
    ) -> float:
        """Leverage for this trade, and record how it was reached.

        Returns the flat configured leverage when confidence sizing is off, so the
        existing behaviour is untouched unless it is asked for explicitly.
        """
        if not self.settings.confidence_leverage:
            return self.settings.leverage
        evidence = self._evidence_for(bar, sa, side, stop, objective, source)
        confidence = score_confidence(evidence)
        leverage = self._leverage_for(bar, confidence, stop)
        self.confidences[bar.symbol] = confidence
        self.leverages[bar.symbol] = leverage
        if decision is not None:
            decision.confidence = confidence
            decision.leverage = leverage
        return leverage.leverage
    def _portfolio_state(self) -> PortfolioState:
        return PortfolioState(
            equity=self.equity,
            committed={
                sym: abs(p.market_value(p.last_price)) / max(1.0, self.settings.leverage)
                for sym, p in self.positions.items()
                if not p.is_flat
            },
            groups={sym: _group_of(sym) for sym in self.analysis},
            peak_equity=self.peak_equity,
            start_equity=self.start_equity,
        )

    # ----------------------------------------------------------------- sizing
    def _size(
        self,
        bar: Bar,
        sa: SymbolAnalysis,
        target: TargetPosition,
        decision: BarDecision | None = None,
    ) -> PositionPlan | None:
        """Convert a target weight into a size with a real stop and objective.

        The stop comes from the chart wherever possible - a pattern's own protective
        level, otherwise the nearest structural level, otherwise an ATR multiple.
        Murphy's whole objection to arbitrary stops is that they ignore where the
        market has actually shown it will turn.
        """
        long_side = target.weight > 0
        side_bias = Bias.BULLISH if long_side else Bias.BEARISH
        entry = bar.close

        stop, objective, source = self._stop_and_target(sa, side_bias, entry)
        if stop <= 0:
            return None

        portfolio = PortfolioState(
            equity=self.equity,
            committed={
                sym: abs(p.market_value(p.last_price)) / max(1.0, self.settings.leverage)
                for sym, p in self.positions.items()
                if not p.is_flat
            },
            groups={sym: _group_of(sym) for sym in self.analysis},
            peak_equity=self.peak_equity,
            start_equity=self.start_equity,
        )
        leverage = self._sizing_leverage(
            bar, sa, 1 if long_side else -1, stop, objective, source, decision
        )
        plan = self.money.plan(
            symbol=bar.symbol,
            entry=entry,
            stop=stop,
            objective=objective,
            portfolio=portfolio,
            conviction=min(1.0, abs(target.weight) / max(1e-9, self.settings.max_position_weight)),
            leverage=leverage,
        )
        plan.notes = f"stop from {source}"
        return plan

    def _stop_and_target(
        self, sa: SymbolAnalysis, side: Bias, entry: float
    ) -> tuple[float, float, str]:
        """Pick the protective stop and profit objective, best source first."""
        atr = sa.atr.value or entry * 0.01
        long_side = side is Bias.BULLISH

        def usable(stop: float, objective: float) -> bool:
            """Stop behind the entry, objective still ahead of it.

            The second half matters more than it looks: a pattern whose target price
            has already been reached leaves an objective *behind* the current price,
            which would size a trade with negative reward. Checking it here is what
            keeps stale patterns from producing nonsense plans.
            """
            if stop <= 0 or objective <= 0:
                return False
            if long_side:
                return stop < entry < objective
            return objective < entry < stop

        # 1. A completed pattern carries its own stop and measured objective, which is
        # strictly better information than anything derived generically.
        for pattern in sa.patterns.actionable(min_reward_risk=self.settings.min_reward_risk):
            if pattern.bias is side and usable(pattern.stop, pattern.objective):
                return pattern.stop, pattern.objective, f"{pattern.kind.label} pattern"

        # 2. The nearest structural level beyond which the premise is wrong.
        level = (
            sa.levels.underlying_support(entry)
            if long_side
            else sa.levels.overhead_resistance(entry)
        )
        if level is not None:
            # Sit clear of the level rather than exactly on it, and keep the stop off
            # round numbers where everyone else's orders cluster.
            raw = level.price - 0.25 * atr if long_side else level.price + 0.25 * atr
            from quantdesk.analysis.levels import offset_stop_from_round

            stop = offset_stop_from_round(raw, long_side)
            # A stop inside the noise is not protection, it is a guaranteed loss with
            # extra steps. If the structural level sits closer than the market's own
            # bar-to-bar range, push the stop out to the volatility floor and let
            # position sizing shrink to keep the risk budget unchanged. Murphy makes
            # the same point when he warns against resting stops too close to obvious
            # levels: everyone's orders are there, and price routinely pokes through.
            floor = self.settings.min_stop_atr * atr
            if abs(entry - stop) < floor:
                stop = entry - floor if long_side else entry + floor
            risk = abs(entry - stop)
            if risk > 0:
                # Aim at the next opposing level if there is one and it is far enough
                # to justify the trade; otherwise project the yardstick from the stop
                # rather than abandoning an otherwise sound structural stop.
                target_level = (
                    sa.levels.overhead_resistance(entry)
                    if long_side
                    else sa.levels.underlying_support(entry)
                )
                projected = (
                    entry + self.settings.min_reward_risk * risk
                    if long_side
                    else entry - self.settings.min_reward_risk * risk
                )
                objective = projected
                if target_level is not None:
                    reach = abs(target_level.price - entry)
                    if reach >= self.settings.min_reward_risk * risk:
                        # A real level beyond the yardstick is a better target than an
                        # arbitrary multiple, because that is where price will stall.
                        objective = target_level.price
                if usable(stop, objective):
                    return stop, objective, f"{level.role.value} level"

        # 3. Fall back to a volatility stop, with the objective set at the yardstick.
        risk = max(2.0, self.settings.min_stop_atr) * atr
        stop = entry - risk if long_side else entry + risk
        objective = (
            entry + self.settings.min_reward_risk * risk
            if long_side
            else entry - self.settings.min_reward_risk * risk
        )
        return stop, objective, "2 ATR volatility stop"

    def _orders_for(
        self, bar: Bar, plan: PositionPlan, target: TargetPosition
    ) -> list[Order]:
        """Turn a sized plan into the order that moves us from here to there."""
        current = self.positions.get(bar.symbol)
        current_qty = current.qty if current else 0.0
        desired = plan.qty if target.weight > 0 else -plan.qty
        delta = desired - current_qty

        # Don't churn: a trivial adjustment costs spread and gains nothing.
        if abs(delta) * bar.close < 0.001 * max(1.0, self.equity):
            return []

        side = Side.BUY if delta > 0 else Side.SELL
        # Risk travels with the order. Scaled to *this order's* quantity rather than
        # copied from the plan, because the order is the delta from what is already
        # held and only that delta puts new risk on.
        return [
            Order(
                symbol=bar.symbol,
                side=side,
                qty=abs(delta),
                type=OrderType.MARKET,
                id=next_order_id(),
                ts=bar.ts,
                tag=f"{target.reason or 'blend'}|{plan.binding_limit}",
                planned_risk=plan.risk_per_unit * abs(delta),
            )
        ]

    # ------------------------------------------------------------------ state
    def _risk_appetite(self) -> float:
        if not self._regime_views:
            return 1.0
        return min(v.risk_appetite for v in self._regime_views)

    def _roll_day(self, ts: datetime) -> None:
        """Reset the day's starting equity when the date changes.

        Without this, ``day_start_equity`` stays pinned to inception and the daily
        loss limit becomes a *lifetime* loss limit: one 3% drawdown puts the desk into
        reduce-only permanently and it never trades again. The symptom is thousands of
        identical risk vetoes, which is exactly what a year-long backtest showed.
        """
        day = ts.date()
        if self._current_day is None:
            self._current_day = day
            return
        if day != self._current_day:
            self._current_day = day
            self.start_equity = self.equity
            # A new day clears reduce-only. A drawdown limit that never lifts is a
            # kill switch wearing a daily label.
            if self.risk.state.value == "reduce_only":
                self.risk.resume()

    def _snapshot(self, ts: datetime) -> AccountSnapshot:
        self._roll_day(ts)
        gross = sum(abs(p.market_value(p.last_price)) for p in self.positions.values())
        net = sum(p.market_value(p.last_price) for p in self.positions.values())
        unrealized = sum(p.unrealized_pnl(p.last_price) for p in self.positions.values())
        realized = sum(p.realized_pnl for p in self.positions.values())
        return AccountSnapshot(
            ts=ts,
            equity=self.equity,
            cash=self.equity - gross,
            gross_exposure=gross,
            net_exposure=net,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            day_start_equity=self.start_equity,
            peak_equity=self.peak_equity,
        )

    def sync_account(self, equity: float, positions: dict[str, Position]) -> None:
        """Adopt broker truth. Live trading must never trust its own bookkeeping."""
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        self.positions = positions
        self.risk.evaluate_account(self._snapshot(datetime.now().astimezone()))

    def latest_signals(self, symbol: str) -> list[Signal]:
        return list(self._latest_signals.get(symbol, []))

    @property
    def risk_appetite(self) -> float:
        return self._risk_appetite()

    @property
    def bars_seen(self) -> int:
        return self._bars_seen


def _base_timeframe(raw: str) -> Timeframe:
    """Map the configured Alpaca timeframe string onto our Timeframe enum."""
    return {
        "1Min": Timeframe.M1,
        "5Min": Timeframe.M5,
        "15Min": Timeframe.M15,
        "1Hour": Timeframe.H1,
        "1Day": Timeframe.D1,
    }.get(raw, Timeframe.M5)


def _group_of(symbol: str) -> str:
    """Market group for Murphy's group-margin ceiling.

    "Markets within groups tend to move together." Every crypto pair quoted in USD
    is one group for this purpose, because in a risk-off move they all fall together
    and treating them as independent would defeat the diversification the limit
    exists to enforce.
    """
    if "/" in symbol:
        base, _, quote = symbol.partition("/")
        if quote in ("USD", "USDT", "USDC"):
            return "crypto_usd"
    return symbol
