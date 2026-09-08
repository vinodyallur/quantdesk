"""Money management, Murphy chapter 16.

His summary of the division of labour: "price forecasting tells the trader what to
do (buy or sell), timing helps decide when to do it, and money management determines
how much to commit to the trade." This module is the third of those. It answers the
last four questions on his chapter 19 checklist - how many units, how much am I
prepared to risk, what is my objective, where does the stop go - and it is the
component that decides whether a signal becomes a trade at all.

His allocation limits, quoted and encoded verbatim in :class:`MoneyRules`:

* "Total invested funds should be limited to 50% of total capital."
* "Total commitment in any one market should be limited to 10-15% of total equity."
* "The total amount risked in any one market should be limited to 5% of total
  equity. This 5% refers to how much the trader is willing to lose if the trade
  doesn't work."
* "Total margin in any market group should be limited to 20-25% of total equity."
* "A commonly used yardstick is a 3 to 1 reward-to-risk ratio. The profit potential
  must be at least three times the possible loss if a trade is to be considered."

Note the distinction his 10% and 5% rules draw, because conflating them is the
classic sizing error: 10-15% caps how much *capital the position uses*, while 5%
caps how much is *lost if the stop is hit*. They bind at different times, so both
are checked and the tighter one wins.

Pyramiding follows his four rules exactly: each layer smaller than the last, add
only to winners, never to losers, and move stops to breakeven as you go.

**On streaks, he declines to prescribe.** He poses the question - what to do after
doubling your money, or halving it - and answers "the answers to these two questions
aren't as simple or obvious as they seem". He does give one firm warning: doubling
position size after a winning streak means that in the inevitable losing period
"you'll wind up giving it all back". So :attr:`MoneyRules.streak_growth_cap` limits
how far sizing may expand on a hot run, and no claim is made that he endorses
increasing size into a drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SizingRefusal(str, Enum):
    """Why a trade was refused. Refusals are outcomes, not errors."""

    NONE = "none"
    NO_STOP = "no protective stop, so risk is unbounded"
    STOP_WRONG_SIDE = "stop is on the wrong side of entry"
    REWARD_TOO_SMALL = "reward-to-risk below the 3:1 yardstick"
    PORTFOLIO_FULL = "50% invested-capital ceiling reached"
    GROUP_FULL = "market-group margin ceiling reached"
    SIZE_ZERO = "limits leave no tradable size"
    RISK_TOO_WIDE = "stop so wide that even one unit breaches the 5% risk cap"


@dataclass(slots=True)
class MoneyRules:
    """Murphy's chapter 16 limits. Defaults are the conservative end of his ranges."""

    #: "Total invested funds should be limited to 50% of total capital."
    max_invested_pct: float = 0.50
    #: "Total commitment in any one market ... 10-15% of total equity."
    max_market_commitment_pct: float = 0.10
    #: "The total amount risked in any one market ... 5% of total equity."
    max_market_risk_pct: float = 0.05
    #: *Target* risk per trade, as a fraction of equity. Murphy's 5% is a ceiling, and
    #: sizing only against a ceiling lets actual risk vary enormously from trade to
    #: trade - whichever limit happens to bind first decides it. That destroys the
    #: arithmetic that makes an edge compound: a +3.4R win on a tiny position earns less
    #: than a -1R loss on a large one, so a strategy with a genuinely positive
    #: expectancy in R can still lose money. Equalising risk per trade is what turns
    #: R-multiples into currency. Set to 0 to size purely against the caps.
    risk_target_pct: float = 0.005
    #: "Total margin in any market group ... 20-25% of total equity."
    max_group_commitment_pct: float = 0.20
    #: "The profit potential must be at least three times the possible loss."
    min_reward_risk: float = 3.0
    #: Cap on how far equity growth may expand unit size, per his warning about
    #: doubling up after a winning streak.
    streak_growth_cap: float = 1.5
    #: Each pyramid layer is this fraction of the one before it: "Each successive
    #: layer should be smaller than before."
    pyramid_decay: float = 0.5
    max_pyramid_layers: int = 3

    def __post_init__(self) -> None:
        if not 0 < self.max_market_risk_pct <= self.max_market_commitment_pct:
            # Risking more than you commit is incoherent; catch it at construction
            # rather than producing silently wrong sizes.
            raise ValueError(
                "max_market_risk_pct must be positive and no greater than "
                "max_market_commitment_pct"
            )


@dataclass(slots=True)
class PositionPlan:
    """The full answer to "how much, where, and what do I risk?"."""

    symbol: str
    approved: bool
    qty: float = 0.0
    entry: float = 0.0
    stop: float = 0.0
    objective: float = 0.0
    risk_amount: float = 0.0
    """Currency lost if the stop is hit."""
    risk_target: float = 0.0
    """Currency this trade *intended* to risk, before the ceilings and conviction.

    Kept next to :attr:`risk_amount` because the gap between the two is the whole
    story on an unleveraged account: the commitment ceilings routinely allow far less
    size than the risk target asks for, so actual risk per trade ends up set by stop
    distance rather than by the budget. That makes risk per trade vary by orders of
    magnitude, and a plain average of R across such trades is dominated by whichever
    trades happened to risk least. See :attr:`risk_delivered` .
    """
    risk_pct: float = 0.0
    notional: float = 0.0
    margin: float = 0.0
    """Capital actually tied up. Equals notional when unleveraged."""
    commitment_pct: float = 0.0
    reward_risk: float = 0.0
    layer: int = 1
    refusal: SizingRefusal = SizingRefusal.NONE
    binding_limit: str = ""
    """Which of Murphy's limits actually determined the size."""
    notes: str = ""

    @property
    def long_side(self) -> bool:
        return self.stop < self.entry

    @property
    def risk_per_unit(self) -> float:
        """Distance to the stop. The per-unit loss if the premise is wrong."""
        return abs(self.entry - self.stop)

    @property
    def risk_delivered(self) -> float:
        """Actual risk as a fraction of the intended risk.

        1.0 means the trade risks exactly what it set out to. Well below 1.0 means a
        ceiling or conviction cut the size down, and that this trade will carry less
        weight than others in any R average.
        """
        if self.risk_target <= 1e-12:
            return 1.0
        return self.risk_amount / self.risk_target

    def describe(self) -> str:
        if not self.approved:
            return f"{self.symbol}: REFUSED - {self.refusal.value}"
        side = "long" if self.long_side else "short"
        return (
            f"{self.symbol}: {side} {self.qty:.6g} @ {self.entry:.6g} "
            f"stop {self.stop:.6g} target {self.objective:.6g} "
            f"risk {self.risk_amount:,.0f} ({self.risk_pct:.2%}) "
            f"of {self.risk_target:,.0f} intended ({self.risk_delivered:.0%}) "
            f"R:R {self.reward_risk:.1f} [{self.binding_limit}]"
        )


@dataclass(slots=True)
class PortfolioState:
    """What is already committed, so the portfolio-level ceilings can be applied."""

    equity: float
    #: Absolute notional currently committed, by symbol.
    committed: dict[str, float] = field(default_factory=dict)
    #: Symbol to market-group name, for the group ceiling. Murphy's point is that
    #: "markets within groups tend to move together", so gold and silver share one.
    groups: dict[str, str] = field(default_factory=dict)
    peak_equity: float = 0.0
    start_equity: float = 0.0

    @property
    def total_committed(self) -> float:
        return sum(abs(v) for v in self.committed.values())

    @property
    def invested_pct(self) -> float:
        if self.equity <= 0:
            return 0.0
        return self.total_committed / self.equity

    def group_committed(self, group: str) -> float:
        return sum(
            abs(v)
            for sym, v in self.committed.items()
            if self.groups.get(sym, sym) == group
        )

    def group_of(self, symbol: str) -> str:
        return self.groups.get(symbol, symbol)


class MoneyManager:
    """Turns a directional view plus a stop into a concrete, limit-respecting size."""

    def __init__(self, rules: MoneyRules | None = None) -> None:
        self.rules = rules or MoneyRules()

    # ------------------------------------------------------------------ sizing
    def plan(
        self,
        symbol: str,
        entry: float,
        stop: float,
        objective: float,
        portfolio: PortfolioState,
        layer: int = 1,
        conviction: float = 1.0,
        min_qty: float = 0.0,
        qty_step: float = 0.0,
        leverage: float = 1.0,
    ) -> PositionPlan:
        """Size a trade against every one of Murphy's limits.

        ``conviction`` scales *down* from the maximum permitted size; it can never
        scale up past a limit. Weight of evidence decides how much of the allowance
        to use, not whether the allowance applies.

        ``leverage`` matters more than it looks. Murphy is writing about futures, and
        his 10-15% figure is explicitly margin: "only $10,000 to $15,000 would be
        available for *margin deposit* in any one market". So the commitment ceilings
        apply to capital tied up, not to notional exposure. On an unleveraged spot
        account the two are identical, and the consequence is that the commitment
        ceiling binds before the 5% risk cap for any stop closer than 50% away -
        which is conservative, and correct for cash trading. Supplying real leverage
        makes the 5% risk cap the operative constraint, as he intends it to be.
        """
        r = self.rules
        plan = PositionPlan(
            symbol=symbol, approved=False, entry=entry, stop=stop, objective=objective,
            layer=layer,
        )

        if entry <= 0 or stop <= 0:
            plan.refusal = SizingRefusal.NO_STOP
            return plan
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            plan.refusal = SizingRefusal.NO_STOP
            return plan

        long_side = stop < entry
        if objective > 0:
            if long_side and objective <= entry:
                plan.refusal = SizingRefusal.STOP_WRONG_SIDE
                return plan
            if not long_side and objective >= entry:
                plan.refusal = SizingRefusal.STOP_WRONG_SIDE
                return plan
            plan.reward_risk = abs(objective - entry) / risk_per_unit
            if plan.reward_risk < r.min_reward_risk:
                # His 3:1 yardstick is a gate, not a preference.
                plan.refusal = SizingRefusal.REWARD_TOO_SMALL
                return plan

        equity = portfolio.equity
        if equity <= 0:
            plan.refusal = SizingRefusal.SIZE_ZERO
            return plan

        # The plan sizes the *target* position, not an increment on top of what is
        # already held. So this symbol's own existing commitment is excluded from the
        # portfolio and group ceilings: we are replacing that exposure, not stacking
        # on it. Treating it as an increment instead makes a position that has reached
        # its cap look like a refusal on every subsequent bar, when the correct answer
        # is simply "hold what you have".
        already = abs(portfolio.committed.get(symbol, 0.0))
        others = max(0.0, portfolio.total_committed - already)

        room_invested = r.max_invested_pct * equity - others
        if room_invested <= 0:
            plan.refusal = SizingRefusal.PORTFOLIO_FULL
            return plan

        # Group ceiling: correlated markets share an allowance.
        group = portfolio.group_of(symbol)
        group_others = max(0.0, portfolio.group_committed(group) - already)
        room_group = r.max_group_commitment_pct * equity - group_others
        if room_group <= 0:
            plan.refusal = SizingRefusal.GROUP_FULL
            return plan

        # Per-market ceilings. The 5% risk cap and the 10% commitment cap bind at
        # different times - a tight stop lets you hold a big position within the 5%,
        # a wide stop makes the 5% bite long before the 10% does - so both are
        # converted to a quantity and the smaller wins.
        room_market = r.max_market_commitment_pct * equity

        # Target a consistent risk per trade, never exceeding Murphy's ceiling. This is
        # what makes every trade carry the same weight, so the measured expectancy in R
        # actually translates into money.
        target_pct = r.risk_target_pct if r.risk_target_pct > 0 else r.max_market_risk_pct
        risk_budget = min(target_pct, r.max_market_risk_pct) * equity
        qty_by_risk = risk_budget / risk_per_unit
        # Commitment ceilings govern capital tied up, so leverage converts an
        # allowance of margin into an allowance of quantity. Risk is unaffected by
        # leverage: the stop distance decides the loss either way.
        lev = max(1e-9, leverage)
        margin_per_unit = entry / lev
        qty_by_market = room_market / margin_per_unit
        qty_by_invested = room_invested / margin_per_unit
        qty_by_group = room_group / margin_per_unit

        risk_label = (
            f"{target_pct:.2%} risk target"
            if r.risk_target_pct > 0
            else f"{r.max_market_risk_pct:.0%} risk cap"
        )
        limits = {
            risk_label: qty_by_risk,
            "10% market commitment": qty_by_market,
            "50% invested capital": qty_by_invested,
            "20% group margin": qty_by_group,
        }
        binding, qty = min(limits.items(), key=lambda kv: kv[1])
        qty_at_ceiling = qty

        # Pyramiding: each successive layer smaller than the one before.
        if layer > 1:
            if layer > r.max_pyramid_layers:
                plan.refusal = SizingRefusal.SIZE_ZERO
                plan.notes = f"beyond {r.max_pyramid_layers} pyramid layers"
                return plan
            qty *= r.pyramid_decay ** (layer - 1)

        qty *= max(0.0, min(1.0, conviction))
        qty = self._round_qty(qty, min_qty, qty_step)

        # Conviction and pyramid decay are applied *after* the ceilings, so either can
        # cut the size below whichever ceiling was nominally binding. Reporting the
        # ceiling in that case is misleading: it names a constraint that was not the
        # one that actually decided the size.
        # Kept low-cardinality on purpose: this string is grouped and counted, so
        # interpolating the actual percentage would make every trade its own category.
        # The size of the reduction is available numerically as ``risk_delivered``.
        if qty < qty_at_ceiling * (1.0 - 1e-9):
            reducers = []
            if layer > 1:
                reducers.append("pyramid decay")
            if conviction < 1.0 - 1e-9:
                reducers.append("conviction")
            if reducers:
                binding = f"{binding}, cut by {' and '.join(reducers)}"

        if qty <= 0:
            # A stop so wide that even the minimum tradable size breaches the 5% cap
            # is a distinct failure from simply running out of portfolio room.
            if min_qty > 0 and min_qty * risk_per_unit > risk_budget:
                plan.refusal = SizingRefusal.RISK_TOO_WIDE
            else:
                plan.refusal = SizingRefusal.SIZE_ZERO
            plan.binding_limit = binding
            plan.risk_target = risk_budget
            return plan

        plan.qty = qty
        plan.notional = qty * entry
        plan.margin = plan.notional / lev
        plan.risk_amount = qty * risk_per_unit
        plan.risk_target = risk_budget
        plan.risk_pct = plan.risk_amount / equity
        plan.commitment_pct = plan.margin / equity
        plan.binding_limit = binding
        plan.approved = True

        # Sizing rounding can only ever reduce risk below the cap, never above it,
        # but assert it rather than assume it - this is the number that matters most.
        if plan.risk_pct > r.max_market_risk_pct + 1e-9:
            plan.approved = False
            plan.refusal = SizingRefusal.RISK_TOO_WIDE
        return plan

    def _round_qty(self, qty: float, min_qty: float, step: float) -> float:
        """Round *down* to a tradable size, so limits are never rounded through."""
        if step > 0:
            qty = (qty // step) * step
        if min_qty > 0 and qty < min_qty:
            return 0.0
        return max(0.0, qty)

    # -------------------------------------------------------------- pyramiding
    def may_add(self, unrealized_pnl: float, layer: int) -> tuple[bool, str]:
        """Murphy's pyramiding rules: "Add only to winning positions. Never add to a
        losing position."
        """
        if layer > self.rules.max_pyramid_layers:
            return False, f"already at {layer - 1} layers"
        if unrealized_pnl <= 0:
            return False, "never add to a losing position"
        return True, "adding to a winner"

    def breakeven_stop(self, entry: float, long_side: bool, buffer: float = 0.0) -> float:
        """Stop at entry once a pyramid layer is added: "Adjust protective stops to
        the breakeven point."
        """
        return entry + buffer if long_side else entry - buffer

    # ------------------------------------------------------------------ streaks
    def equity_scalar(self, portfolio: PortfolioState) -> float:
        """How much equity growth may expand unit size.

        Sizing is a percentage of equity, so it already grows and shrinks with the
        account. This caps the growth side, because Murphy's one firm answer on
        streaks is that doubling up after a winning run means giving it all back
        when the losing run arrives. Drawdowns are left uncapped - equity-proportional
        sizing reduces exposure automatically, and he does not prescribe overriding
        that.
        """
        base = portfolio.start_equity or portfolio.equity
        if base <= 0:
            return 1.0
        ratio = max(0.0, portfolio.equity / base)
        if ratio > 1.0:
            return min(ratio, self.rules.streak_growth_cap)
        return ratio

    def risk_budget(self, equity: float) -> float:
        """Currency at risk allowed on one market."""
        return self.rules.max_market_risk_pct * equity

    def summary(self, portfolio: PortfolioState) -> list[str]:
        r = self.rules
        return [
            f"equity {portfolio.equity:,.0f} | invested {portfolio.invested_pct:.1%} "
            f"of {r.max_invested_pct:.0%} ceiling",
            f"risk budget per market {self.risk_budget(portfolio.equity):,.0f} "
            f"({r.max_market_risk_pct:.0%})",
            f"reward:risk gate {r.min_reward_risk:.1f}:1",
        ]
