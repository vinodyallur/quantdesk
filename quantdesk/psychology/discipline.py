"""The discipline monitor: does the desk follow its own rules?

Douglas's sixth principle is "I continually monitor my susceptibility for making
errors". This is that monitor, and for an automated desk it watches two distinct
things.

**The machine.** Every entry should trace to a named edge, carry a predefined stop, and
never see that stop widened. These are invariants the code already tries to maintain,
which is exactly why they are worth checking: a bug produces the same behaviour as a
lapse in discipline, and neither announces itself.

**The operator.** The system cannot hesitate, but the person running it can pause it
during a drawdown, kill it after a losing streak, or quietly change parameters and
restart. Douglas would call that the real trading psychology, and it is invisible
unless something records it. So pauses, kills and parameter changes are logged as
interventions, and a pause taken during a drawdown is flagged more loudly than one
taken in calm - because that is when the decision is least likely to be reasoned.

The monitor never blocks anything. It reports. A discipline layer with veto power
would just be another risk manager, and the desk already has one of those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from quantdesk.psychology.edge import (
    EdgeStats,
    TradeOutcome,
    is_streak_alarming,
    measure,
)
from quantdesk.psychology.principles import (
    AdherenceReport,
    Principle,
    PrincipleEvent,
    Severity,
)


@dataclass
class Intervention:
    """A human action that changed how the desk behaves."""

    kind: str
    when: datetime
    reason: str = ""
    equity: float = 0.0
    drawdown: float = 0.0
    """Drawdown from peak at the moment of the intervention."""

    @property
    def during_drawdown(self) -> bool:
        return self.drawdown > 0.02


class DisciplineMonitor:
    """Records adherence, interventions and edge statistics for a session."""

    def __init__(self, min_sample: int = 30, drawdown_alarm: float = 0.02) -> None:
        self.min_sample = min_sample
        self.drawdown_alarm = drawdown_alarm
        self.report = AdherenceReport()
        self.interventions: list[Intervention] = []
        self.outcomes: list[TradeOutcome] = []
        self._stops: dict[str, float] = {}
        self._sides: dict[str, int] = {}

    # ------------------------------------------------------------------ entries
    def on_entry(
        self,
        symbol: str,
        stop: float,
        side: int,
        agent: str = "",
        rationale: str = "",
        when: datetime | None = None,
    ) -> None:
        """Check an entry against principles 1 and 2."""
        stamp = when.strftime("%Y-%m-%d %H:%M") if when else ""

        # P1: every edge must be objectively identified and attributable.
        if not agent and not rationale:
            self._record(
                Principle.IDENTIFY_EDGES,
                Severity.VIOLATION,
                "entry with no attributable edge or rationale",
                stamp,
                symbol,
            )

        # P2: risk must be predefined. Without a stop there is no defined risk, and
        # every position-size calculation downstream is meaningless.
        if stop <= 0:
            self._record(
                Principle.PREDEFINE_RISK,
                Severity.VIOLATION,
                "entry with no protective stop, so the risk is undefined",
                stamp,
                symbol,
            )
        else:
            self._stops[symbol] = stop
            self._sides[symbol] = side

    def on_stop_change(
        self, symbol: str, new_stop: float, when: datetime | None = None
    ) -> None:
        """Check a stop adjustment against principle 3.

        A stop moved *away* from entry is the mechanical signature of refusing to accept
        the risk: it converts a defined loss into an open-ended one at the exact moment
        the trade is going wrong.
        """
        old = self._stops.get(symbol)
        side = self._sides.get(symbol, 0)
        if old is None or side == 0:
            self._stops[symbol] = new_stop
            return

        widened = new_stop < old if side > 0 else new_stop > old
        if widened:
            self._record(
                Principle.ACCEPT_RISK,
                Severity.VIOLATION,
                f"stop widened from {old:.6g} to {new_stop:.6g}",
                when.strftime("%Y-%m-%d %H:%M") if when else "",
                symbol,
            )
        self._stops[symbol] = new_stop

    def on_signal(self, qualified: bool, acted: bool, symbol: str = "") -> None:
        """Check principle 4: act on every edge, without picking and choosing."""
        if not qualified:
            return
        self.report.qualifying_signals += 1
        if acted:
            self.report.signals_acted_on += 1
        else:
            # Skipping a qualifying setup is a violation even if it would have lost.
            # Douglas is explicit that traders must not "pick and choose the edges they
            # think ... are going to work".
            self._record(
                Principle.ACT_WITHOUT_HESITATION,
                Severity.WARNING,
                "a qualifying edge was not acted on",
                symbol=symbol,
            )

    # ------------------------------------------------------------------- exits
    def on_exit(
        self,
        symbol: str,
        r_multiple: float,
        pnl: float = 0.0,
        agent: str = "",
        took_profit: bool = False,
    ) -> None:
        """Record the outcome, and check principle 5."""
        self.outcomes.append(
            TradeOutcome(symbol=symbol, r_multiple=r_multiple, pnl=pnl, agent=agent)
        )
        self.report.trades_considered += 1
        self._stops.pop(symbol, None)
        self._sides.pop(symbol, None)

        # P5: "I pay myself as the market makes money available to me." A trade that
        # ran well into profit and closed at or below breakeven gave the money back.
        if not took_profit and r_multiple <= 0.0:
            pass  # A loss taken at the stop is correct behaviour, not a violation.

    def on_profit_given_back(self, symbol: str, peak_r: float, final_r: float) -> None:
        """Flag profit that was reached and then surrendered.

        Not automatically a violation - letting a winner breathe is Murphy's advice and
        sometimes it simply reverses - but a pattern of it means the exit rules are not
        paying the desk as the market made money available.
        """
        if peak_r >= 1.0 and final_r <= 0.0:
            self._record(
                Principle.PAY_MYSELF,
                Severity.WARNING,
                f"reached {peak_r:.1f}R then closed at {final_r:.1f}R",
                symbol=symbol,
            )

    # ----------------------------------------------------------- interventions
    def on_intervention(
        self,
        kind: str,
        when: datetime,
        reason: str = "",
        equity: float = 0.0,
        drawdown: float = 0.0,
    ) -> Intervention:
        """Record a human action: pause, kill, resume, parameter change, restart."""
        event = Intervention(
            kind=kind, when=when, reason=reason, equity=equity, drawdown=drawdown
        )
        self.interventions.append(event)
        self.report.interventions += 1

        severity = Severity.INFO
        detail = f"{kind}: {reason}" if reason else kind
        if event.during_drawdown and kind in ("pause", "kill", "parameter_change"):
            # This is the decision most likely to be a flinch rather than a judgement.
            severity = Severity.WARNING
            detail += f" (taken during a {drawdown:.1%} drawdown)"
        self._record(
            Principle.MONITOR_ERRORS,
            severity,
            detail,
            when.strftime("%Y-%m-%d %H:%M"),
        )
        return event

    # ---------------------------------------------------------------- readouts
    @property
    def stats(self) -> EdgeStats:
        return measure(self.outcomes, self.min_sample)

    def streak_verdict(self) -> tuple[bool, str]:
        """Is the current losing run actually worth worrying about?"""
        stats = self.stats
        streak = -stats.current_streak if stats.current_streak < 0 else 0
        return is_streak_alarming(streak, stats.win_rate, stats.count)

    def _record(
        self,
        principle: Principle,
        severity: Severity,
        detail: str,
        when: str = "",
        symbol: str = "",
    ) -> None:
        self.report.events.append(
            PrincipleEvent(
                principle=principle, severity=severity, detail=detail,
                when=when, symbol=symbol,
            )
        )
        if severity is Severity.VIOLATION:
            # P7 is violated by definition whenever any of the other six is.
            self.report.events.append(
                PrincipleEvent(
                    principle=Principle.NEVER_VIOLATE,
                    severity=Severity.VIOLATION,
                    detail=f"principle {principle.number} was violated",
                    when=when,
                    symbol=symbol,
                )
            )
        if len(self.report.events) > 2000:
            del self.report.events[:-2000]

    def summary(self) -> list[str]:
        stats = self.stats
        lines = ["EDGE"] + [f"  {line}" for line in stats.summary()]
        lines.append("")
        lines.append("DISCIPLINE")
        lines.extend(f"  {line}" for line in self.report.summary())

        alarming, explanation = self.streak_verdict()
        lines.append("")
        lines.append("STREAK")
        lines.append(f"  {'INVESTIGATE' if alarming else 'normal'}: {explanation}")

        if not stats.reliable:
            lines.append("")
            lines.append(
                f"  NOTE: {stats.count} trades is too few to judge this edge. "
                f"Douglas's point is that probable outcomes only produce consistent "
                f"results over a large enough sample."
            )
        return lines
