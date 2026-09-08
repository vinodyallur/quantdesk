"""Mark Douglas's framework from *Trading in the Zone*, as checkable invariants.

The book is about human psychology, so the first question is what it can possibly
mean for a machine. The answer turns out to be more than it sounds, for two reasons.

**An automated desk is the ideal implementation of Douglas.** Almost every failure he
describes is an ego defending itself: hesitating on a valid signal, moving a stop to
avoid being wrong, taking profit early to feel like a winner, doubling down to get
even. A program does none of that by default. What his principles become, then, is a
specification the desk can be *audited against* - and the audit is worth running,
because a bug produces the same behaviour as a flinch.

**The remaining psychology is the operator's.** The human can still override the
system, pause it during a drawdown, or restart it with different parameters after a
losing streak. Douglas's principle 6 - "I continually monitor my susceptibility for
making errors" - therefore points at the person, and this module makes their
interventions visible rather than invisible.

His five fundamental truths and seven principles of consistency are quoted exactly.
The mapping from each principle to a mechanical check is *our* interpretation, and is
labelled as such on every check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

#: "A probabilistic mind-set pertaining to trading consists of five fundamental
#: truths." Quoted verbatim.
FUNDAMENTAL_TRUTHS: tuple[str, ...] = (
    "Anything can happen.",
    "You don't need to know what is going to happen next in order to make money.",
    "There is a random distribution between wins and losses for any given set of "
    "variables that define an edge.",
    "An edge is nothing more than an indication of a higher probability of one thing "
    "happening over another.",
    "Every moment in the market is unique.",
)


class Principle(str, Enum):
    """The seven principles of consistency: "I am a consistent winner because..."."""

    IDENTIFY_EDGES = "I objectively identify my edges"
    PREDEFINE_RISK = "I predefine the risk of every trade"
    ACCEPT_RISK = "I completely accept the risk or I am willing to let go of the trade"
    ACT_WITHOUT_HESITATION = "I act on my edges without reservation or hesitation"
    PAY_MYSELF = "I pay myself as the market makes money available to me"
    MONITOR_ERRORS = "I continually monitor my susceptibility for making errors"
    NEVER_VIOLATE = (
        "I understand the absolute necessity of these principles of consistent "
        "success and, therefore, I never violate them"
    )

    @property
    def number(self) -> int:
        return list(Principle).index(self) + 1

    @property
    def mechanical_check(self) -> str:
        """How this desk tests the principle. Our interpretation, not Douglas's words."""
        return _CHECKS[self]


_CHECKS: dict[Principle, str] = {
    Principle.IDENTIFY_EDGES: (
        "Every entry must trace to a named agent signal with a recorded rationale. A "
        "trade with no attributable edge is a bug or an override, not a trade."
    ),
    Principle.PREDEFINE_RISK: (
        "No position may be sized without a protective stop. The money manager "
        "refuses to size against a missing stop, so risk is defined before entry by "
        "construction rather than by intention."
    ),
    Principle.ACCEPT_RISK: (
        "Stops are honoured on the bar they are touched and may only ever tighten. A "
        "widened stop is the mechanical signature of not having accepted the risk."
    ),
    Principle.ACT_WITHOUT_HESITATION: (
        "Every signal meeting the desk's own criteria must produce an order. Douglas "
        "is explicit that traders must not 'pick and choose the edges they think ... "
        "are going to work', so skipping qualifying setups is a violation even when "
        "the skipped trade would have lost."
    ),
    Principle.PAY_MYSELF: (
        "Profit is realised on a rule, not on a feeling: objectives are taken and "
        "stops trail. Left unimplemented, open profit is repeatedly given back."
    ),
    Principle.MONITOR_ERRORS: (
        "Operator interventions - pauses, kills, restarts, parameter changes mid-run - "
        "are logged and counted. For an automated desk this is where the psychology "
        "actually lives."
    ),
    Principle.NEVER_VIOLATE: (
        "Any violation of the six above is recorded with its timestamp and reason. The "
        "count is reported rather than reset, because a rule that is quietly forgiven "
        "is not a rule."
    ),
}


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    VIOLATION = "violation"


@dataclass(slots=True)
class PrincipleEvent:
    """A recorded adherence event against one of the seven principles."""

    principle: Principle
    severity: Severity
    detail: str
    when: str = ""
    symbol: str = ""

    def line(self) -> str:
        mark = {
            Severity.INFO: "ok",
            Severity.WARNING: "warn",
            Severity.VIOLATION: "VIOLATION",
        }[self.severity]
        stamp = f"{self.when} " if self.when else ""
        sym = f"{self.symbol} " if self.symbol else ""
        return f"{stamp}[{mark}] {sym}P{self.principle.number}: {self.detail}"


@dataclass
class AdherenceReport:
    """How well the desk followed its own rules over a run."""

    events: list[PrincipleEvent] = field(default_factory=list)
    trades_considered: int = 0
    qualifying_signals: int = 0
    signals_acted_on: int = 0
    interventions: int = 0

    @property
    def violations(self) -> list[PrincipleEvent]:
        return [e for e in self.events if e.severity is Severity.VIOLATION]

    @property
    def warnings(self) -> list[PrincipleEvent]:
        return [e for e in self.events if e.severity is Severity.WARNING]

    @property
    def action_rate(self) -> float:
        """Fraction of qualifying edges actually acted on. Douglas wants this at 1.0."""
        if self.qualifying_signals <= 0:
            return 1.0
        return self.signals_acted_on / self.qualifying_signals

    def by_principle(self) -> dict[Principle, int]:
        counts: dict[Principle, int] = {p: 0 for p in Principle}
        for event in self.violations:
            counts[event.principle] += 1
        return counts

    @property
    def clean(self) -> bool:
        return not self.violations

    def summary(self) -> list[str]:
        lines = [
            f"adherence: {len(self.violations)} violation(s), "
            f"{len(self.warnings)} warning(s)",
            f"edges acted on: {self.signals_acted_on}/{self.qualifying_signals} "
            f"({self.action_rate:.0%})",
            f"operator interventions: {self.interventions}",
        ]
        counts = self.by_principle()
        for principle, count in counts.items():
            if count:
                lines.append(f"  P{principle.number} {principle.value}: {count}")
        return lines


def principles_text() -> list[str]:
    """The framework as it appears in the book, for display in the terminal."""
    out = ["THE FIVE FUNDAMENTAL TRUTHS"]
    for i, truth in enumerate(FUNDAMENTAL_TRUTHS, 1):
        out.append(f"  {i}. {truth}")
    out.append("")
    out.append("THE SEVEN PRINCIPLES OF CONSISTENCY")
    out.append("  I am a consistent winner because:")
    for principle in Principle:
        out.append(f"  {principle.number}. {principle.value}")
    return out
