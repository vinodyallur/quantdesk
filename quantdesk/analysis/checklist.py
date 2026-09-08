"""Murphy's technical checklist, chapter 19 - the weight-of-evidence layer.

His closing chapter is the one that makes the rest usable. Twenty-odd techniques
have been introduced, they overlap, and they routinely disagree; the checklist is
how a technician turns that into a single decision:

    "All of these approaches overlap to some extent and complement one another. The
    day the user sees these interrelationships, and is able to view technical
    analysis as the sum of its parts, is the day that person deserves the title of
    technical analyst."

The questions below are his, in his order. Each is answered from
:class:`~quantdesk.analysis.symbol.SymbolAnalysis` and contributes a weighted vote.

**Two deliberate honesty choices.**

*Weights are mine, not his.* Murphy never assigns numbers to these questions. The
weights here reflect what he emphasises - trend direction and the long-term picture
carry most, a single candlestick reading carries least - but they are a judgement
call and are declared as such rather than presented as doctrine.

*Unanswerable questions are marked unanswerable.* Three of his items need techniques
this desk does not implement: contrary-opinion sentiment numbers, cycle analysis
(chapter 14), and point and figure charts (chapter 11). They are reported as
UNAVAILABLE and excluded from the tally rather than quietly scored as neutral,
because a checklist that hides its own blind spots is worse than one that admits
them. :attr:`Verdict.coverage` is how much of the checklist was actually answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from quantdesk.analysis.patterns import Bias, PatternFamily
from quantdesk.analysis.retracement import classify_retracement
from quantdesk.analysis.symbol import SymbolAnalysis


class ItemStatus(str, Enum):
    ANSWERED = "answered"
    UNAVAILABLE = "unavailable"
    """The technique is not implemented or the data does not exist."""
    NOT_READY = "not_ready"
    """Implemented, but there is not yet enough history to answer."""


@dataclass(slots=True)
class ChecklistItem:
    """One of Murphy's questions, answered."""

    number: int
    question: str
    bias: Bias = Bias.NEUTRAL
    weight: float = 1.0
    confidence: float = 1.0
    answer: str = ""
    status: ItemStatus = ItemStatus.ANSWERED

    @property
    def contribution(self) -> float:
        """Signed weighted vote, zero unless answered."""
        if self.status is not ItemStatus.ANSWERED:
            return 0.0
        return self.bias.sign * self.weight * self.confidence

    @property
    def counted_weight(self) -> float:
        if self.status is not ItemStatus.ANSWERED:
            return 0.0
        return self.weight * self.confidence

    def line(self) -> str:
        mark = {
            Bias.BULLISH: "+",
            Bias.BEARISH: "-",
            Bias.NEUTRAL: "=",
        }[self.bias]
        if self.status is not ItemStatus.ANSWERED:
            mark = "?"
        return f"{mark} [{self.number:2d}] {self.question} -> {self.answer}"


@dataclass(slots=True)
class Verdict:
    """The checklist's overall read."""

    symbol: str
    bias: Bias
    score: float
    """Weighted evidence in [-1, 1]."""
    conviction: float
    """How strongly to act, in [0, 1]. Combines score magnitude with coverage."""
    items: list[ChecklistItem] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    @property
    def answered(self) -> list[ChecklistItem]:
        return [i for i in self.items if i.status is ItemStatus.ANSWERED]

    @property
    def coverage(self) -> float:
        """Fraction of the checklist that could actually be answered."""
        if not self.items:
            return 0.0
        return len(self.answered) / len(self.items)

    @property
    def bullish(self) -> list[ChecklistItem]:
        return [i for i in self.answered if i.bias is Bias.BULLISH]

    @property
    def bearish(self) -> list[ChecklistItem]:
        return [i for i in self.answered if i.bias is Bias.BEARISH]

    @property
    def agreement(self) -> float:
        """How one-sided the answered evidence is, in [0, 1]."""
        bull = sum(i.counted_weight for i in self.bullish)
        bear = sum(i.counted_weight for i in self.bearish)
        total = bull + bear
        if total <= 0:
            return 0.0
        return abs(bull - bear) / total

    @property
    def tradable(self) -> bool:
        """Is there a directional case with nothing blocking it?"""
        return self.bias is not Bias.NEUTRAL and not self.blockers

    def summary(self) -> list[str]:
        head = [
            f"{self.symbol}: {self.bias.value.upper()} "
            f"score {self.score:+.2f} conviction {self.conviction:.2f} "
            f"agreement {self.agreement:.2f} coverage {self.coverage:.0%}",
        ]
        if self.blockers:
            head.append("BLOCKED: " + "; ".join(self.blockers))
        return head + [i.line() for i in self.items]


class TechnicalChecklist:
    """Runs Murphy's chapter 19 checklist against a symbol's analysis.

    Parameters
    ----------
    min_coverage:
        Below this fraction of answered questions the verdict is treated as
        unreliable and reported as blocked. Guards against acting on a nearly empty
        checklist during warmup.
    require_timeframe_agreement:
        Enforce his "work from the long term to the short term" rule as a hard gate:
        a verdict that opposes the higher-timeframe direction is blocked outright
        rather than merely down-weighted.
    """

    def __init__(
        self,
        min_coverage: float = 0.55,
        require_timeframe_agreement: bool = True,
    ) -> None:
        self.min_coverage = min_coverage
        self.require_timeframe_agreement = require_timeframe_agreement

    def evaluate(self, sa: SymbolAnalysis) -> Verdict:
        items = self._items(sa)
        total = sum(i.contribution for i in items)
        weight = sum(i.counted_weight for i in items)
        score = 0.0 if weight <= 0 else max(-1.0, min(1.0, total / weight))
        bias = Bias.from_sign(score) if abs(score) >= 0.12 else Bias.NEUTRAL

        verdict = Verdict(
            symbol=sa.symbol,
            bias=bias,
            score=score,
            conviction=0.0,
            items=items,
        )
        # Conviction is deliberately not just |score|. A strong reading from a
        # quarter of the checklist is not strong evidence, and agreement matters as
        # much as direction.
        verdict.conviction = min(
            1.0, abs(score) * (0.4 + 0.6 * verdict.coverage) * (0.5 + 0.5 * verdict.agreement)
        )
        verdict.blockers = self._blockers(sa, verdict)
        return verdict

    # ------------------------------------------------------------------ items
    def _items(self, sa: SymbolAnalysis) -> list[ChecklistItem]:
        out: list[ChecklistItem] = []
        add = out.append

        # 1-3: the long-term picture first, which is his procedural rule.
        permitted = sa.timeframes.permitted_direction()
        alignment = sa.timeframes.alignment()
        add(
            ChecklistItem(
                1,
                "What is the direction of the overall market?",
                bias=permitted,
                weight=2.5,
                confidence=min(1.0, abs(alignment) * 2.0) if permitted is not Bias.NEUTRAL else 0.5,
                answer=f"higher timeframes permit {permitted.value} (alignment {alignment:+.2f})",
                status=ItemStatus.ANSWERED if sa.ready else ItemStatus.NOT_READY,
            )
        )
        add(
            ChecklistItem(
                2,
                "What is the direction of the various market sectors?",
                weight=1.0,
                answer="cross-sectional read lives in the relative-strength agent, not here",
                status=ItemStatus.UNAVAILABLE,
            )
        )
        higher = [
            f"{tf.value} {sa.timeframes.reads[tf].bias.value}"
            for tf in sa.timeframes.direction_frames
        ]
        add(
            ChecklistItem(
                3,
                "What are the weekly and monthly charts showing?",
                bias=permitted,
                weight=1.5,
                answer=", ".join(higher) if higher else "no higher-timeframe structure",
                status=ItemStatus.ANSWERED if higher else ItemStatus.NOT_READY,
            )
        )

        # 4: Dow trend and its degree.
        trend = sa.trend
        add(
            ChecklistItem(
                4,
                "Are the major, intermediate and minor trends up, down or sideways?",
                bias=sa.trend_bias,
                weight=2.5,
                confidence=trend.confidence if trend else 0.0,
                answer=(
                    f"{trend.direction.value} ({sa.degree}), {trend.health.value}"
                    if trend
                    else "unclassified"
                ),
                status=ItemStatus.ANSWERED if trend and trend.pivots_used >= 2 else ItemStatus.NOT_READY,
            )
        )

        # 5: support and resistance, read as headroom rather than raw levels, since
        # what matters for a decision is whether there is room to the next barrier.
        up_room = sa.headroom(True)
        down_room = sa.headroom(False)
        level_bias = Bias.NEUTRAL
        if up_room != down_room:
            level_bias = Bias.BULLISH if up_room > down_room else Bias.BEARISH
        add(
            ChecklistItem(
                5,
                "Where are the important support and resistance levels?",
                bias=level_bias,
                weight=1.5,
                answer=f"{_fmt_atr(up_room)} ATR to resistance, {_fmt_atr(down_room)} to support",
                status=ItemStatus.ANSWERED if sa.levels.levels else ItemStatus.NOT_READY,
            )
        )

        # 6: trendlines and channels.
        active = sa.trendlines.active
        valid = sa.trendlines.valid_lines
        line_bias = Bias.NEUTRAL
        if valid:
            primary = max(valid, key=lambda ln: ln.touches)
            line_bias = Bias.BULLISH if primary.kind.value == "uptrend" else Bias.BEARISH
        add(
            ChecklistItem(
                6,
                "Where are the important trendlines or channels?",
                bias=line_bias,
                weight=1.5,
                answer=(
                    f"{len(valid)} validated of {len(active)} active"
                    + (f", fan count {sa.trendlines.fan_count}" if sa.trendlines.fan_count else "")
                ),
                status=ItemStatus.ANSWERED if active else ItemStatus.NOT_READY,
            )
        )

        # 7: volume confirmation. Murphy's asymmetry - it matters most on the upside.
        vol_ratio = sa.volume_ratio
        vol_bias = Bias.NEUTRAL
        if vol_ratio >= 1.3 and sa.trend_bias is not Bias.NEUTRAL:
            vol_bias = sa.trend_bias
        elif vol_ratio < 0.7 and sa.trend_bias is Bias.BULLISH:
            vol_bias = Bias.BEARISH
        add(
            ChecklistItem(
                7,
                "Are volume and open interest confirming the price action?",
                bias=vol_bias,
                weight=1.5,
                answer=f"volume x{vol_ratio:.2f} of average, OBV {'rising' if sa.obv.rising else 'falling'}",
                status=ItemStatus.ANSWERED if sa.volume_avg > 0 else ItemStatus.NOT_READY,
            )
        )

        # 8: retracement position. 40-60% is his entry zone.
        retr = sa.retracements
        retr_bias = Bias.NEUTRAL
        retr_answer = "no measurable leg"
        if retr is not None:
            frac = retr.signed_retracement_of(sa.price)
            leg_up = retr.is_up_leg
            if frac < 0:
                # Still extending past the leg's end. Nothing has been retraced, so
                # the leg's own direction is the only reading available.
                retr_bias = Bias.BULLISH if leg_up else Bias.BEARISH
                retr_answer = f"extended {-frac:.0%} beyond the last leg, no pullback yet"
            elif frac <= 0.66:
                zone = classify_retracement(frac).value
                if 0.38 <= frac <= 0.62:
                    # Murphy's 40-60% entry zone: a healthy correction against the
                    # leg favours the leg resuming.
                    retr_bias = Bias.BULLISH if leg_up else Bias.BEARISH
                retr_answer = f"retraced {frac:.0%} of the last leg ({zone})"
            else:
                # "beyond 66% is a reversal, not a retracement"
                retr_bias = Bias.BEARISH if leg_up else Bias.BULLISH
                retr_answer = f"retraced {frac:.0%} - past 66%, a reversal not a correction"
        add(
            ChecklistItem(
                8,
                "Where are the 33%, 50% and 66% retracements?",
                bias=retr_bias,
                weight=1.5,
                answer=retr_answer,
                status=ItemStatus.ANSWERED if retr is not None else ItemStatus.NOT_READY,
            )
        )

        # 9: gaps.
        recent_gaps = [g for g in sa.gaps.gaps if sa.index - g.index <= 20]
        gap_bias = Bias.NEUTRAL
        if recent_gaps:
            newest = recent_gaps[-1]
            gap_bias = Bias.BULLISH if newest.up else Bias.BEARISH
            # An exhaustion gap appears at the *end* of a move, so it argues the
            # other way from the direction it opened in.
            if newest.gap_type.value == "exhaustion":
                gap_bias = gap_bias.opposite
        add(
            ChecklistItem(
                9,
                "Are there any price gaps and what type are they?",
                bias=gap_bias,
                weight=1.0,
                answer=(
                    ", ".join(
                        f"{g.gap_type.value} {'up' if g.up else 'down'}"
                        for g in recent_gaps[-3:]
                    )
                    if recent_gaps
                    else "none recent"
                ),
            )
        )

        # 10-12: patterns and their objectives.
        reversals = sa.patterns.actionable(family=PatternFamily.REVERSAL)
        continuations = sa.patterns.actionable(family=PatternFamily.CONTINUATION)
        add(self._pattern_item(10, "Are there any major reversal patterns visible?", sa, reversals, 2.0))
        add(self._pattern_item(11, "Are there any continuation patterns visible?", sa, continuations, 1.5))
        objectives = [
            f"{p.kind.label} -> {p.objective:.4g} (R:R {p.reward_risk:.1f})"
            for p in (reversals + continuations)[:3]
        ]
        add(
            ChecklistItem(
                12,
                "What are the price objectives from those patterns?",
                bias=Bias.NEUTRAL,
                weight=0.0,
                answer="; ".join(objectives) if objectives else "no completed patterns",
                status=ItemStatus.ANSWERED if objectives else ItemStatus.NOT_READY,
            )
        )

        # 13: moving averages.
        add(
            ChecklistItem(
                13,
                "Which way are the moving averages pointing?",
                bias=sa.ma_slope_bias,
                weight=2.0,
                answer=f"fast {sa.ma_fast.value:.4g} vs slow {sa.ma_slow.value:.4g}, {sa.ma_slope_bias.value}",
                status=ItemStatus.ANSWERED if sa.ma_slow.ready else ItemStatus.NOT_READY,
            )
        )

        # 14: oscillator extremes. In a trend these warn rather than reverse, which
        # is why the weight is modest.
        osc_bias = Bias.NEUTRAL
        rsi = sa.rsi.value
        if rsi >= 70:
            osc_bias = Bias.BEARISH
        elif rsi <= 30:
            osc_bias = Bias.BULLISH
        add(
            ChecklistItem(
                14,
                "Are the oscillators overbought or oversold?",
                bias=osc_bias,
                weight=1.0,
                answer=f"RSI {rsi:.0f}, stochastic {sa.stoch.k:.0f}",
                status=ItemStatus.ANSWERED if sa.rsi.ready else ItemStatus.NOT_READY,
            )
        )

        # 15: divergence, which Murphy rates highly.
        divs = sa.divergences
        div_bias = Bias.NEUTRAL
        if divs:
            bull = sum(1 for d in divs if d.bias is Bias.BULLISH)
            bear = sum(1 for d in divs if d.bias is Bias.BEARISH)
            div_bias = Bias.from_sign(bull - bear)
        add(
            ChecklistItem(
                15,
                "Are any divergences apparent on the oscillators?",
                bias=div_bias,
                weight=2.0,
                answer="; ".join(d.describe() for d in divs[:2]) if divs else "none",
            )
        )

        # 16: sentiment. No data source, so no answer.
        add(
            ChecklistItem(
                16,
                "Are contrary opinion numbers showing any extremes?",
                weight=1.5,
                answer="no sentiment feed wired up",
                status=ItemStatus.UNAVAILABLE,
            )
        )

        # 17-19: Elliott and Fibonacci.
        best = sa.elliott.best
        add(
            ChecklistItem(
                17,
                "What is the Elliott Wave pattern showing?",
                bias=sa.elliott.bias(),
                weight=1.5,
                confidence=best.confidence if best else 0.0,
                answer=sa.elliott.position(),
                status=ItemStatus.ANSWERED if best else ItemStatus.NOT_READY,
            )
        )
        add(
            ChecklistItem(
                18,
                "Are there any obvious 3 or 5 wave patterns?",
                bias=Bias.NEUTRAL,
                weight=0.0,
                answer=(
                    f"{best.kind.value}, {len(best.waves)} waves, "
                    + ("valid" if best.valid else "; ".join(best.violations))
                    if best
                    else "no count"
                ),
                status=ItemStatus.ANSWERED if best else ItemStatus.NOT_READY,
            )
        )
        fib = sa.elliott.targets()
        add(
            ChecklistItem(
                19,
                "What about Fibonacci retracements or projections?",
                bias=Bias.NEUTRAL,
                weight=0.0,
                answer=", ".join(f"{k} {v:.4g}" for k, v in list(fib.items())[:3]) or "none",
                status=ItemStatus.ANSWERED if fib else ItemStatus.NOT_READY,
            )
        )

        # 20-21: cycles. Chapter 14 is not implemented.
        add(
            ChecklistItem(
                20,
                "Are there any cycle tops or bottoms due?",
                weight=1.0,
                answer="cycle analysis (ch. 14) not implemented",
                status=ItemStatus.UNAVAILABLE,
            )
        )
        add(
            ChecklistItem(
                21,
                "Is the market showing right or left translation?",
                weight=0.5,
                answer="needs cycle analysis (ch. 14)",
                status=ItemStatus.UNAVAILABLE,
            )
        )

        # 22: the trend-following indicator read.
        adx_bias = Bias.NEUTRAL
        if sa.dmi.trending:
            adx_bias = Bias.from_sign(sa.dmi.direction)
        add(
            ChecklistItem(
                22,
                "Which way is the computer trend moving: up, down or sideways?",
                bias=adx_bias,
                weight=2.0,
                confidence=min(1.0, sa.dmi.adx / 40.0) if sa.dmi.ready else 0.0,
                answer=(
                    f"ADX {sa.dmi.adx:.0f} "
                    f"({'trending' if sa.dmi.trending else 'no trend'}), "
                    f"+DI {sa.dmi.plus_di:.0f} / -DI {sa.dmi.minus_di:.0f}"
                ),
                status=ItemStatus.ANSWERED if sa.dmi.ready else ItemStatus.NOT_READY,
            )
        )

        # 23: candlesticks. Point and figure (ch. 11) is not implemented.
        candle_bias = Bias.from_sign(sa.candles.net_bias())
        signals = sa.candle_signals
        add(
            ChecklistItem(
                23,
                "What are the point and figure charts or candlesticks showing?",
                bias=candle_bias,
                weight=1.0,
                answer=(
                    "; ".join(s.kind.label for s in signals[:3])
                    if signals
                    else "no candle signal (point and figure not implemented)"
                ),
                status=ItemStatus.ANSWERED if signals else ItemStatus.NOT_READY,
            )
        )
        return out

    def _pattern_item(
        self, number: int, question: str, sa: SymbolAnalysis, patterns, weight: float
    ) -> ChecklistItem:
        if not patterns:
            return ChecklistItem(
                number, question, weight=weight, answer="none completed",
                status=ItemStatus.NOT_READY,
            )
        total = sum(p.significance(sa.index, sa.atr.value) * p.bias.sign for p in patterns)
        mass = sum(p.significance(sa.index, sa.atr.value) for p in patterns)
        bias = Bias.from_sign(total)
        return ChecklistItem(
            number,
            question,
            bias=bias,
            weight=weight,
            confidence=min(1.0, mass / max(1, len(patterns))),
            answer="; ".join(
                f"{p.kind.label} {p.stage.value}" for p in patterns[:3]
            ),
        )

    # --------------------------------------------------------------- blockers
    def _blockers(self, sa: SymbolAnalysis, verdict: Verdict) -> list[str]:
        """Hard gates. A blocked verdict must not be traded regardless of score."""
        out: list[str] = []
        if not sa.ready:
            out.append("analysis still warming up")
        if verdict.coverage < self.min_coverage:
            out.append(
                f"only {verdict.coverage:.0%} of the checklist answered "
                f"(need {self.min_coverage:.0%})"
            )
        if self.require_timeframe_agreement and verdict.bias is not Bias.NEUTRAL:
            if not sa.timeframes.agrees(verdict.bias):
                out.append(
                    f"higher timeframes do not permit {verdict.bias.value} "
                    "(work from the long term to the short term)"
                )
        return out


def _fmt_atr(value: float) -> str:
    return "clear" if value == float("inf") else f"{value:.1f}"
