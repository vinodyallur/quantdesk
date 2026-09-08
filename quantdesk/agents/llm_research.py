"""Optional LLM research agent.

Off by default, and deliberately so. A language model is not a price predictor,
and putting one in the per-bar decision path would be slow, expensive and
non-deterministic - three properties you do not want in an execution loop.

Where a model does add something is the slow loop: reading the state of the book
across symbols and forming a contextual view that the numeric agents, which each
see one indicator family, cannot express. Treat its output as one weak opinion
among several, which is why its default weight is the lowest on the desk.

The design constraints this file enforces:

* The model runs on its own timer (``llm_interval_seconds``), never on a bar.
* ``evaluate`` only ever reads a cached view. It cannot block or do I/O.
* Cached views decay with a half-life, so a stale opinion fades out instead of
  being trusted indefinitely if the API starts failing.
* Every failure mode - missing package, missing key, API error, malformed JSON -
  degrades to "no opinion" rather than raising.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from quantdesk.agents.base import AgentContext, SignalAgent
from quantdesk.core.types import Signal, utcnow

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a risk-aware discretionary analyst on a small systematic \
trading desk. You are given current technical state for a universe of instruments.

Return a JSON object of the form:
{"views": [{"symbol": "<symbol>", "bias": <number -1..1>, "confidence": <number 0..1>, \
"reason": "<max 15 words>"}]}

Rules:
- bias: -1 strongly bearish, 0 neutral, +1 strongly bullish, over the next few hours.
- confidence: how much you'd stake on it. Use low values when the picture is mixed.
- Prefer neutral (bias near 0, low confidence) when signals conflict. Being \
non-committal is correct more often than being decisive.
- Judge only from the data given. Do not invent news, prices or events.
- One entry per symbol provided. No prose outside the JSON."""


@dataclass(slots=True)
class LLMView:
    symbol: str
    bias: float
    confidence: float
    reason: str
    ts: datetime

    def decayed_confidence(self, now: datetime, halflife_seconds: float) -> float:
        """Confidence after exponential time decay."""
        age = max(0.0, (now - self.ts).total_seconds())
        if halflife_seconds <= 0:
            return self.confidence
        return self.confidence * math.pow(0.5, age / halflife_seconds)


class LLMResearchAgent(SignalAgent):
    """Slow-loop LLM analyst producing a decaying contextual bias per symbol."""

    name = "llm_research"
    weight = 0.5

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        interval_seconds: int = 900,
        max_age_multiple: float = 4.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.interval_seconds = max(60, interval_seconds)
        self.halflife_seconds = float(self.interval_seconds) * 2.0
        self.max_age = timedelta(seconds=self.interval_seconds * max_age_multiple)
        self._views: dict[str, LLMView] = {}
        self._client = None
        self._last_error: str = ""
        self._last_refresh: datetime | None = None
        self.calls_made = 0

        if not api_key:
            self.enabled = False
            self._last_error = "no OPENAI api key configured"
            return
        try:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            self.enabled = False
            self._last_error = "openai package not installed (pip install -r requirements-llm.txt)"

    # ------------------------------------------------------------- hot path
    def evaluate(self, ctx: AgentContext) -> Signal | None:
        """Read the cached view. No I/O, no blocking."""
        view = self._views.get(ctx.symbol)
        if view is None:
            return None
        if ctx.now - view.ts > self.max_age:
            return None
        conf = view.decayed_confidence(ctx.now, self.halflife_seconds)
        if conf < 0.1 or abs(view.bias) < 0.1:
            return None
        return self.signal(
            ctx,
            score=view.bias,
            confidence=conf,
            horizon_bars=48,
            rationale=f"LLM: {view.reason}",
            bias=view.bias,
            age_seconds=(ctx.now - view.ts).total_seconds(),
        )

    # ------------------------------------------------------------ slow loop
    async def run_forever(self, snapshot_provider) -> None:
        """Refresh views on a timer. ``snapshot_provider`` returns current features."""
        if not self.enabled:
            log.info("LLM agent disabled: %s", self._last_error)
            return
        # Small initial delay so the feature engine has warmed up.
        await asyncio.sleep(5)
        while True:
            try:
                await self.refresh_once(snapshot_provider())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)
                log.warning("LLM refresh failed: %s", exc)
            await asyncio.sleep(self.interval_seconds)

    async def refresh_once(self, snapshots: dict) -> int:
        """One research pass. Returns the number of views updated."""
        if not self.enabled or self._client is None:
            return 0
        ready = {s: snap for s, snap in snapshots.items() if snap.ready}
        if not ready:
            return 0

        prompt = self._build_prompt(ready)
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=800,
            )
            self.calls_made += 1
            content = resp.choices[0].message.content or "{}"
        except Exception as exc:  # noqa: BLE001 - network/API issues are expected
            self._last_error = f"api call failed: {exc}"
            log.warning("LLM call failed: %s", exc)
            return 0

        updated = self._ingest(content, set(ready))
        self._last_refresh = utcnow()
        return updated

    def _build_prompt(self, snapshots: dict) -> str:
        lines = [
            "Current technical state (all values as of the latest closed bar):",
            "",
            f"{'symbol':<12}{'chg_slow':>10}{'trend_atr':>11}{'rsi':>7}"
            f"{'zscore':>9}{'vol_ann':>9}{'vol_pct':>9}{'chan_pos':>10}",
        ]
        for sym, f in sorted(snapshots.items()):
            trend_atr = f.ema_spread / f.atr_pct if f.atr_pct > 1e-9 else 0.0
            lines.append(
                f"{sym:<12}{f.roc_slow:>9.2%}{trend_atr:>11.2f}{f.rsi:>7.0f}"
                f"{f.zscore:>9.2f}{f.realized_vol:>8.1%}{f.vol_percentile:>9.0%}"
                f"{f.channel_pos:>10.0%}"
            )
        lines += [
            "",
            "Legend: chg_slow = momentum over the long lookback; trend_atr = EMA spread",
            "in ATR units (>1 is a firm trend); zscore = price vs its rolling mean;",
            "vol_pct = where current volatility sits in its own history; chan_pos =",
            "position in the recent range (0 = lows, 1 = highs).",
            "",
            f"Provide one view for each of: {', '.join(sorted(snapshots))}",
        ]
        return "\n".join(lines)

    def _ingest(self, content: str, allowed: set[str]) -> int:
        """Parse and validate the model's JSON response."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            self._last_error = f"bad JSON: {exc}"
            return 0

        views = data.get("views")
        if not isinstance(views, list):
            self._last_error = "response missing 'views' list"
            return 0

        now = utcnow()
        updated = 0
        for item in views:
            if not isinstance(item, dict):
                continue
            sym = str(item.get("symbol", "")).strip().upper()
            # Only accept symbols we actually asked about.
            if sym not in allowed:
                continue
            try:
                bias = float(item.get("bias", 0.0))
                conf = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(bias) and math.isfinite(conf)):
                continue
            self._views[sym] = LLMView(
                symbol=sym,
                bias=max(-1.0, min(1.0, bias)),
                confidence=max(0.0, min(1.0, conf)),
                reason=str(item.get("reason", ""))[:120],
                ts=now,
            )
            updated += 1
        if updated:
            self._last_error = ""
        return updated

    # ---------------------------------------------------------------- status
    def status(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "views": len(self._views),
            "calls": self.calls_made,
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "last_error": self._last_error,
        }
