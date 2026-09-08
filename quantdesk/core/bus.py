"""A tiny async publish/subscribe bus.

Used to decouple the trading pipeline from observers (persistence, the terminal
UI, logging). Deliberately in-process and unordered across topics: this is a
single-process desk, not a distributed system.

Design notes:

* A subscriber that raises is logged and skipped, never allowed to bring down
  the publisher. A crashing UI panel must not stop trading.
* ``publish`` awaits all handlers. Handlers are expected to be fast and to
  offload slow work themselves. This keeps event ordering per topic intuitive.
* A bounded history ring buffer lets pull-based consumers (the TUI) catch up
  without needing their own queue.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Deque, Iterable

log = logging.getLogger(__name__)

Handler = Callable[[Any], Awaitable[None] | None]


class Topic:
    """Well-known topic names. Plain strings, grouped for discoverability."""

    BAR = "bar"
    SIGNAL = "signal"
    TARGETS = "targets"
    ORDER = "order"
    FILL = "fill"
    ACCOUNT = "account"
    RISK = "risk"
    LOG = "log"
    ERROR = "error"
    STATUS = "status"


@dataclass(slots=True)
class Event:
    topic: str
    payload: Any


class EventBus:
    """In-process async event bus with per-topic fan-out."""

    def __init__(self, history: int = 500) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self._history: Deque[Event] = deque(maxlen=history)
        self._history_by_topic: dict[str, Deque[Any]] = defaultdict(
            lambda: deque(maxlen=history)
        )

    # ------------------------------------------------------------ subscribe
    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        """Register ``handler`` for ``topic``. Returns an unsubscribe callable."""
        self._subs[topic].append(handler)

        def _unsubscribe() -> None:
            try:
                self._subs[topic].remove(handler)
            except ValueError:
                pass

        return _unsubscribe

    def subscribe_many(self, topics: Iterable[str], handler: Handler) -> None:
        for topic in topics:
            self.subscribe(topic, handler)

    # -------------------------------------------------------------- publish
    async def publish(self, topic: str, payload: Any) -> None:
        """Deliver ``payload`` to every subscriber of ``topic``."""
        self._history.append(Event(topic, payload))
        self._history_by_topic[topic].append(payload)

        for handler in list(self._subs.get(topic, ())):
            try:
                result = handler(payload)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - observers must never break the desk
                log.exception("event handler failed for topic %r", topic)

    def publish_soon(self, topic: str, payload: Any) -> None:
        """Fire-and-forget publish from sync code inside a running loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop (e.g. inside a backtest run from sync code): record only.
            self._history.append(Event(topic, payload))
            self._history_by_topic[topic].append(payload)
            return
        loop.create_task(self.publish(topic, payload))

    # -------------------------------------------------------------- history
    def recent(self, topic: str | None = None, limit: int = 50) -> list[Any]:
        """Most recent payloads, newest last."""
        if topic is None:
            return [e.payload for e in list(self._history)[-limit:]]
        return list(self._history_by_topic[topic])[-limit:]

    def clear(self) -> None:
        self._history.clear()
        self._history_by_topic.clear()
