"""shelf-ops alert broker: SSE fan-out with per-(kind,slot) cooldown.

Consumes the events emitted by analytics.record() (stockout / restock /
occupancy) and fans them out to connected SSE subscribers at /api/events.
Cooldown (cfg.alert_cooldown_seconds) dedupes bursts of identical events so a
flapping slot can't flood the stream.
"""
from __future__ import annotations

import json
import time
from typing import Any


class AlertBroker:
    def __init__(self, cooldown_seconds: int = 30) -> None:
        self.cooldown_seconds = int(cooldown_seconds)
        self._subscribers: list[Any] = []   # queue-ish objects with put_nowait
        self._last_emit: dict[str, float] = {}

    def subscribe(self, sink: Any) -> None:
        """sink: object with .put_nowait(str) for SSE message delivery."""
        self._subscribers.append(sink)

    def unsubscribe(self, sink: Any) -> None:
        if sink in self._subscribers:
            self._subscribers.remove(sink)

    def emit(self, event: dict, now: float | None = None) -> bool:
        """Forward one analytics event if cooldown allows. Returns bool sent."""
        now = now if now is not None else time.time()
        key = f"{event.get('slot_id')}:{event.get('kind')}"
        if now - self._last_emit.get(key, 0.0) < self.cooldown_seconds:
            return False
        self._last_emit[key] = now
        payload = json.dumps(event, ensure_ascii=False)
        dead: list[Any] = []
        for s in self._subscribers:
            try:
                s.put_nowait(payload)
            except Exception:  # full/broken queue -> drop it
                dead.append(s)
        for s in dead:
            self.unsubscribe(s)
        return True

    def fanout(self, events: list[dict]) -> int:
        """Emit a batch from analytics.record(); returns count forwarded."""
        return sum(1 for ev in events if self.emit(ev))