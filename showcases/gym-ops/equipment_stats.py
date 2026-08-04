"""Equipment utilization tracking: usage duration, idle time, turnover rate.

Tracks per-equipment bind/unbind events and computes utilization over a
sliding window. Used by B-end dashboard for space-efficiency analysis.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class _UsageInterval:
    start_ts: float
    end_ts: float | None = None  # None = still active
    member_id: str | None = None


class EquipmentStats:
    """Track equipment usage intervals and compute utilization."""

    def __init__(self, max_intervals: int = 1000):
        self._history: dict[str, list[_UsageInterval]] = {}
        self._active: dict[str, _UsageInterval] = {}
        self._max_intervals = max_intervals

    def on_bind(self, equipment_id: str, member_id: str | None, now: float) -> None:
        """Person binds to equipment. Starts a usage interval."""
        if equipment_id in self._active:
            return  # already bound — shouldn't happen, but guard
        interval = _UsageInterval(start_ts=now, member_id=member_id)
        self._active[equipment_id] = interval
        self._history.setdefault(equipment_id, []).append(interval)

    def on_unbind(self, equipment_id: str, now: float) -> None:
        """Person leaves equipment. Closes the usage interval."""
        interval = self._active.pop(equipment_id, None)
        if interval is not None:
            interval.end_ts = now
        # Trim old history per equipment
        hist = self._history.get(equipment_id, [])
        if len(hist) > self._max_intervals:
            self._history[equipment_id] = hist[-self._max_intervals:]

    def utilization(self, equipment_id: str, window_hours: float = 24.0,
                    now: float | None = None) -> float:
        """Compute utilization ratio (0.0-1.0) over the last window_hours.

        Includes both closed intervals and the currently active interval.
        """
        now = now or time.time()
        window_start = now - window_hours * 3600.0
        hist = self._history.get(equipment_id, [])
        used_seconds = 0.0
        for iv in hist:
            start = max(iv.start_ts, window_start)
            end = iv.end_ts if iv.end_ts is not None else now
            if end < window_start:
                continue
            used_seconds += max(0.0, end - start)
        window_seconds = window_hours * 3600.0
        return min(1.0, used_seconds / window_seconds) if window_seconds > 0 else 0.0

    def turnover_count(self, equipment_id: str, window_hours: float = 24.0,
                       now: float | None = None) -> int:
        """Count completed usage intervals (turnovers) in the window."""
        now = now or time.time()
        window_start = now - window_hours * 3600.0
        hist = self._history.get(equipment_id, [])
        count = 0
        for iv in hist:
            if iv.end_ts is not None and iv.end_ts >= window_start:
                count += 1
        return count

    def snapshot(self, now: float | None = None) -> dict:
        """Return per-equipment summary: current state + 24h utilization."""
        now = now or time.time()
        result: dict[str, dict] = {}
        all_ids = set(self._history.keys()) | set(self._active.keys())
        for eid in all_ids:
            active = self._active.get(eid)
            result[eid] = {
                "occupied": active is not None,
                "member_id": active.member_id if active else None,
                "duration_s": round(now - active.start_ts, 1) if active else 0.0,
                "utilization_24h": round(self.utilization(eid, 24.0, now), 3),
                "turnover_24h": self.turnover_count(eid, 24.0, now),
            }
        return result
