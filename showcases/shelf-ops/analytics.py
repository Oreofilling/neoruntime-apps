"""shelf-ops analytics: sqlite time-bucket heatmap + stockout/restock events.

Mirrors gym-ops trajectory/equipment_stats but for shelf occupancy, deriving
the user's "销售情况热力图" ask directly from slot snapshots:

  * shelf_counts — PEAK items per (bucket, slot, A–E code). A bucket spans
    cfg.bucket_seconds (default 600s); each tick UPSERTs the max count seen so
    far, so a heatmap cell shows peak fill per time window.
  * shelf_states  — tick-counters per (bucket, slot, STATE) so the UI can render
    EMPTY/PARTIAL/FULL proportions per window.
  * shelf_events  — occupancy transitions + confirmed stockout/restock.

stockout logic: a slot sustains EMPTY >= cfg.stockout_seconds -> one "stockout"
event (recorded once, so it does not spam); the first non-EMPTY observation
afterwards emits "restock". alert_cooldown_seconds is reserved for the SSE layer.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any

_STATE_ORDER = {"EMPTY": 0, "PARTIAL": 1, "FULL": 2}


class ShelfAnalytics:
    """Owns the sqlite db; thread-safe via one lock around writes."""

    def __init__(
        self,
        db_path: str,
        bucket_seconds: int = 600,
        retention_hours: int = 24 * 14,
        stockout_seconds: int = 120,
        alert_cooldown_seconds: int = 30,
    ) -> None:
        self.bucket_seconds = int(bucket_seconds)
        self.retention_hours = int(retention_hours)
        self.stockout_seconds = int(stockout_seconds)
        self.cooldown_seconds = int(alert_cooldown_seconds)
        self._lock = threading.RLock()   # reentrant: heatmap() -> _latest_bucket()
        self._empty_since: dict[str, float] = {}   # slot_id -> ts entered EMPTY
        self._stockout_fired: set[str] = set()     # slots currently confirmed
        self._conn = self._connect(db_path)

    # ---------- schema ----------
    @staticmethod
    def _connect(db_path: str) -> sqlite3.Connection:
        if db_path:
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        # check_same_thread=False: writes come from the scan worker, reads from
        # Flask request threads; every access is serialized by self._lock, so a
        # shared connection is safe (one lock guards all execute/commit pairs).
        conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS shelf_counts(
                bucket INTEGER NOT NULL,
                slot_id TEXT NOT NULL,
                code TEXT NOT NULL,
                max_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (bucket, slot_id, code)
            );
            CREATE TABLE IF NOT EXISTS shelf_states(
                bucket INTEGER NOT NULL,
                slot_id TEXT NOT NULL,
                state TEXT NOT NULL,
                ticks INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (bucket, slot_id, state)
            );
            CREATE TABLE IF NOT EXISTS shelf_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                slot_id TEXT NOT NULL,
                kind TEXT NOT NULL,          -- occupancy | stockout | restock
                detail TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_counts_bucket ON shelf_counts(bucket);
            CREATE INDEX IF NOT EXISTS idx_events_ts ON shelf_events(ts);
            """
        )
        conn.commit()
        return conn

    # ---------- per-tick recording ----------
    def record(
        self,
        snapshots: list[Any],
        now: float | None = None,
    ) -> list[dict]:
        """Record a tick of slot snapshots.

        snapshots: list of SlotSnapshot (slots.py). Returns emitted events list
        ({"kind","slot_id","detail"}) for the SSE layer to forward.
        """
        now = now if now is not None else time.time()
        bucket = int(now // self.bucket_seconds) * self.bucket_seconds
        events: list[dict] = []
        with self._lock:
            for snap in snapshots:
                sid = snap.slot_id
                self._upsert_counts(bucket, sid, snap)
                self._upsert_state(bucket, sid, snap.state)
                events.extend(self._evaluate(sid, snap, now))
            self._prune(now)
        return events

    def _upsert_counts(self, bucket: int, sid: str, snap: Any) -> None:
        rows = list(snap.by_code.items()) + [("all", snap.count)]
        self._conn.executemany(
            """INSERT INTO shelf_counts(bucket, slot_id, code, max_count)
               VALUES(?,?,?,?)
               ON CONFLICT(bucket, slot_id, code)
               DO UPDATE SET max_count = MAX(max_count, excluded.max_count)""",
            [(bucket, sid, code, count) for code, count in rows],
        )
        self._conn.commit()

    def _upsert_state(self, bucket: int, sid: str, state: str) -> None:
        self._conn.execute(
            """INSERT INTO shelf_states(bucket, slot_id, state, ticks)
               VALUES(?,?,?,1)
               ON CONFLICT(bucket, slot_id, state)
               DO UPDATE SET ticks = ticks + 1""",
            (bucket, sid, state),
        )
        self._conn.commit()

    def _evaluate(self, sid: str, snap: Any, now: float) -> list[dict]:
        out: list[dict] = []
        if snap.state == "EMPTY":
            self._empty_since.setdefault(sid, now)
            elapsed = now - self._empty_since[sid]
            if elapsed >= self.stockout_seconds and sid not in self._stockout_fired:
                self._stockout_fired.add(sid)
                out.append(self._event("stockout", sid, {"empty_seconds": round(elapsed)}))
        else:
            since = self._empty_since.pop(sid, None)
            if sid in self._stockout_fired:
                self._stockout_fired.discard(sid)
                out.append(self._event("restock", sid, {
                    "empty_seconds": round(now - since) if since else None}))
        return out

    def _event(self, kind: str, sid: str, detail: dict) -> dict:
        d = json.dumps(detail, ensure_ascii=False)
        ts = int(time.time())
        self._conn.execute(
            "INSERT INTO shelf_events(ts, slot_id, kind, detail) VALUES(?,?,?,?)",
            (ts, sid, kind, d),
        )
        self._conn.commit()
        return {"kind": kind, "slot_id": sid, "detail": detail, "ts": ts}

    def _prune(self, now: float) -> None:
        cutoff = now - self.retention_hours * 3600
        self._conn.execute("DELETE FROM shelf_counts WHERE bucket < ?", (cutoff,))
        self._conn.execute("DELETE FROM shelf_states WHERE bucket < ?", (cutoff,))
        self._conn.execute("DELETE FROM shelf_events WHERE ts < ?", (cutoff,))
        self._conn.commit()

    # ---------- queries ----------
    def heatmap(self, buckets: int = 6) -> list[dict]:
        """Most recent `buckets` time windows: per slot per code peak counts
        + per-state tick proportions. Sorted bucket asc, slot, code."""
        latest = self._latest_bucket()
        start = latest - (buckets - 1) * self.bucket_seconds
        with self._lock:
            rows = self._conn.execute(
                """SELECT bucket, slot_id, code, max_count
                   FROM shelf_counts WHERE bucket BETWEEN ? AND ?
                   ORDER BY bucket ASC, slot_id ASC, code ASC""",
                (start, latest),
            ).fetchall()
            state_rows = self._conn.execute(
                """SELECT bucket, slot_id, state, ticks
                   FROM shelf_states WHERE bucket BETWEEN ? AND ?""",
                (start, latest),
            ).fetchall()
        states_by: dict[tuple[int, str], dict[str, int]] = {}
        for b, sid, st, tk in state_rows:
            states_by.setdefault((b, sid), {})[st] = tk

        out: list[dict] = []
        for b, sid, code, mc in rows:
            states = states_by.get((b, sid), {})
            majority = max(states, key=states.get) if states else None
            out.append({
                "bucket": b,
                "slot_id": sid,
                "code": code,
                "max_count": mc,
                "state": majority,
                "state_ticks": states,
            })
        return out

    def recent_events(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT ts, slot_id, kind, detail FROM shelf_events
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {"ts": int(ts), "slot_id": sid, "kind": kind,
             "detail": json.loads(d) if d else {}}
            for ts, sid, kind, d in reversed(rows)
        ]

    def _latest_bucket(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(bucket) FROM shelf_counts").fetchone()
        if not row or row[0] is None:
            return int(time.time() // self.bucket_seconds) * self.bucket_seconds
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()