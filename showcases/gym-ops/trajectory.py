"""Semantic zone-visit trajectories.

Aggregates per-frame tracker zone assignments into durable "visits" (a person
dwelling in one zone past a debounce threshold) and per-tracker sessions
(continuous presence with close timeout). Emits two event types consumed by
app.py:

  visit_confirmed  — dwell crossed min_visit_seconds; app.py may rebuild the
                     person's counter for the zone's exercise
  session_summary  — person left (close timeout elapsed); carries the full
                     visit list with per-visit counter deltas, plus a
                     possible_continuation flag when a tracker ID split likely
                     reused the same person.

Persisted summaries go through TrajectoryStore (JSONL, daily files, retention
pruning, optional member-identity stripping).
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field

from pose import Pt, body_center

# A NEW session is flagged possible_continuation=True when it starts this soon
# after a recently closed session at a nearby last position.
_CONTINUATION_GAP_SECONDS = 5.0
_CONTINUATION_MAX_DIST = 0.2


@dataclass
class ZoneVisit:
    """One continuous (rejoin-merged) stay in a zone."""

    zone_id: str
    enter_ts: float
    exit_ts: float | None = None          # None while ongoing
    confirmed: bool = False               # dwell crossed min_visit_seconds
    exercise: str | None = None           # zone's exercise_type, "" -> None
    # Counter baseline captured lazily AFTER confirmation, from the first
    # snapshot whose exercise matches self.exercise. Deltas at exit are
    # computed against it (handles app.py rebuilding counters on zone change).
    entry_counter: dict | None = None
    exit_counter: dict | None = None

    def dwell(self, now: float) -> float:
        end = self.exit_ts if self.exit_ts is not None else now
        return end - self.enter_ts

    def to_dict(self) -> dict:
        reps = sets = bad = 0
        if (self.entry_counter is not None and self.exit_counter is not None
                and self.exit_counter.get("exercise")
                == self.entry_counter.get("exercise")):
            reps = max(0, self.exit_counter.get("reps", 0)
                       - self.entry_counter.get("reps", 0))
            sets = max(0, self.exit_counter.get("sets", 0)
                       - self.entry_counter.get("sets", 0))
            bad = max(0, self.exit_counter.get("bad_reps", 0)
                      - self.entry_counter.get("bad_reps", 0))
        exit_ts = self.exit_ts if self.exit_ts is not None else self.enter_ts
        return {
            "zone_id": self.zone_id,
            "zone_name": None,  # filled by TrajectoryTracker
            "enter_ts": round(self.enter_ts, 2),
            "exit_ts": round(exit_ts, 2),
            "dwell_s": round(exit_ts - self.enter_ts, 1),
            "exercise": self.exercise,
            "reps": reps,
            "sets": sets,
            "bad_reps": bad,
        }


@dataclass
class Session:
    tracker_id: str
    first_seen: float
    last_seen: float
    member_id: str | None = None
    last_pos: tuple[float, float] | None = None
    possible_continuation: bool = False
    current: ZoneVisit | None = None        # ongoing visit
    pending: list[ZoneVisit] = field(default_factory=list)  # exited, rejoin window open
    visits: list[ZoneVisit] = field(default_factory=list)   # finalized


class TrajectoryTracker:
    """Per-frame update; returns visit_confirmed / session_summary events."""

    def __init__(self, *, zone_names: dict[str, str],
                 zone_exercise_map: dict[str, str],
                 min_visit_seconds: float = 3.0,
                 rejoin_seconds: float = 30.0,
                 session_close_seconds: float = 10.0,
                 trail_points: int = 60,
                 store: "TrajectoryStore | None" = None):
        self.zone_names = zone_names
        self.zone_exercise_map = zone_exercise_map
        self.min_visit_seconds = min_visit_seconds
        self.rejoin_seconds = rejoin_seconds
        self.session_close_seconds = session_close_seconds
        self.trail_points = trail_points
        self.store = store
        self.sessions: dict[str, Session] = {}
        self._trails: dict[str, deque] = {}
        # Recently closed sessions: (close_ts, last_pos) for continuation flag.
        self._recent_closes: deque[tuple[float, tuple[float, float]]] = \
            deque(maxlen=16)

    # ---- main entry, called once per inference frame ----

    def update(self, persons: list[tuple[str, list]],
               assignments: dict[str, str | None],
               counter_snapshots: dict[str, dict],
               member_ids: dict[str, str],
               now: float) -> list[dict]:
        events: list[dict] = []
        present = {tid for tid, _ in persons}

        # 1. Close sessions unseen past the timeout. A person present in THIS
        # frame is never closed (the check runs before ingest, so exclude them).
        for tid in [t for t, s in self.sessions.items()
                    if t not in present
                    and now - s.last_seen >= self.session_close_seconds]:
            events.append(self._close_session(tid, now))

        # 2. Expire rejoin windows on live sessions.
        for s in self.sessions.values():
            still_pending = []
            for v in s.pending:
                if now - v.exit_ts < self.rejoin_seconds:
                    still_pending.append(v)
                elif v.confirmed:
                    s.visits.append(v)
            s.pending = still_pending

        # 3. Ingest this frame's persons.
        for tid, kp in persons:
            is_new = tid not in self.sessions
            session = self.sessions.get(tid)
            if session is None:
                session = self._new_session(tid, now)
            session.last_seen = now
            mid = (member_ids or {}).get(tid)
            if mid:
                session.member_id = mid

            center = body_center(kp) if kp else None
            if center is not None:
                session.last_pos = (center.x, center.y)
                self._trails[tid].append((center.x, center.y))
                if is_new:
                    # Tracker-ID split heuristic: first position near a
                    # recently closed session's last position in time & space.
                    self.check_continuation(tid, now)

            zone_id = assignments.get(tid) if assignments else None
            snap = counter_snapshots.get(tid) if counter_snapshots else None
            self._update_session(session, zone_id, snap, now, events)

        return events

    # ---- per-session state machine ----

    def _new_session(self, tid: str, now: float) -> Session:
        session = Session(tracker_id=tid, first_seen=now, last_seen=now)
        self.sessions[tid] = session
        self._trails[tid] = deque(maxlen=self.trail_points)
        return session

    def _update_session(self, session: Session, zone_id: str | None,
                        snap: dict | None, now: float,
                        events: list[dict]) -> None:
        if zone_id != (session.current.zone_id if session.current else None):
            if session.current is not None:
                session.current.exit_ts = now
                if session.current.confirmed:
                    session.pending.append(session.current)
                session.current = None
            if zone_id is not None:
                # Rejoin: same zone re-entered within the window resumes the
                # visit (strict <; a gap == rejoin_seconds starts a new visit).
                resumed = None
                for i, v in enumerate(session.pending):
                    if v.zone_id == zone_id and now - v.exit_ts < self.rejoin_seconds:
                        resumed = session.pending.pop(i)
                        break
                if resumed is not None:
                    resumed.exit_ts = None
                    session.current = resumed
                else:
                    ex = self.zone_exercise_map.get(zone_id) or None
                    session.current = ZoneVisit(zone_id=zone_id, enter_ts=now,
                                                exercise=ex)

        cur = session.current
        if cur is None:
            return

        # Confirmation: dwell crossed the debounce threshold.
        if not cur.confirmed and cur.dwell(now) >= self.min_visit_seconds:
            cur.confirmed = True
            events.append({
                "type": "visit_confirmed",
                "tracker_id": session.tracker_id,
                "zone_id": cur.zone_id,
                "exercise": cur.exercise,
                "ts": now,
            })

        # Lazy counter baseline: first post-confirmation snapshot whose
        # exercise matches the visit's (app.py rebuilds counters on zone
        # change, so the confirming frame may still carry the old exercise).
        if cur.confirmed and snap is not None:
            if cur.entry_counter is None:
                if cur.exercise is None or snap.get("exercise") == cur.exercise:
                    cur.entry_counter = dict(snap)
            elif snap.get("exercise") == cur.entry_counter.get("exercise"):
                cur.exit_counter = dict(snap)

    def _close_session(self, tid: str, now: float) -> dict:
        s = self.sessions.pop(tid)
        # Finalize every visit; unconfirmed ones are dropped (counted as
        # transit time in the summary).
        all_visits = list(s.visits) + [v for v in s.pending if v.confirmed]
        if s.current is not None:
            s.current.exit_ts = s.last_seen
            if s.current.confirmed:
                all_visits.append(s.current)
        all_visits.sort(key=lambda v: v.enter_ts)

        confirmed_dwell = sum(v.dwell(s.last_seen) for v in all_visits)
        # Session effectively lasted until the close timeout elapsed.
        duration = (s.last_seen + self.session_close_seconds) - s.first_seen
        visits = []
        for v in all_visits:
            d = v.to_dict()
            d["zone_name"] = self.zone_names.get(v.zone_id)
            visits.append(d)

        payload = {
            "type": "session_summary",
            "session_id": f"{s.tracker_id}-{int(s.first_seen)}",
            "tracker_id": s.tracker_id,
            "member_id": s.member_id,
            "first_seen": round(s.first_seen, 2),
            "last_seen": round(s.last_seen, 2),
            "duration_s": round(max(0.0, duration), 1),
            "possible_continuation": s.possible_continuation,
            "visits": visits,
            "transit_s": round(max(0.0, duration - confirmed_dwell), 1),
        }

        if s.last_pos is not None:
            self._recent_closes.append(
                (s.last_seen + self.session_close_seconds, s.last_pos))
        self._trails.pop(tid, None)
        if self.store is not None:
            self.store.append(payload, now=now)
        return payload

    # ---- queries ----

    def active_snapshot(self, now: float) -> list[dict]:
        out = []
        for s in sorted(self.sessions.values(), key=lambda x: x.tracker_id):
            cur = s.current
            out.append({
                "tracker_id": s.tracker_id,
                "member_id": s.member_id,
                "current_zone": cur.zone_id if cur else None,
                "zone_dwell_s": round(cur.dwell(now), 1) if cur else 0.0,
                "confirmed": bool(cur and cur.confirmed),
                "since": round(s.first_seen, 2),
            })
        return out

    def trails_snapshot(self) -> dict[str, list[tuple[float, float]]]:
        return {tid: list(pts) for tid, pts in self._trails.items()}

    def trails(self) -> dict[str, list[tuple[float, float]]]:
        """Alias of trails_snapshot — the raw pixel-space trail per tracker."""
        return self.trails_snapshot()

    # The continuation flag is decided when a NEW session records its first
    # position: near a recently closed session's last position in time & space.
    def check_continuation(self, tid: str, now: float) -> None:
        s = self.sessions.get(tid)
        if s is None or s.last_pos is None or s.possible_continuation:
            return
        for close_ts, pos in self._recent_closes:
            if abs(now - close_ts) > _CONTINUATION_GAP_SECONDS:
                continue
            dx = s.last_pos[0] - pos[0]
            dy = s.last_pos[1] - pos[1]
            if (dx * dx + dy * dy) ** 0.5 <= _CONTINUATION_MAX_DIST:
                s.possible_continuation = True
                return


class TrajectoryStore:
    """JSONL persistence: one file per local day, retention-pruned."""

    def __init__(self, dir_path: str, retention_days: int = 30,
                 link_member_identity: bool = False):
        self.dir_path = dir_path
        self.retention_days = retention_days
        self.link_member_identity = link_member_identity

    # ---- writing ----

    def append(self, payload: dict, now: float) -> None:
        record = dict(payload)
        record["_ts"] = now
        if not self.link_member_identity:
            record["member_id"] = None
        try:
            os.makedirs(self.dir_path, exist_ok=True)
            path = self._file_for(now)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[trajectory] WARN: persist failed: {e}")

    # ---- reading ----

    def load_sessions(self, limit: int = 50, member_id: str | None = None,
                      now: float | None = None) -> list[dict]:
        records = self._read_all()
        if member_id is not None:
            records = [r for r in records if r.get("member_id") == member_id]
        records.sort(key=lambda r: r.get("_ts", 0), reverse=True)
        out = []
        for r in records[:limit]:
            r = dict(r)
            r.pop("_ts", None)
            out.append(r)
        return out

    def get_session(self, session_id: str,
                    now: float | None = None) -> dict | None:
        for r in self._read_all():
            if r.get("session_id") == session_id:
                r = dict(r)
                r.pop("_ts", None)
                return r
        return None

    def prune(self, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        cutoff = now - self.retention_days * 86400
        try:
            names = os.listdir(self.dir_path)
        except OSError:
            return
        for name in names:
            date_str = self._date_from_name(name)
            if date_str is None:
                continue
            try:
                file_ts = time.mktime(time.strptime(date_str, "%Y%m%d"))
            except ValueError:
                continue
            if file_ts < cutoff:
                try:
                    os.remove(os.path.join(self.dir_path, name))
                except OSError:
                    pass

    # ---- internals ----

    def _file_for(self, now: float) -> str:
        return os.path.join(
            self.dir_path,
            f"trajectory-{time.strftime('%Y%m%d', time.localtime(now))}.jsonl")

    @staticmethod
    def _date_from_name(name: str) -> str | None:
        if not (name.startswith("trajectory-") and name.endswith(".jsonl")):
            return None
        return name[len("trajectory-"):-len(".jsonl")]

    def _read_all(self) -> list[dict]:
        try:
            names = sorted(os.listdir(self.dir_path))
        except OSError:
            return []
        records: list[dict] = []
        for name in names:
            if self._date_from_name(name) is None:
                continue
            path = os.path.join(self.dir_path, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue
        return records
