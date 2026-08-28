"""Unit tests for trajectory.py — zone-visit tracking and session reports.

Covers: visit debounce (min dwell), rejoin merging, session close timing,
per-visit counter deltas, split continuation flag, overlay trails, active
snapshot, and JSONL persistence with retention pruning.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pose import Pt  # noqa: E402
from trajectory import TrajectoryStore, TrajectoryTracker  # noqa: E402


# ---- helpers ----

def make_kp(cx: float = 0.5, cy: float = 0.5) -> list:
    """Minimal 17-keypoint set whose body_center lands at (cx, cy).

    Shoulders at indices 5/6, hips at 11/12 (COCO-17), mirroring
    pose.body_center's midpoint-of-midpoints computation.
    """
    kp: list = [None] * 17
    kp[5] = Pt(cx - 0.05, cy - 0.10, 1.0)
    kp[6] = Pt(cx + 0.05, cy - 0.10, 1.0)
    kp[11] = Pt(cx - 0.05, cy + 0.10, 1.0)
    kp[12] = Pt(cx + 0.05, cy + 0.10, 1.0)
    return kp


def snap(exercise: str = "squat", reps: int = 0, sets: int = 0,
         bad: int = 0) -> dict:
    """Counter snapshot stub matching counter.BaseCounter.snapshot()."""
    return {"exercise": exercise, "reps": reps, "sets": sets,
            "bad_reps": bad}


ZONE_NAMES = {
    "power_rack": "Power Rack",
    "cardio": "Cardio Area",
    "walkway": "Front Walkway",
}
EXERCISE_MAP = {"power_rack": "squat", "cardio": ""}


def make_tracker(**overrides) -> TrajectoryTracker:
    params = dict(
        zone_names=ZONE_NAMES,
        zone_exercise_map=EXERCISE_MAP,
        min_visit_seconds=3.0,
        rejoin_seconds=30.0,
        session_close_seconds=10.0,
        trail_points=60,
        store=None,
    )
    params.update(overrides)
    return TrajectoryTracker(**params)


# ---- visit debounce ----

class TestVisitDebounce:
    def test_short_pass_through_does_not_confirm_visit(self):
        # Arrange
        t = make_tracker()
        # Act: p0 in power_rack for 2s (< min_visit 3s), then session closes
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap()}, {}, now=100.0)
        events = t.update([("p0", make_kp())], {"p0": "power_rack"},
                          {"p0": snap()}, {}, now=102.0)
        summaries = t.update([], {}, {}, {}, now=112.0)
        # Assert: no visit_confirmed; summary has no visits, all transit
        assert all(e["type"] != "visit_confirmed" for e in events)
        s = summaries[0]
        assert s["type"] == "session_summary"
        assert s["visits"] == []
        assert s["transit_s"] > 0

    def test_visit_confirms_after_min_dwell(self):
        # Arrange
        t = make_tracker()
        # Act: dwell 4s >= 3s
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap()}, {}, now=100.0)
        events = t.update([("p0", make_kp())], {"p0": "power_rack"},
                          {"p0": snap()}, {}, now=104.0)
        # Assert
        confirms = [e for e in events if e["type"] == "visit_confirmed"]
        assert len(confirms) == 1
        assert confirms[0]["zone_id"] == "power_rack"
        assert confirms[0]["exercise"] == "squat"

    def test_visit_confirmed_fires_exactly_once(self):
        # Arrange
        t = make_tracker()
        # Act: keep dwelling many frames
        all_events = []
        for now in (100.0, 101.0, 102.0, 103.0, 104.0, 110.0, 120.0):
            all_events += t.update([("p0", make_kp())], {"p0": "power_rack"},
                                    {"p0": snap()}, {}, now=now)
        # Assert
        confirms = [e for e in all_events if e["type"] == "visit_confirmed"]
        assert len(confirms) == 1

    def test_zone_without_equipment_confirms_with_null_exercise(self):
        # Arrange
        t = make_tracker()
        # Act: cardio has no exercise in EXERCISE_MAP
        events = t.update([("p0", make_kp())], {"p0": "cardio"},
                          {"p0": snap()}, {}, now=100.0)
        events += t.update([("p0", make_kp())], {"p0": "cardio"},
                           {"p0": snap()}, {}, now=104.0)
        # Assert
        confirms = [e for e in events if e["type"] == "visit_confirmed"]
        assert len(confirms) == 1
        assert confirms[0]["exercise"] is None


# ---- rejoin merging ----

class TestRejoinMerge:
    def test_quick_return_merges_into_single_visit(self):
        # Arrange
        t = make_tracker()
        # Act: rack 100->110 (confirmed), walkway 110->111 (transit),
        # back to rack 111->130 (confirmed) — gap 1s < rejoin 30s
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap()}, {}, now=100.0)
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap()}, {}, now=110.0)   # confirmed here
        t.update([("p0", make_kp())], {"p0": "walkway"},
                 {"p0": snap()}, {}, now=111.0)   # leaves
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap()}, {}, now=112.0)   # back after 1s
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap()}, {}, now=130.0)
        summaries = t.update([], {}, {}, {}, now=141.0)
        # Assert: one merged rack visit, not two
        s = summaries[0]
        rack_visits = [v for v in s["visits"] if v["zone_id"] == "power_rack"]
        assert len(rack_visits) == 1
        assert rack_visits[0]["dwell_s"] == pytest.approx(30.0, abs=1.5)

    def test_late_return_creates_separate_visit(self):
        # Arrange
        t = make_tracker(rejoin_seconds=10.0)
        # Act: rack 100->110, away 10s (gap == rejoin, not merged), rack again
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap()}, {}, now=100.0)
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap()}, {}, now=110.0)
        t.update([("p0", make_kp())], {"p0": "walkway"},
                 {"p0": snap()}, {}, now=125.0)
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap()}, {}, now=135.0)
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap()}, {}, now=145.0)  # 2nd visit dwell 10s, confirmed
        summaries = t.update([], {}, {}, {}, now=156.0)
        # Assert: two separate visits
        s = summaries[0]
        rack_visits = [v for v in s["visits"] if v["zone_id"] == "power_rack"]
        assert len(rack_visits) == 2


# ---- session lifecycle ----

class TestSessionLifecycle:
    def test_session_closes_after_unseen_timeout(self):
        # Arrange
        t = make_tracker(session_close_seconds=10.0)
        t.update([("p0", make_kp())], {"p0": "cardio"},
                 {"p0": snap()}, {}, now=100.0)
        t.update([("p0", make_kp())], {"p0": "cardio"},
                 {"p0": snap()}, {}, now=101.0)
        # Act: at 112 (>10s after last seen) the close fires
        events = t.update([], {}, {}, {}, now=112.0)
        # Assert
        assert len(events) == 1
        assert events[0]["type"] == "session_summary"
        assert events[0]["tracker_id"] == "p0"
        assert events[0]["duration_s"] == pytest.approx(11.0, abs=1.0)

    def test_reappear_within_close_window_continues_session(self):
        # Arrange
        t = make_tracker(session_close_seconds=10.0)
        t.update([("p0", make_kp())], {"p0": "cardio"},
                 {"p0": snap()}, {}, now=100.0)
        # Act: gone 8s (< close window), reappears
        events = t.update([], {}, {}, {}, now=108.0)
        events += t.update([("p0", make_kp())], {"p0": "cardio"},
                           {"p0": snap()}, {}, now=109.0)
        # Assert: no summary fired; session continues
        assert all(e["type"] != "session_summary" for e in events)
        assert t.active_snapshot(now=109.0)[0]["tracker_id"] == "p0"

    def test_member_id_included_in_summary(self):
        # Arrange
        t = make_tracker()
        t.update([("p0", make_kp())], {"p0": "cardio"},
                 {"p0": snap()}, {"p0": "member_42"}, now=100.0)
        # Act
        events = t.update([], {}, {}, {}, now=111.0)
        # Assert
        assert events[0]["member_id"] == "member_42"

    def test_summary_contains_zone_names_and_visit_details(self):
        # Arrange
        t = make_tracker()
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap()}, {}, now=100.0)
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap()}, {}, now=110.0)  # confirmed
        # Act
        events = t.update([], {}, {}, {}, now=121.0)
        s = events[0]
        # Assert
        v = s["visits"][0]
        assert v["zone_id"] == "power_rack"
        assert v["zone_name"] == "Power Rack"
        assert v["enter_ts"] == pytest.approx(100.0)
        assert v["dwell_s"] == pytest.approx(10.0, abs=1.0)


# ---- counter attribution ----

class TestCounterAttribution:
    def test_rep_deltas_attributed_to_visit(self):
        # Arrange
        t = make_tracker()
        # Act: squat in rack; visit confirmed at ~103, baseline captured at
        # the confirming frame, 8 reps by 120
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap("squat", 0)}, {}, now=100.0)
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap("squat", 0)}, {}, now=104.0)  # confirmed+baseline
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap("squat", 8, 2, 1)}, {}, now=120.0)
        t.update([("p0", make_kp())], {"p0": "walkway"},
                 {"p0": snap()}, {}, now=121.0)            # exit closes visit
        closed = t.update([], {}, {}, {}, now=132.0)       # session close
        # Assert
        s = closed[0]
        v = [x for x in s["visits"] if x["zone_id"] == "power_rack"][0]
        assert v["reps"] == 8
        assert v["sets"] == 2
        assert v["bad_reps"] == 1
        assert v["exercise"] == "squat"

    def test_counter_rebind_mid_session_resets_baseline(self):
        # Arrange: app rebuilds the counter when the zone's exercise differs,
        # so the snapshot exercise changes mid-visit. Baseline must come from
        # the counter matching the visit's exercise, not a stale one.
        t = make_tracker()
        # Act
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap("unknown", 5)}, {}, now=100.0)
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap("unknown", 5)}, {}, now=104.0)  # confirmed
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap("squat", 0)}, {}, now=105.0)    # rebound counter
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap("squat", 6)}, {}, now=120.0)    # 6 squats done
        closed = t.update([], {}, {}, {}, now=131.0)
        # Assert: 6 reps attributed (not 6-5, not 6+5)
        v = [x for x in closed[0]["visits"] if x["zone_id"] == "power_rack"][0]
        assert v["reps"] == 6


# ---- split continuation ----

class TestSplitContinuation:
    def test_new_session_near_recently_closed_flags_continuation(self):
        # Arrange
        t = make_tracker(session_close_seconds=10.0)
        t.update([("p0", make_kp(0.5, 0.5))], {"p0": "cardio"},
                 {"p0": snap()}, {}, now=100.0)
        t.update([], {}, {}, {}, now=111.0)  # p0 session closed
        # Act: p1 appears 2s later at the same spot
        t.update([("p1", make_kp(0.5, 0.5))], {"p1": "cardio"},
                 {"p1": snap()}, {}, now=113.0)
        closed = t.update([], {}, {}, {}, now=124.0)
        # Assert
        assert closed[0]["tracker_id"] == "p1"
        assert closed[0]["possible_continuation"] is True

    def test_new_session_far_away_no_flag(self):
        # Arrange
        t = make_tracker(session_close_seconds=10.0)
        t.update([("p0", make_kp(0.5, 0.5))], {"p0": "cardio"},
                 {"p0": snap()}, {}, now=100.0)
        t.update([], {}, {}, {}, now=111.0)
        # Act: p1 appears at a distant position
        t.update([("p1", make_kp(0.1, 0.1))], {"p1": "cardio"},
                 {"p1": snap()}, {}, now=113.0)
        closed = t.update([], {}, {}, {}, now=124.0)
        # Assert
        assert closed[0]["possible_continuation"] is False


# ---- overlay trails ----

class TestTrails:
    def test_trail_ring_buffer_bounded(self):
        # Arrange
        t = make_tracker(trail_points=5)
        # Act: 10 frames of movement
        for i in range(10):
            t.update([("p0", make_kp(0.1 * i, 0.5))], {"p0": "cardio"},
                     {"p0": snap()}, {}, now=100.0 + i)
        # Assert
        trails = t.trails()
        assert len(trails["p0"]) == 5
        assert trails["p0"][-1] == (pytest.approx(0.9), pytest.approx(0.5))

    def test_trail_cleared_when_session_closes(self):
        # Arrange
        t = make_tracker()
        t.update([("p0", make_kp())], {"p0": "cardio"},
                 {"p0": snap()}, {}, now=100.0)
        # Act
        t.update([], {}, {}, {}, now=111.0)
        # Assert
        assert "p0" not in t.trails()


# ---- active snapshot ----

class TestActiveSnapshot:
    def test_active_snapshot_reports_current_zone_and_dwell(self):
        # Arrange
        t = make_tracker()
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap()}, {}, now=100.0)
        # Act
        t.update([("p0", make_kp())], {"p0": "power_rack"},
                 {"p0": snap()}, {}, now=105.0)
        active = t.active_snapshot(now=105.0)
        # Assert
        assert len(active) == 1
        a = active[0]
        assert a["tracker_id"] == "p0"
        assert a["current_zone"] == "power_rack"
        assert a["zone_dwell_s"] == pytest.approx(5.0, abs=0.1)
        assert a["member_id"] is None


# ---- persistence ----

class TestTrajectoryStore:
    def test_append_and_load_roundtrip(self, tmp_path):
        # Arrange
        store = TrajectoryStore(str(tmp_path), retention_days=30,
                                link_member_identity=False)
        payload = {
            "type": "session_summary", "session_id": "p0-100",
            "tracker_id": "p0", "member_id": "secret_member",
            "duration_s": 60.0, "visits": [], "transit_s": 10.0,
        }
        # Act
        store.append(payload, now=100.0)
        loaded = store.load_sessions(limit=10)
        # Assert
        assert len(loaded) == 1
        assert loaded[0]["session_id"] == "p0-100"
        # privacy: member_id stripped when link disabled
        assert loaded[0]["member_id"] is None

    def test_member_id_kept_when_link_enabled(self, tmp_path):
        # Arrange
        store = TrajectoryStore(str(tmp_path), link_member_identity=True)
        payload = {"type": "session_summary", "session_id": "p0-100",
                   "tracker_id": "p0", "member_id": "m1",
                   "visits": [], "transit_s": 0.0, "duration_s": 1.0}
        # Act
        store.append(payload, now=100.0)
        loaded = store.load_sessions()
        # Assert
        assert loaded[0]["member_id"] == "m1"

    def test_load_filters_by_member_id(self, tmp_path):
        # Arrange
        store = TrajectoryStore(str(tmp_path), link_member_identity=True)
        for i, mid in enumerate(("alice", "bob", "alice")):
            store.append({"type": "session_summary",
                          "session_id": f"p{i}-10{i}",
                          "tracker_id": f"p{i}", "member_id": mid,
                          "visits": [], "transit_s": 0.0, "duration_s": 1.0},
                         now=100.0 + i)
        # Act
        loaded = store.load_sessions(member_id="alice")
        # Assert
        assert len(loaded) == 2
        assert all(s["member_id"] == "alice" for s in loaded)

    def test_get_session_by_id(self, tmp_path):
        # Arrange
        store = TrajectoryStore(str(tmp_path))
        store.append({"type": "session_summary", "session_id": "p0-100",
                      "tracker_id": "p0", "member_id": None,
                      "visits": [], "transit_s": 0.0, "duration_s": 1.0},
                     now=100.0)
        # Act
        found = store.get_session("p0-100")
        missing = store.get_session("nope-1")
        # Assert
        assert found is not None and found["session_id"] == "p0-100"
        assert missing is None

    def test_prune_removes_old_files(self, tmp_path):
        # Arrange
        store = TrajectoryStore(str(tmp_path), retention_days=7)
        store.append({"type": "session_summary", "session_id": "old-1",
                      "tracker_id": "p0", "member_id": None,
                      "visits": [], "transit_s": 0.0, "duration_s": 1.0},
                     now=time.time() - 30 * 86400)  # 30 days ago
        store.append({"type": "session_summary", "session_id": "new-1",
                      "tracker_id": "p1", "member_id": None,
                      "visits": [], "transit_s": 0.0, "duration_s": 1.0},
                     now=time.time())
        # Act
        store.prune()
        # Assert
        loaded = store.load_sessions()
        assert [s["session_id"] for s in loaded] == ["new-1"]

    def test_tracker_writes_summaries_to_store(self, tmp_path):
        # Arrange
        store = TrajectoryStore(str(tmp_path))
        t = make_tracker(store=store)
        # Act
        t.update([("p0", make_kp())], {"p0": "cardio"},
                 {"p0": snap()}, {}, now=100.0)
        t.update([], {}, {}, {}, now=111.0)
        # Assert
        loaded = store.load_sessions()
        assert len(loaded) == 1
        assert loaded[0]["tracker_id"] == "p0"

    def test_load_newest_first_and_limit(self, tmp_path):
        # Arrange
        store = TrajectoryStore(str(tmp_path), link_member_identity=True)
        for i in range(5):
            store.append({"type": "session_summary", "session_id": f"s{i}",
                          "tracker_id": f"p{i}", "member_id": None,
                          "visits": [], "transit_s": 0.0, "duration_s": 1.0},
                         now=100.0 + i)
        # Act
        loaded = store.load_sessions(limit=3)
        # Assert
        assert [s["session_id"] for s in loaded] == ["s4", "s3", "s2"]
