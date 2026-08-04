"""Zone headcount, capacity/crowd, and equipment-occupancy tracking.

Persons are assigned to a zone by testing their body center (normalized) against
each zone polygon. Equipment occupancy requires a person to remain in the
equipment's zone >= EQUIPMENT_OCCUPANCY_SECONDS. Long occupation is flagged at
LONG_OCCUPATION_SECONDS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from pose import Pt, body_center, point_in_polygon, normalize_polygon, KeyPoints


@dataclass
class Zone:
    id: str
    name: str
    polygon: list[tuple[float, float]]   # normalized (x,y) vertices
    capacity: int = 0                     # 0 = unlimited


@dataclass
class Equipment:
    id: str
    name: str
    zone_id: str
    exercise_type: str = ""  # e.g. "squat", "deadlift" — used by ExerciseClassifier


@dataclass
class OccupancyRecord:
    tracker_id: str
    equipment_id: str
    start_ts: float
    last_seen_ts: float
    confirmed: bool = False   # True once >= occupancy_seconds

    def duration(self, now: float) -> float:
        return now - self.start_ts


class ZoneManager:
    def __init__(self, zones_cfg: list[dict], equipment_cfg: list[dict],
                 occupancy_seconds: int, long_occupation_seconds: int):
        self.zones: list[Zone] = []
        for z in zones_cfg:
            self.zones.append(Zone(
                id=str(z.get("id", "")),
                name=str(z.get("name", z.get("id", ""))),
                polygon=normalize_polygon(z.get("polygon", [])),
                capacity=int(z.get("capacity", 0)),
            ))
        self.equipment: list[Equipment] = [
            Equipment(id=str(e.get("id", "")),
                      name=str(e.get("name", e.get("id", ""))),
                      zone_id=str(e.get("zone_id", "")),
                      exercise_type=str(e.get("exercise_type", "")))
            for e in equipment_cfg
        ]
        self.equipment_by_zone: dict[str, list[Equipment]] = {}
        for eq in self.equipment:
            self.equipment_by_zone.setdefault(eq.zone_id, []).append(eq)

        self.occupancy_seconds = occupancy_seconds
        self.long_occupation_seconds = long_occupation_seconds

        # tracker_id → OccupancyRecord (one tracker per equipment at a time)
        self._active: dict[tuple[str, str], OccupancyRecord] = {}
        # last snapshot
        self.last_counts: dict[str, int] = {z.id: 0 for z in self.zones}
        self.last_total: int = 0

    @property
    def zone_exercise_map(self) -> dict[str, str]:
        """Build {zone_id: exercise_type} from equipment definitions.

        Used by ExerciseClassifier for zone-based exercise identification.
        If multiple equipment in same zone, first one wins.
        """
        mapping: dict[str, str] = {}
        for eq in self.equipment:
            if eq.exercise_type and eq.zone_id not in mapping:
                mapping[eq.zone_id] = eq.exercise_type
        return mapping

    def _zone_of(self, pt: Pt | None) -> Zone | None:
        if pt is None:
            return None
        for z in self.zones:
            if point_in_polygon(pt.x, pt.y, z.polygon):
                return z
        return None

    def update(self, persons: list[tuple[str, KeyPoints]], now: float,
               detect_persons: list[dict] | None = None,
               member_ids: dict[str, str] | None = None) -> dict:
        """persons: list of (tracker_id, keypoints). Returns snapshot dict.

        detect_persons: optional list of detection bbox dicts from a secondary
        detection model (e.g. yolov8n), each with {x, y, w, h, score,
        aspect_ratio}. When a person has keypoints but the body_center is
        outside all zones, the detection bbox center is tried as fallback.
        When a detection bbox has no matching keypoint person, it still
        contributes to zone headcount (supplementary counting for occluded
        people where keypoints are missing).

        member_ids: optional {tracker_id: member_id} mapping from face
        recognition. Included in equipment_occupy and equipment_release events.

        Snapshot:
          total, counts: {zone_id: n}, crowded: [zone_id...],
          equipment: [{equipment_id, occupied, tracker_id, member_id, duration_s, long}],
          events: [{type, ...}]  # occupancy start/confirm/long/release
        """
        centers: dict[str, Pt | None] = {}
        zone_assignment: dict[str, Zone | None] = {}
        counts: dict[str, int] = {z.id: 0 for z in self.zones}
        total = 0

        for tid, kp in persons:
            bc = body_center(kp)
            centers[tid] = bc
            z = self._zone_of(bc)
            zone_assignment[tid] = z
            if z is not None:
                counts[z.id] += 1
                total += 1

        # Supplementary zone membership from detection bboxes.
        # For persons already assigned via keypoints, try bbox as fallback
        # if keypoint-based assignment failed (body_center outside all zones).
        # For unmatched bboxes (no corresponding keypoint person), count them
        # as additional zone members (handles occluded people).
        if detect_persons:
            # Collect keypoint body centers for nearest-match
            kp_centers = [body_center(kp) for _, kp in persons]
            matched_bbox_indices: set[int] = set()

            # First pass: use bbox centers as fallback for unassigned persons
            for i, (tid, kp) in enumerate(persons):
                if zone_assignment.get(tid) is not None:
                    continue  # already assigned via keypoints
                bc = kp_centers[i] if i < len(kp_centers) else None
                if bc is None:
                    continue
                # Find closest detection bbox
                best_j = -1
                best_dist = float("inf")
                for j, dp in enumerate(detect_persons):
                    dist = ((dp["x"] - bc.x) ** 2 + (dp["y"] - bc.y) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
                if best_j >= 0 and best_dist < 0.05:  # close enough match
                    dp = detect_persons[best_j]
                    z = self._zone_of(Pt(dp["x"], dp["y"], 1.0))
                    if z is not None:
                        zone_assignment[tid] = z
                        counts[z.id] += 1
                        total += 1
                        matched_bbox_indices.add(best_j)

            # Second pass: count unmatched bboxes as additional persons
            for j, dp in enumerate(detect_persons):
                if j in matched_bbox_indices:
                    continue
                z = self._zone_of(Pt(dp["x"], dp["y"], 1.0))
                if z is not None:
                    counts[z.id] += 1
                    total += 1

        crowded = [z.id for z in self.zones if z.capacity and counts[z.id] > z.capacity]

        # ---- equipment occupancy ----
        events: list[dict] = []
        seen_keys: set[tuple[str, str]] = set()

        for tid, kp in persons:
            z = zone_assignment.get(tid)
            if z is None:
                continue
            for eq in self.equipment_by_zone.get(z.id, []):
                key = (eq.id, tid)
                seen_keys.add(key)
                rec = self._active.get(key)
                if rec is None:
                    rec = OccupancyRecord(tracker_id=tid, equipment_id=eq.id,
                                          start_ts=now, last_seen_ts=now)
                    self._active[key] = rec
                    events.append({"type": "equipment_enter",
                                   "equipment_id": eq.id, "tracker_id": tid})
                else:
                    rec.last_seen_ts = now
                dur = rec.duration(now)
                if not rec.confirmed and dur >= self.occupancy_seconds:
                    rec.confirmed = True
                    evt = {"type": "equipment_occupy",
                           "equipment_id": eq.id, "tracker_id": tid,
                           "duration_s": round(dur, 1)}
                    if member_ids and tid in member_ids:
                        evt["member_id"] = member_ids[tid]
                    events.append(evt)
                if dur >= self.long_occupation_seconds and not getattr(rec, "_long_flagged", False):
                    rec._long_flagged = True  # type: ignore[attr-defined]
                    events.append({"type": "long_occupation",
                                   "equipment_id": eq.id, "tracker_id": tid,
                                   "duration_s": round(dur, 1)})

        # release records whose tracker left the equipment zone
        to_release = [k for k in self._active if k not in seen_keys]
        for k in to_release:
            rec = self._active.pop(k)
            events.append({"type": "equipment_release",
                           "equipment_id": rec.equipment_id,
                           "tracker_id": rec.tracker_id,
                           "duration_s": round(rec.duration(now), 1),
                           "member_id": (member_ids or {}).get(rec.tracker_id)})

        equipment_snap = []
        for eq in self.equipment:
            recs = [r for (eid, _), r in self._active.items() if eid == eq.id and r.confirmed]
            for r in recs:
                equipment_snap.append({
                    "equipment_id": eq.id,
                    "name": eq.name,
                    "occupied": True,
                    "tracker_id": r.tracker_id,
                    "member_id": (member_ids or {}).get(r.tracker_id),
                    "duration_s": round(r.duration(now), 1),
                    "long": r.duration(now) >= self.long_occupation_seconds,
                })
            if not recs:
                equipment_snap.append({
                    "equipment_id": eq.id, "name": eq.name,
                    "occupied": False, "tracker_id": None,
                    "member_id": None,
                    "duration_s": 0.0, "long": False,
                })

        self.last_counts = counts
        self.last_total = total

        return {
            "total": total,
            "counts": counts,
            "crowded": crowded,
            "equipment": equipment_snap,
            "events": events,
        }
