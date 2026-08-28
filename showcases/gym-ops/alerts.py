"""Safety alerts: cooldown-gated Event Bus publishing + fall heuristic.

Alert types are deduplicated by a per-type cooldown to avoid spamming the bus.
Fall detection is heuristic-only (no depth model): a person whose torso is
horizontal and hips are low in frame, sustained for >= FALL_CANDIDATE_SECONDS,
raises a fall alert. After-hours presence, forbidden-zone entry, and long
static (no keypoint displacement) are also detected here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from pose import KeyPoints, body_center, hip_near_bottom, torso_horizontal


# Topics (must match app.yaml permissions.events.publish)
TOPIC_OCCUPANCY = "gym/occupancy"
TOPIC_EQUIPMENT = "gym/equipment"
TOPIC_POSE = "gym/pose"
TOPIC_ALERTS = "gym/alerts"
TOPIC_HEALTH = "gym/health"
TOPIC_IDENTITY = "gym/identity"
TOPIC_TRAJECTORY = "gym/trajectory"


@dataclass
class _TrackerStatic:
    last_pos: tuple[float, float] | None = None
    static_since: float | None = None
    horizontal_since: float | None = None
    fall_alerted: bool = False
    forbidden_alerted: bool = False
    long_static_alerted: bool = False


class AlertManager:
    def __init__(self, event_client, cooldown_seconds: int,
                 fall_candidate_seconds: int, long_static_seconds: int):
        self.events = event_client
        self.cooldown = float(cooldown_seconds)
        self.fall_candidate_seconds = float(fall_candidate_seconds)
        self.long_static_seconds = float(long_static_seconds)
        # (alert_type, key) → last_sent_ts
        self._sent: dict[tuple[str, str], float] = {}
        self._trackers: dict[str, _TrackerStatic] = {}

    # ---- publish with cooldown ----
    def publish(self, topic: str, alert_type: str, key: str,
                payload: dict, now: float, persistent: bool = False) -> bool:
        """Returns True if actually published (not cooldown-suppressed)."""
        gate = (alert_type, key)
        last = self._sent.get(gate)
        if last is not None and (now - last) < self.cooldown:
            return False
        self._sent[gate] = now
        envelope = {"type": alert_type, "ts": now, **payload}
        try:
            self.events.publish(topic, envelope, persistent=persistent)
        except Exception as e:  # noqa: BLE001 — bus must not crash the infer loop
            print(f"[alerts] publish failed topic={topic} type={alert_type}: {e}")
            return False
        return True

    # ---- per-tracker safety evaluation ----
    def evaluate_tracker(self, tracker_id: str, kp: KeyPoints, now: float,
                         zone_id: str | None, forbidden_zones: set[str],
                         bbox_aspect_ratio: float | None = None) -> list[dict]:
        """Run fall / long-static / forbidden-zone checks for one person.

        Returns list of alert dicts that were published this tick.

        bbox_aspect_ratio: optional width/height ratio from detection bbox.
        Standing person ~0.3-0.5; fallen person > 0.8 (wider than tall).
        When provided alongside keypoint signals, strengthens fall detection
        for cases where keypoints are noisy but bbox is clearly horizontal.
        """
        tr = self._trackers.setdefault(tracker_id, _TrackerStatic())
        bc = body_center(kp)
        published: list[dict] = []

        # --- fall: horizontal torso + low hip, sustained ---
        # Bbox aspect ratio > 0.8 is a strong independent fall signal
        # (person wider than tall). Combine with keypoint heuristic:
        #   - keypoints say fall (horiz+low) → fall (high confidence)
        #   - bbox says fall (aspect>0.8) + low hip → fall (medium confidence)
        #   - keypoints say fall + bbox confirms → fall (highest confidence)
        BBOX_FALL_THRESHOLD = 0.8
        bbox_fall = (bbox_aspect_ratio is not None
                     and bbox_aspect_ratio > BBOX_FALL_THRESHOLD)

        horiz = torso_horizontal(kp)
        low = hip_near_bottom(kp)
        # Keypoint-based fall OR bbox-based fall (with low hip corroboration)
        fall_signal = (horiz and low) or (bbox_fall and low)
        if fall_signal:
            if tr.horizontal_since is None:
                tr.horizontal_since = now
            elif (now - tr.horizontal_since) >= self.fall_candidate_seconds and not tr.fall_alerted:
                tr.fall_alerted = True
                if self.publish(TOPIC_ALERTS, "fall", tracker_id,
                                {"tracker_id": tracker_id, "zone_id": zone_id},
                                now, persistent=True):
                    published.append({"type": "fall", "tracker_id": tracker_id})
        else:
            # recovered upright posture clears the fall latch
            # only clear when BOTH keypoint and bbox agree person is upright
            if tr.horizontal_since is not None and not horiz and not bbox_fall:
                tr.horizontal_since = None
                tr.fall_alerted = False

        # --- long static: minimal keypoint displacement ---
        if bc is not None:
            if tr.last_pos is not None:
                dx = bc.x - tr.last_pos[0]
                dy = bc.y - tr.last_pos[1]
                moved = (dx * dx + dy * dy) ** 0.5
                if moved < 0.01:  # <1% frame displacement
                    if tr.static_since is None:
                        tr.static_since = now
                    elif (now - tr.static_since) >= self.long_static_seconds \
                            and not tr.long_static_alerted:
                        tr.long_static_alerted = True
                        if self.publish(TOPIC_ALERTS, "long_static", tracker_id,
                                        {"tracker_id": tracker_id, "zone_id": zone_id,
                                         "duration_s": round(now - tr.static_since, 1)},
                                        now):
                            published.append({"type": "long_static", "tracker_id": tracker_id})
                else:
                    tr.static_since = None
                    tr.long_static_alerted = False
            tr.last_pos = (bc.x, bc.y)

        # --- forbidden zone entry ---
        if zone_id is not None and zone_id in forbidden_zones and not tr.forbidden_alerted:
            tr.forbidden_alerted = True
            if self.publish(TOPIC_ALERTS, "forbidden_zone", tracker_id,
                            {"tracker_id": tracker_id, "zone_id": zone_id},
                            now, persistent=True):
                published.append({"type": "forbidden_zone", "tracker_id": tracker_id,
                                  "zone_id": zone_id})
        elif zone_id not in forbidden_zones:
            tr.forbidden_alerted = False

        return published

    # ---- zone-level crowd alert ----
    def evaluate_crowd(self, crowded: list[str], counts: dict[str, int],
                       zone_names: dict[str, str], now: float) -> list[dict]:
        out = []
        for zid in crowded:
            payload = {"zone_id": zid, "name": zone_names.get(zid, zid),
                       "count": counts.get(zid, 0)}
            if self.publish(TOPIC_ALERTS, "crowd", zid, payload, now):
                out.append({"type": "crowd", "zone_id": zid, **payload})
        return out

    # ---- equipment events passthrough (cooldown by equipment+type) ----
    def forward_equipment(self, events: list[dict], now: float) -> list[dict]:
        out = []
        for ev in events:
            etype = ev.get("type", "equipment")
            eid = ev.get("equipment_id", "?")
            if self.publish(TOPIC_EQUIPMENT, etype, f"{eid}:{etype}", ev, now):
                out.append(ev)
        return out

    # ---- after-hours presence alert ----
    def evaluate_after_hours(self, after_hours: dict, total_persons: int,
                             now: float) -> list[dict]:
        """Publish after-hours alert when people are still present during
        closed hours.  after_hours = {"start": "22:00", "end": "06:00"}.

        Handles overnight wrap (e.g. 22:00→06:00).  Cooldown is keyed by
        alert_type="after_hours", key="gym" so at most one alert per cooldown
        period regardless of how many people are present.
        """
        if not after_hours or total_persons <= 0:
            return []
        start_s = after_hours.get("start")
        end_s = after_hours.get("end")
        if not start_s or not end_s:
            return []

        def _hhmm(s: str) -> int:
            """'HH:MM' → minutes since midnight."""
            try:
                h, m = s.split(":")
                return int(h) * 60 + int(m)
            except (ValueError, AttributeError):
                return -1

        start_min = _hhmm(start_s)
        end_min = _hhmm(end_s)
        if start_min < 0 or end_min < 0:
            return []

        now_local = time.localtime(now)
        now_min = now_local.tm_hour * 60 + now_local.tm_min

        # Check if current time falls within the after-hours window.
        # Overnight wrap: start=22:00 end=06:00 → in-range if now≥22:00 OR now<06:00
        if start_min <= end_min:
            in_range = start_min <= now_min < end_min
        else:
            in_range = now_min >= start_min or now_min < end_min

        if not in_range:
            return []

        if self.publish(TOPIC_ALERTS, "after_hours", "gym",
                        {"total_persons": total_persons,
                         "hours": after_hours},
                        now):
            return [{"type": "after_hours", "total_persons": total_persons}]
        return []

    def drop_tracker(self, tracker_id: str) -> None:
        self._trackers.pop(tracker_id, None)
        # also purge cooldown keys for that tracker
        self._sent = {k: v for k, v in self._sent.items() if k[1] != tracker_id}
