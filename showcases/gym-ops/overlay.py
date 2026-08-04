"""Overlay rendering: draw COCO-17 skeleton, zone polygons, and annotations
onto a cv2 frame (BGR) for the MJPEG preview stream.

All keypoint coords are normalized [0,1]; here they are scaled to pixel space.
Zone polygons are likewise normalized.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import SKELETON_EDGES
from pose import KeyPoints, Pt


# BGR colors
_SKELETON_COLOR = (0, 255, 255)     # yellow
_KPT_COLOR = (0, 0, 255)            # red
_ZONE_EDGE = (0, 255, 0)            # green
_ZONE_FILL = (0, 255, 0)
_CROWD_EDGE = (0, 0, 255)           # red
_TEXT_COLOR = (255, 255, 255)
_TEXT_BG = (0, 0, 0)


def _to_px(pt: Pt, w: int, h: int) -> tuple[int, int]:
    return (int(pt.x * w), int(pt.y * h))


def draw_keypoints(frame: np.ndarray, kp: KeyPoints, min_conf: float = 0.15) -> None:
    h, w = frame.shape[:2]
    # edges
    for a, b in SKELETON_EDGES:
        pa, pb = kp[a] if a < len(kp) else None, kp[b] if b < len(kp) else None
        if pa and pb and pa.conf >= min_conf and pb.conf >= min_conf:
            cv2.line(frame, _to_px(pa, w, h), _to_px(pb, w, h),
                     _SKELETON_COLOR, 2, cv2.LINE_AA)
    # points
    for p in kp:
        if p and p.conf >= min_conf:
            cv2.circle(frame, _to_px(p, w, h), 4, _KPT_COLOR, -1, cv2.LINE_AA)


def _poly_px(polygon, w: int, h: int):
    return np.array([[int(x * w), int(y * h)] for x, y in polygon], dtype=np.int32)


def draw_zones(frame: np.ndarray, zones, crowded: set[str] | None = None) -> None:
    h, w = frame.shape[:2]
    crowded = crowded or set()
    for z in zones:
        if len(z.polygon) < 3:
            continue
        pts = _poly_px(z.polygon, w, h)
        color = _CROWD_EDGE if z.id in crowded else _ZONE_EDGE
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
        # label at first vertex
        x0, y0 = pts[0]
        label = f"{z.name}"
        if z.capacity:
            label += f" {z.id in crowded and '!' or ''}{z.capacity}"
        _put_text(frame, label, (x0 + 4, y0 + 4))


def draw_exercise_tag(frame: np.ndarray, tracker_id: str, snap: dict,
                      y_off: int, member_id: str | None = None) -> int:
    """Draw a small per-person exercise summary. Returns next y offset."""
    prefix = f"[{tracker_id}]"
    if member_id:
        prefix += f" {member_id}"
    line = f"{prefix} {snap['exercise']} reps={snap['reps']} " \
           f"sets={snap.get('sets', 0)} bad={snap['bad_reps']} phase={snap['phase']}"
    if snap.get("angle"):
        line += f" {snap['angle']}°"
    if snap.get("rest_seconds", 0) > 0:
        line += f" rest={snap['rest_seconds']:.0f}s"
    if snap.get("quality_flags"):
        line += " ⚠" + ",".join(snap["quality_flags"])
    _put_text(frame, line, (8, y_off))
    return y_off + 22


def draw_status_bar(frame: np.ndarray, total: int, counts: dict, fps: float | None) -> None:
    h, w = frame.shape[:2]
    parts = [f"total={total}"]
    for zid, n in counts.items():
        parts.append(f"{zid}={n}")
    if fps is not None:
        parts.append(f"{fps:.0f}fps")
    _put_text(frame, "  ".join(parts), (8, h - 24))


def _put_text(frame: np.ndarray, text: str, org: tuple[int, int]) -> None:
    cv2.rectangle(frame, (org[0] - 2, org[1] - 16),
                  (org[0] + 8 * len(text) + 4, org[1] + 4), _TEXT_BG, -1)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                _TEXT_COLOR, 1, cv2.LINE_AA)
