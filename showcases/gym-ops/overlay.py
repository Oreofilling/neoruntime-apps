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
_CROWD_COLOR = (93, 93, 255)        # red (#ff5d5d) — crowd override
_AMBER = (66, 179, 245)             # amber (#f5b342) — forbidden zones
_HEX_FALLBACK = (255, 168, 110)     # soft blue (#6ea8ff) — malformed color
_TEXT_COLOR = (255, 255, 255)
_TEXT_BG = (0, 0, 0)
_TRAIL_COLOR = (255, 200, 80)        # warm amber for motion trails

_ZONE_FILL_ALPHA = 0.12             # low-saturation tint fill
_CROWD_FILL_ALPHA = 0.18
_BADGE_BG = (15, 11, 8)             # #080b0f
_BADGE_ALPHA = 0.75
_BADGE_TEXT = (230, 235, 242)       # #e6ebf2


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


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert '#rrggbb' to BGR; falls back to soft blue on malformed input."""
    h = (hex_color or "").lstrip("#")
    if len(h) != 6:
        return _HEX_FALLBACK
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return _HEX_FALLBACK
    return (b, g, r)


def _fill_poly_alpha(frame: np.ndarray, pts: np.ndarray,
                     color: tuple[int, int, int], alpha: float) -> None:
    """Alpha-blend a polygon fill, copying only the polygon's bbox ROI."""
    h, w = frame.shape[:2]
    x0 = max(int(pts[:, 0].min()), 0)
    x1 = min(int(pts[:, 0].max()) + 1, w)
    y0 = max(int(pts[:, 1].min()), 0)
    y1 = min(int(pts[:, 1].max()) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    layer = roi.copy()
    cv2.fillPoly(layer, [pts - np.array([x0, y0], dtype=np.int32)], color)
    cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0, roi)


def _dashed_line(frame: np.ndarray, p0, p1, color, thickness: int,
                 dash: int = 12, gap: int = 8) -> None:
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    dist = float(np.hypot(x1 - x0, y1 - y0))
    if dist < 1.0:
        return
    steps = np.arange(0.0, 1.0, (dash + gap) / dist)
    for k in range(0, len(steps) - 1, 2):
        a, b = steps[k], steps[k + 1]
        cv2.line(frame, (int(x0 + a * (x1 - x0)), int(y0 + a * (y1 - y0))),
                 (int(x0 + b * (x1 - x0)), int(y0 + b * (y1 - y0))),
                 color, thickness, cv2.LINE_AA)


def _dashed_polylines(frame: np.ndarray, pts: np.ndarray,
                      color, thickness: int = 2) -> None:
    for i in range(len(pts)):
        _dashed_line(frame, pts[i], pts[(i + 1) % len(pts)], color, thickness)


def _blend_rounded_chip(frame: np.ndarray, box, bg, alpha: float,
                        radius: int = 6) -> None:
    """Blend a rounded-rect dark chip onto frame within `box` = (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    if cw < 4 or ch < 4:
        return
    r = max(1, min(radius, cw // 2, ch // 2))
    mask = np.zeros((ch, cw), dtype=np.float32)
    cv2.rectangle(mask, (r, 0), (cw - 1 - r, ch - 1), 1.0, -1)
    cv2.rectangle(mask, (0, r), (cw - 1, ch - 1 - r), 1.0, -1)
    for cx, cy in ((r, r), (cw - 1 - r, r), (r, ch - 1 - r), (cw - 1 - r, ch - 1 - r)):
        cv2.circle(mask, (cx, cy), r, 1.0, -1)
    m3 = cv2.merge([mask, mask, mask]) * alpha
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    chip = np.array(bg, dtype=np.float32)
    blended = roi * (1.0 - m3) + chip * m3
    frame[y0:y1, x0:x1] = blended.astype(np.uint8)


def _zone_badge(frame: np.ndarray, z, pts: np.ndarray,
                dot_color, count: int | None) -> None:
    """Rounded label chip at the polygon bbox top-left (8px inset).

    Content: color dot + name [+ live count/capacity when count is given].
    """
    h, w = frame.shape[:2]
    x0 = int(pts[:, 0].min()) + 8
    y0 = int(pts[:, 1].min()) + 8
    label = z.name
    if count is not None:
        label += f"  {count}/{z.capacity}" if z.capacity else f"  {count}"
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    pad, dot_r = 8, 4
    chip_h = th + base + 10
    chip_w = pad + dot_r * 2 + 8 + tw + pad
    _blend_rounded_chip(frame, (x0, y0, min(x0 + chip_w, w), min(y0 + chip_h, h)),
                        _BADGE_BG, _BADGE_ALPHA)
    cy = y0 + chip_h // 2
    cv2.circle(frame, (x0 + pad + dot_r, cy), dot_r, dot_color, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (x0 + pad + dot_r * 2 + 8, y0 + 5 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BADGE_TEXT, 1, cv2.LINE_AA)


def draw_zones(frame: np.ndarray, zones, crowded: set[str] | None = None,
               counts: dict[str, int] | None = None) -> None:
    """Draw zone polygons with per-zone identity colors and label badges.

    Style: low-alpha fill (0.12; 0.18 when crowded) + 2px AA border in the
    zone's own color; crowded zones overridden red, forbidden zones amber
    with a dashed border. Badge shows name + live count/capacity when
    `counts` is supplied.
    """
    crowded = crowded or set()
    counts = counts or {}
    for z in zones:
        if len(z.polygon) < 3:
            continue
        h, w = frame.shape[:2]
        pts = _poly_px(z.polygon, w, h)
        if z.forbidden:
            color = _AMBER
        elif z.id in crowded:
            color = _CROWD_COLOR
        else:
            color = _hex_to_bgr(getattr(z, "color", "") or "")
        _fill_poly_alpha(frame, pts, color,
                         _CROWD_FILL_ALPHA if z.id in crowded else _ZONE_FILL_ALPHA)
        if z.forbidden:
            _dashed_polylines(frame, pts, color, 2)
        else:
            cv2.polylines(frame, [pts], True, color,
                          3 if z.id in crowded else 2, cv2.LINE_AA)
        _zone_badge(frame, z, pts, color, counts.get(z.id))


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


def draw_trail(frame: np.ndarray, path: list[tuple[float, float]],
               color: tuple[int, int, int] = _TRAIL_COLOR) -> None:
    """Draw a fading motion trail from normalized (x, y) body-center points.

    Older segments are thinner and blended toward the frame so the trail
    dissolves behind the person. No-op for paths shorter than 2 points.
    """
    if len(path) < 2:
        return
    h, w = frame.shape[:2]
    n = len(path) - 1
    for i in range(n):
        p0 = (int(path[i][0] * w), int(path[i][1] * h))
        p1 = (int(path[i + 1][0] * w), int(path[i + 1][1] * h))
        t = (i + 1) / n               # 0 = oldest, 1 = newest
        thickness = max(1, int(round(1 + 2 * t)))
        overlay = frame.copy()
        cv2.line(overlay, p0, p1, color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25 + 0.55 * t, frame, 0.75 - 0.55 * t, 0, frame)


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
