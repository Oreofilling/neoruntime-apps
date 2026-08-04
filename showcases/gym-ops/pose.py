"""Pose geometry helpers operating on COCO-17 keypoints.

All coordinates are normalized [0,1] in image space (x right, y down — same
convention as the SDK LandmarkPoint). Pixel conversion happens only in
overlay.py at draw time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from config import (
    L_ANKLE, L_EAR, L_ELBOW, L_EYE, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    NOSE, R_ANKLE, R_EAR, R_ELBOW, R_EYE, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
)


@dataclass(frozen=True)
class Pt:
    """A single normalized keypoint."""
    x: float
    y: float
    conf: float

    @property
    def visible(self) -> bool:
        return self.conf > 0.0


# Index 0..16 → Pt. None where confidence below threshold.
KeyPoints = list[Pt | None]


def parse_keypoints(points, conf_threshold: float) -> KeyPoints:
    """Convert SDK LandmarkPoint list to KeyPoints (len 17).

    `points` is an iterable of objects with .x/.y/.confidence (the SDK
    LandmarkPoint). Points below conf_threshold become None.
    """
    out: KeyPoints = []
    for p in points:
        c = float(getattr(p, "confidence", 0.0))
        if c < conf_threshold:
            out.append(None)
        else:
            out.append(Pt(float(p.x), float(p.y), c))
    # pad/truncate to 17
    while len(out) < 17:
        out.append(None)
    return out[:17]


def _get(kp: KeyPoints, idx: int) -> Pt | None:
    if idx < 0 or idx >= len(kp):
        return None
    return kp[idx]


def joint_angle(a: Pt | None, b: Pt | None, c: Pt | None) -> float | None:
    """Angle in degrees at vertex b, formed by a-b-c.

    Returns None if any of the three points is missing. Range [0, 180].
    """
    if a is None or b is None or c is None:
        return None
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return None
    cos_ = (ba[0] * bc[0] + ba[1] * bc[1]) / (mag_ba * mag_bc)
    cos_ = max(-1.0, min(1.0, cos_))
    return math.degrees(math.acos(cos_))


def midpoint(a: Pt | None, b: Pt | None) -> Pt | None:
    if a is None or b is None:
        return None
    return Pt((a.x + b.x) / 2, (a.y + b.y) / 2, min(a.conf, b.conf))


def body_center(kp: KeyPoints) -> Pt | None:
    """Torso center: midpoint of shoulder-midpoint and hip-midpoint."""
    ls, rs = _get(kp, L_SHOULDER), _get(kp, R_SHOULDER)
    lh, rh = _get(kp, L_HIP), _get(kp, R_HIP)
    sh = midpoint(ls, rs)
    hp = midpoint(lh, rh)
    return midpoint(sh, hp)


def hip_center(kp: KeyPoints) -> Pt | None:
    return midpoint(_get(kp, L_HIP), _get(kp, R_HIP))


def shoulder_center(kp: KeyPoints) -> Pt | None:
    return midpoint(_get(kp, L_SHOULDER), _get(kp, R_SHOULDER))


def torso_angle(kp: KeyPoints) -> float | None:
    """Torso inclination from vertical, in degrees.

    0 = upright (standing), 90 = horizontal (lying). Uses shoulder-center to
    hip-center vector. None if either midpoint missing.
    """
    sh = shoulder_center(kp)
    hp = hip_center(kp)
    if sh is None or hp is None:
        return None
    dx = hp.x - sh.x
    dy = hp.y - sh.y
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return None
    # vertical vector is (0, +1) in image coords (y down). angle from vertical:
    cos_v = dy / mag
    cos_v = max(-1.0, min(1.0, cos_v))
    return math.degrees(math.acos(cos_v))


# ---- Per-exercise joint-angle helpers ----

def squat_knee_angle(kp: KeyPoints, side: str = "right") -> float | None:
    """Hip-knee-ankle angle. side in {'left','right'}."""
    if side == "left":
        return joint_angle(_get(kp, L_HIP), _get(kp, L_KNEE), _get(kp, L_ANKLE))
    return joint_angle(_get(kp, R_HIP), _get(kp, R_KNEE), _get(kp, R_ANKLE))


def pushup_elbow_angle(kp: KeyPoints, side: str = "right") -> float | None:
    """Shoulder-elbow-wrist angle."""
    if side == "left":
        return joint_angle(_get(kp, L_SHOULDER), _get(kp, L_ELBOW), _get(kp, L_WRIST))
    return joint_angle(_get(kp, R_SHOULDER), _get(kp, R_ELBOW), _get(kp, R_WRIST))


def best_side_angle(
    kp: KeyPoints,
    left_idx: tuple[int, int, int],
    right_idx: tuple[int, int, int],
) -> tuple[float | None, str]:
    """Pick the side with higher total confidence for a 3-point angle.

    Returns (angle_deg, side_str). side_str in {'left','right','none'}.
    """
    la = joint_angle(_get(kp, left_idx[0]), _get(kp, left_idx[1]), _get(kp, left_idx[2]))
    ra = joint_angle(_get(kp, right_idx[0]), _get(kp, right_idx[1]), _get(kp, right_idx[2]))

    def _score(a, idxs):
        if a is None:
            return -1.0
        return sum((_get(kp, i).conf if _get(kp, i) else 0.0) for i in idxs)

    ls, rs = _score(la, left_idx), _score(ra, right_idx)
    if ls < 0 and rs < 0:
        return None, "none"
    if ls >= rs:
        return la, "left"
    return ra, "right"


# ---- Knee-valgus / imbalance diagnostics ----

def knee_valgus(kp: KeyPoints, side: str = "right") -> float | None:
    """Knee medial offset relative to hip-ankle line, in normalized x.

    Positive = knee pulled inward (valgus). None if points missing.
    """
    if side == "left":
        hip, knee, ankle = _get(kp, L_HIP), _get(kp, L_KNEE), _get(kp, L_ANKLE)
    else:
        hip, knee, ankle = _get(kp, R_HIP), _get(kp, R_KNEE), _get(kp, R_ANKLE)
    if hip is None or knee is None or ankle is None:
        return None
    # x of knee vs the hip-ankle segment at knee's y
    if abs(ankle.y - hip.y) < 1e-6:
        return knee.x - hip.x
    t = (knee.y - hip.y) / (ankle.y - hip.y)
    ref_x = hip.x + t * (ankle.x - hip.x)
    return knee.x - ref_x


def l_r_imbalance(kp: KeyPoints, left_idx: tuple, right_idx: tuple,
                  triple: bool = False) -> float | None:
    """Symmetric angle difference between left and right limbs.

    Pass triple=(a,b,c) indices. Returns |left - right| in degrees, or None.
    """
    la = joint_angle(_get(kp, left_idx[0]), _get(kp, left_idx[1]), _get(kp, left_idx[2]))
    ra = joint_angle(_get(kp, right_idx[0]), _get(kp, right_idx[1]), _get(kp, right_idx[2]))
    if la is None or ra is None:
        return None
    return abs(la - ra)


# ---- Point-in-polygon (normalized) ----

def point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting test. polygon is a list of (x,y) normalized vertices."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > py) != (yj > py):
            x_int = (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
            if px < x_int:
                inside = not inside
        j = i
    return inside


def normalize_polygon(raw: list) -> list[tuple[float, float]]:
    """Coerce [[x,y],...] or [(x,y),...] to list of (float,float) tuples."""
    out = []
    for p in raw:
        if len(p) >= 2:
            out.append((float(p[0]), float(p[1])))
    return out


# ---- Fall heuristic primitives ----

def torso_horizontal(kp: KeyPoints) -> bool:
    """True if torso angle >= 60 deg from vertical (near-lying)."""
    ang = torso_angle(kp)
    return ang is not None and ang >= 60.0


def hip_near_bottom(kp: KeyPoints, frac: float = 0.6) -> bool:
    """True if hip-center y exceeds frac of frame height (low in image)."""
    hp = hip_center(kp)
    return hp is not None and hp.y >= frac
