"""Rule-based exercise classifier from keypoint geometry + zone equipment mapping.

No NPU inference — pure CPU heuristics. Priority:
1. Zone → equipment → exercise mapping (most reliable)
2. Keypoint posture heuristics (fallback when zone unknown)
"""
from __future__ import annotations

from pose import KeyPoints, joint_angle, torso_angle, body_center
from config import (
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
    NOSE,
)


def _get(kp: KeyPoints, idx: int):
    if idx < 0 or idx >= len(kp):
        return None
    return kp[idx]


class ExerciseClassifier:
    """Classify current exercise from keypoint pose + optional zone context.

    Usage:
        cls = ExerciseClassifier(zone_exercise_map)
        exercise = cls.classify(kp, zone_id="squat_rack_A")
    """

    def __init__(self, zone_exercise_map: dict[str, str] | None = None):
        """zone_exercise_map: {zone_id: exercise_name}, e.g. {"squat_rack_A": "squat"}."""
        self.zone_exercise_map = zone_exercise_map or {}

    def classify(self, kp: KeyPoints, zone_id: str | None = None) -> str:
        """Return the most likely exercise name.

        Priority:
        1. If zone_id is in zone_exercise_map → that exercise (highest confidence).
        2. Otherwise, use keypoint posture heuristics.
        3. If uncertain → "unknown".
        """
        # 1. Zone-based (most reliable)
        if zone_id and zone_id in self.zone_exercise_map:
            return self.zone_exercise_map[zone_id]

        # 2. Keypoint heuristics
        return self._classify_from_pose(kp)

    def _classify_from_pose(self, kp: KeyPoints) -> str:
        """Heuristic classification from skeletal geometry.

        Decision tree (rough):
        - Torso upright + knee bending cycle → squat or deadlift
          - Hip hinge (shoulder-hip-knee aligned forward) → deadlift
          - Knee-dominant (knee angle < 120) → squat
        - Torso horizontal-ish + elbow bending → bench_press
        - Seated/upright + arms pulling down → lat_pulldown
        - Upright + single-arm curl motion → bicep_curl
        - Upright + elbow bending (push motion) → pushup
        """
        t_ang = torso_angle(kp)

        # Get key angles
        l_knee = joint_angle(_get(kp, L_HIP), _get(kp, L_KNEE), _get(kp, L_ANKLE))
        r_knee = joint_angle(_get(kp, R_HIP), _get(kp, R_KNEE), _get(kp, R_ANKLE))
        l_elbow = joint_angle(_get(kp, L_SHOULDER), _get(kp, L_ELBOW), _get(kp, L_WRIST))
        r_elbow = joint_angle(_get(kp, R_SHOULDER), _get(kp, R_ELBOW), _get(kp, R_WRIST))
        # Spine alignment: shoulder-hip-knee
        l_spine = joint_angle(_get(kp, L_SHOULDER), _get(kp, L_HIP), _get(kp, L_KNEE))
        r_spine = joint_angle(_get(kp, R_SHOULDER), _get(kp, R_HIP), _get(kp, R_KNEE))

        knee_ang = l_knee if l_knee is not None else r_knee
        elbow_ang = l_elbow if l_elbow is not None else r_elbow
        spine_ang = l_spine if l_spine is not None else r_spine

        # Torso horizontal → likely bench press or pushup
        if t_ang is not None and t_ang > 45.0:
            if elbow_ang is not None and elbow_ang < 130.0:
                return "bench_press"
            return "pushup"

        # Torso upright
        if t_ang is not None and t_ang < 30.0:
            # Knee bending significantly
            if knee_ang is not None and knee_ang < 130.0:
                # Hip hinge vs knee-dominant
                if spine_ang is not None and spine_ang < 150.0:
                    return "deadlift"
                return "squat"

            # Elbow bending, no significant knee bend
            if elbow_ang is not None and elbow_ang < 90.0:
                # Check if arms are above shoulder (lat pulldown position)
                l_wrist = _get(kp, L_WRIST)
                r_wrist = _get(kp, R_WRIST)
                l_shoulder = _get(kp, L_SHOULDER)
                r_shoulder = _get(kp, R_SHOULDER)

                # Wrist above shoulder (y < shoulder.y in image coords) → pulldown
                arms_up = False
                if l_wrist and l_shoulder and l_wrist.y < l_shoulder.y:
                    arms_up = True
                if r_wrist and r_shoulder and r_wrist.y < r_shoulder.y:
                    arms_up = True

                if arms_up:
                    return "lat_pulldown"
                return "bicep_curl"

        return "unknown"
