"""Squat / pushup rep counting state machines + pose-quality diagnostics.

State machine: UP ⇄ DOWN. A rep is a DOWN→UP transition that also satisfied
the DOWN threshold (full-amplitude guard). Each tracked person owns one
ExerciseCounter; app.py keys them by tracker id.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from pose import (
    KeyPoints, Pt, body_center, best_side_angle, hip_center, joint_angle,
    knee_valgus, l_r_imbalance, pushup_elbow_angle, squat_knee_angle,
)
from config import (
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
)


class Phase(str, Enum):
    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"


# ---- Thresholds (degrees) ----
# Standing knee ~170-180, deep squat ~70-90.
SQUAT_UP_ANGLE = 160.0
SQUAT_DOWN_ANGLE = 110.0
# Pushup top elbow ~170, bottom ~80-90.
PUSHUP_UP_ANGLE = 160.0
PUSHUP_DOWN_ANGLE = 100.0
# Deadlift hip-knee-ankle: standing ~170, bent ~130.
DEADLIFT_UP_ANGLE = 160.0
DEADLIFT_DOWN_ANGLE = 130.0
# Bench press elbow: top ~170, bottom ~90.
BENCH_UP_ANGLE = 160.0
BENCH_DOWN_ANGLE = 100.0
# Lat pulldown elbow: top (arms up) ~170, bottom (pull down) ~100.
LATPD_UP_ANGLE = 160.0
LATPD_DOWN_ANGLE = 110.0
# Bicep curl elbow: top (arm straight) ~170, bottom (curl) ~40.
CURL_UP_ANGLE = 150.0
CURL_DOWN_ANGLE = 60.0

# Tempo: a rep faster than this (sec) is flagged too-fast.
MIN_REP_SECONDS = 0.6
# Quality: knee valgus (normalized x) above this = valgus flag.
KNEE_VALGUS_THRESHOLD = 0.04
L_R_IMBALANCE_THRESHOLD = 25.0  # degrees
# Sets: pause longer than this (sec) counts as end-of-set.
SET_REST_SECONDS = 15.0


@dataclass
class ExerciseState:
    phase: Phase = Phase.UNKNOWN
    reps: int = 0
    sets: int = 0                       # completed sets (groups of reps)
    last_phase: Phase = Phase.UNKNOWN
    last_transition_ts: float = 0.0
    last_rep_ts: float = 0.0            # timestamp of most recent rep
    reached_down: bool = False           # amplitude guard
    bad_reps: int = 0                    # reps flagged low quality
    last_quality_flags: list[str] = field(default_factory=list)
    last_angle: float | None = None
    last_side: str = "none"
    rest_since: float | None = None     # None = active; float = resting since ts


class ExerciseCounter:
    """Generic up/down rep counter parameterized by angle thresholds."""

    exercise: str = "generic"

    def __init__(self, up_angle: float, down_angle: float, now: float = 0.0):
        self.up_angle = up_angle
        self.down_angle = down_angle
        self.state = ExerciseState(last_transition_ts=now)

    def _compute_angle(self, kp: KeyPoints) -> tuple[float | None, str]:
        raise NotImplementedError

    def update(self, kp: KeyPoints, now: float) -> ExerciseState:
        angle, side = self._compute_angle(kp)
        self.state.last_angle = angle
        self.state.last_side = side
        if angle is None:
            self.state.phase = Phase.UNKNOWN
            return self.state

        prev = self.state.phase
        if angle >= self.up_angle:
            new_phase = Phase.UP
        elif angle <= self.down_angle:
            new_phase = Phase.DOWN
        else:
            new_phase = prev  # hysteresis band: hold previous phase

        if new_phase == Phase.DOWN:
            self.state.reached_down = True

        # rep = DOWN→UP only if a true DOWN was reached
        if prev == Phase.DOWN and new_phase == Phase.UP and self.state.reached_down:
            self.state.reps += 1
            self.state.last_rep_ts = now
            dt = now - self.state.last_transition_ts if self.state.last_transition_ts else 0.0
            flags = self._quality(kp, dt)
            self.state.last_quality_flags = flags
            if flags:
                self.state.bad_reps += 1
            self.state.reached_down = False
            # Active rep ends any ongoing rest
            self.state.rest_since = None

        # Sets logic: if no rep for SET_REST_SECONDS, close the set.
        if self.state.reps > 0 and self.state.last_rep_ts > 0:
            idle = now - self.state.last_rep_ts
            if idle >= SET_REST_SECONDS and self.state.rest_since is None:
                # Start rest period, increment sets
                self.state.sets += 1
                self.state.rest_since = now

        if new_phase != prev and new_phase != Phase.UNKNOWN:
            self.state.last_transition_ts = now
        self.state.last_phase = prev
        self.state.phase = new_phase
        return self.state

    def _quality(self, kp: KeyPoints, rep_dt: float) -> list[str]:
        """Subclass hook: return list of quality-issue flag strings."""
        return []

    def snapshot(self) -> dict:
        rest_seconds = 0.0
        if self.state.rest_since is not None and self.state.last_rep_ts > 0:
            rest_seconds = time.time() - self.state.rest_since
        return {
            "exercise": self.exercise,
            "phase": self.state.phase.value,
            "reps": self.state.reps,
            "sets": self.state.sets,
            "bad_reps": self.state.bad_reps,
            "angle": round(self.state.last_angle, 1) if self.state.last_angle else None,
            "side": self.state.last_side,
            "quality_flags": list(self.state.last_quality_flags),
            "rest_seconds": round(rest_seconds, 1),
        }


class SquatCounter(ExerciseCounter):
    exercise = "squat"

    def __init__(self, now: float = 0.0):
        super().__init__(SQUAT_UP_ANGLE, SQUAT_DOWN_ANGLE, now)

    def _compute_angle(self, kp: KeyPoints):
        ang, side = best_side_angle(
            kp,
            (L_HIP, L_KNEE, L_ANKLE),
            (R_HIP, R_KNEE, R_ANKLE),
        )
        return ang, side

    def _quality(self, kp: KeyPoints, rep_dt: float) -> list[str]:
        flags: list[str] = []
        if not self.state.reached_down:
            flags.append("depth_insufficient")
        # knee valgus on the active side
        side = self.state.last_side
        kv = knee_valgus(kp, side)
        if kv is not None and abs(kv) > KNEE_VALGUS_THRESHOLD:
            flags.append("knee_valgus")
        # L/R knee angle imbalance
        imb = l_r_imbalance(kp, (L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE))
        if imb is not None and imb > L_R_IMBALANCE_THRESHOLD:
            flags.append("l_r_imbalance")
        if 0 < rep_dt < MIN_REP_SECONDS:
            flags.append("tempo_too_fast")
        return flags


class PushupCounter(ExerciseCounter):
    exercise = "pushup"

    def __init__(self, now: float = 0.0):
        super().__init__(PUSHUP_UP_ANGLE, PUSHUP_DOWN_ANGLE, now)

    def _compute_angle(self, kp: KeyPoints):
        ang, side = best_side_angle(
            kp,
            (L_SHOULDER, L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW, R_WRIST),
        )
        return ang, side

    def _quality(self, kp: KeyPoints, rep_dt: float) -> list[str]:
        flags: list[str] = []
        if not self.state.reached_down:
            flags.append("depth_insufficient")
        imb = l_r_imbalance(
            kp,
            (L_SHOULDER, L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW, R_WRIST),
        )
        if imb is not None and imb > L_R_IMBALANCE_THRESHOLD:
            flags.append("l_r_imbalance")
        if 0 < rep_dt < MIN_REP_SECONDS:
            flags.append("tempo_too_fast")
        return flags


class DeadliftCounter(ExerciseCounter):
    """Deadlift: hip-knee-ankle angle. Standing ~170, bent ~130.

    Quality checks: spine neutral (shoulder-hip-knee alignment), knee valgus,
    tempo.
    """
    exercise = "deadlift"

    def __init__(self, now: float = 0.0):
        super().__init__(DEADLIFT_UP_ANGLE, DEADLIFT_DOWN_ANGLE, now)

    def _compute_angle(self, kp: KeyPoints):
        ang, side = best_side_angle(
            kp,
            (L_HIP, L_KNEE, L_ANKLE),
            (R_HIP, R_KNEE, R_ANKLE),
        )
        return ang, side

    def _quality(self, kp: KeyPoints, rep_dt: float) -> list[str]:
        flags: list[str] = []
        # Spine neutral: shoulder-hip-knee angle should stay > ~150°
        # If the back rounds, shoulder-hip-knee drops.
        spine_ang = best_side_angle(
            kp,
            (L_SHOULDER, L_HIP, L_KNEE),
            (R_SHOULDER, R_HIP, R_KNEE),
        )
        if spine_ang[0] is not None and spine_ang[0] < 145.0:
            flags.append("spine_not_neutral")
        if not self.state.reached_down:
            flags.append("depth_insufficient")
        if 0 < rep_dt < MIN_REP_SECONDS:
            flags.append("tempo_too_fast")
        return flags


class BenchPressCounter(ExerciseCounter):
    """Bench press: shoulder-elbow-wrist angle. Top ~170, bottom ~90.

    Quality checks: depth, L/R imbalance, tempo.
    """
    exercise = "bench_press"

    def __init__(self, now: float = 0.0):
        super().__init__(BENCH_UP_ANGLE, BENCH_DOWN_ANGLE, now)

    def _compute_angle(self, kp: KeyPoints):
        ang, side = best_side_angle(
            kp,
            (L_SHOULDER, L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW, R_WRIST),
        )
        return ang, side

    def _quality(self, kp: KeyPoints, rep_dt: float) -> list[str]:
        flags: list[str] = []
        if not self.state.reached_down:
            flags.append("depth_insufficient")
        imb = l_r_imbalance(
            kp,
            (L_SHOULDER, L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW, R_WRIST),
        )
        if imb is not None and imb > L_R_IMBALANCE_THRESHOLD:
            flags.append("l_r_imbalance")
        if 0 < rep_dt < MIN_REP_SECONDS:
            flags.append("tempo_too_fast")
        return flags


class LatPulldownCounter(ExerciseCounter):
    """Lat pulldown: shoulder-elbow-wrist angle. Arms up ~170, pull down ~100.

    Quality checks: depth, L/R imbalance, tempo.
    """
    exercise = "lat_pulldown"

    def __init__(self, now: float = 0.0):
        super().__init__(LATPD_UP_ANGLE, LATPD_DOWN_ANGLE, now)

    def _compute_angle(self, kp: KeyPoints):
        ang, side = best_side_angle(
            kp,
            (L_SHOULDER, L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW, R_WRIST),
        )
        return ang, side

    def _quality(self, kp: KeyPoints, rep_dt: float) -> list[str]:
        flags: list[str] = []
        if not self.state.reached_down:
            flags.append("depth_insufficient")
        imb = l_r_imbalance(
            kp,
            (L_SHOULDER, L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW, R_WRIST),
        )
        if imb is not None and imb > L_R_IMBALANCE_THRESHOLD:
            flags.append("l_r_imbalance")
        if 0 < rep_dt < MIN_REP_SECONDS:
            flags.append("tempo_too_fast")
        return flags


class BicepCurlCounter(ExerciseCounter):
    """Bicep curl: shoulder-elbow-wrist angle. Straight ~170, curled ~40.

    Quality checks: depth, L/R imbalance, tempo.
    """
    exercise = "bicep_curl"

    def __init__(self, now: float = 0.0):
        super().__init__(CURL_UP_ANGLE, CURL_DOWN_ANGLE, now)

    def _compute_angle(self, kp: KeyPoints):
        ang, side = best_side_angle(
            kp,
            (L_SHOULDER, L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW, R_WRIST),
        )
        return ang, side

    def _quality(self, kp: KeyPoints, rep_dt: float) -> list[str]:
        flags: list[str] = []
        if not self.state.reached_down:
            flags.append("depth_insufficient")
        imb = l_r_imbalance(
            kp,
            (L_SHOULDER, L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW, R_WRIST),
        )
        if imb is not None and imb > L_R_IMBALANCE_THRESHOLD:
            flags.append("l_r_imbalance")
        if 0 < rep_dt < MIN_REP_SECONDS:
            flags.append("tempo_too_fast")
        return flags


COUNTER_CLASSES = {
    "squat": SquatCounter,
    "pushup": PushupCounter,
    "deadlift": DeadliftCounter,
    "bench_press": BenchPressCounter,
    "lat_pulldown": LatPulldownCounter,
    "bicep_curl": BicepCurlCounter,
}


def make_counter(exercise: str, now: float = 0.0) -> ExerciseCounter:
    cls = COUNTER_CLASSES.get(exercise)
    if cls is None:
        raise ValueError(f"Unknown exercise '{exercise}'. Available: {list(COUNTER_CLASSES)}")
    return cls(now)
