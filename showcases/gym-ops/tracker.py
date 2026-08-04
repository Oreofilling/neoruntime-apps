"""Greedy body-center nearest-neighbor multi-person tracker.

The yolov8s_pose stream has no per-person IDs — `result.landmarks` is an
unordered list whose index order can swap frame-to-frame as people enter/leave
or cross. A positional `p{i}` key (the previous scheme) therefore reassigns a
person's rep counter to whoever happens to land at index i, corrupting counts.

This module associates each frame's detections to the previous frame's tracks
by body-center (shoulder/hip mid) distance in normalized [0,1] image space.
Greedy nearest-neighbor within `max_dist` — simple, cheap, good enough for a
gym floor with sparse, slow-moving people. No Kalman / Hungarian; that's
overkill here and the current stage only asks for a simple tracker.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from pose import Pt, body_center


@dataclass
class _Track:
    track_id: int
    last_center: Pt
    missed: int = 0            # consecutive frames unseen before expiry


class BodyTracker:
    """Assigns stable `p{n}` ids to per-frame body centers.

    Call `update()` once per frame with the parsed keypoint sets; it returns a
    list aligned to the input, where each entry is the assigned id string
    (or None when the keypoint set had no computable body center).
    """

    def __init__(self, max_dist: float = 0.3, max_missed: int = 5) -> None:
        self._max_dist = max_dist
        self._max_missed = max_missed
        self._next_id = 0
        self._tracks: dict[int, _Track] = {}

    def update(self, keypoint_sets: list) -> list[str | None]:
        """Return per-input assigned id strings (None where no body center)."""
        centers: list[tuple[int, Pt]] = []  # (orig_index, center)
        for i, kp in enumerate(keypoint_sets):
            c = body_center(kp)
            if c is not None:
                centers.append((i, c))

        track_ids = list(self._tracks.keys())

        # Greedy pair (track, detection) by ascending center distance.
        candidates: list[tuple[float, int, int]] = []  # (dist, track_pos, det_pos)
        for di, (_i, dc) in enumerate(centers):
            for ti, tid in enumerate(track_ids):
                pc = self._tracks[tid].last_center
                d = math.hypot(dc.x - pc.x, dc.y - pc.y)
                if d <= self._max_dist:
                    candidates.append((d, ti, di))
        candidates.sort(key=lambda t: t[0])

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        assign: dict[int, int] = {}  # det_pos -> track_id
        for _d, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            assign[di] = track_ids[ti]

        ids: list[str | None] = [None] * len(keypoint_sets)
        new_centers: dict[int, Pt] = {}  # track_id -> latest center
        for di, (orig_i, dc) in enumerate(centers):
            if di in assign:
                tid = assign[di]
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _Track(track_id=tid, last_center=dc)
            ids[orig_i] = f"p{tid}"
            new_centers[tid] = dc

        # Refresh matched track centers + missed counters; expire stale tracks.
        for tid, tr in list(self._tracks.items()):
            if tid in new_centers:
                tr.last_center = new_centers[tid]
                tr.missed = 0
            else:
                tr.missed += 1
                if tr.missed > self._max_missed:
                    self._tracks.pop(tid, None)

        return ids

    def reset(self) -> None:
        """Drop all tracks (e.g. on video-source switch)."""
        self._tracks.clear()
        self._next_id = 0
