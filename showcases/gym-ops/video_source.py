"""Uploaded-video frame source.

Used when the operator uploads a test clip via /api/video/upload. Emits frames
on demand (generator) for the inference loop and the MJPEG preview. Loops the
clip by default so a long-running demo doesn't stop.
"""
from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np


class VideoFrameSource:
    def __init__(self, path: str, loop: bool = True, target_fps: float | None = None):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        self.loop = loop
        self.target_fps = target_fps
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cv2 failed to open {path}")
        self._lock = threading.Lock()
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._frame_interval = 1.0 / self._fps if target_fps is None else 1.0 / target_fps
        self._last_emit = 0.0

    @property
    def fps(self) -> float:
        return self._fps

    def read(self) -> np.ndarray | None:
        """Return next BGR frame (resized to max 1280 wide), or None on EOF."""
        with self._lock:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._cap.read()
                if not ok or frame is None:
                    return None
            h, w = frame.shape[:2]
            if w > 1280:
                nh = int(h * 1280 / w)
                frame = cv2.resize(frame, (1280, nh))
            return frame

    def frames(self):
        """Generator yielding frames at the configured target FPS."""
        while True:
            now = time.monotonic()
            if (now - self._last_emit) < self._frame_interval:
                time.sleep(self._frame_interval - (now - self._last_emit))
            frame = self.read()
            if frame is None:
                return
            self._last_emit = time.monotonic()
            yield frame

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None  # type: ignore[assignment]
