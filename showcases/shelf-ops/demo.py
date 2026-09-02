"""Built-in demo video source (customer demos).

demo.mp4 (bundled in assets/) shows a two-row drink fridge: fully stocked
at the start, purchases opening gaps toward the end, and the loop restart
snapping back to full. Looping it through the LIVE pipeline gives a
walk-away demo of the whole story — empty-slot detection, stockout and
restock events, ITEMS counting down and back up, heatmap accumulating —
without depending on a camera operator or a real fridge on cue.

Decode-on-demand: pre-decoding 367 frames of 540x960 RGB would cost
~0.6 GB of RAM for no benefit, so each read maps the current wall clock
to a video
position. Normal pacing decodes forward frame-by-frame; a backward jump
(loop restart) or a big forward jump (a stalled consumer) seeks instead.
Both consumers (the ~2 Hz scan loop and the ~15 Hz preview loop) share one
cv2.VideoCapture behind a lock — cv2 is not thread-safe per capture.
"""
from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np

# If the clock asks for a frame this far ahead of the decoder, seek instead
# of decode-forward (>= 1.5 s of video at 30 fps).
_SEEK_JUMP_FRAMES = 45


class DemoVideoSource:
    """Looping, wall-clock-paced reader for the built-in demo video."""

    def __init__(self, path: str, speed: float = 0.5,
                 loop: bool = True) -> None:
        if speed <= 0:
            raise ValueError(f"demo speed must be > 0, got {speed}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"demo video not found: {path}")

        probe = cv2.VideoCapture(path)
        if not probe.isOpened():
            raise ValueError(f"demo video unreadable: {path}")
        self.path = path
        self.speed = float(speed)
        self.loop = bool(loop)
        self.fps = float(probe.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
        self.frame_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        probe.release()
        if self.frame_count <= 1:
            raise ValueError(f"demo video has no frames: {path}")
        self.duration = self.frame_count / self.fps   # seconds of video

        self._lock = threading.Lock()
        self._cap = cv2.VideoCapture(path)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._cap.release()
            raise ValueError(f"demo video first frame unreadable: {path}")
        self._frame = frame          # last decoded frame (BGR)
        self._seq = 0                # its index in the video
        self._t0 = time.monotonic()  # wall clock the loop started at

    # ---- introspection ------------------------------------------------------
    @property
    def info(self) -> dict:
        """Status block for /api/demo + /api/config."""
        return {
            "path": self.path,
            "speed": self.speed,
            "fps": round(self.fps, 3),
            "frames": self.frame_count,
            "width": self.width,
            "height": self.height,
            "video_seconds": round(self.duration, 2),
            "loop_seconds": round(self.duration / self.speed, 2),
        }

    # ---- pacing -------------------------------------------------------------
    def _target_seq(self, now: float) -> int:
        """Wall clock -> video frame index (speed-scaled, wrapped on loop)."""
        elapsed = max(0.0, now - self._t0) * self.speed
        if not self.loop and elapsed >= self.duration:
            return self.frame_count - 1
        return int((elapsed % self.duration) * self.fps)

    def _restart_locked(self) -> bool:
        """Seek back to frame 0 (loop wrap or read past EOF)."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        ok, frame = self._cap.read()
        if ok and frame is not None:
            self._frame, self._seq = frame, 0
            return True
        return False

    def _read_locked(self) -> np.ndarray:
        idx = self._target_seq(time.monotonic())
        if idx < self._seq or idx > self._seq + _SEEK_JUMP_FRAMES:
            # backward (loop restart) or far-forward (stalled consumer) jump
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = self._cap.read()
            if ok and frame is not None:
                self._frame, self._seq = frame, idx
            elif not self._restart_locked():
                return self._frame   # decoder wedged: hold the last frame
        else:
            while self._seq < idx:
                ok, frame = self._cap.read()
                if ok and frame is not None:
                    self._seq += 1
                elif not (self.loop and self._restart_locked()):
                    break            # EOF without loop: hold the last frame
                else:
                    continue         # wrapped to 0; keep decoding toward idx
                self._frame = frame
        return self._frame

    # ---- public reads ---------------------------------------------------------
    def read_bgr(self) -> np.ndarray:
        """One BGR frame at the paced position (a copy — callers draw on it)."""
        with self._lock:
            return self._read_locked().copy()

    def read_rgb(self) -> np.ndarray:
        with self._lock:
            return cv2.cvtColor(self._read_locked(), cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        with self._lock:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001 - release is best-effort
                pass
