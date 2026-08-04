"""Unit tests for the threaded VideoFrameSource decoder.

These verify the high-res-video stutter fix: the decode thread decouples
playback from INFER_FPS, decodes straight to a downscaled buffer (full-res
never leaks), and pause/seek/speed/close are concurrency-safe.

Runs wherever the app deps exist (cv2, numpy, flask, PIL). The Hailo SDK and
flask_sock (container-only) are stubbed so importing ``main`` works in a plain
unit-test environment.
"""

from __future__ import annotations

import sys
import time
import types
from typing import Optional

import numpy as np
import pytest
from unittest.mock import MagicMock

# --- stub container-only / unavailable imports so `import main` succeeds ----
for _name in ("hailo_ipc_sdk", "flask_sock"):
    if _name not in sys.modules:
        _stub = types.ModuleType(_name)
        _stub.__getattr__ = lambda attr: MagicMock()  # any name -> MagicMock
        sys.modules[_name] = _stub

import main  # noqa: E402  (must follow the stubs above)

import cv2  # noqa: E402

VideoFrameSource = main.VideoFrameSource


# --- helpers ---------------------------------------------------------------

def _make_clip(path: str, w: int, h: int, fps: float, n_frames: int) -> bool:
    """Write a clip of distinct solid-grey frames. Returns True on success."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        writer.release()
        return False
    try:
        for i in range(n_frames):
            val = int(np.clip(30 + i * 6, 0, 230))  # distinct grey per frame
            frame = np.full((h, w, 3), val, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return True


def _wait_first_frame(vs: VideoFrameSource, timeout: float = 3.0) -> Optional[np.ndarray]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        f = vs.latest_frame()
        if f is not None:
            return f
        time.sleep(0.01)
    return None


def _sample_distinct(vs: VideoFrameSource, window: float) -> set:
    """Distinct frame greys seen over a wall-clock window (dups collapsed)."""
    seen: set = set()
    deadline = time.time() + window
    while time.time() < deadline:
        f = vs.latest_frame()
        if f is not None:
            seen.add(round(float(f.mean()), 1))
        time.sleep(0.02)
    return seen


@pytest.fixture
def clip(tmp_path):
    """A 640x480 / 10fps / 6s clip; skipped if no MJPEG writer backend."""
    p = str(tmp_path / "clip.avi")
    if not _make_clip(p, 640, 480, 10.0, 60):
        pytest.skip("cv2.VideoWriter MJPEG backend unavailable")
    return p


# --- tests -----------------------------------------------------------------

def test_decode_downscales_to_target_dims(clip):
    # Arrange: source is 640x480; ask for 160x90 decode output.
    vs = VideoFrameSource(clip, playback_fps=10, decode_w=160, decode_h=90)
    try:
        # Act
        vs.start()
        f = _wait_first_frame(vs)
        # Assert
        assert f is not None
        assert f.shape == (90, 160, 3)
    finally:
        vs.close()


def test_no_full_res_leak_from_1080p(tmp_path):
    # Arrange: a 1920x1080 source; decode target 960x540 (16:9).
    p = str(tmp_path / "hd.avi")
    if not _make_clip(p, 1920, 1080, 10.0, 12):
        pytest.skip("cv2.VideoWriter MJPEG backend unavailable for 1080p")
    vs = VideoFrameSource(p, playback_fps=10, decode_w=960, decode_h=540)
    try:
        # Act
        vs.start()
        f = _wait_first_frame(vs)
        # Assert: full-res (1080p) never reaches consumers.
        assert f is not None
        assert f.shape == (540, 960, 3)
    finally:
        vs.close()


def test_playback_runs_near_realtime(clip):
    # Arrange: 10fps playback. If decode were coupled to a slow consumer the
    # distinct-frame rate would collapse; if 2x-fast it would double.
    vs = VideoFrameSource(clip, playback_fps=10, decode_w=160, decode_h=90)
    try:
        vs.start()
        assert _wait_first_frame(vs) is not None
        # Act
        seen = _sample_distinct(vs, window=1.2)
        # Assert: ~10 unique frames in 1.2s (tolerant for CI variance).
        assert 7 <= len(seen) <= 16
    finally:
        vs.close()


def test_pause_freezes_then_resumes(clip):
    vs = VideoFrameSource(clip, playback_fps=10, decode_w=160, decode_h=90)
    try:
        vs.start()
        assert _wait_first_frame(vs) is not None
        # Act 1: pause -> frames must freeze.
        vs.paused = True
        frozen = _sample_distinct(vs, window=0.6)
        assert len(frozen) == 1, f"expected frozen frame, got {len(frozen)}"
        # Act 2: resume -> frames advance again.
        vs.paused = False
        moving = _sample_distinct(vs, window=0.8)
        assert len(moving) >= 2, f"expected motion after resume, got {len(moving)}"
    finally:
        vs.close()


def test_seek_jumps_position(clip):
    vs = VideoFrameSource(clip, playback_fps=10, decode_w=160, decode_h=90)
    try:
        vs.start()
        assert _wait_first_frame(vs) is not None
        time.sleep(0.3)  # play a little from the start
        assert vs.position_sec < 2.0
        # Act: seek to the middle (3s of a 6s clip).
        vs.seek(3.0)
        time.sleep(0.6)  # let the decode thread apply the pending seek
        # Assert: position jumped near 3s, well past where it had played.
        assert vs.position_sec >= 2.5, f"position={vs.position_sec}"
    finally:
        vs.close()


def test_speed_2x_advances_faster(clip):
    vs = VideoFrameSource(clip, playback_fps=10, decode_w=160, decode_h=90)
    try:
        vs.start()
        assert _wait_first_frame(vs) is not None
        vs.speed = 2.0
        seen = _sample_distinct(vs, window=1.0)
        # ~20 unique frames/s at 2x; allow wide tolerance.
        assert len(seen) >= 14, f"expected ~2x rate, got {len(seen)}"
    finally:
        vs.close()


def test_close_is_idempotent_and_joins(clip):
    vs = VideoFrameSource(clip, playback_fps=10, decode_w=160, decode_h=90)
    vs.start()
    thread = vs._decode_thread
    assert thread is not None and thread.is_alive()
    # Act
    vs.close()
    # Assert: thread joined, cap released, second close is a no-op.
    assert not thread.is_alive()
    assert vs.cap is None
    vs.close()  # must not raise
    assert vs.cap is None
