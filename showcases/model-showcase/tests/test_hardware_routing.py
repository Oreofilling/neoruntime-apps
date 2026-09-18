"""Tests for SDK 0.7.4+/0.8.0 hardware-first integration points.

Container-only imports are stubbed so importing ``main`` works in a plain
unit-test environment; individual SDK helper globals are monkeypatched with
small fakes where we need to observe routing.
"""

from __future__ import annotations

import sys
import threading
import types
from unittest.mock import MagicMock

import numpy as np

for _name in ("neoruntime_ipc_sdk", "flask_sock"):
    if _name not in sys.modules:
        _stub = types.ModuleType(_name)
        _stub.__getattr__ = lambda attr: MagicMock()
        sys.modules[_name] = _stub

import main  # noqa: E402


class _FakeFrame:
    def __init__(
        self,
        width: int,
        height: int,
        fmt: str = "NV12",
        image: np.ndarray | None = None,
        handle: object | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.format = fmt
        self.image = image
        self.handle = handle
        self.sequence = 7
        self.resize_calls = []
        self.released = False

    def resize(self, width: int, height: int, mode: str = "letterbox"):
        self.resize_calls.append((width, height, mode))
        if self.format == "NV12":
            image = np.zeros((height * 3 // 2, width), dtype=np.uint8)
        else:
            image = np.zeros((height, width, 3), dtype=np.uint8)
        return _FakeFrame(width, height, self.format, image=image)

    def to_array(self):
        if self.image is None:
            if self.format == "NV12":
                self.image = np.zeros((self.height * 3 // 2, self.width), dtype=np.uint8)
            else:
                self.image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return self.image

    def release(self):
        self.released = True


def _minimal_showcase() -> main.ModelShowcase:
    sc = main.ModelShowcase.__new__(main.ModelShowcase)
    sc._model_lock = threading.Lock()
    sc._video_active = False
    sc._active_infer_stream = "third"
    return sc


def test_get_infer_raw_requests_retained_fd():
    sc = _minimal_showcase()
    frame = _FakeFrame(640, 384, handle=object())
    sc.media_infer = MagicMock()
    sc.media_infer.get_frame.return_value = frame

    raw = sc._get_infer_raw()

    assert raw == (frame, 640, 384, "NV12")
    sc.media_infer.get_frame.assert_called_once_with(
        "third", timeout_ms=3000, keep_fd=True,
    )


def test_get_infer_bgr_releases_retained_frame():
    sc = _minimal_showcase()
    frame = _FakeFrame(
        32, 18, fmt="BGR", image=np.zeros((18, 32, 3), dtype=np.uint8), handle=object()
    )
    sc.media_infer = MagicMock()
    sc.media_infer.get_frame.return_value = frame

    out = sc._get_infer_bgr()

    assert out.shape == (18, 32, 3)
    assert frame.released
    sc.media_infer.get_frame.assert_called_once_with(
        "third", timeout_ms=3000, keep_fd=True,
    )


def test_prepare_input_uses_frame_resize_before_flattening():
    sc = _minimal_showcase()
    frame = _FakeFrame(1280, 720, handle=object())

    out = sc._prepare_input(frame, 640, 384, "nv12")

    assert frame.resize_calls == [(640, 384, "stretch")]
    assert out.shape == (640 * 384 * 3 // 2,)


def test_prepare_input_prefers_persistent_dsp_resize(monkeypatch):
    sc = _minimal_showcase()
    sc._hw_frame_prep_enabled = True
    sc._infer_hw_retry_at = 0
    sc._infer_hw_retry_interval = 5.0
    resized = np.full((384 * 3 // 2, 640), 7, dtype=np.uint8)
    dsp = MagicMock()
    dsp.resize_hw.return_value = resized
    sc._infer_dsp = dsp

    class _FrameResult(_FakeFrame):
        def __init__(self, **kwargs):
            super().__init__(
                kwargs["width"],
                kwargs["height"],
                kwargs["format"],
                image=kwargs["image"],
            )
            self.sequence = kwargs["sequence"]
            self.timestamp_ns = kwargs["timestamp_ns"]
            self.metadata = kwargs["metadata"]

    monkeypatch.setattr(main, "Frame", _FrameResult)
    frame = _FakeFrame(1280, 720, handle=object())
    frame.timestamp_ns = 99
    frame.metadata = {"camera": "main"}

    out = sc._prepare_input(frame, 640, 384, "nv12")

    assert frame.resize_calls == []
    dsp.resize_hw.assert_called_once_with(
        frame,
        640,
        384,
        scaling="stretch",
        cpu_fallback=False,
    )
    assert np.array_equal(out.reshape(384 * 3 // 2, 640), resized)


def test_prepare_rgb_input_uses_sdk_nv12_to_rgb_after_resize(monkeypatch):
    sc = _minimal_showcase()
    frame = _FakeFrame(1280, 720, handle=object())
    sentinel = np.full((384, 640, 3), 9, dtype=np.uint8)
    calls = []

    def fake_nv12_to_rgb(nv12, width, height):
        calls.append((nv12.shape, width, height))
        return sentinel

    monkeypatch.setattr(main, "sdk_nv12_to_rgb", fake_nv12_to_rgb)

    out = sc._prepare_input(frame, 640, 384, "rgb")

    assert frame.resize_calls == [(640, 384, "stretch")]
    assert calls == [((576, 640), 640, 384)]
    assert np.array_equal(out.reshape(384, 640, 3), sentinel)


def test_bgr_to_nv12_prefers_sdk_rgb_route(monkeypatch):
    sentinel = np.full((12, 16), 42, dtype=np.uint8)
    calls = []

    def fake_rgb_to_nv12(rgb):
        calls.append(rgb.copy())
        return sentinel

    monkeypatch.setattr(main, "sdk_rgb_to_nv12", fake_rgb_to_nv12)
    monkeypatch.setattr(main, "sdk_bgr_to_nv12", MagicMock())
    bgr = np.zeros((8, 16, 3), dtype=np.uint8)
    bgr[:, :, 0] = 10
    bgr[:, :, 2] = 200

    out = main.ModelShowcase._bgr_to_nv12(bgr)

    assert np.array_equal(out, sentinel)
    assert calls[0][0, 0].tolist() == [200, 0, 10]
    main.sdk_bgr_to_nv12.assert_not_called()


class _FakeJob:
    buffer_id = 1234

    def __init__(self):
        self.waited = False
        self.released = False

    def wait_result(self, timeout_s=None):
        self.waited = True
        self.timeout_s = timeout_s
        return self

    def release(self):
        self.released = True


class _FakeDsp:
    def __init__(self, job: _FakeJob):
        self.job = job
        self.resize_calls = []
        self.encode_calls = []
        self.closed = False

    def resize_hw(self, *args, **kwargs):
        self.resize_calls.append((args, kwargs))
        return self.job

    def encode_jpeg_hw(self, *args, **kwargs):
        self.encode_calls.append((args, kwargs))
        return b"\xff\xd8hw\xff\xd9"

    def close(self):
        self.closed = True


def test_preview_hw_encode_chains_resize_result_to_encode():
    sc = _minimal_showcase()
    sc._hw_preview_enabled = True
    sc.preview_width = 640
    sc.preview_height = 360
    sc.jpeg_quality = 70
    sc._preview_hw_retry_at = 0
    sc._preview_hw_retry_interval = 5.0
    job = _FakeJob()
    dsp = _FakeDsp(job)
    sc._preview_dsp = dsp
    frame = _FakeFrame(1280, 720, handle=object())

    jpeg = sc._encode_preview_frame_hw(frame)

    assert jpeg == b"\xff\xd8hw\xff\xd9"
    assert job.waited and job.released
    assert dsp.resize_calls[0][0][:3] == (frame, 640, 360)
    assert dsp.resize_calls[0][1]["wait"] is False
    assert dsp.resize_calls[0][1]["cpu_fallback"] is False
    assert dsp.encode_calls == [
        ((None,), {"quality": 70, "src_buffer_id": 1234, "cpu_fallback": False})
    ]
