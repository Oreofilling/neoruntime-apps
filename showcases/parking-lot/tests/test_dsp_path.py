"""Unit tests for parking-lot DSP offload routing (keep_fd + DspClient).

Covers the app-side decision logic only — which sources are eligible for
resize_hw/multi_crop_hw, how results are routed back, and the CPU-fallback
policy (quota cooldown vs permanent disable).  The DspClient is faked; no
SDK daemon is required.
"""

import sys
import os
import time

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import parking_lot.app as app_module
from parking_lot.app import ParkingLotApp
from parking_lot.config import MODEL_DEFS
from neoruntime_ipc_sdk import DspError


class FakeFrame:
    """Stand-in for a keep-fd NV12 Frame (only attrs the helpers read)."""

    def __init__(self, w: int = 1920, h: int = 1080, fmt: str = "NV12") -> None:
        self.width = w
        self.height = h
        self.format = fmt


class FakeDspClient:
    """Records hw calls; can be armed to raise a DspError."""

    def __init__(self) -> None:
        self.calls = []
        self.next_error: DspError | None = None

    def _maybe_raise(self) -> None:
        if self.next_error is not None:
            raise self.next_error

    def resize_hw(self, src, width, height, **kwargs):
        self._maybe_raise()
        self.calls.append(("resize_hw", width, height))
        # NV12 layout: Y plane + interleaved UV plane
        return np.zeros((height + height // 2, width), dtype=np.uint8)

    def multi_crop_hw(self, src, rects, **kwargs):
        self._maybe_raise()
        self.calls.append(("multi_crop_hw", list(rects), dict(kwargs)))
        return [
            np.zeros((dh + dh // 2, dw), dtype=np.uint8)
            for (_x, _y, _w, _h, dw, dh) in rects
        ]


def make_app(client=None) -> ParkingLotApp:
    """ParkingLotApp with only the DSP fields set (no __init__ side effects)."""
    app = object.__new__(ParkingLotApp)
    app._dsp_client = client
    app._dsp_enabled = True
    app._dsp_ok = True
    app._dsp_retry_ts = 0.0
    app._dsp_stats = {
        "enabled": True, "hw_jobs": 0, "cpu_fallbacks": 0, "last_error": "",
    }
    return app


# ---------------------------------------------------------------------------
# _dsp_active gating
# ---------------------------------------------------------------------------

class TestDspActive:
    def test_active_by_default(self) -> None:
        assert make_app()._dsp_active() is True

    def test_disabled_via_flag(self) -> None:
        app = make_app()
        app._dsp_enabled = False
        assert app._dsp_active() is False

    def test_disabled_after_fatal_error(self) -> None:
        app = make_app()
        app._dsp_ok = False
        assert app._dsp_active() is False

    def test_inactive_during_quota_cooldown(self) -> None:
        app = make_app()
        app._dsp_retry_ts = time.monotonic() + 5.0
        assert app._dsp_active() is False


# ---------------------------------------------------------------------------
# _dsp_model_input routing
# ---------------------------------------------------------------------------

class TestDspModelInput:
    def test_uses_resize_hw_for_nv12_source(self) -> None:
        client = FakeDspClient()
        app = make_app(client)
        # license_plate_det: rgb 416x416 != 1920x1080 source
        out = app._dsp_model_input(FakeFrame(), MODEL_DEFS["license_plate_det"])
        assert out is not None
        assert client.calls == [("resize_hw", 416, 416)]
        assert out.shape == (416 * 416 * 3,)
        assert app._dsp_stats["hw_jobs"] == 1

    def test_nv12_model_returns_flattened_nv12(self) -> None:
        client = FakeDspClient()
        app = make_app(client)
        mdef = dict(MODEL_DEFS["plate_recognition"])  # nv12 320x48
        out = app._dsp_model_input(FakeFrame(320, 240), mdef)
        assert out is not None
        assert client.calls == [("resize_hw", 320, 48)]
        assert out.shape == (320 * 48 * 3 // 2,)

    def test_same_dims_skips_dsp(self) -> None:
        client = FakeDspClient()
        app = make_app(client)
        # yolov5m_vehicles input is 1920x1080 == source dims
        out = app._dsp_model_input(
            FakeFrame(1920, 1080), MODEL_DEFS["yolov5m_vehicles"])
        assert out is None
        assert client.calls == []

    def test_non_nv12_source_skips_dsp(self) -> None:
        client = FakeDspClient()
        app = make_app(client)
        out = app._dsp_model_input(
            FakeFrame(fmt="BGR"), MODEL_DEFS["license_plate_det"])
        assert out is None
        assert client.calls == []

    def test_quota_error_sets_cooldown_not_disable(self) -> None:
        client = FakeDspClient()
        client.next_error = DspError("quota exceeded", code=-3)
        app = make_app(client)
        out = app._dsp_model_input(FakeFrame(), MODEL_DEFS["license_plate_det"])
        assert out is None
        assert app._dsp_stats["cpu_fallbacks"] == 1
        assert app._dsp_ok is True
        assert app._dsp_active() is False  # cooling down

        # After the cooldown lapses the hw path retries and succeeds.
        client.next_error = None
        app._dsp_retry_ts = 0.0
        out = app._dsp_model_input(FakeFrame(), MODEL_DEFS["license_plate_det"])
        assert out is not None

    def test_other_error_disables_for_the_run(self) -> None:
        client = FakeDspClient()
        client.next_error = DspError("dsp job failed: no such buffer id", code=-2)
        app = make_app(client)
        out = app._dsp_model_input(FakeFrame(), MODEL_DEFS["license_plate_det"])
        assert out is None
        assert app._dsp_ok is False

        # Subsequent calls short-circuit before reaching the client.
        client.next_error = None
        out = app._dsp_model_input(FakeFrame(), MODEL_DEFS["license_plate_det"])
        assert out is None
        assert len(client.calls) == 0


# ---------------------------------------------------------------------------
# _dsp_plate_tiles routing
# ---------------------------------------------------------------------------

class TestDspPlateTiles:
    BBOXES = [
        (0.30, 0.30, 0.20, 0.10),  # valid
        (0.50, 0.50, 0.00, 0.00),  # degenerate -> None slot
        (0.60, 0.20, 0.10, 0.10),  # valid
    ]

    def test_one_letterbox_multi_crop_job(self) -> None:
        client = FakeDspClient()
        app = make_app(client)
        tiles = app._dsp_plate_tiles(FakeFrame(), self.BBOXES)
        assert tiles is not None
        assert len(client.calls) == 1
        op, rects, kwargs = client.calls[0]
        assert op == "multi_crop_hw"
        assert kwargs.get("scaling") == "letterbox"
        # Degenerate box excluded from the job
        assert len(rects) == 2
        # Results scattered back in bbox order, None kept for the degenerate
        assert tiles[0] is not None
        assert tiles[1] is None
        assert tiles[2] is not None
        assert app._dsp_stats["hw_jobs"] == 1

    def test_non_nv12_source_returns_none(self) -> None:
        client = FakeDspClient()
        app = make_app(client)
        assert app._dsp_plate_tiles(FakeFrame(fmt="BGR"), self.BBOXES) is None
        assert client.calls == []

    def test_all_degenerate_boxes_return_none(self) -> None:
        client = FakeDspClient()
        app = make_app(client)
        out = app._dsp_plate_tiles(FakeFrame(), [(0.5, 0.5, 0.0, 0.0)])
        assert out is None
        assert client.calls == []

    def test_error_falls_back_to_cpu(self) -> None:
        client = FakeDspClient()
        client.next_error = DspError("dsp service not running")
        app = make_app(client)
        assert app._dsp_plate_tiles(FakeFrame(), self.BBOXES) is None
        assert app._dsp_stats["cpu_fallbacks"] == 1


# ---------------------------------------------------------------------------
# _get_dsp lazy creation
# ---------------------------------------------------------------------------

class TestGetDsp:
    def test_creates_client_once(self, monkeypatch) -> None:
        created = []

        def fake_client():
            c = FakeDspClient()
            created.append(c)
            return c

        monkeypatch.setattr(app_module, "DspClient", fake_client)
        app = make_app()
        first = app._get_dsp()
        second = app._get_dsp()
        assert first is second
        assert len(created) == 1

    def test_creation_failure_disables(self, monkeypatch) -> None:
        def boom():
            raise OSError("camera.sock: no such file")

        monkeypatch.setattr(app_module, "DspClient", boom)
        app = make_app()
        assert app._get_dsp() is None
        assert app._dsp_ok is False
        assert app._dsp_stats["cpu_fallbacks"] == 1
        assert "client-init" in app._dsp_stats["last_error"]

    def test_no_client_when_inactive(self) -> None:
        app = make_app()
        app._dsp_ok = False
        assert app._get_dsp() is None
