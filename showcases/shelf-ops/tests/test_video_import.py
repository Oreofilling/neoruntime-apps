"""Tests for the offline video import (/api/import/video): validation
rejections at the route (415), the unopenable-container 422, the single-job
409 while a worker holds the import lock, and the full sim-mode job
lifecycle — 202 + polled status ending in a done result carrying the
timeline / keyframes / aggregate fields the modal renders."""
import io
import tempfile
import time

import pytest

from app import ShelfOpsApp
from config import ShelfConfig

from test_demo import _write_video

_MP4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 64


def _make_app(tmp_path, detect_delay=0.0):
    cfg = ShelfConfig()
    cfg.analytics_db = str(tmp_path / "shelf.db")
    sa = ShelfOpsApp(cfg)
    sa._register_routes()          # start() normally does this
    if detect_delay:
        def slow(rgb):
            time.sleep(detect_delay)
            return sa._synthetic_detections(time.time())
        sa._detect_import_frame = slow   # instance attr shadows the method
    return sa, sa.app.test_client()


def _post_video(client, raw, name="clip.mp4"):
    return client.post(
        "/api/import/video",
        data={"file": (io.BytesIO(raw), name)},
        content_type="multipart/form-data")


def _poll_done(client, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get("/api/import/video/status").get_json()["job"]
        if job and job["state"] == "done":
            return job
        if job and job["state"] == "error":
            raise AssertionError(f"job errored: {job['error']}")
        time.sleep(0.1)
    raise AssertionError("job did not finish in time")


# ---- validation ------------------------------------------------------------


def test_status_is_null_before_any_import(tmp_path):
    _, client = _make_app(tmp_path)
    body = client.get("/api/import/video/status").get_json()
    assert body == {"ok": True, "job": None}


def test_rejects_garbage_upload(tmp_path):
    _, client = _make_app(tmp_path)
    assert _post_video(client, b"definitely not a video").status_code == 415
    # valid ftyp magic but truncated payload: cv2 cannot open it -> 422
    assert _post_video(client, _MP4).status_code == 422


# ---- job lifecycle ---------------------------------------------------------


def test_video_import_full_job(tmp_path, monkeypatch):
    raw = open(_write_video(tmp_path / "src.mp4", frames=12), "rb").read()
    # keep the worker's temp file inside tmp_path so cleanup is assertable
    real_mkstemp = tempfile.mkstemp

    def mkstemp_here(suffix=None, **kw):
        return real_mkstemp(suffix=suffix, dir=str(tmp_path))
    monkeypatch.setattr(tempfile, "mkstemp", mkstemp_here)

    sa, client = _make_app(tmp_path)
    res = _post_video(client, raw)
    assert res.status_code == 202
    started = res.get_json()["job"]
    assert started["state"] == "running"
    assert started["total"] == 12           # 12 frames <= cap 48 -> all kept
    assert started["video"]["frames"] == 12
    assert started["video"]["duration_seconds"] == pytest.approx(1.2, abs=0.2)

    job = _poll_done(client)
    assert job["done"] == job["total"] == 12
    r = job["result"]
    assert r["sampled"] == 12
    assert len(r["timeline"]) == 12
    assert all(0 <= s["items"] <= 10 for s in r["timeline"])
    # sim detections are 3-5 goods -> the aggregates are populated
    assert 3 <= r["items_peak"] <= 5
    assert 0.0 <= r["peak_t"] <= 1.3
    assert 0 < r["items_mean"] <= 5
    assert sum(r["items_by_category"].values()) == r["items_peak"]
    # keyframes: 6 of the 12 samples, annotated JPEG stills
    assert len(r["keyframes"]) == 6
    assert {kf["t"] for kf in r["keyframes"]} <= {s["t"] for s in r["timeline"]}
    for kf in r["keyframes"]:
        assert len(kf["image"]) > 100 and kf["image"][:4] == "/9j/"
        assert 3 <= kf["items"] <= 5
    # the worker cleaned its temp file up (src.mp4 itself stays)
    leftovers = [p for p in tmp_path.glob("*.mp4") if p.name != "src.mp4"]
    assert leftovers == []


def test_second_import_conflicts_with_running_job(tmp_path):
    raw = open(_write_video(tmp_path / "src.mp4", frames=8), "rb").read()
    _, client = _make_app(tmp_path, detect_delay=0.8)
    assert _post_video(client, raw).status_code == 202
    # the worker sleeps per frame -> the lock is held; a second POST 409s
    assert _post_video(client, raw).status_code == 409
    job = _poll_done(client)
    assert job["state"] == "done" and job["result"]["sampled"] == 8
    # lock released after done -> a new import is accepted again
    assert _post_video(client, raw).status_code == 202
    _poll_done(client)


def test_broken_stream_reports_error_state(tmp_path, monkeypatch):
    raw = open(_write_video(tmp_path / "src.mp4", frames=6), "rb").read()

    def boom(rgb):
        raise RuntimeError("npu exploded")
    sa, client = _make_app(tmp_path)
    monkeypatch.setattr(sa, "_detect_import_frame", boom)
    assert _post_video(client, raw).status_code == 202
    deadline = time.time() + 10
    job = None
    while time.time() < deadline:
        job = client.get("/api/import/video/status").get_json()["job"]
        if job and job["state"] == "error":
            break
        time.sleep(0.1)
    assert job and job["state"] == "error"
    assert "npu exploded" in job["error"]
    # the failed worker still released the import lock
    assert _post_video(client, raw).status_code == 202
