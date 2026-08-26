"""Tests for the built-in demo video mode (assets/demo.mp4 looped through
the live pipeline): DemoVideoSource pacing/loop/errors, the config demo:*
fields (defaults + env/yaml overrides + validation), the /api/demo +
/api/config demo view with the fast-timer swap/restore, and the grid
overlay's highlight_empty red marking."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from app import ShelfOpsApp
from config import load_config
from demo import DemoVideoSource
from overlay import (CATEGORY_BGR, DET_NEG_COLOR, STATE_COLORS, draw_detections,
                     draw_slots)
from slots import SlotManager, build_grid_slots

DEMO_ENV = (
    "CONFIG_PATH", "DEMO_VIDEO_PATH", "DEMO_SPEED", "DEMO_GRID_ROWS",
    "DEMO_GRID_COLS", "DEMO_STOCKOUT_SECONDS", "DEMO_ALERT_COOLDOWN_SECONDS",
    "DEMO_DETECT_TILES", "DEMO_DETECT_THRESHOLD", "DEMO_DETECT_MAX_ITEMS",
)


@pytest.fixture(autouse=True)
def _clean_demo_env(monkeypatch):
    """Demo knobs must come from the tested source, not the host shell."""
    for var in DEMO_ENV:
        monkeypatch.delenv(var, raising=False)


def _write_video(path, frames=12, w=64, h=48, fps=10.0):
    """Tiny synthetic clip: frame 0 solid blue (BGR), the rest green."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    if not writer.isOpened():
        pytest.skip("mp4v encoder unavailable")
    for i in range(frames):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:, :] = (255, 0, 0) if i == 0 else (0, 255, 0)
        writer.write(frame)
    writer.release()
    return str(path)


# ---- DemoVideoSource -------------------------------------------------------

def test_source_info_and_first_frame(tmp_path):
    path = _write_video(tmp_path / "demo.mp4")
    src = DemoVideoSource(path, speed=0.5)
    try:
        info = src.info
        assert info["width"] == 64 and info["height"] == 48
        assert info["fps"] == pytest.approx(10.0, rel=0.01)
        assert info["frames"] == 12
        assert info["video_seconds"] == pytest.approx(1.2, rel=0.05)
        assert info["loop_seconds"] == pytest.approx(2.4, rel=0.05)
        frame = src.read_bgr()
        assert frame.shape == (48, 64, 3)
        assert frame.dtype == np.uint8
        assert frame[..., 0].mean() > 150       # BGR blue -> B channel high
        rgb = src.read_rgb()
        assert rgb[..., 2].mean() > 150         # same frame as RGB -> R high
        assert rgb[..., 0].mean() < 100
    finally:
        src.close()


def test_source_slow_speed_holds_first_frame(tmp_path):
    path = _write_video(tmp_path / "demo.mp4")
    src = DemoVideoSource(path, speed=0.001)
    try:
        for _ in range(5):
            rgb = src.read_rgb()
        assert rgb[..., 2].mean() > 150        # never left frame 0 (blue)
    finally:
        src.close()


def test_source_seek_wrap_and_rapid_reads(tmp_path, monkeypatch):
    # 90 frames @ 10 fps = 9 s of video; speed 500 -> wall loop = 18 ms.
    # A fake clock pins the paced positions: seek, decode-forward and the
    # loop-restart backward seek each get exercised deterministically.
    path = _write_video(tmp_path / "demo.mp4", frames=90)
    import demo as demo_mod
    clock = {"now": 1000.0}
    monkeypatch.setattr(demo_mod.time, "monotonic", lambda: clock["now"])
    src = DemoVideoSource(path, speed=500.0)
    try:
        def blue(frame):        # frame 0 is the only blue (BGR) frame
            return frame[..., 0].mean() > 150

        assert blue(src.read_bgr())            # t0 -> frame 0
        clock["now"] += 0.05                   # 25 s in -> seek to frame 60
        assert not blue(src.read_bgr())
        clock["now"] += 0.002                  # 26 s in -> decode forward
        assert not blue(src.read_bgr())
        clock["now"] = 1000.0 + 9.05 / 500     # wrap -> target back at 0
        assert blue(src.read_bgr())            # loop restarted
        clock["now"] += 0.05                   # stalled consumer catch-up
        for _ in range(50):
            assert src.read_bgr().shape == (48, 64, 3)
    finally:
        src.close()


def test_source_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        DemoVideoSource(str(tmp_path / "a.mp4"), speed=0)
    with pytest.raises(FileNotFoundError):
        DemoVideoSource(str(tmp_path / "missing.mp4"))
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video" * 64)
    with pytest.raises(ValueError):
        DemoVideoSource(str(bad))


# ---- config demo fields ----------------------------------------------------

def test_config_demo_defaults():
    cfg = load_config()
    assert cfg.demo_video_path == "assets/demo.mp4"
    assert cfg.demo_speed == 0.5
    assert cfg.demo_grid_region == (0.02, 0.04, 0.98, 0.99)
    assert cfg.demo_grid_rows == 2
    assert cfg.demo_grid_cols == 8
    assert cfg.demo_stockout_seconds == 3.0
    assert cfg.demo_alert_cooldown_seconds == 3
    assert cfg.demo_detect_tiles == 4
    assert cfg.demo_detect_threshold == pytest.approx(0.15)
    assert cfg.demo_detect_max_items == 200


def test_config_demo_env_overrides(monkeypatch):
    monkeypatch.setenv("DEMO_SPEED", "1.25")
    monkeypatch.setenv("DEMO_GRID_ROWS", "3")
    monkeypatch.setenv("DEMO_GRID_COLS", "7")
    monkeypatch.setenv("DEMO_VIDEO_PATH", "/tmp/custom.mp4")
    cfg = load_config()
    assert cfg.demo_speed == 1.25
    assert cfg.demo_grid_rows == 3
    assert cfg.demo_grid_cols == 7
    assert cfg.demo_video_path == "/tmp/custom.mp4"


def test_config_demo_yaml_overlay(tmp_path):
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump({
        "demo": {
            "speed": 2.0,
            "grid_rows": 5,
            "grid_cols": 9,
            "video_path": "/tmp/v.mp4",
            "grid_region": [0.1, 0.2, 0.9, 0.95],
            "stockout_seconds": 4.5,
            "alert_cooldown_seconds": 7,
        },
    }))
    import os
    os.environ["CONFIG_PATH"] = str(overlay)
    try:
        cfg = load_config()
    finally:
        del os.environ["CONFIG_PATH"]
    assert cfg.demo_speed == 2.0
    assert cfg.demo_grid_rows == 5
    assert cfg.demo_grid_cols == 9
    assert cfg.demo_video_path == "/tmp/v.mp4"
    assert cfg.demo_grid_region == (0.1, 0.2, 0.9, 0.95)
    assert cfg.demo_stockout_seconds == 4.5
    assert cfg.demo_alert_cooldown_seconds == 7


def test_config_demo_yaml_bad_region(tmp_path):
    overlay = tmp_path / "bad.yaml"
    overlay.write_text(yaml.safe_dump({"demo": {"grid_region": [0.1, 0.2]}}))
    import os
    os.environ["CONFIG_PATH"] = str(overlay)
    try:
        with pytest.raises(ValueError, match="demo.grid_region"):
            load_config()
    finally:
        del os.environ["CONFIG_PATH"]


@pytest.mark.parametrize("env,val,match", [
    ("DEMO_SPEED", "0", "DEMO_SPEED"),
    ("DEMO_STOCKOUT_SECONDS", "0", "DEMO_STOCKOUT_SECONDS"),
    ("DEMO_ALERT_COOLDOWN_SECONDS", "-1", "DEMO_ALERT_COOLDOWN_SECONDS"),
    ("DEMO_DETECT_TILES", "3", "DEMO_DETECT_TILES"),
])
def test_config_demo_validation(monkeypatch, env, val, match):
    monkeypatch.setenv(env, val)
    with pytest.raises(ValueError, match=match):
        load_config()


# ---- /api/demo route + timer swap ------------------------------------------

def _make_app(tmp_path, video):
    from config import ShelfConfig
    cfg = ShelfConfig()
    cfg.analytics_db = str(tmp_path / "shelf.db")
    cfg.demo_video_path = str(video)
    cfg.demo_speed = 0.5
    cfg.demo_grid_rows = 2
    cfg.demo_grid_cols = 3
    cfg.demo_stockout_seconds = 1.5
    cfg.demo_alert_cooldown_seconds = 2
    sa = ShelfOpsApp(cfg)
    sa._register_routes()          # start() normally does this
    return sa, sa.app.test_client()


def test_demo_route_get_initial_state(tmp_path):
    video = _write_video(tmp_path / "demo.mp4")
    sa, client = _make_app(tmp_path, video)
    body = client.get("/api/demo").get_json()
    assert body["ok"] is True
    demo = body["demo"]
    assert demo["enabled"] is False
    assert demo["available"] is True
    assert demo["video"] is None
    assert demo["grid"]["rows"] == 2
    assert demo["grid"]["cols"] == 3
    assert demo["grid"]["show_cells"] is True


def test_demo_route_post_validates_body(tmp_path):
    video = _write_video(tmp_path / "demo.mp4")
    sa, client = _make_app(tmp_path, video)
    assert client.post("/api/demo", json={"enabled": "yes"}).status_code == 400
    assert client.post("/api/demo", data="not json").status_code == 400
    assert client.post("/api/demo", json={}).status_code == 400


def test_demo_toggle_swaps_and_restores_timers(tmp_path):
    video = _write_video(tmp_path / "demo.mp4")
    sa, client = _make_app(tmp_path, video)
    live_stockout = sa.analytics.stockout_seconds
    live_cooldown = sa.broker.cooldown_seconds
    assert live_stockout != 1.5          # demo value really is different

    resp = client.post("/api/demo", json={"enabled": True})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["demo"]["enabled"] is True
    assert body["demo"]["video"]["loop_seconds"] > 0
    assert sa.analytics.stockout_seconds == 1.5
    assert sa.broker.cooldown_seconds == 2
    assert client.get("/api/health").get_json()["demo"] is True

    # idempotent re-toggle keeps the swapped timers
    assert client.post("/api/demo", json={"enabled": True}).status_code == 200
    assert sa.analytics.stockout_seconds == 1.5

    resp = client.post("/api/demo", json={"enabled": False})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["demo"]["enabled"] is False
    assert body["demo"]["video"] is None
    assert sa.analytics.stockout_seconds == live_stockout
    assert sa.broker.cooldown_seconds == live_cooldown
    assert client.get("/api/health").get_json()["demo"] is False


def test_demo_config_view_uses_d_prefixed_cells(tmp_path):
    video = _write_video(tmp_path / "demo.mp4")
    sa, client = _make_app(tmp_path, video)
    live_ids = [s["id"] for s in client.get("/api/config").get_json()["slots"]]
    assert not any(i.startswith("d") for i in live_ids)

    client.post("/api/demo", json={"enabled": True})
    body = client.get("/api/config").get_json()
    assert [s["id"] for s in body["slots"]] == [
        "d1-1", "d1-2", "d1-3", "d2-1", "d2-2", "d2-3"]
    assert body["mode"] == "grid"
    assert body["grid"]["rows"] == 2
    assert body["grid"]["show_cells"] is True
    assert body["demo"]["enabled"] is True


def test_demo_missing_video_returns_409(tmp_path):
    sa, client = _make_app(tmp_path, tmp_path / "never.mp4")
    live_stockout = sa.analytics.stockout_seconds
    resp = client.post("/api/demo", json={"enabled": True})
    assert resp.status_code == 409
    body = resp.get_json()
    assert body["ok"] is False
    assert body["demo"]["enabled"] is False
    assert sa.analytics.stockout_seconds == live_stockout


# ---- grid overlay highlight_empty ------------------------------------------

def _snap(sid, state, count, code="", score=0.9, by_code=None):
    return SimpleNamespace(slot_id=sid, state=state, count=count, code=code,
                           score=score, by_code=by_code or {})


def _red_pixels(frame):
    return int((frame == STATE_COLORS["EMPTY"]).all(axis=2).sum())


def test_grid_overlay_marks_confirmed_empty_red():
    slots = SlotManager(build_grid_slots((0.0, 0.0, 1.0, 1.0), 1, 2)).slots
    snaps = [_snap("g1-1", "FULL", 2, "A", 0.9, {"A": 2}),
             _snap("g1-2", "EMPTY", 0)]
    frame_on = np.full((120, 160, 3), 40, np.uint8)
    draw_slots(frame_on, snaps, slots, grid_region=(0.0, 0.0, 1.0, 1.0),
               items=3, highlight_empty=True)
    assert _red_pixels(frame_on) > 0

    frame_off = np.full((120, 160, 3), 40, np.uint8)
    draw_slots(frame_off, snaps, slots, grid_region=(0.0, 0.0, 1.0, 1.0),
               items=3, highlight_empty=False)
    assert _red_pixels(frame_off) == 0


def test_grid_overlay_never_read_cell_not_marked():
    slots = SlotManager(build_grid_slots((0.0, 0.0, 1.0, 1.0), 1, 2)).slots
    frame = np.full((120, 160, 3), 40, np.uint8)
    draw_slots(frame, [], slots, grid_region=(0.0, 0.0, 1.0, 1.0),
               highlight_empty=True)
    assert _red_pixels(frame) == 0


# ---- demo view: items-only overlay -----------------------------------------

def _det(box, label, goods, category="other", score=0.5, cell=None):
    d = {"box": box, "label": label, "score": score,
         "category": category, "goods": goods}
    if cell:
        d["cell"] = cell
    return d


def test_draw_detections_goods_only_hides_negatives():
    dets = [_det((0.1, 0.1, 0.4, 0.4), "bottle", True, "bottle", 0.5, "d1-1"),
            _det((0.5, 0.5, 0.8, 0.8), "shelf", False, 0.4)]
    frame_all = np.full((120, 160, 3), 40, np.uint8)
    draw_detections(frame_all, dets)
    assert (frame_all == DET_NEG_COLOR).all(axis=2).sum() > 0   # gray box drawn

    frame_items = np.full((120, 160, 3), 40, np.uint8)
    draw_detections(frame_items, dets, goods_only=True)
    assert (frame_items == DET_NEG_COLOR).all(axis=2).sum() == 0
    assert (frame_items == CATEGORY_BGR["bottle"]).all(axis=2).sum() > 0


def test_demo_overlay_policy_is_items_only(tmp_path, monkeypatch):
    video = _write_video(tmp_path / "demo.mp4")
    sa, client = _make_app(tmp_path, video)
    import app as app_mod
    calls = {"slots": 0, "goods_only": []}
    monkeypatch.setattr(app_mod, "draw_slots",
                        lambda *a, **k: calls.__setitem__("slots",
                                                          calls["slots"] + 1))
    monkeypatch.setattr(app_mod, "draw_detections",
                        lambda f, d, goods_only=False:
                            calls["goods_only"].append(goods_only))

    frame = np.full((48, 64, 3), 20, np.uint8)
    snaps = [_snap("g1-1", "FULL", 1, "A")]
    dets = [_det((0.1, 0.1, 0.3, 0.3), "bottle", True, "bottle")]

    sa._draw_overlay(frame.copy(), snaps, dets)      # live: grid + both kinds
    assert calls["slots"] == 1
    assert calls["goods_only"] == [False]

    client.post("/api/demo", json={"enabled": True})
    sa._draw_overlay(frame.copy(), snaps, dets)      # demo: boxes only
    assert calls["slots"] == 1                        # grid layer skipped
    assert calls["goods_only"] == [False, True]
