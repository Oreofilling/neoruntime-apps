"""Tests for playing an IMPORTED video through the live pipeline: the
/api/import/video worker persists its upload into the media slot next to
the analytics db, POST /api/demo gains a "source" switch ("builtin" vs
"imported"), a missing import or bad source fails cleanly without
touching demo state, and switching source while demo runs swaps the
decoder in place."""
import io
import os
import tempfile
import time

from app import ShelfOpsApp
from config import ShelfConfig

from test_demo import _write_video


def _make_app(tmp_path):
    cfg = ShelfConfig()
    cfg.analytics_db = str(tmp_path / "shelf.db")
    cfg.demo_video_path = str(tmp_path / "builtin.mp4")
    cfg.demo_speed = 0.5
    cfg.demo_grid_rows = 2
    cfg.demo_grid_cols = 3
    cfg.demo_stockout_seconds = 1.5
    cfg.demo_alert_cooldown_seconds = 2
    sa = ShelfOpsApp(cfg)
    sa._register_routes()          # start() normally does this
    return sa, sa.app.test_client()


def _post_demo(client, body):
    return client.post("/api/demo", json=body)


# ---- payload + validation ----------------------------------------------------


def test_initial_payload_reports_source_and_import_slot(tmp_path):
    _write_video(tmp_path / "builtin.mp4")
    sa, client = _make_app(tmp_path)
    d = client.get("/api/demo").get_json()["demo"]
    assert d["enabled"] is False
    assert d["source"] == "builtin"
    assert d["available"] is True           # bundled video exists
    assert d["imported_available"] is False  # nothing imported yet
    assert sa._imported_video_path() == str(tmp_path / "media" / "imported.mp4")


def test_bad_source_is_rejected_without_state_change(tmp_path):
    _write_video(tmp_path / "builtin.mp4")
    _, client = _make_app(tmp_path)
    res = _post_demo(client, {"enabled": True, "source": "dvd"})
    assert res.status_code == 400
    assert "source" in res.get_json()["error"]
    assert client.get("/api/demo").get_json()["demo"]["enabled"] is False


def test_imported_source_without_upload_is_409(tmp_path):
    _write_video(tmp_path / "builtin.mp4")
    _, client = _make_app(tmp_path)
    res = _post_demo(client, {"enabled": True, "source": "imported"})
    assert res.status_code == 409
    assert "no imported video" in res.get_json()["message"]
    assert client.get("/api/demo").get_json()["demo"]["enabled"] is False


# ---- source selection ---------------------------------------------------------


def test_enable_imported_then_swap_to_builtin_while_running(tmp_path):
    _write_video(tmp_path / "builtin.mp4")
    imported = _write_video(tmp_path / "upload.mp4", frames=8, w=32, h=24)
    sa, client = _make_app(tmp_path)
    dest = sa._imported_video_path()
    os.makedirs(os.path.dirname(dest))
    os.replace(imported, dest)

    res = _post_demo(client, {"enabled": True, "source": "imported"})
    assert res.status_code == 200
    d = res.get_json()["demo"]
    assert d["enabled"] is True and d["source"] == "imported"
    assert d["video"]["path"] == dest
    assert d["video"]["width"] == 32          # the upload's geometry, not the
    assert d["video"]["height"] == 24         # bundled video's

    # swapping while running keeps demo on and only changes the decoder
    res = _post_demo(client, {"enabled": True, "source": "builtin"})
    assert res.status_code == 200
    assert res.get_json()["message"] == "demo source swapped"
    d = res.get_json()["demo"]
    assert d["enabled"] is True and d["source"] == "builtin"
    assert d["video"]["path"] == str(tmp_path / "builtin.mp4")

    # disable restores timers; the source choice persists for the next enable
    assert _post_demo(client, {"enabled": False}).status_code == 200
    d = client.get("/api/demo").get_json()["demo"]
    assert d["enabled"] is False and d["source"] == "builtin"


def test_enable_without_source_reuses_last_choice(tmp_path):
    _write_video(tmp_path / "builtin.mp4")
    _write_video(tmp_path / "upload.mp4", frames=8)
    sa, client = _make_app(tmp_path)
    dest = sa._imported_video_path()
    os.makedirs(os.path.dirname(dest))
    os.replace(tmp_path / "upload.mp4", dest)

    res = _post_demo(client, {"enabled": True, "source": "imported"})
    assert res.status_code == 200
    assert _post_demo(client, {"enabled": False}).status_code == 200
    # no source given: plays the imported upload again, not the builtin
    res = _post_demo(client, {"enabled": True})
    assert res.status_code == 200
    assert res.get_json()["demo"]["video"]["path"] == dest


# ---- import -> playback integration -------------------------------------------


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


def test_video_import_persists_upload_for_playback(tmp_path):
    _write_video(tmp_path / "builtin.mp4")
    raw = open(_write_video(tmp_path / "src.mp4", frames=8), "rb").read()
    sa, client = _make_app(tmp_path)

    assert _post_video(client, raw).status_code == 202
    assert _poll_done(client)["state"] == "done"
    # the analyzed upload now sits in the persistent media slot, byte-identical
    dest = sa._imported_video_path()
    assert os.path.isfile(dest)
    assert open(dest, "rb").read() == raw
    assert client.get("/api/demo").get_json()["demo"]["imported_available"] is True

    # and demo can immediately play it through the live pipeline
    res = _post_demo(client, {"enabled": True, "source": "imported"})
    assert res.status_code == 200
    assert res.get_json()["demo"]["enabled"] is True


def test_unwritable_media_dir_does_not_fail_the_import(tmp_path):
    _write_video(tmp_path / "builtin.mp4")
    raw = open(_write_video(tmp_path / "src.mp4", frames=6), "rb").read()
    sa, client = _make_app(tmp_path)
    # a FILE where the media/ dir would go makes persistence impossible
    (tmp_path / "media").write_text("occupied")
    assert _post_video(client, raw).status_code == 202
    job = _poll_done(client)
    assert job["state"] == "done" and job["result"]["sampled"] == 6
    assert not os.path.isfile(sa._imported_video_path())


def test_cross_device_fallback_copies_and_cleans_up(tmp_path, monkeypatch):
    """On device the media slot lives on a bind mount while the upload
    temp file is container-local: os.replace fails EXDEV and the worker
    must fall back to copy — keeping the destination AND removing the
    original instead of leaking it in the temp dir."""
    _write_video(tmp_path / "builtin.mp4")
    raw = open(_write_video(tmp_path / "src.mp4", frames=6), "rb").read()
    sa, client = _make_app(tmp_path)
    updir = tmp_path / "up"
    updir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(updir))

    def _exdev(*_a, **_k):
        raise OSError(18, "cross-device link")
    monkeypatch.setattr(os, "replace", _exdev)

    assert _post_video(client, raw).status_code == 202
    job = _poll_done(client)
    assert job["state"] == "done" and job["result"]["sampled"] == 6
    dest = sa._imported_video_path()
    assert os.path.isfile(dest)
    assert open(dest, "rb").read() == raw
    assert list(updir.iterdir()) == []       # original upload not leaked
