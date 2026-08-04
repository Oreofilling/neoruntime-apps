"""Integration tests for the Flask web API layer.

Uses Flask test client to verify endpoints without running a server.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parking_lot.web import create_flask_app, FrameBuffer, UPLOAD_DIR


def _make_mock_app() -> MagicMock:
    """Create a mock ParkingLotApp with all methods needed by routes."""
    app = MagicMock()
    app.frame_buffer = FrameBuffer()
    app.get_stats.return_value = {
        "fps": 20.0,
        "infer_ms": 45.2,
        "frame_count": 100,
        "vehicles": 3,
        "plates": 2,
        "alerts": 1,
    }
    app.get_alerts.return_value = [
        {"type": "spoof_detected", "detail": "car score=0.85", "time": "10:30:00"},
    ]
    app.get_mode.return_value = {"mode": "live", "progress": {"status": "idle"}}
    app.get_plate_snapshots.return_value = [
        {"id": "abc123def456", "plate": "ABC123", "confidence": 0.95, "time": "10:30:00"},
    ]
    app.get_plate_snapshot_image.return_value = b"\xff\xd8\xff\xe0fake_jpeg_data"
    app.video_control.return_value = {
        "active": True, "paused": False, "speed": 1.0,
        "position_sec": 5.2, "duration": 30.0, "progress": 0.17,
        "total_frames": 750, "fps": 25.0,
    }
    app.get_video_status.return_value = {
        "active": True, "paused": False, "speed": 1.0,
        "position_sec": 5.2, "duration": 30.0, "progress": 0.17,
        "total_frames": 750, "fps": 25.0,
    }
    return app


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_app = _make_mock_app()
        self.client = create_flask_app(self.mock_app).test_client()

    def test_health_returns_ok(self) -> None:
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")


class TestStatsEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_app = _make_mock_app()
        self.client = create_flask_app(self.mock_app).test_client()

    def test_stats_returns_expected_fields(self) -> None:
        resp = self.client.get("/api/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("fps", data)
        self.assertIn("infer_ms", data)
        self.assertIn("frame_count", data)
        self.assertIn("vehicles", data)
        self.assertIn("plates", data)
        self.assertIn("alerts", data)

    def test_stats_values_match_mock(self) -> None:
        resp = self.client.get("/api/stats")
        data = resp.get_json()
        self.assertEqual(data["vehicles"], 3)
        self.assertEqual(data["plates"], 2)


class TestAlertsEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_app = _make_mock_app()
        self.client = create_flask_app(self.mock_app).test_client()

    def test_alerts_returns_list(self) -> None:
        resp = self.client.get("/api/alerts")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["type"], "spoof_detected")


class TestUploadEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_app = _make_mock_app()
        self.client = create_flask_app(self.mock_app).test_client()

    def test_upload_rejects_no_file(self) -> None:
        resp = self.client.post("/api/upload")
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)

    def test_upload_rejects_empty_filename(self) -> None:
        data = {"file": (tempfile.SpooledTemporaryFile(max_size=100), "")}
        resp = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_upload_rejects_bad_extension(self) -> None:
        data = {"file": (tempfile.SpooledTemporaryFile(max_size=100), "evil.exe")}
        resp = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        result = resp.get_json()
        self.assertIn("Unsupported", result["error"])


class TestModeEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_app = _make_mock_app()
        self.client = create_flask_app(self.mock_app).test_client()

    def test_get_mode(self) -> None:
        resp = self.client.get("/api/mode")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["mode"], "live")

    def test_set_mode_live(self) -> None:
        resp = self.client.post(
            "/api/mode",
            data=json.dumps({"mode": "live"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["mode"], "live")
        self.mock_app.switch_to_live.assert_called_once()

    def test_set_mode_upload_rejected(self) -> None:
        resp = self.client.post(
            "/api/mode",
            data=json.dumps({"mode": "upload"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)


class TestDownloadSecurity(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_app = _make_mock_app()
        self.client = create_flask_app(self.mock_app).test_client()

    def test_download_rejects_traversal(self) -> None:
        resp = self.client.get("/api/download/../../etc/passwd")
        self.assertIn(resp.status_code, [400, 403, 404])

    def test_download_rejects_absolute_path(self) -> None:
        resp = self.client.get("/api/download//etc/passwd")
        self.assertIn(resp.status_code, [400, 403, 404])

    def test_download_rejects_dotdot(self) -> None:
        resp = self.client.get("/api/download/../../../etc/shadow")
        self.assertIn(resp.status_code, [400, 403, 404])

    def test_download_missing_file(self) -> None:
        resp = self.client.get("/api/download/nonexistent.mp4")
        self.assertEqual(resp.status_code, 404)


class TestIndexRoute(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_app = _make_mock_app()
        flask_app = create_flask_app(self.mock_app)
        # Point template folder at the actual templates directory
        template_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
        flask_app.template_folder = template_dir
        self.client = flask_app.test_client()

    def test_index_returns_html(self) -> None:
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Parking Lot Monitor", resp.data)


class TestPlateSnapshotEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_app = _make_mock_app()
        self.client = create_flask_app(self.mock_app).test_client()

    def test_plates_returns_list(self) -> None:
        resp = self.client.get("/api/plates")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["plate"], "ABC123")
        self.assertIn("id", data[0])
        self.assertIn("confidence", data[0])
        self.assertIn("time", data[0])

    def test_plate_image_returns_jpeg(self) -> None:
        resp = self.client.get("/api/plates/abc123def456/image")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content_type, "image/jpeg")
        self.assertTrue(len(resp.data) > 0)

    def test_plate_image_not_found(self) -> None:
        self.mock_app.get_plate_snapshot_image.return_value = None
        resp = self.client.get("/api/plates/nonexistent/image")
        self.assertEqual(resp.status_code, 404)
        data = resp.get_json()
        self.assertIn("error", data)


class TestVideoControlEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_app = _make_mock_app()
        self.client = create_flask_app(self.mock_app).test_client()

    def test_video_control_pause(self) -> None:
        resp = self.client.post(
            "/api/video/control",
            data=json.dumps({"action": "pause"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["active"])
        # Route strips 'action' from forwarded kwargs (web.py) to avoid
        # TypeError: video_control() got multiple values for argument 'action'.
        # So only the positional action is passed; no other kwargs for pause.
        self.mock_app.video_control.assert_called_once_with("pause")

    def test_video_control_seek(self) -> None:
        resp = self.client.post(
            "/api/video/control",
            data=json.dumps({"action": "seek", "position_sec": 10.0}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        # 'action' is stripped by the route; only position_sec survives as a kwarg.
        self.mock_app.video_control.assert_called_once_with(
            "seek", position_sec=10.0,
        )

    def test_video_control_rejects_missing_action(self) -> None:
        resp = self.client.post(
            "/api/video/control",
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_video_control_rejects_unknown_action(self) -> None:
        self.mock_app.video_control.return_value = {"error": "Unknown action: foo"}
        resp = self.client.post(
            "/api/video/control",
            data=json.dumps({"action": "foo"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_video_status(self) -> None:
        resp = self.client.get("/api/video/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["active"])
        self.assertIn("paused", data)
        self.assertIn("speed", data)
        self.assertIn("progress", data)

    def test_video_status_no_video(self) -> None:
        self.mock_app.get_video_status.return_value = {"active": False}
        resp = self.client.get("/api/video/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data["active"])


if __name__ == "__main__":
    unittest.main()
