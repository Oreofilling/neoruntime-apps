"""
Flask web layer for the Parking Lot application.

Provides MJPEG streaming, SSE event broadcast, REST API endpoints,
and video upload/download with security hardening.
"""

import json
import logging
import os
import queue
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
)

from .config import (
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE,
    UPLOAD_DIR,
    WEB_PORT,
)
from .postprocess import SpoofResult

logger = logging.getLogger("parking-lot")


# ---------------------------------------------------------------------------
# FrameBuffer — thread-safe MJPEG frame buffer
# ---------------------------------------------------------------------------


class FrameBuffer:
    """Thread-safe buffer that holds the latest JPEG frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[bytes] = None
        self._frame_id = 0
        self._event = threading.Event()

    def put(self, jpeg_bytes: bytes) -> int:
        with self._lock:
            self._frame = jpeg_bytes
            self._frame_id += 1
            fid = self._frame_id
        self._event.set()
        return fid

    def wait_for_new(
        self, last_id: int, timeout: float = 2.0,
    ) -> Tuple[Optional[bytes], int]:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._frame_id > last_id:
                    return self._frame, self._frame_id
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self._lock:
                    return self._frame, self._frame_id
            self._event.clear()
            self._event.wait(min(remaining, 0.5))


# ---------------------------------------------------------------------------
# SSE broadcast helpers
# ---------------------------------------------------------------------------

_SSE_CLIENTS: List[queue.Queue] = []
_SSE_LOCK = threading.Lock()


def _json_default(o: Any) -> Any:
    """Coerce numpy scalars/arrays to native Python for JSON serialization.

    Model outputs (confidence, bbox coords) are often numpy.float32, which
    json.dumps cannot serialize — an unhandled TypeError here crashes the
    inference thread, since sse_broadcast runs on that thread. Duck-typed so
    we don't hard-depend on numpy at import time.
    """
    if hasattr(o, "item"):
        try:
            return o.item()  # np.float32/np.int64 -> python scalar
        except Exception:
            pass
    if hasattr(o, "tolist"):
        return o.tolist()  # np.ndarray -> list
    raise TypeError(
        f"Object of type {o.__class__.__name__} is not JSON serializable",
    )


def sse_broadcast(data: Dict[str, Any]) -> None:
    """Push a JSON payload to all connected SSE clients."""
    payload = json.dumps(data, ensure_ascii=False, default=_json_default)
    with _SSE_LOCK:
        dead: List[queue.Queue] = []
        for q in _SSE_CLIENTS:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _SSE_CLIENTS.remove(q)


# ---------------------------------------------------------------------------
# Upload filename security helper
# ---------------------------------------------------------------------------


def _safe_upload_path(filename: str) -> str:
    """Generate a safe, unique file path for an uploaded file.

    Uses a UUID-based name to prevent path-traversal attacks.
    Preserves the original file extension (lowercased, validated).
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext.lstrip(".") not in ALLOWED_EXTENSIONS:
        ext = ".mp4"
    safe_name = f"{uuid.uuid4().hex}{ext}"
    return os.path.join(UPLOAD_DIR, safe_name)


def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# HD preview config (reuses platform-api hardware H.264 stream)
# ---------------------------------------------------------------------------


def _build_hd_preview(req: Any) -> Dict[str, Any]:
    """Build the HD 1080P preview config injected into the template.

    Reuses the platform-api hardware H.264 stream
    (``ws://<lan-ip>:<port>/api/v1/h264/main``) decoded in the browser via MSE,
    so 1080P preview costs zero Python CPU. The host is taken from the request
    the browser used to reach us, so it is correct regardless of which LAN
    address was accessed. ``PLATFORM_API_TOKEN`` is forwarded as ``?token=``
    for platform-api token auth; omit it when auth is disabled.
    """
    host = (req.host or "").rsplit(":", 1)[0] or "127.0.0.1"
    port = int(os.environ.get("PLATFORM_API_PORT", "8080"))
    token = os.environ.get("PLATFORM_API_TOKEN", "")
    ws_url = f"ws://{host}:{port}/api/v1/h264/main"
    if token:
        ws_url += "?token=" + token
    enabled = os.environ.get("HD_PREVIEW_ENABLED", "1") != "0"
    return {"enabled": enabled, "wsUrl": ws_url, "token": token}


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------


def create_flask_app(parking_app: Any) -> Flask:
    """Build and configure the Flask application with all routes."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

    @app.route("/")
    def index():  # type: ignore[override]
        return render_template("index.html", hd_preview=_build_hd_preview(request))

    @app.route("/stream")
    def mjpeg_stream():  # type: ignore[override]
        def generate() -> Any:
            last_id = 0
            while True:
                frame_bytes, frame_id = parking_app.frame_buffer.wait_for_new(
                    last_id, timeout=2.0,
                )
                if frame_bytes is None:
                    continue
                last_id = frame_id
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode()
                    + frame_bytes
                    + b"\r\n"
                )

        return Response(
            generate(), mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/stats")
    def api_stats():  # type: ignore[override]
        return jsonify(parking_app.get_stats())

    @app.route("/api/alerts")
    def api_alerts():  # type: ignore[override]
        return jsonify(parking_app.get_alerts())

    @app.route("/api/events")
    def api_events():  # type: ignore[override]
        def generate() -> Any:
            q: queue.Queue = queue.Queue(maxsize=64)
            with _SSE_LOCK:
                _SSE_CLIENTS.append(q)
            try:
                while True:
                    try:
                        data = q.get(timeout=30)
                        yield f"data: {data}\n\n"
                    except queue.Empty:
                        yield ": keepalive\n\n"
            finally:
                with _SSE_LOCK:
                    if q in _SSE_CLIENTS:
                        _SSE_CLIENTS.remove(q)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/upload", methods=["POST"])
    def api_upload():  # type: ignore[override]
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not _allowed_file(f.filename):
            return jsonify(
                {"error": "Unsupported format. Use mp4/avi/mov/mkv"},
            ), 400

        # Security: generate safe path instead of using raw filename
        save_path = _safe_upload_path(f.filename)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        f.save(save_path)
        logger.info(
            "Uploaded video: %s (%d bytes)",
            save_path,
            os.path.getsize(save_path),
        )
        parking_app.switch_to_upload(save_path)
        return jsonify({"status": "processing", "file": os.path.basename(save_path)})

    @app.route("/api/mode", methods=["GET"])
    def api_get_mode():  # type: ignore[override]
        return jsonify(parking_app.get_mode())

    @app.route("/api/mode", methods=["POST"])
    def api_set_mode():  # type: ignore[override]
        data = request.get_json(silent=True) or {}
        mode = data.get("mode", "live")
        if mode == "live":
            parking_app.switch_to_live()
            return jsonify({"status": "ok", "mode": "live"})
        return jsonify({"error": "Use POST /api/upload for upload mode"}), 400

    @app.route("/api/download/<filename>")
    def api_download(filename: str):  # type: ignore[override]
        # Security: prevent path traversal
        safe_name = os.path.basename(filename)
        if safe_name != filename:
            return jsonify({"error": "Invalid filename"}), 400
        path = os.path.join(UPLOAD_DIR, safe_name)
        # Ensure resolved path is within UPLOAD_DIR
        real_upload = os.path.abspath(UPLOAD_DIR)
        real_path = os.path.abspath(path)
        if not real_path.startswith(real_upload + os.sep) and real_path != real_upload:
            return jsonify({"error": "Forbidden"}), 403
        if not os.path.isfile(path):
            return jsonify({"error": "File not found"}), 404
        return send_file(path, as_attachment=True, download_name=safe_name)

    @app.route("/api/video/control", methods=["POST"])
    def api_video_control():  # type: ignore[override]
        data = request.get_json(silent=True) or {}
        action = data.get("action", "")
        if not action:
            return jsonify({"error": "Missing 'action'"}), 400
        result = parking_app.video_control(action, **{k: v for k, v in data.items() if k != "action"})
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)

    @app.route("/api/video/status", methods=["GET"])
    def api_video_status():  # type: ignore[override]
        return jsonify(parking_app.get_video_status())

    @app.route("/api/plates")
    def api_plates():  # type: ignore[override]
        return jsonify(parking_app.get_plate_snapshots())

    @app.route("/api/plates/<snapshot_id>/image")
    def api_plate_image(snapshot_id: str):  # type: ignore[override]
        jpeg = parking_app.get_plate_snapshot_image(snapshot_id)
        if jpeg is None:
            return jsonify({"error": "Snapshot not found"}), 404
        return Response(jpeg, mimetype="image/jpeg")

    @app.route("/api/health")
    def api_health():  # type: ignore[override]
        return jsonify({"status": "ok"})

    return app
