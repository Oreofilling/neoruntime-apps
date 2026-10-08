"""HTTP layer + entrypoint for the teaching app.

Single port (HTTP_PORT, default 8090): the page, the MJPEG stream and
the JSON API all live here, so the app works both directly and behind
the console's /apps/sdk-teaching-demo/ reverse proxy. The engine side
lives in teaching_app.py / sdk_helpers.py; this module only serves.
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from station_snippets import payload as snippets_payload
from teaching_app import TeachingApp

HTTP_PORT = int(os.environ.get("HTTP_PORT", "8090"))
INDEX_PATH = Path(__file__).resolve().parent / "templates" / "index.html"


def _load_index():
    try:
        return INDEX_PATH.read_bytes()
    except OSError:
        return (b"<!doctype html><title>sdk-teaching-demo</title>"
                b"<p>templates/index.html missing from the image</p>")


INDEX_HTML = _load_index()


class ApiHandler(BaseHTTPRequestHandler):
    app: TeachingApp = None  # injected via the BoundHandler subclass
    server_version = "sdk-teaching-demo/1.2"

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_bytes(200, INDEX_HTML, "text/html; charset=utf-8")
        elif path == "/stream.mjpg":
            self._stream_mjpeg()
        elif path == "/api/health":
            self._send_json(200, {"success": True, "data": {
                "ok": True,
                "app": self.app.app_id,
                "uptime_s": round(
                    time.monotonic() - self.app.started_monotonic, 1),
                "bform_running": self.app.bform.running,
                "aform_running": self.app.aform.running,
            }})
        elif path == "/api/status":
            self._send_json(200, {"success": True,
                                  "data": self.app.status()})
        elif path == "/api/snippets":
            self._send_json(200, {"success": True,
                                  "data": snippets_payload()})
        else:
            self._send_json(404, {"success": False, "error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/settings":
            body, error = self._read_json()
            if error:
                self._send_json(400, {"success": False, "error": error})
                return
            ok, payload = self.app.apply_settings(body)
            if ok:
                self._send_json(200, {"success": True, "data": payload})
            else:
                self._send_json(400, {"success": False, "error": payload})
        elif path == "/api/routing/refusal":
            self._send_json(200, {"success": True, "data": {
                "message": self.app.refusal_demo()}})
        else:
            self._send_json(404, {"success": False, "error": "not found"})

    # -- helpers -----------------------------------------------------------

    def _stream_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        last_id = 0
        try:
            while self.app.running:
                jpeg, frame_id = self.app.buffer.wait_for_new(
                    last_id, timeout=2.0)
                if jpeg is None or frame_id <= last_id:
                    continue
                last_id = frame_id
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode()
                    + b"\r\n\r\n" + jpeg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser closed the tab; nothing to clean up

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw or b"{}"), None
        except (ValueError, json.JSONDecodeError) as exc:
            return None, f"invalid JSON body: {exc}"

    def _send_json(self, code, envelope):
        self._send_bytes(code, json.dumps(envelope).encode(),
                         "application/json; charset=utf-8")

    def _send_bytes(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # route noise stays out of app logs
        pass


def make_server(app):
    handler = type("BoundHandler", (ApiHandler,), {"app": app})
    return ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), handler)


def main():
    return TeachingApp().run(make_server)


if __name__ == "__main__":
    sys.exit(main())
