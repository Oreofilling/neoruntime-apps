#!/usr/bin/env python3
"""05-live-detection — a live detection loop with its own web page.

Rung 5 of the examples ladder (01-07) — capstone of the pixel path and the code-level
twin of sdk-teaching-demo station 1. Pixels flow through YOUR app —
the "B form": you pull frames, you run inference, you draw, you serve.

    FdMediaClient.subscribe     InferencePipeline.run      draw_detections
        (your frames)       (model + geometry handled)       (your pixels)
                                                                        |
                                                     Frame(...).to_jpeg_bytes
                                                                        |
                                    latest-JPEG buffer --> /  (status JSON)
                                                            /snapshot.jpg
                                                            /stream  (MJPEG)

New ideas vs rungs 02-04: the loop never exits, the buffer keeps only
the newest JPEG (viewers always get "now", memory stays flat), and the
app is a web server — network mode host + an inbound port in app.yaml,
plus register_web_url("/") so the console can reverse-proxy it.
"""

import json
import logging
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from neoruntime_ipc_sdk import (
    AppClient,
    FdMediaClient,
    Frame,
    InferenceClient,
    InferencePipeline,
    Preprocessor,
    draw_detections,
)

STREAM_ID = os.environ.get("STREAM_ID", "main")
# Injected from app.yaml spec.models (key "detector"), like rung 03.
MODEL_ID = os.environ.get("AIPC_MODEL_detector", "yolov8n")
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8090"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.3"))

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="[%(levelname)s] %(message)s")
logger = logging.getLogger("live-detection")

INDEX_HTML = b"""<!doctype html>
<html><head><title>05-live-detection</title>
<style>body{background:#111;color:#ddd;font-family:sans-serif;margin:0;
display:grid;place-items:center;height:100vh}img{max-width:96vw;
max-height:90vh;border:1px solid #444}</style></head>
<body><div><img src="/stream" alt="live detection stream">
<p id="s" style="text-align:center"></p></div>
<script>setInterval(async()=>{const r=await fetch("/status");
document.getElementById("s").textContent=JSON.stringify(await r.json());
},2000)</script></body></html>"""


class LatestJpeg:
    """One-slot buffer: writers keep only the newest JPEG."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = b""
        self.meta = {"sequence": 0, "objects": 0, "fps": 0.0}

    def update(self, jpeg, **meta):
        with self._cond:
            self._jpeg = jpeg
            self.meta = meta
            self._cond.notify_all()

    def wait(self, timeout):
        """Block until the next update (or timeout); returns the JPEG."""
        with self._cond:
            self._cond.wait(timeout)
            return self._jpeg

    def snapshot(self):
        with self._cond:
            return self._jpeg, dict(self.meta)


def as_frame(array):
    """Wrap an ndarray as a Frame for JPEG encoding (NV12 or RGB)."""
    if array.ndim == 2:  # SDK NV12 layout: (h*3//2, w)
        height, width = array.shape[0] * 2 // 3, array.shape[1]
        fmt = "NV12"
    else:
        height, width = array.shape[:2]
        fmt = "RGB"
    return Frame(sequence=0, timestamp_ns=0, width=width,
                 height=height, format=fmt, image=array)


def multipart_chunk(jpeg):
    """One MJPEG part: boundary + headers + JPEG bytes."""
    return (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")


class Loop(threading.Thread):
    """subscribe -> pipeline -> draw -> JPEG, until stop() closes media."""

    def __init__(self, buffer):
        super().__init__(daemon=True, name="detection-loop")
        self.buffer = buffer
        self._media = None

    def stop(self):
        if self._media is not None:
            try:
                self._media.close()  # unblocks the subscribe iterator
            except Exception:
                logger.exception("media close during stop")

    def run(self):
        inference = InferenceClient()
        pipeline = InferencePipeline(
            client=inference, model_id=MODEL_ID,
            preprocessor=Preprocessor.from_model(inference, MODEL_ID))
        self._media = FdMediaClient()
        window = []  # frame timestamps within the last second -> fps
        try:
            for frame in self._media.subscribe(STREAM_ID):
                with frame:
                    out = pipeline.run(frame)
                    try:
                        objects = [obj for obj in out.objects
                                   if obj.score >= MIN_SCORE]
                        # Normalized bboxes draw correctly on the FULL
                        # frame (stream geometry), not just model input.
                        annotated = draw_detections(
                            frame.to_array(), objects)
                        jpeg = as_frame(annotated).to_jpeg_bytes(
                            quality=JPEG_QUALITY)
                    finally:
                        out.release()  # never leak a retained input
                    now = time.monotonic()
                    window = [t for t in window if now - t < 1.0] + [now]
                    self.buffer.update(
                        jpeg, sequence=frame.sequence,
                        objects=len(objects), fps=round(len(window), 1))
        except Exception:
            logger.exception("detection loop died")
        finally:
            inference.close()


class Handler(BaseHTTPRequestHandler):
    buffer = None  # the LatestJpeg shared with Loop

    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html", INDEX_HTML)
        elif self.path == "/status":
            _jpeg, meta = self.buffer.snapshot()
            self._send(200, "application/json", json.dumps(meta).encode())
        elif self.path == "/snapshot.jpg":
            jpeg, _meta = self.buffer.snapshot()
            if jpeg:
                self._send(200, "image/jpeg", jpeg)
            else:
                self._send(503, "text/plain", b"no frame yet")
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpeg = self.buffer.wait(2.0)
                    if jpeg:
                        self.wfile.write(multipart_chunk(jpeg))
            except (BrokenPipeError, ConnectionResetError):
                pass  # viewer closed the tab
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass  # keep per-request noise out of the app log


def main():
    buffer = LatestJpeg()
    Handler.buffer = buffer
    loop = Loop(buffer)
    stopping = threading.Event()

    def _stop(signum, _frame):
        logger.info("signal %d -> stopping", signum)
        stopping.set()
        loop.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Lets the console reverse-proxy this app under /apps/live-detection/.
    # Non-fatal: direct http://<device>:8090 works regardless.
    try:
        client = AppClient()
        try:
            client.register_web_url("/")
        finally:
            client.close()
    except Exception as exc:
        logger.warning("register_web_url failed (non-fatal): %r", exc)

    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True,
                     name="http").start()
    logger.info("serving http://0.0.0.0:%d/ "
                "(page, /stream, /snapshot.jpg, /status)", HTTP_PORT)
    loop.start()
    try:
        while not stopping.is_set():
            stopping.wait(1.0)
            if not loop.is_alive():
                break  # loop died on its own; do not spin forever
    finally:
        loop.stop()
        loop.join(timeout=3.0)
        server.shutdown()
    logger.info("bye (last meta: %s)", buffer.snapshot()[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
