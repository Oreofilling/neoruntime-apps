#!/usr/bin/env python3
"""
Visualizer - Subscribe to inference events and draw on video frames

Real-time preview via HTTP MJPEG stream:
  - http://device:8080/         -> HTML page with embedded stream
  - http://device:8080/stream   -> Raw MJPEG stream
  - http://device:8080/snapshot -> Single JPEG snapshot

Subscribes to:
  - Video frames from SHM
  - Inference results from Event Bus
"""

import os
import sys
import time
import signal
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import numpy as np

# AIPC SDK
from neoruntime_ipc_sdk import FdMediaClient as MediaClient, EventClient, Frame

# OpenCV for drawing
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("ERROR: OpenCV is required for visualizer")
    sys.exit(1)


@dataclass
class Config:
    stream_id: str = "main"
    subscribe_topic: str = "inference/face_cascade/*"
    output_dir: str = "/app/output"
    fps: int = 15
    save_frames: bool = False
    save_interval: int = 30  # Save every N frames
    http_port: int = 8080
    jpeg_quality: int = 80

    # Drawing options
    box_color: tuple = (0, 255, 0)       # Green BGR
    landmark_color: tuple = (0, 0, 255)  # Red BGR
    text_color: tuple = (255, 255, 255)  # White
    box_thickness: int = 2
    landmark_radius: int = 3
    font_scale: float = 0.6

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            stream_id=os.getenv("STREAM_ID", "main"),
            subscribe_topic=os.getenv("SUBSCRIBE_TOPIC", "inference/face_cascade/*"),
            output_dir=os.getenv("OUTPUT_DIR", "/app/output"),
            fps=int(os.getenv("FPS", "15")),
            save_frames=os.getenv("SAVE_FRAMES", "false").lower() == "true",
            save_interval=int(os.getenv("SAVE_INTERVAL", "30")),
            http_port=int(os.getenv("HTTP_PORT", "8080")),
            jpeg_quality=int(os.getenv("JPEG_QUALITY", "80")),
        )


# ============================================
# Global frame buffer for HTTP streaming
# ============================================

class FrameBuffer:
    """Thread-safe frame buffer for MJPEG streaming"""

    def __init__(self):
        self.frame: Optional[bytes] = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.frame_id = 0

    def update(self, jpeg_bytes: bytes):
        with self.condition:
            self.frame = jpeg_bytes
            self.frame_id += 1
            self.condition.notify_all()

    def get(self, timeout: float = 1.0) -> Optional[bytes]:
        with self.condition:
            if self.condition.wait_for(lambda: self.frame is not None, timeout):
                return self.frame
            return None

    def wait_for_new(self, last_id: int, timeout: float = 1.0) -> tuple:
        """Wait for a new frame, returns (frame_bytes, frame_id)"""
        with self.condition:
            if self.condition.wait_for(lambda: self.frame_id > last_id, timeout):
                return self.frame, self.frame_id
            return None, last_id


# Global frame buffer
frame_buffer = FrameBuffer()


# ============================================
# HTTP MJPEG Server
# ============================================

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Thread-per-request HTTP server"""
    daemon_threads = True


class MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP handler for MJPEG streaming"""

    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_index_page()
        elif self.path == "/stream":
            self.send_mjpeg_stream()
        elif self.path == "/snapshot":
            self.send_snapshot()
        elif self.path == "/status":
            self.send_status()
        else:
            self.send_error(404)

    def send_index_page(self):
        """Send HTML page with embedded MJPEG stream"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>AIPC Visualizer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 { color: #00d9ff; margin-bottom: 10px; }
        .container {
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        img {
            border-radius: 5px;
            max-width: 100%;
        }
        .info {
            margin-top: 15px;
            font-size: 14px;
            color: #888;
        }
        .links a {
            color: #00d9ff;
            margin: 0 10px;
        }
    </style>
</head>
<body>
    <h1>🎥 AIPC Inference Visualizer</h1>
    <div class="container">
        <img src="/stream" alt="Live Stream" />
    </div>
    <div class="info">
        <div class="links">
            <a href="/stream">MJPEG Stream</a> |
            <a href="/snapshot">Snapshot</a> |
            <a href="/status">Status</a>
        </div>
    </div>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html.encode())

    def send_mjpeg_stream(self):
        """Send continuous MJPEG stream"""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

        last_id = 0
        try:
            while True:
                frame_bytes, frame_id = frame_buffer.wait_for_new(last_id, timeout=2.0)
                if frame_bytes is None:
                    continue
                last_id = frame_id

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode())
                self.wfile.write(frame_bytes)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_snapshot(self):
        """Send single JPEG snapshot"""
        frame_bytes = frame_buffer.get(timeout=2.0)
        if frame_bytes:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", len(frame_bytes))
            self.end_headers()
            self.wfile.write(frame_bytes)
        else:
            self.send_error(503, "No frame available")

    def send_status(self):
        """Send JSON status"""
        import json
        status = {
            "status": "running",
            "frame_id": frame_buffer.frame_id,
            "has_frame": frame_buffer.frame is not None
        }
        data = json.dumps(status).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)


# ============================================
# Inference Cache
# ============================================

class InferenceCache:
    """Cache recent inference results for matching with frames"""

    def __init__(self, max_size: int = 30, max_age_ms: int = 500):
        self.cache: deque = deque(maxlen=max_size)
        self.max_age_ms = max_age_ms
        self.lock = threading.Lock()

    def add(self, result: Dict[str, Any]):
        with self.lock:
            self.cache.append({
                "timestamp": time.time(),
                "data": result
            })

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            now = time.time()
            for item in reversed(self.cache):
                age_ms = (now - item["timestamp"]) * 1000
                if age_ms < self.max_age_ms:
                    return item["data"]
            return None


# ============================================
# Visualizer
# ============================================

class Visualizer:
    """Main visualizer class"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("visualizer")

        self.media: Optional[MediaClient] = None
        self.events: Optional[EventClient] = None

        self.inference_cache = InferenceCache()
        self.running = False
        self.frame_count = 0
        self.detection_count = 0

    def connect(self) -> bool:
        try:
            self.media = MediaClient()
            self.media.connect()
            self.logger.info("Connected to Media service")

            self.events = EventClient()
            self.events.connect()
            self.logger.info("Connected to Event Bus")

            return True
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        if self.media:
            self.media.close()
        if self.events:
            self.events.close()

    def _event_subscriber_thread(self):
        """Background thread to subscribe to inference events"""
        self.logger.info(f"Subscribing to: {self.config.subscribe_topic}")

        try:
            for event in self.events.subscribe(self.config.subscribe_topic):
                if not self.running:
                    break
                self.inference_cache.add(event.payload)

        except Exception as e:
            if self.running:
                self.logger.error(f"Event subscriber error: {e}")

    def draw_overlay(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Draw detection boxes and landmarks on frame"""
        h, w = frame.shape[:2]

        # Handle face_cascade format
        faces = result.get("faces", [])
        for face in faces:
            self._draw_face(frame, face, w, h)
            self.detection_count += 1

        # Handle generic detection format
        detections = result.get("detections", result.get("objects", []))
        for det in detections:
            self._draw_detection(frame, det, w, h)
            self.detection_count += 1

        return frame

    def _draw_face(self, frame: np.ndarray, face: dict, w: int, h: int):
        """Draw face bbox and landmarks"""
        bbox = face.get("bbox", {})
        confidence = face.get("confidence", 0)
        landmarks = face.get("landmarks", [])
        face_id = face.get("face_id", 0)

        # Bounding box
        x1 = int(bbox.get("x", 0) * w)
        y1 = int(bbox.get("y", 0) * h)
        x2 = int((bbox.get("x", 0) + bbox.get("w", 0)) * w)
        y2 = int((bbox.get("y", 0) + bbox.get("h", 0)) * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2),
                     self.config.box_color, self.config.box_thickness)

        # Label
        label = f"Face {face_id}: {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       self.config.font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1),
                     self.config.box_color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                   self.config.text_color, 1)

        # Landmarks
        for lm in landmarks:
            lx = int(lm.get("x", 0) * w)
            ly = int(lm.get("y", 0) * h)
            cv2.circle(frame, (lx, ly), self.config.landmark_radius,
                      self.config.landmark_color, -1)

    def _draw_detection(self, frame: np.ndarray, det: dict, w: int, h: int):
        """Draw generic detection bbox"""
        bbox = det.get("bbox", det)
        label = det.get("label", "object")
        confidence = det.get("confidence", det.get("score", 0))

        x1 = int(bbox.get("x", 0) * w)
        y1 = int(bbox.get("y", 0) * h)
        bw = bbox.get("w", bbox.get("width", 0))
        bh = bbox.get("h", bbox.get("height", 0))
        x2 = int((bbox.get("x", 0) + bw) * w)
        y2 = int((bbox.get("y", 0) + bh) * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2),
                     self.config.box_color, self.config.box_thickness)

        text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                   self.config.box_color, 2)

    def draw_stats(self, frame: np.ndarray, result: Optional[dict], fps: float):
        """Draw statistics overlay"""
        h, w = frame.shape[:2]

        # Top-left stats
        lines = [
            f"FPS: {fps:.1f}",
            f"Frame: {self.frame_count}",
        ]

        if result:
            num_faces = len(result.get("faces", []))
            num_dets = len(result.get("detections", result.get("objects", [])))
            if num_faces > 0:
                lines.append(f"Faces: {num_faces}")
            if num_dets > 0:
                lines.append(f"Objects: {num_dets}")

            # Inference time
            total_time = result.get("total_time_us", 0)
            if total_time:
                lines.append(f"Infer: {total_time/1000:.1f}ms")

        # Draw semi-transparent background
        y_offset = 10
        for i, line in enumerate(lines):
            y = y_offset + i * 25
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (5, y - th - 2), (tw + 15, y + 5), (0, 0, 0), -1)
            cv2.putText(frame, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    def frame_to_bgr(self, frame: Frame) -> np.ndarray:
        """Convert frame to BGR format for OpenCV"""
        if frame.format == "NV12":
            return cv2.cvtColor(frame.image, cv2.COLOR_YUV2BGR_NV12)
        elif frame.format == "RGB":
            return cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
        elif frame.format == "BGR":
            return frame.image
        else:
            rgb = frame.to_rgb()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def run(self):
        """Main processing loop"""
        self.running = True

        # Start event subscriber thread
        event_thread = threading.Thread(target=self._event_subscriber_thread, daemon=True)
        event_thread.start()

        # Create output directory
        if self.config.save_frames:
            os.makedirs(self.config.output_dir, exist_ok=True)

        self.logger.info(f"Processing stream: {self.config.stream_id}")
        self.logger.info(f"HTTP server: http://0.0.0.0:{self.config.http_port}/")

        frame_interval = 1.0 / self.config.fps
        last_frame_time = 0.0
        fps_counter = 0
        fps_start = time.time()
        current_fps = 0.0

        try:
            for frame in self.media.subscribe(self.config.stream_id):
                if not self.running:
                    break

                # Rate limiting
                now = time.time()
                if now - last_frame_time < frame_interval:
                    continue
                last_frame_time = now

                # FPS calculation
                fps_counter += 1
                if now - fps_start >= 1.0:
                    current_fps = fps_counter / (now - fps_start)
                    fps_counter = 0
                    fps_start = now

                # Convert to BGR
                bgr_frame = self.frame_to_bgr(frame)

                # Get inference result and draw
                result = self.inference_cache.get_latest()
                if result:
                    bgr_frame = self.draw_overlay(bgr_frame, result)

                # Draw stats
                self.draw_stats(bgr_frame, result, current_fps)

                self.frame_count += 1

                # Encode to JPEG and update buffer
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
                _, jpeg = cv2.imencode('.jpg', bgr_frame, encode_params)
                frame_buffer.update(jpeg.tobytes())

                # Save frames periodically
                if self.config.save_frames and self.frame_count % self.config.save_interval == 0:
                    filename = f"{self.config.output_dir}/frame_{self.frame_count:06d}.jpg"
                    cv2.imwrite(filename, bgr_frame)
                    self.logger.debug(f"Saved: {filename}")

                # Log progress
                if self.frame_count % 300 == 0:
                    self.logger.info(f"Frames: {self.frame_count}, "
                                    f"Detections: {self.detection_count}, "
                                    f"FPS: {current_fps:.1f}")

        except Exception as e:
            self.logger.error(f"Processing error: {e}")
        finally:
            self.logger.info(f"Stopped: {self.frame_count} frames processed")

    def stop(self):
        self.running = False


# ============================================
# Main
# ============================================

def setup_logging(level: str = "info"):
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    config = Config.from_env()
    setup_logging(os.getenv("LOG_LEVEL", "info"))

    logger = logging.getLogger("main")
    logger.info("=" * 50)
    logger.info("  AIPC Inference Visualizer")
    logger.info("=" * 50)
    logger.info(f"Stream: {config.stream_id}")
    logger.info(f"Topic: {config.subscribe_topic}")
    logger.info(f"HTTP Port: {config.http_port}")
    logger.info(f"FPS: {config.fps}")

    visualizer = Visualizer(config)

    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        visualizer.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start HTTP server
    http_server = ThreadedHTTPServer(("0.0.0.0", config.http_port), MJPEGHandler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()
    logger.info(f"HTTP server started: http://0.0.0.0:{config.http_port}/")

    # Connect to services
    if not visualizer.connect():
        logger.error("Failed to connect to services")
        sys.exit(1)

    try:
        visualizer.run()
    finally:
        visualizer.disconnect()
        http_server.shutdown()

    logger.info("Visualizer exited")


if __name__ == "__main__":
    main()
