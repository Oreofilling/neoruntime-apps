#!/usr/bin/env python3
"""
CLIP Zero-Shot Viewer Application for NeoRuntime Platform

Features:
- Real-time zero-shot image classification using CLIP model (NPU)
- Web UI for custom text label management
- Event Bus subscription for NPU inference results
- SSE (Server-Sent Events) for real-time result push

Architecture:
  NPU (ai-runtime) does ALL inference: image encoding + text encoding + scoring.
  This app is a thin web UI that:
  1. Sends text labels to ai-runtime via gRPC UpdatePostprocessConfig
  2. Receives classification results via Event Bus subscription
  3. Pushes results to browser via SSE
"""

import os
import sys
import json
import time
import signal
import logging
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

from flask import Flask, render_template, request, jsonify, Response

from neoruntime_ipc_sdk import InferenceClient, Config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt Manager — updates CLIP prompts on ai-runtime via gRPC
# ---------------------------------------------------------------------------
class PromptManager:
    """Sends text labels to ai-runtime via gRPC UpdatePostprocessConfig."""

    def __init__(self):
        self.client = InferenceClient()
        self.labels: List[str] = []
        self._lock = threading.Lock()
        try:
            self.client.connect()
            logger.info("PromptManager: connected to ai-runtime")
        except Exception as e:
            logger.warning("PromptManager: failed to connect to ai-runtime: %s", e)

    def set_labels(self, labels: List[str]) -> bool:
        """Update CLIP zero-shot prompts on ai-runtime."""
        if not labels:
            return False
        with self._lock:
            self.labels = list(labels)
        config = json.dumps({"prompts": labels})
        try:
            result = self.client.update_postprocess_config("clip_image_encoder", config)
            logger.info("Prompts updated: %s", labels)
            return result
        except Exception as e:
            logger.error("Failed to update prompts: %s", e)
            return False

    def set_labels_with_retry(self, labels: List[str], max_retries: int = 15, interval: int = 2) -> None:
        """Update prompts repeatedly until ai-runtime accepts them or times out."""
        def _retry_loop():
            for i in range(1, max_retries + 1):
                if self.set_labels(labels):
                    return
                logger.warning("Retry %d/%d: ai-runtime not ready, waiting to push default labels...", i, max_retries)
                time.sleep(interval)
            logger.error("Failed to push default labels after %d retries.", max_retries)
            
        t = threading.Thread(target=_retry_loop, daemon=True)
        t.start()

    def get_labels(self) -> List[str]:
        with self._lock:
            return list(self.labels)

    def close(self):
        self.client.close()


# ---------------------------------------------------------------------------
# Stream Infer Subscriber — receives classification results via gRPC StreamInfer
# ---------------------------------------------------------------------------
class StreamInferSubscriber:
    """Subscribe to CLIP inference results via active gRPC StreamInfer."""

    def __init__(self):
        self.client: Optional[InferenceClient] = None
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_result: Dict[str, Any] = {}
        self._result_lock = threading.Lock()
        self._subscribers: List = []  # SSE response objects
        self._sub_lock = threading.Lock()
        # Stats
        self.frame_count = 0
        self.last_frame_time = 0.0

    def start(self) -> bool:
        try:
            self.client = InferenceClient()
            self.running = True
            self._thread = threading.Thread(target=self._subscribe_loop, daemon=True)
            self._thread.start()
            logger.info("StreamInfer subscriber started")
            return True
        except Exception as e:
            logger.error("Failed to start StreamInfer subscriber: %s", e)
            return False

    def stop(self) -> None:
        self.running = False
        if self.client:
            self.client.close()

    def _subscribe_loop(self) -> None:
        stream_id = os.environ.get("CLIP_STREAM_ID", "third")
        model_id = "clip_image_encoder"
        fps = 5
        logger.info("Calling StreamInfer: stream=%s, model=%s, fps=%d", stream_id, model_id, fps)

        while self.running:
            try:
                # This is a blocking generator that yields results as frames are processed
                for frame_seq, result in self.client.subscribe(stream=stream_id, model=model_id, fps=fps):
                    if not self.running:
                        break
                    self._handle_result(frame_seq, result)
            except Exception as e:
                if self.running:
                    logger.error("StreamInfer error: %s, reconnecting in 2s...", e)
                    time.sleep(2)

    def _handle_result(self, frame_seq: int, result: Any) -> None:
        self.frame_count += 1
        now = time.time()
        self.last_frame_time = now

        logger.debug("Received StreamInfer result: frame_seq=%s", frame_seq)

        # The result object has a list of Classification objects
        if not result.classifications:
            return

        # Build result for SSE
        results = []
        top_label = "-"
        top_score = 0.0

        for cls in result.classifications:
            results.append({
                "label": cls.label,
                "score": round(cls.confidence, 4),
                "display_score": round(cls.confidence * 100, 1),
            })

        if results:
            top_label = results[0]["label"]
            top_score = results[0]["score"]

        classification = {
            "timestamp": datetime.now().isoformat(),
            "frame_sequence": frame_seq,
            "top_label": top_label,
            "top_score": top_score,
            "results": results,
            "frame_count": self.frame_count,
        }

        with self._result_lock:
            self._latest_result = classification

        self._push_sse(classification)

    def _push_sse(self, data: Dict[str, Any]) -> None:
        msg = f"data: {json.dumps(data)}\n\n"
        with self._sub_lock:
            dead = []
            for q in self._subscribers:
                try:
                    q.put(msg)
                except Exception:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    def get_latest(self) -> Dict[str, Any]:
        with self._result_lock:
            return dict(self._latest_result)

    def get_stats(self) -> Dict[str, Any]:
        fps = 0.0
        if self.frame_count > 0 and self.last_frame_time > 0:
            elapsed = time.time() - self.last_frame_time
            if elapsed < 5:
                fps = 1.0 / max(elapsed, 0.001)
        return {
            "frame_count": self.frame_count,
            "last_frame_time": self.last_frame_time,
            "estimated_fps": round(fps, 1),
            "connected": self.running,
        }

    def add_sse_subscriber(self, queue) -> None:
        with self._sub_lock:
            self._subscribers.append(queue)

    def remove_sse_subscriber(self, queue) -> None:
        with self._sub_lock:
            try:
                self._subscribers.remove(queue)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Flask Web Application
# ---------------------------------------------------------------------------
def create_app() -> Flask:
    app = Flask(__name__)

    # Initialize prompt manager
    prompt_mgr = PromptManager()

    # Set default labels
    default_labels_str = os.environ.get("DEFAULT_LABELS", "a person,a car,a dog,a cat,empty scene")
    default_labels = [l.strip() for l in default_labels_str.split(",") if l.strip()]
    prompt_mgr.set_labels_with_retry(default_labels)

    # Initialize inference subscriber
    subscriber = StreamInferSubscriber()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "app_id": Config.get_app_id()})

    @app.route("/api/labels", methods=["GET"])
    def get_labels():
        return jsonify({"labels": prompt_mgr.get_labels()})

    @app.route("/api/labels", methods=["POST"])
    def update_labels():
        data = request.get_json(force=True)
        labels = data.get("labels", [])
        if not isinstance(labels, list) or len(labels) == 0:
            return jsonify({"error": "labels must be a non-empty list of strings"}), 400
        if len(labels) > 20:
            return jsonify({"error": "maximum 20 labels allowed"}), 400
        cleaned = []
        for l in labels:
            if not isinstance(l, str) or not l.strip():
                continue
            cleaned.append(l.strip()[:100])
        if not cleaned:
            return jsonify({"error": "no valid labels provided"}), 400
        ok = prompt_mgr.set_labels(cleaned)
        if not ok:
            return jsonify({"error": "failed to update prompts on ai-runtime"}), 500
        return jsonify({"labels": prompt_mgr.get_labels()})

    @app.route("/api/stream", methods=["GET"])
    def sse_stream():
        """SSE endpoint for real-time classification results."""
        import queue
        q = queue.Queue(maxsize=50)
        subscriber.add_sse_subscriber(q)

        def generate():
            try:
                while True:
                    try:
                        msg = q.get(timeout=30)
                        yield msg
                    except queue.Empty:
                        yield ": keepalive\n\n"
            except GeneratorExit:
                pass
            finally:
                subscriber.remove_sse_subscriber(q)

        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/api/stats", methods=["GET"])
    def stats():
        return jsonify({
            "subscriber": subscriber.get_stats(),
            "labels": prompt_mgr.get_labels(),
            "latest": subscriber.get_latest(),
        })

    @app.route("/api/latest", methods=["GET"])
    def latest():
        return jsonify(subscriber.get_latest())

    # Store references for shutdown
    app.prompt_mgr = prompt_mgr
    app.subscriber = subscriber

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    app_id = Config.get_app_id()
    web_port = int(os.environ.get("WEB_PORT", "8888"))

    logger.info("=" * 60)
    logger.info("  CLIP Zero-Shot Viewer v2.0.0 (NPU text encoding)")
    logger.info("  App ID: %s", app_id)
    logger.info("  Web Port: %d", web_port)
    logger.info("=" * 60)

    app = create_app()

    # Start inference subscriber
    if not app.subscriber.start():
        logger.warning("Failed to start inference subscriber — running in offline mode")

    # Signal handler for graceful shutdown
    running = True

    def _signal_handler(signum, frame):
        nonlocal running
        logger.info("Received signal %d, shutting down...", signum)
        running = False
        app.subscriber.stop()
        app.prompt_mgr.close()
        os._exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Start Flask (blocking)
    logger.info("Starting web server on port %d...", web_port)
    try:
        app.run(host="0.0.0.0", port=web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        app.subscriber.stop()
        app.prompt_mgr.close()
        logger.info("CLIP Viewer stopped")


if __name__ == "__main__":
    main()
