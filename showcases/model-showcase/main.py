#!/usr/bin/env python3
"""
AI Model Showcase for NeoRuntime Platform

Real-time AI inference showcase that bundles model metadata, auto-registers
models with ai-runtime at startup, and renders type-specific visualizations.

Architecture:
  - Main thread: FrameSource -> get frame -> overlay inference result -> MJPEG
  - Infer thread: InferenceClient -> continuous inference -> store result
  - Flask: REST API + SSE + MJPEG streaming
"""

import os
import sys
import base64
import json
import time
import signal
import sqlite3
import logging
import subprocess
import threading
import urllib.parse
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import cv2
# OpenCV's pthreads parallel_for_ pool spin-waits (sched_yield) on its worker
# threads even when idle. With the default 4 workers this burned ~37% CPU
# (3 threads × ~50K sched_yield/s) on 4-core Hailo15H. The hot cv2 ops here
# (resize 640x640, rectangle/putText draws, VideoCapture decode) are small
# enough that 4-way parallelism adds dispatch overhead with no net gain.
# Force 1 thread → pool workers are not spawned, spin disappears.
# gdb bt: spinners were in cv2.abi3.so → pthread_cond_wait.
cv2.setNumThreads(1)
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from flask_sock import Sock
from PIL import Image, ImageDraw, ImageFont

from neoruntime_ipc_sdk import (
    FdMediaClient,
    InferenceClient,
    BatchInferItem,
    AppClient,
    DeviceClient,
    Config,
    Frame,
    EventClient,
)
import urllib.error

# ---------------------------------------------------------------------------
# Gallery Manager — CLIP text-to-image search
# ---------------------------------------------------------------------------

logger = logging.getLogger("model-showcase")

# When the infer loop recovers from a burst of NPU stalls, log a benign INFO
# line if at least this many consecutive transient failures were absorbed.
# Visibility without an alarm: a recovered burst is not an error, only genuine
# sustained degradation (the _degraded block) warns. See the infer-loop handler
# for why "consecutive failures" tracks stall duration, not stall count.
_STALL_BURST_INFO_THRESHOLD = 3


class GalleryManager:
    """SQLite + numpy in-memory cosine similarity gallery for CLIP image search."""

    def __init__(self, data_dir: str, max_images: int = 5000,
                 capture_interval: float = 5.0):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images")
        self.db_path = os.path.join(data_dir, "gallery.db")
        self.max_images = max_images
        self.capture_interval = capture_interval
        self.enabled = False
        self._last_capture = 0.0
        self._lock = threading.Lock()

        os.makedirs(self.image_dir, exist_ok=True)
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._init_db()
        self._ids: np.ndarray = np.array([], dtype=np.int64)
        self._matrix: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self._load_embeddings()

    def _init_db(self) -> None:
        self.db.execute("""CREATE TABLE IF NOT EXISTS gallery_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            embedding BLOB NOT NULL,
            timestamp INTEGER NOT NULL,
            width INTEGER,
            height INTEGER
        )""")
        self.db.commit()

    def _load_embeddings(self) -> None:
        rows = self.db.execute(
            "SELECT id, embedding FROM gallery_images ORDER BY id"
        ).fetchall()
        if rows:
            ids = [r[0] for r in rows]
            vecs = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
            self._ids = np.array(ids, dtype=np.int64)
            self._matrix = np.stack(vecs).astype(np.float32)
            logger.info("Gallery loaded %d images from DB", len(rows))
        else:
            self._ids = np.array([], dtype=np.int64)
            self._matrix = np.empty((0, 512), dtype=np.float32)

    def add_image(self, jpeg_bytes: bytes, embedding_data: List[float],
                  timestamp: int, width: int, height: int) -> int:
        with self._lock:
            img_id = self.db.execute(
                "SELECT COALESCE(MAX(id),0)+1 FROM gallery_images"
            ).fetchone()[0]
            path = os.path.join(self.image_dir, f"{img_id:06d}.jpg")
            with open(path, "wb") as f:
                f.write(jpeg_bytes)

            emb_blob = np.array(embedding_data, dtype=np.float32).tobytes()
            self.db.execute(
                "INSERT INTO gallery_images (path, embedding, timestamp, width, height) "
                "VALUES (?,?,?,?,?)",
                (path, emb_blob, timestamp, width, height),
            )
            self.db.commit()

            vec = np.array(embedding_data, dtype=np.float32).reshape(1, -1)
            if len(self._matrix) > 0:
                self._matrix = np.vstack([self._matrix, vec])
            else:
                self._matrix = vec
            self._ids = np.append(self._ids, img_id)
            self._enforce_cap()
            return img_id

    def search(self, query_embedding: List[float],
               top_k: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            if len(self._matrix) == 0:
                return []
            query = np.array(query_embedding, dtype=np.float32)
            qnorm = np.linalg.norm(query)
            if qnorm > 0:
                query /= qnorm
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normed = self._matrix / norms
            scores = normed @ query
            top_idx = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_idx:
                img_id = int(self._ids[idx])
                row = self.db.execute(
                    "SELECT path, timestamp, width, height FROM gallery_images WHERE id=?",
                    (img_id,),
                ).fetchone()
                if row:
                    results.append({
                        "id": img_id,
                        "path": row[0],
                        "score": round(float(scores[idx]), 4),
                        "timestamp": row[1],
                        "width": row[2],
                        "height": row[3],
                    })
            return results

    def get_page(self, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        offset = (page - 1) * page_size
        with self._lock:
            total = self.db.execute(
                "SELECT COUNT(*) FROM gallery_images"
            ).fetchone()[0]
            rows = self.db.execute(
                "SELECT id, path, timestamp, width, height FROM gallery_images "
                "ORDER BY id DESC LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()
        items = [
            {"id": r[0], "path": r[1], "timestamp": r[2],
             "width": r[3], "height": r[4]}
            for r in rows
        ]
        return {"items": items, "total": total, "page": page, "page_size": page_size}

    def get_image_path(self, img_id: int) -> Optional[str]:
        with self._lock:
            row = self.db.execute(
                "SELECT path FROM gallery_images WHERE id=?", (img_id,)
            ).fetchone()
            return row[0] if row else None

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            count = len(self._ids)
        disk_bytes = 0
        if os.path.isdir(self.image_dir):
            for f in os.listdir(self.image_dir):
                fp = os.path.join(self.image_dir, f)
                if os.path.isfile(fp):
                    disk_bytes += os.path.getsize(fp)
        return {"count": count, "disk_mb": round(disk_bytes / (1024 * 1024), 2)}

    def delete_older_than(self, seconds: int) -> int:
        cutoff = int(time.time() * 1000) - seconds * 1000
        with self._lock:
            rows = self.db.execute(
                "SELECT id, path FROM gallery_images WHERE timestamp < ?",
                (cutoff,),
            ).fetchall()
            for img_id, path in rows:
                if os.path.exists(path):
                    os.remove(path)
            self.db.execute(
                "DELETE FROM gallery_images WHERE timestamp < ?", (cutoff,)
            )
            self.db.commit()
            self._load_embeddings()
            return len(rows)

    def _enforce_cap(self) -> None:
        while len(self._ids) > self.max_images:
            old_id = int(self._ids[0])
            row = self.db.execute(
                "SELECT path FROM gallery_images WHERE id=?", (old_id,)
            ).fetchone()
            if row and os.path.exists(row[0]):
                os.remove(row[0])
            self.db.execute("DELETE FROM gallery_images WHERE id=?", (old_id,))
            self.db.commit()
            self._ids = self._ids[1:]
            self._matrix = self._matrix[1:]


# ---------------------------------------------------------------------------
# Frame Source abstraction — camera or video file
# ---------------------------------------------------------------------------

_VIDEO_DIR = os.environ.get("VIDEO_DIR", "/tmp/neoruntime-videos")
os.makedirs(_VIDEO_DIR, exist_ok=True)
_MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB

# --- Image upload inference (one-shot) ---
# Uploaded stills live here ephemerally (tmpfs); they are not persisted across
# restarts. Decoded to BGR, run through the same single-shot infer + overlay
# pipeline as the live loop, then returned as one annotated JPEG.
_IMAGE_DIR = os.environ.get("IMAGE_DIR", "/tmp/neoruntime-images")
os.makedirs(_IMAGE_DIR, exist_ok=True)
_MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50 MB
_ALLOWED_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Normalize uploaded stills to this longest side before drawing the overlay.
# The draw handlers use fixed pixel scales tuned for ~1280-wide preview frames;
# without this, a tiny phone thumbnail renders a vanishingly small overlay and a
# huge photo renders a microscopic one. 1280 makes the result readable for both.
_IMAGE_RENDER_MAX_SIDE = 1280


def _resize_to_side(bgr: np.ndarray, target_side: int) -> np.ndarray:
    """Aspect-preserving resize so the longest side equals ``target_side``.

    Returns the original array unchanged if it's already within 2% of the target
    (avoids a needless resample). Uses INTER_AREA for downscale (anti-alias) and
    INTER_LINEAR for upscale.
    """
    h, w = bgr.shape[:2]
    longest = max(w, h)
    if longest == 0 or abs(longest - target_side) / max(target_side, 1) <= 0.02:
        return bgr
    scale = target_side / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(bgr, (new_w, new_h), interpolation=interp)


class VideoFrameSource:
    """Threaded video decoder with a latest-frame slot.

    A background decode thread owns ALL ``cv2.VideoCapture`` access (OpenCV
    ``VideoCapture`` is not safe to share across threads) and decodes at a
    configurable playback FPS, downscaling each frame to ``decode_w x decode_h``
    in-thread so the full-resolution buffer never flows through the rest of the
    pipeline. Both the MJPEG display loop and the inference loop read the newest
    decoded frame via :meth:`latest_frame`; *neither advances the file*, so
    playback is decoupled from the (slow) ``INFER_FPS`` — this is the fix for
    high-res imported-video stutter.

    Control (pause / speed / seek) mutates thread-safe fields read by the decode
    loop; the API thread never touches ``cap`` directly (avoiding torn seeks and
    use-after-free on close). Public attribute surface (``fps``, ``total_frames``,
    ``width``, ``height``, ``duration``, ``paused``, ``speed``,
    ``position_sec``, ``progress``) is preserved.
    """

    def __init__(self, path: str,
                 playback_fps: Optional[float] = None,
                 decode_w: Optional[int] = None,
                 decode_h: Optional[int] = None) -> None:
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        # Keep the demuxer's internal buffer tiny to cut latency / jitter.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        # Decode dims: 16:9 cover for the 640x360 preview + 640x640 infer resize.
        self._decode_w = int(decode_w or os.environ.get("VIDEO_DECODE_W", "960"))
        self._decode_h = int(decode_h or os.environ.get("VIDEO_DECODE_H", "540"))
        # Pace playback at min(source_fps, 30) unless overridden explicitly/env.
        src_fps = self.fps if (self.fps and self.fps > 0) else 25.0
        default_playback = min(src_fps, 30.0)
        if playback_fps is not None and playback_fps > 0:
            self._playback_fps = float(playback_fps)
        else:
            env_pf = os.environ.get("VIDEO_PLAYBACK_FPS")
            self._playback_fps = (float(env_pf)
                                  if (env_pf and float(env_pf) > 0)
                                  else default_playback)

        # Latest decoded (already-downscaled) frame slot.
        self._latest: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # Control fields guarded by _ctl_lock. The decode loop is the sole
        # cap reader; these fields tell it what to do next.
        self._ctl_lock = threading.Lock()
        self._paused = False
        self._speed = 1.0
        self._pending_seek: Optional[int] = None  # frame number, applied by loop
        self._eof = False
        # _frame_idx has a single writer (the decode thread); reads in the
        # properties below are atomic int reads on CPython, so no lock needed.
        self._frame_idx = 0

        self._resume_event = threading.Event()   # set == playing
        self._resume_event.set()
        self._stop_event = threading.Event()
        self._decode_thread: Optional[threading.Thread] = None
        self._closed = False

    # -- control surface (thread-safe; safe to call from the API thread) ----

    @property
    def paused(self) -> bool:
        with self._ctl_lock:
            return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        with self._ctl_lock:
            self._paused = bool(value)
            playing = not self._paused
        if playing:
            self._resume_event.set()
        else:
            self._resume_event.clear()

    @property
    def speed(self) -> float:
        with self._ctl_lock:
            return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        with self._ctl_lock:
            self._speed = max(0.25, min(4.0, float(value)))

    def seek(self, position_sec: float) -> None:
        """Queue a seek; the decode thread applies it on the next iteration."""
        frame_no = int(position_sec * self.fps)
        if self.total_frames > 0:
            frame_no = max(0, min(frame_no, self.total_frames - 1))
        else:
            frame_no = max(0, frame_no)
        with self._ctl_lock:
            self._pending_seek = frame_no
            self._eof = False  # a seek implies "play from here"

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Spawn the background decode thread (idempotent)."""
        if self._decode_thread is not None:
            return
        self._stop_event.clear()
        self._resume_event.set()
        self._decode_thread = threading.Thread(
            target=self._decode_loop, name="video-decode", daemon=True)
        self._decode_thread.start()
        logger.info("Video decode thread started (playback=%.1ffps, decode=%dx%d)",
                    self._playback_fps, self._decode_w, self._decode_h)

    def close(self) -> None:
        """Stop the decode thread and release the capture (idempotent).

        On a join timeout the thread is leaked and ``cap`` is left unreleased
        rather than yanked out from under a live read (segfault risk).
        """
        with self._ctl_lock:
            if self._closed:
                return
            self._closed = True
        self._stop_event.set()
        self._resume_event.set()  # unblock a paused loop
        thread = self._decode_thread
        self._decode_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning(
                    "video decode thread did not join within 2s; leaving cap "
                    "unreleased to avoid a use-after-free")
                return
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # -- decode loop (runs in the background thread) ------------------------

    def _decode_loop(self) -> None:
        next_ts = time.time()
        while not self._stop_event.is_set():
            # Apply a pending seek — only this thread touches cap.
            with self._ctl_lock:
                pending = self._pending_seek
                self._pending_seek = None
            if pending is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, pending)
                self._frame_idx = pending
                # Drop the stale latest so consumers briefly wait for the new
                # frame rather than showing the pre-seek one.
                with self._frame_lock:
                    self._latest = None
                next_ts = time.time()

            with self._ctl_lock:
                paused = self._paused
            if paused:
                # Wake promptly on resume (or stop via the 0.2s poll).
                self._resume_event.wait(timeout=0.2)
                next_ts = time.time()
                continue

            with self._ctl_lock:
                eof = self._eof
            if eof:
                # Hold the last frame; wait for seek / restart / stop.
                if self._stop_event.wait(timeout=0.2):
                    break
                continue

            with self._ctl_lock:
                speed = self._speed

            # speed > 1: grab-skip frames to preserve existing speed semantics.
            skip = max(1, int(speed)) - 1
            eof_now = False
            for _ in range(skip):
                if not self.cap.grab():
                    eof_now = True
                    break
                self._frame_idx += 1
            if eof_now:
                with self._ctl_lock:
                    self._eof = True
                logger.info("Video decode reached EOF (grab), holding last frame")
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                with self._ctl_lock:
                    self._eof = True
                logger.info("Video decode reached EOF (read), holding last frame")
                continue
            self._frame_idx += 1

            # Downscale in-thread so full-res never leaves the decoder. cv2.resize
            # allocates a fresh buffer, so _latest always owns an isolated array.
            if frame.shape[1] != self._decode_w or frame.shape[0] != self._decode_h:
                frame = cv2.resize(frame, (self._decode_w, self._decode_h),
                                   interpolation=cv2.INTER_LINEAR)
            with self._frame_lock:
                self._latest = frame

            # Pace to playback FPS (scaled by speed). Wake promptly on stop.
            interval = (1.0 / self._playback_fps) / max(speed, 0.0001)
            next_ts += interval
            drift = next_ts - time.time()
            if drift > 0:
                if self._stop_event.wait(timeout=drift):
                    break
            elif drift < -interval:
                # Fell behind realtime by more than a frame (heavy decode):
                # skip ahead so playback does not drift into slow-motion.
                catch = min(int(-drift / interval), 30)
                caught_eof = False
                for _ in range(catch):
                    if not self.cap.grab():
                        caught_eof = True
                        break
                    self._frame_idx += 1
                if caught_eof:
                    with self._ctl_lock:
                        self._eof = True
                    continue
                next_ts = time.time()
        logger.debug("Video decode loop exiting")

    # -- read accessors -----------------------------------------------------

    def latest_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the newest decoded (already-downscaled) frame.

        ``None`` only briefly: before the first frame decodes, or right after a
        seek (the loop clears the slot). On EOF the last frame is held.
        """
        with self._frame_lock:
            frame = self._latest
        return None if frame is None else frame.copy()

    @property
    def position_sec(self) -> float:
        return self._frame_idx / self.fps if self.fps > 0 else 0.0

    @property
    def progress(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return self._frame_idx / self.total_frames


# ---------------------------------------------------------------------------
# Model Catalog — built-in manifest of all supported models
# ---------------------------------------------------------------------------

# Model directory inside the container.
# AIPC_HOST_PREFIX is the HOST-side prefix, not the container path.
def _select_model_root() -> str:
    candidates = [
        os.environ.get("MODEL_ROOT"),
        os.environ.get("AIPC_MODEL_ROOT"),
        "/data/aipc/models",
        "/opt/aipc/models",
    ]
    for root in candidates:
        if root and os.path.isdir(root):
            return root
    return next(root for root in candidates if root)


_MODEL_ROOT = _select_model_root()


# Bundled model copies ship in the image at a flat layout (Dockerfile COPY
# models/ /opt/aipc/bundled-models/). NOT /opt/aipc/models — app.yaml
# bind-mounts the host model store there, which would shadow image content.
_BUNDLED_MODEL_ROOT = "/opt/aipc/bundled-models"


def _model_path(category: str, filename: str) -> str:
    """Host-provisioned copy first; image-bundled fallback otherwise.

    Devices that pre-provision /data/aipc/models keep using their own copy;
    fresh installs fall back to the file bundled in the image so installs
    are plug-and-play (same pattern as gym-ops _resolve_model_path).
    """
    host = os.path.join(_MODEL_ROOT, category, filename)
    if os.path.isfile(host):
        return host
    return os.path.join(_BUNDLED_MODEL_ROOT, filename)


def _entry_available(m: Dict[str, Any]) -> bool:
    """Whether every file a catalog entry needs exists on this device.

    Genai is always considered available: it is served interactively and a
    missing weight file is surfaced as an early warning instead (see
    _discover_models) rather than hiding the chat entry.
    """
    if m.get("type") == "genai":
        return True
    if "stages" in m:
        return all(os.path.isfile(s["path"]) for s in m["stages"])
    return os.path.isfile(m.get("path", ""))


def _missing_models() -> List[Dict[str, Any]]:
    """Catalog entries this device can't run yet, with acquisition hints.

    Backs the /api/models ``missing`` list so the UI can show a dimmed
    "not installed" card with copy-paste instructions instead of silently
    hiding the model. Genai entries are excluded (see _entry_available);
    they stay listed with a warning badge instead.
    """
    keep = ("id", "name", "type", "description", "category", "acquisition")
    return [
        {k: m[k] for k in keep if k in m}
        for m in MODEL_CATALOG
        if not _entry_available(m)
    ]


MODEL_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "yolov8n_detection",
        "path": _model_path("detection", "hailo_yolov8n_384_640.hef"),
        "type": "detection",
        "input_format": "nv12",
        "input_width": 640,
        "input_height": 384,
        "name": "YOLOv8n Detection",
        "description": "Real-time object detection — person, vehicle, face (4-class)",
        "category": "detection",
    },
    {
        "id": "yolov5m_vehicles",
        "path": _model_path("detection", "yolov5m_vehicles.hef"),
        "type": "detection",
        "input_format": "rgb",
        "input_width": 1920,
        "input_height": 1080,
        "name": "YOLOv5m Vehicles",
        "description": "Vehicle-specific detection with built-in NMS — car, truck, bus, motorcycle",
        "category": "detection",
        "register_type": "",
    },
    {
        "id": "vit_classification",
        "path": _model_path("classification", "vit_large.hef"),
        "type": "classification",
        "input_format": "rgb",
        "input_width": 224,
        "input_height": 224,
        "name": "ViT Classification",
        "description": "Image classification — identify the main subject in the scene",
        "category": "classification",
        # Not image-bundled (267 MB would bloat the install package). When the
        # file is absent the UI shows a "not installed" card built from these
        # hints instead of silently hiding the entry. sha256 verified
        # byte-identical to the device factory artifact.
        "acquisition": {
            "url": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.3.0/hailo15h/vit_large.hef",
            "sha256": "e5160e0c3647315bb027ac162200fa90860a708ebf97daa4bbdbe9858c9cea89",
            "size_bytes": 280158208,
            "target": "/data/aipc/models/classification/vit_large.hef",
        },
    },
    {
        "id": "linknet_segmentation",
        "path": _model_path("segmentation", "linknet_mbv1_ss_dpm_256.hef"),
        "type": "segmentation",
        "input_format": "nv12",
        "input_width": 256,
        "input_height": 256,
        "name": "Linknet Segmentation",
        "description": "Pixel-level scene parsing — road, sidewalk, sky, vegetation",
        "category": "segmentation",
    },
    {
        "id": "face_landmarks",
        "path": _model_path("keypoint", "face_landmarks_lite.hef"),
        "type": "keypoint",
        "register_type": "",
        "input_format": "rgb",
        "input_width": 192,
        "input_height": 192,
        "name": "Face Landmarks",
        "description": "Facial feature detection — eyes, nose, mouth, jaw outline",
        "category": "keypoint",
    },
    {
        "id": "clip_encoder",
        "path": _model_path("clip", "clip_vit_b_32_image_encoder_nv12.hef"),
        "type": "clip",
        "input_format": "nv12",
        "input_width": 224,
        "input_height": 224,
        "name": "CLIP Zero-Shot",
        "description": "Zero-shot image classification with custom text prompts",
        "category": "classification",
        "postprocess_json": json.dumps({
            "prompts": ["a person", "a car", "a dog", "a cat", "empty scene"],
        }),
    },
    {
        "id": "lpr_pipeline",
        "type": "pipeline_lpr",
        "name": "License Plate Recognition",
        "description": "Detect license plates and read plate numbers in real-time",
        "category": "pipeline",
        "stages": [
            {
                "id": "license_plate_det",
                "role": "detector",
                "path": _model_path("detection", "tiny_yolov4_license_plates.hef"),
                "type": "detection",
                "register_type": "",
                "input_format": "rgb",
                "input_width": 416,
                "input_height": 416,
            },
            {
                "id": "lprnet",
                "role": "recognizer",
                "path": _model_path("ocr", "lprnet.hef"),
                "type": "ocr_recognition",
                "input_format": "rgb",
                "input_width": 300,
                "input_height": 75,
            },
        ],
    },
    {
        "id": "ocr_pipeline",
        "type": "pipeline_ocr",
        "name": "Text Recognition (OCR)",
        "description": "Detect text regions and recognize content in the scene",
        "category": "pipeline",
        "stages": [
            {
                "id": "ocr_detection",
                "role": "detector",
                "path": _model_path("ocr", "paddle_ocr_v5_mobile_detection.hef"),
                "type": "ocr_detection",
                "input_format": "rgb",
                "input_width": 960,
                "input_height": 544,
            },
            {
                "id": "ocr_recognition",
                "role": "recognizer",
                "path": _model_path("ocr", "paddle_ocr_v5_mobile_recognition_nv12.hef"),
                "type": "ocr_recognition",
                "input_format": "nv12",
                "input_width": 320,
                "input_height": 48,
            },
        ],
    },
    {
        "id": "scdepth",
        "path": _model_path("depth", "scdepthv3.hef"),
        "type": "depth",
        "input_format": "rgb",
        "input_width": 320,
        "input_height": 256,
        "name": "SCDepthV3 Depth Estimation",
        "description": "Monocular depth estimation — colorized depth map overlay",
        "category": "depth",
    },
    {
        "id": "qwen3_vl_2b",
        "type": "genai",
        "path": _model_path("genai", "Qwen3-VL-2B-Instruct.hef"),
        "kind": "vlm",
        "name": "Qwen3-VL-2B",
        "description": "Vision-Language Model — multimodal chat with image understanding",
        "category": "genai",
        "optimize_memory": False,
        "vlm_width": 512,
        "vlm_height": 288,
        # GB-scale weight file ships with the factory device image, never in
        # the app package. Devices lacking it keep the chat entry (genai is
        # always "available") but get an early warning badge + copy hint.
        "acquisition": {
            "source": "device",
            "size_bytes": 3185773502,
            "target": "/data/aipc/models/genai/Qwen3-VL-2B-Instruct.hef",
        },
    },
]
LPR_CHARSET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ-"


def _load_paddle_charset() -> List[str]:
    """Load PaddleOCR v5 recognition charset from dict file.

    Order: index 0 = CTC blank, then CJK, digits, letters, and symbols.
    Must be a list (not str) because 10 flag emojis are multi-codepoint.
    """
    dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_dict.txt")
    charset: List[str] = [""]  # index 0 = CTC blank
    try:
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                ch = line.rstrip("\n")
                if not ch:
                    ch = " "
                charset.append(ch)
    except FileNotFoundError:
        logging.warning("ocr_dict.txt not found at %s, OCR will produce empty text", dict_path)
    return charset


OCR_CHARSET: List[str] = _load_paddle_charset()

AVAILABLE_MODELS: List[Dict[str, str]] = []

# ---------------------------------------------------------------------------
# Pipeline result wrapper
# ---------------------------------------------------------------------------

class _OcrLineWrap:
    __slots__ = ("text", "confidence", "bbox")
    def __init__(self, text: str, confidence: float,
                 bbox_x: float, bbox_y: float,
                 bbox_w: float, bbox_h: float) -> None:
        self.text = text
        self.confidence = confidence
        self.bbox = type("_BBox", (), {
            "x": bbox_x, "y": bbox_y,
            "width": bbox_w, "height": bbox_h,
        })()


class _OcrAccumulator:
    """Sliding-window accumulator that merges OCR detections across frames.

    The detection model typically returns 1 bbox per frame, but different
    text lines appear across consecutive frames. This accumulator keeps
    recent detections alive for a configurable TTL, deduplicates by spatial
    overlap + text similarity, and returns the merged set.
    """

    def __init__(self, ttl_sec: float = 3.0, iou_threshold: float = 0.3) -> None:
        self._ttl = ttl_sec
        self._iou_thr = iou_threshold
        self._entries: List[Tuple[float, _OcrLineWrap]] = []  # (timestamp, line)
        self._lock = threading.Lock()

    def add(self, lines: List[_OcrLineWrap]) -> None:
        now = time.monotonic()
        with self._lock:
            for line in lines:
                merged = False
                for i, (ts, existing) in enumerate(self._entries):
                    if self._similar(existing, line):
                        self._entries[i] = (now, line)
                        merged = True
                        break
                if not merged:
                    self._entries.append((now, line))

    def snapshot(self) -> List[_OcrLineWrap]:
        now = time.monotonic()
        with self._lock:
            self._entries = [
                (ts, e) for ts, e in self._entries if now - ts < self._ttl
            ]
            return [e for _, e in self._entries]

    def _similar(self, a: _OcrLineWrap, b: _OcrLineWrap) -> bool:
        iou = self._iou(a.bbox, b.bbox)
        if iou > self._iou_thr:
            return True
        if a.text and b.text and (a.text == b.text or a.text in b.text or b.text in a.text):
            if iou > 0.05:
                return True
        return False

    @staticmethod
    def _iou(b1: Any, b2: Any) -> float:
        x1 = max(b1.x, b2.x)
        y1 = max(b1.y, b2.y)
        x2 = min(b1.x + b1.width, b2.x + b2.width)
        y2 = min(b1.y + b1.height, b2.y + b2.height)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = b1.width * b1.height
        area2 = b2.width * b2.height
        return inter / max(area1 + area2 - inter, 1e-9)


class _PipelineResult:
    __slots__ = ("ocr_lines", "objects", "classifications",
                 "landmarks", "masks", "embeddings")
    def __init__(self, ocr_lines: list = None, objects: list = None, **_: Any) -> None:
        self.ocr_lines = ocr_lines or []
        self.objects = objects or []
        self.classifications = []
        self.landmarks = []
        self.masks = []
        self.embeddings = []


class _DetWrap:
    __slots__ = ("label", "score", "class_id", "bbox")
    def __init__(self, label: str, score: float, class_id: int,
                 x: float, y: float, w: float, h: float) -> None:
        self.label = label
        self.score = score
        self.class_id = class_id
        self.bbox = type("_BBox", (), {
            "x": x, "y": y, "width": w, "height": h,
        })()


class _NmsResult:
    __slots__ = ("objects", "classifications", "landmarks",
                 "masks", "ocr_lines", "embeddings", "infer_time_us")
    def __init__(self, objects: list = None) -> None:
        self.objects = objects or []
        self.classifications = []
        self.landmarks = []
        self.masks = []
        self.ocr_lines = []
        self.embeddings = []
        self.infer_time_us = 0


class _ClsResult:
    __slots__ = ("objects", "classifications", "landmarks",
                 "masks", "ocr_lines", "embeddings", "infer_time_us",
                 "raw_outputs")
    def __init__(self, classifications: list = None) -> None:
        self.objects = []
        self.classifications = classifications or []
        self.landmarks = []
        self.masks = []
        self.ocr_lines = []
        self.embeddings = []
        self.infer_time_us = 0
        self.raw_outputs = None


_IMAGENET_NAMES: Dict[int, str] = {
    0: "tench", 1: "goldfish", 2: "great white shark", 3: "tiger shark", 4: "hammerhead", 5: "electric ray", 6: "stingray", 7: "cock", 8: "hen", 9: "ostrich",
    10: "brambling", 11: "goldfinch", 12: "house finch", 13: "junco", 14: "indigo bunting", 15: "robin", 16: "bulbul", 17: "jay", 18: "magpie", 19: "chickadee",
    20: "water ouzel", 21: "kite", 22: "bald eagle", 23: "vulture", 24: "great grey owl", 25: "European fire salamander", 26: "common newt", 27: "eft", 28: "spotted salamander", 29: "axolotl",
    30: "bullfrog", 31: "tree frog", 32: "tailed frog", 33: "loggerhead", 34: "leatherback turtle", 35: "mud turtle", 36: "terrapin", 37: "box turtle", 38: "banded gecko", 39: "common iguana",
    40: "American chameleon", 41: "whiptail", 42: "agama", 43: "frilled lizard", 44: "alligator lizard", 45: "Gila monster", 46: "green lizard", 47: "African chameleon", 48: "Komodo dragon", 49: "African crocodile",
    50: "American alligator", 51: "triceratops", 52: "thunder snake", 53: "ringneck snake", 54: "hognose snake", 55: "green snake", 56: "king snake", 57: "garter snake", 58: "water snake", 59: "vine snake",
    60: "night snake", 61: "boa constrictor", 62: "rock python", 63: "Indian cobra", 64: "green mamba", 65: "sea snake", 66: "horned viper", 67: "diamondback", 68: "sidewinder", 69: "trilobite",
    70: "harvestman", 71: "scorpion", 72: "black and gold garden spider", 73: "barn spider", 74: "garden spider", 75: "black widow", 76: "tarantula", 77: "wolf spider", 78: "tick", 79: "centipede",
    80: "black grouse", 81: "ptarmigan", 82: "ruffed grouse", 83: "prairie chicken", 84: "peacock", 85: "quail", 86: "partridge", 87: "African grey", 88: "macaw", 89: "sulphur-crested cockatoo",
    90: "lorikeet", 91: "coucal", 92: "bee eater", 93: "hornbill", 94: "hummingbird", 95: "jacamar", 96: "toucan", 97: "drake", 98: "red-breasted merganser", 99: "goose",
    100: "black swan", 101: "tusker", 102: "echidna", 103: "platypus", 104: "wallaby", 105: "koala", 106: "wombat", 107: "jellyfish", 108: "sea anemone", 109: "brain coral",
    110: "flatworm", 111: "nematode", 112: "conch", 113: "snail", 114: "slug", 115: "sea slug", 116: "chiton", 117: "chambered nautilus", 118: "Dungeness crab", 119: "rock crab",
    120: "fiddler crab", 121: "king crab", 122: "American lobster", 123: "spiny lobster", 124: "crayfish", 125: "hermit crab", 126: "isopod", 127: "white stork", 128: "black stork", 129: "spoonbill",
    130: "flamingo", 131: "little blue heron", 132: "American egret", 133: "bittern", 134: "crane", 135: "limpkin", 136: "European gallinule", 137: "American coot", 138: "bustard", 139: "ruddy turnstone",
    140: "red-backed sandpiper", 141: "redshank", 142: "dowitcher", 143: "oystercatcher", 144: "pelican", 145: "king penguin", 146: "albatross", 147: "grey whale", 148: "killer whale", 149: "dugong",
    150: "sea lion", 151: "Chihuahua", 152: "Japanese spaniel", 153: "Maltese dog", 154: "Pekinese", 155: "Shih-Tzu", 156: "Blenheim spaniel", 157: "papillon", 158: "toy terrier", 159: "Rhodesian ridgeback",
    160: "Afghan hound", 161: "basset", 162: "beagle", 163: "bloodhound", 164: "bluetick", 165: "black-and-tan coonhound", 166: "Walker hound", 167: "English foxhound", 168: "redbone", 169: "borzoi",
    170: "Irish wolfhound", 171: "Italian greyhound", 172: "whippet", 173: "Ibizan hound", 174: "Norwegian elkhound", 175: "otterhound", 176: "Saluki", 177: "Scottish deerhound", 178: "Weimaraner", 179: "Staffordshire bullterrier",
    180: "American Staffordshire terrier", 181: "Bedlington terrier", 182: "Border terrier", 183: "Kerry blue terrier", 184: "Irish terrier", 185: "Norfolk terrier", 186: "Norwich terrier", 187: "Yorkshire terrier", 188: "wire-haired fox terrier", 189: "Lakeland terrier",
    190: "Sealyham terrier", 191: "Airedale", 192: "cairn", 193: "Australian terrier", 194: "Dandie Dinmont", 195: "Boston bull", 196: "miniature schnauzer", 197: "giant schnauzer", 198: "standard schnauzer", 199: "Scotch terrier",
    200: "Tibetan terrier", 201: "silky terrier", 202: "soft-coated wheaten terrier", 203: "West Highland white terrier", 204: "Lhasa", 205: "flat-coated retriever", 206: "curly-coated retriever", 207: "golden retriever", 208: "Labrador retriever", 209: "Chesapeake Bay retriever",
    210: "German short-haired pointer", 211: "vizsla", 212: "English setter", 213: "Irish setter", 214: "Gordon setter", 215: "Brittany spaniel", 216: "clumber", 217: "English springer", 218: "Welsh springer spaniel", 219: "cocker spaniel",
    220: "Sussex spaniel", 221: "Irish water spaniel", 222: "kuvasz", 223: "schipperke", 224: "groenendael", 225: "malinois", 226: "briard", 227: "kelpie", 228: "komondor", 229: "Old English sheepdog",
    230: "Shetland sheepdog", 231: "collie", 232: "Border collie", 233: "Bouvier des Flandres", 234: "Rottweiler", 235: "German shepherd", 236: "Doberman", 237: "miniature pinscher", 238: "Greater Swiss Mountain dog", 239: "Bernese mountain dog",
    240: "Appenzeller", 241: "EntleBucher", 242: "boxer", 243: "bull mastiff", 244: "Tibetan mastiff", 245: "French bulldog", 246: "Great Dane", 247: "Saint Bernard", 248: "Eskimo dog", 249: "malamute",
    250: "Siberian husky", 251: "dalmatian", 252: "affenpinscher", 253: "basenji", 254: "pug", 255: "Leonberg", 256: "Newfoundland", 257: "Great Pyrenees", 258: "Samoyed", 259: "Pomeranian",
    260: "chow", 261: "keeshond", 262: "Brabancon griffon", 263: "Pembroke", 264: "Cardigan", 265: "toy poodle", 266: "miniature poodle", 267: "standard poodle", 268: "Mexican hairless", 269: "timber wolf",
    270: "white wolf", 271: "red wolf", 272: "coyote", 273: "dingo", 274: "dhole", 275: "African hunting dog", 276: "hyena", 277: "red fox", 278: "kit fox", 279: "Arctic fox",
    280: "grey fox", 281: "tabby", 282: "tiger cat", 283: "Persian cat", 284: "Siamese cat", 285: "Egyptian cat", 286: "cougar", 287: "lynx", 288: "leopard", 289: "snow leopard",
    290: "jaguar", 291: "lion", 292: "tiger", 293: "cheetah", 294: "brown bear", 295: "American black bear", 296: "ice bear", 297: "sloth bear", 298: "mongoose", 299: "meerkat",
    300: "tiger beetle", 301: "ladybug", 302: "ground beetle", 303: "long-horned beetle", 304: "leaf beetle", 305: "dung beetle", 306: "rhinoceros beetle", 307: "weevil", 308: "fly", 309: "bee",
    310: "ant", 311: "grasshopper", 312: "cricket", 313: "walking stick", 314: "cockroach", 315: "mantis", 316: "cicada", 317: "leafhopper", 318: "lacewing", 319: "dragonfly",
    320: "damselfly", 321: "admiral", 322: "ringlet", 323: "monarch", 324: "cabbage butterfly", 325: "sulphur butterfly", 326: "lycaenid", 327: "starfish", 328: "sea urchin", 329: "sea cucumber",
    330: "wood rabbit", 331: "hare", 332: "Angora", 333: "hamster", 334: "porcupine", 335: "fox squirrel", 336: "marmot", 337: "beaver", 338: "guinea pig", 339: "sorrel",
    340: "zebra", 341: "hog", 342: "wild boar", 343: "warthog", 344: "hippopotamus", 345: "ox", 346: "water buffalo", 347: "bison", 348: "ram", 349: "bighorn",
    350: "ibex", 351: "hartebeest", 352: "impala", 353: "gazelle", 354: "Arabian camel", 355: "llama", 356: "weasel", 357: "mink", 358: "polecat", 359: "black-footed ferret",
    360: "otter", 361: "skunk", 362: "badger", 363: "armadillo", 364: "three-toed sloth", 365: "orangutan", 366: "gorilla", 367: "chimpanzee", 368: "gibbon", 369: "siamang",
    370: "guenon", 371: "patas", 372: "baboon", 373: "macaque", 374: "langur", 375: "colobus", 376: "proboscis monkey", 377: "marmoset", 378: "capuchin", 379: "howler monkey",
    380: "titi", 381: "spider monkey", 382: "squirrel monkey", 383: "Madagascar cat", 384: "indri", 385: "Indian elephant", 386: "African elephant", 387: "lesser panda", 388: "giant panda", 389: "barracouta",
    390: "eel", 391: "coho", 392: "rock beauty", 393: "anemone fish", 394: "sturgeon", 395: "gar", 396: "lionfish", 397: "puffer", 398: "abacus", 399: "abaya",
    400: "academic gown", 401: "accordion", 402: "acoustic guitar", 403: "aircraft carrier", 404: "airliner", 405: "airship", 406: "altar", 407: "ambulance", 408: "amphibian", 409: "analog clock",
    410: "apiary", 411: "apron", 412: "ashcan", 413: "assault rifle", 414: "backpack", 415: "bakery", 416: "balance beam", 417: "balloon", 418: "ballpoint", 419: "Band Aid",
    420: "banjo", 421: "bannister", 422: "barbell", 423: "barber chair", 424: "barbershop", 425: "barn", 426: "barometer", 427: "barrel", 428: "barrow", 429: "baseball",
    430: "basketball", 431: "bassinet", 432: "bassoon", 433: "bathing cap", 434: "bath towel", 435: "bathtub", 436: "beach wagon", 437: "beacon", 438: "beaker", 439: "bearskin",
    440: "beer bottle", 441: "beer glass", 442: "bell cote", 443: "bib", 444: "bicycle-built-for-two", 445: "bikini", 446: "binder", 447: "binoculars", 448: "birdhouse", 449: "boathouse",
    450: "bobsled", 451: "bolo tie", 452: "bonnet", 453: "bookcase", 454: "bookshop", 455: "bottlecap", 456: "bow", 457: "bow tie", 458: "brass", 459: "brassiere",
    460: "breakwater", 461: "breastplate", 462: "broom", 463: "bucket", 464: "buckle", 465: "bulletproof vest", 466: "bullet train", 467: "butcher shop", 468: "cab", 469: "caldron",
    470: "candle", 471: "cannon", 472: "canoe", 473: "can opener", 474: "cardigan", 475: "car mirror", 476: "carousel", 477: "carpenter's kit", 478: "carton", 479: "car wheel",
    480: "cash machine", 481: "cassette", 482: "cassette player", 483: "castle", 484: "catamaran", 485: "CD player", 486: "cello", 487: "cellular telephone", 488: "chain", 489: "chainlink fence",
    490: "chain mail", 491: "chain saw", 492: "chest", 493: "chiffonier", 494: "chime", 495: "china cabinet", 496: "Christmas stocking", 497: "church", 498: "cinema", 499: "cleaver",
    500: "cliff dwelling", 501: "cloak", 502: "clog", 503: "cocktail shaker", 504: "coffee mug", 505: "coffeepot", 506: "coil", 507: "combination lock", 508: "computer keyboard", 509: "confectionery",
    510: "container ship", 511: "convertible", 512: "corkscrew", 513: "cornet", 514: "cowboy boot", 515: "cowboy hat", 516: "cradle", 517: "crane", 518: "crash helmet", 519: "crate",
    520: "crib", 521: "Crock Pot", 522: "croquet ball", 523: "crutch", 524: "cuirass", 525: "dam", 526: "desk", 527: "desktop computer", 528: "dial telephone", 529: "diaper",
    530: "digital clock", 531: "digital watch", 532: "dining table", 533: "dishrag", 534: "dishwasher", 535: "disk brake", 536: "dock", 537: "dogsled", 538: "dome", 539: "doormat",
    540: "drilling platform", 541: "drum", 542: "drumstick", 543: "dumbbell", 544: "Dutch oven", 545: "electric fan", 546: "electric guitar", 547: "electric locomotive", 548: "entertainment center", 549: "envelope",
    550: "espresso maker", 551: "face powder", 552: "feather boa", 553: "file", 554: "fireboat", 555: "fire engine", 556: "fire screen", 557: "flagpole", 558: "flute", 559: "folding chair",
    560: "football helmet", 561: "forklift", 562: "fountain", 563: "fountain pen", 564: "four-poster", 565: "freight car", 566: "French horn", 567: "frying pan", 568: "fur coat", 569: "garbage truck",
    570: "gasmask", 571: "gas pump", 572: "goblet", 573: "go-kart", 574: "golf ball", 575: "golfcart", 576: "gondola", 577: "gong", 578: "gown", 579: "grand piano",
    580: "greenhouse", 581: "grille", 582: "grocery store", 583: "guillotine", 584: "hair slide", 585: "hair spray", 586: "half track", 587: "hammer", 588: "hamper", 589: "hand blower",
    590: "hand-held computer", 591: "handkerchief", 592: "hard disc", 593: "harmonica", 594: "harp", 595: "harvester", 596: "hatchet", 597: "holster", 598: "home theater", 599: "honeycomb",
    600: "hook", 601: "hoopskirt", 602: "horizontal bar", 603: "horse cart", 604: "hourglass", 605: "iPod", 606: "iron", 607: "jack-o'-lantern", 608: "jean", 609: "jeep",
    610: "jersey", 611: "jigsaw puzzle", 612: "jinrikisha", 613: "joystick", 614: "kimono", 615: "knee pad", 616: "knot", 617: "lab coat", 618: "ladle", 619: "lampshade",
    620: "laptop", 621: "lawn mower", 622: "lens cap", 623: "letter opener", 624: "library", 625: "lifeboat", 626: "lighter", 627: "limousine", 628: "liner", 629: "lipstick",
    630: "Loafer", 631: "lotion", 632: "loudspeaker", 633: "loupe", 634: "lumbermill", 635: "magnetic compass", 636: "mailbag", 637: "mailbox", 638: "maillot", 639: "maillot",
    640: "manhole cover", 641: "maraca", 642: "marimba", 643: "mask", 644: "matchstick", 645: "maypole", 646: "maze", 647: "measuring cup", 648: "medicine chest", 649: "megalith",
    650: "microphone", 651: "microwave", 652: "military uniform", 653: "milk can", 654: "minibus", 655: "miniskirt", 656: "minivan", 657: "missile", 658: "mitten", 659: "mixing bowl",
    660: "mobile home", 661: "Model T", 662: "modem", 663: "monastery", 664: "monitor", 665: "moped", 666: "mortar", 667: "mortarboard", 668: "mosque", 669: "mosquito net",
    670: "motor scooter", 671: "mountain bike", 672: "mountain tent", 673: "mouse", 674: "mousetrap", 675: "moving van", 676: "muzzle", 677: "nail", 678: "neck brace", 679: "necklace",
    680: "nipple", 681: "notebook", 682: "obelisk", 683: "oboe", 684: "ocarina", 685: "odometer", 686: "oil filter", 687: "organ", 688: "oscilloscope", 689: "overskirt",
    690: "oxcart", 691: "oxygen mask", 692: "packet", 693: "paddle", 694: "paddlewheel", 695: "padlock", 696: "paintbrush", 697: "pajama", 698: "palace", 699: "panpipe",
    700: "paper towel", 701: "parachute", 702: "parallel bars", 703: "park bench", 704: "parking meter", 705: "passenger car", 706: "patio", 707: "pay-phone", 708: "pedestal", 709: "pencil box",
    710: "pencil sharpener", 711: "perfume", 712: "Petri dish", 713: "photocopier", 714: "pick", 715: "pickelhaube", 716: "picket fence", 717: "pickup", 718: "pier", 719: "piggy bank",
    720: "pill bottle", 721: "pillow", 722: "ping-pong ball", 723: "pinwheel", 724: "pirate", 725: "pitcher", 726: "plane", 727: "planetarium", 728: "plastic bag", 729: "plate rack",
    730: "plow", 731: "plunger", 732: "Polaroid camera", 733: "pole", 734: "police van", 735: "poncho", 736: "pool table", 737: "pop bottle", 738: "pot", 739: "potter's wheel",
    740: "power drill", 741: "prayer rug", 742: "printer", 743: "prison", 744: "projectile", 745: "projector", 746: "puck", 747: "punching bag", 748: "purse", 749: "quill",
    750: "quilt", 751: "racer", 752: "racket", 753: "radiator", 754: "radio", 755: "radio telescope", 756: "rain barrel", 757: "recreational vehicle", 758: "reel", 759: "reflex camera",
    760: "refrigerator", 761: "remote control", 762: "restaurant", 763: "revolver", 764: "rifle", 765: "rocking chair", 766: "rotisserie", 767: "rubber eraser", 768: "rugby ball", 769: "rule",
    770: "running shoe", 771: "safe", 772: "safety pin", 773: "saltshaker", 774: "sandal", 775: "sarong", 776: "sax", 777: "scabbard", 778: "scale", 779: "school bus",
    780: "schooner", 781: "scoreboard", 782: "screen", 783: "screw", 784: "screwdriver", 785: "seat belt", 786: "sewing machine", 787: "shield", 788: "shoe shop", 789: "shoji",
    790: "shopping basket", 791: "shopping cart", 792: "shovel", 793: "shower cap", 794: "shower curtain", 795: "ski", 796: "ski mask", 797: "sleeping bag", 798: "slide rule", 799: "sliding door",
    800: "slot", 801: "snorkel", 802: "snowmobile", 803: "snowplow", 804: "soap dispenser", 805: "soccer ball", 806: "sock", 807: "solar dish", 808: "sombrero", 809: "soup bowl",
    810: "space bar", 811: "space heater", 812: "space shuttle", 813: "spatula", 814: "speedboat", 815: "spider web", 816: "spindle", 817: "sports car", 818: "spotlight", 819: "stage",
    820: "steam locomotive", 821: "steel arch bridge", 822: "steel drum", 823: "stethoscope", 824: "stole", 825: "stone wall", 826: "stopwatch", 827: "stove", 828: "strainer", 829: "streetcar",
    830: "stretcher", 831: "studio couch", 832: "stupa", 833: "submarine", 834: "suit", 835: "sundial", 836: "sunglass", 837: "sunglasses", 838: "sunscreen", 839: "suspension bridge",
    840: "swab", 841: "sweatshirt", 842: "swimming trunks", 843: "swing", 844: "switch", 845: "syringe", 846: "table lamp", 847: "tank", 848: "tape player", 849: "teapot",
    850: "teddy", 851: "television", 852: "tennis ball", 853: "thatch", 854: "theater curtain", 855: "thimble", 856: "thresher", 857: "throne", 858: "tile roof", 859: "toaster",
    860: "tobacco shop", 861: "toilet seat", 862: "torch", 863: "totem pole", 864: "tow truck", 865: "toyshop", 866: "tractor", 867: "trailer truck", 868: "tray", 869: "trench coat",
    870: "tricycle", 871: "trimaran", 872: "tripod", 873: "triumphal arch", 874: "trolleybus", 875: "trombone", 876: "tub", 877: "turnstile", 878: "typewriter keyboard", 879: "umbrella",
    880: "unicycle", 881: "upright", 882: "vacuum", 883: "vase", 884: "vault", 885: "velvet", 886: "vending machine", 887: "vestment", 888: "viaduct", 889: "violin",
    890: "volleyball", 891: "waffle iron", 892: "wall clock", 893: "wallet", 894: "wardrobe", 895: "warplane", 896: "washbasin", 897: "washer", 898: "water bottle", 899: "water jug",
    900: "water tower", 901: "whiskey jug", 902: "whistle", 903: "wig", 904: "window screen", 905: "window shade", 906: "Windsor tie", 907: "wine bottle", 908: "wing", 909: "wok",
    910: "wooden spoon", 911: "wool", 912: "worm fence", 913: "wreck", 914: "yawl", 915: "yurt", 916: "web site", 917: "comic book", 918: "crossword puzzle", 919: "street sign",
    920: "traffic light", 921: "book jacket", 922: "menu", 923: "plate", 924: "guacamole", 925: "consomme", 926: "hot pot", 927: "trifle", 928: "ice cream", 929: "ice lolly",
    930: "French loaf", 931: "bagel", 932: "pretzel", 933: "cheeseburger", 934: "hotdog", 935: "mashed potato", 936: "head cabbage", 937: "broccoli", 938: "cauliflower", 939: "zucchini",
    940: "spaghetti squash", 941: "acorn squash", 942: "butternut squash", 943: "cucumber", 944: "artichoke", 945: "bell pepper", 946: "cardoon", 947: "mushroom", 948: "Granny Smith", 949: "strawberry",
    950: "orange", 951: "lemon", 952: "fig", 953: "pineapple", 954: "banana", 955: "jackfruit", 956: "custard apple", 957: "pomegranate", 958: "hay", 959: "carbonara",
    960: "chocolate sauce", 961: "dough", 962: "meat loaf", 963: "pizza", 964: "potpie", 965: "burrito", 966: "red wine", 967: "espresso", 968: "cup", 969: "eggnog",
    970: "alp", 971: "bubble", 972: "cliff", 973: "coral reef", 974: "geyser", 975: "lakeside", 976: "promontory", 977: "sandbar", 978: "seashore", 979: "valley",
    980: "volcano", 981: "ballplayer", 982: "groom", 983: "scuba diver", 984: "rapeseed", 985: "daisy", 986: "yellow lady's slipper", 987: "corn", 988: "acorn", 989: "hip",
    990: "buckeye", 991: "coral fungus", 992: "agaric", 993: "gyromitra", 994: "stinkhorn", 995: "earthstar", 996: "hen-of-the-woods", 997: "bolete", 998: "ear", 999: "toilet tissue",
}


_COCO_NAMES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 12: "parking meter", 13: "bench", 14: "bird",
    15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
    25: "backpack", 26: "umbrella", 28: "tie", 29: "suitcase",
    30: "frisbee", 31: "skis", 32: "snowboard", 33: "sports ball",
    56: "chair", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
    67: "cell phone", 73: "book", 74: "clock", 75: "vase",
}

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

_PALETTE = [
    (48, 255, 48), (48, 200, 255), (255, 180, 48), (255, 48, 180),
    (180, 48, 255), (48, 255, 200), (255, 255, 48), (48, 180, 255),
    (255, 120, 48), (120, 255, 48), (200, 48, 48), (48, 48, 200),
    (200, 200, 48), (48, 200, 200), (200, 48, 200), (180, 255, 180),
    (255, 180, 180), (180, 180, 255), (255, 220, 180), (220, 180, 255),
]


def _draw_detections(bgr: np.ndarray, result: Any) -> None:
    h, w = bgr.shape[:2]
    color_map = {
        "person": (48, 255, 48), "car": (48, 200, 255),
        "truck": (48, 150, 255), "bus": (48, 150, 255),
        "motorcycle": (48, 180, 255), "bicycle": (48, 220, 255),
        "face": (48, 255, 200),
    }
    for obj in result.objects:
        color = color_map.get(obj.label, (48, 255, 48))
        x1, y1 = int(obj.bbox.x * w), int(obj.bbox.y * h)
        x2 = int((obj.bbox.x + obj.bbox.width) * w)
        y2 = int((obj.bbox.y + obj.bbox.height) * h)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{obj.label}: {obj.score:.0%}"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
        lh = th + baseline + 6
        ly = max(y1 - lh, 0)
        overlay = bgr[ly:ly + lh, x1:x1 + tw + 8].copy()
        cv2.rectangle(bgr, (x1, ly), (x1 + tw + 8, ly + lh), (0, 0, 0), -1)
        cv2.addWeighted(bgr[ly:ly + lh, x1:x1 + tw + 8], 0.5,
                        overlay, 0.5, 0, bgr[ly:ly + lh, x1:x1 + tw + 8])
        tx, ty = x1 + 4, ly + th + 2
        cv2.putText(bgr, label, (tx + 1, ty + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(bgr, label, (tx, ty), font, scale, color, thick, cv2.LINE_AA)


def _draw_classifications(bgr: np.ndarray, result: Any) -> None:
    if not result.classifications:
        return
    h, w = bgr.shape[:2]
    bar_h = 28 * min(len(result.classifications), 5) + 10
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, bgr, 0.3, 0, bgr)
    y_off = 22
    for cls in result.classifications[:5]:
        pct = cls.confidence * 100
        text = f"{cls.label}: {pct:.1f}%"
        color = (0, 255, 0) if pct > 50 else (0, 200, 255)
        cv2.putText(bgr, text, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        bar_w = int(min(pct / 100, 1.0) * 150)
        cv2.rectangle(bgr, (10, y_off + 5), (10 + bar_w, y_off + 14), color, -1)
        cv2.rectangle(bgr, (10, y_off + 5), (160, y_off + 14), (100, 100, 100), 1)
        y_off += 28


def _rle_decode(rle: bytes, width: int, height: int) -> Optional[np.ndarray]:
    """Decode varint-encoded RLE mask from ai-runtime.

    ai-runtime encodes masks as pairs of (start, length) where each value
    is a protobuf-style varint (7 bits per byte, MSB continuation flag).
    """
    try:
        if not rle:
            return None
        data = rle if isinstance(rle, (bytes, bytearray)) else rle.encode()
        mask = np.zeros(width * height, dtype=np.uint8)
        pos = 0

        def decode_varint(idx: int) -> tuple:
            val = 0
            shift = 0
            while idx < len(data):
                b = data[idx]
                idx += 1
                val |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            return val, idx

        while pos < len(data):
            start, pos = decode_varint(pos)
            length, pos = decode_varint(pos)
            mask[start:start + length] = 1
        return mask.reshape((height, width))
    except Exception:
        return None


def _reshape_1d_to_2d(tensor_1d: np.ndarray, hint_w: int = 0, hint_h: int = 0) -> Optional[np.ndarray]:
    """Reshape a 1-D flat tensor to 2-D (H, W).

    Uses explicit *hint_w* / *hint_h* when available.  Otherwise falls back
    to square-root heuristics (nearest exact square or factorisation).
    """
    n = tensor_1d.size
    if n == 0:
        return None
    # Explicit hints (e.g. model output dimensions)
    if hint_w > 0 and hint_h > 0 and hint_w * hint_h == n:
        return tensor_1d.reshape(hint_h, hint_w)
    # Try exact square
    side = int(np.sqrt(n))
    if side * side == n:
        return tensor_1d.reshape(side, side)
    # Find closest rectangular factorisation (prefer wider-than-tall)
    best_w, best_h = n, 1
    for h in range(int(np.sqrt(n)), 0, -1):
        if n % h == 0:
            w = n // h
            best_w, best_h = w, h
            break
    return tensor_1d.reshape(best_h, best_w)


# Module-level reference set by ModelShowcase.draw_overlay so draw handlers
# can access model dimensions without needing a self reference.
_current_draw_model_info: Dict[str, Any] = {}

# Module-level overlay cache — populated by draw handlers, consumed by frame loop.
_overlay_cache: Dict[str, Any] = {}


def _seg_class_map_from_result(result: Any) -> Optional[np.ndarray]:
    """Extract a 2-D class-id map (H, W) uint8 from inference result.

    For pixel-level segmentation, raw output tensors (per-pixel class logits)
    are preferred over single-class NMS masks.  Falls back to RLE masks only
    when no raw tensors exist.
    """
    # Primary: raw output tensors (per-pixel class prediction)
    raw = getattr(result, "raw_outputs", None)
    if raw and len(raw) >= 2:
        channels = [np.asarray(t).flatten().astype(np.float32) for t in raw]
        stacked = np.stack(channels, axis=0)  # (C, N)
        n_pixels = stacked.shape[1]
        mi = _current_draw_model_info
        out_h = mi.get("input_height", 0)
        out_w = mi.get("input_width", 0)
        if out_h > 0 and out_w > 0 and out_h * out_w == n_pixels:
            h, w = out_h, out_w
        else:
            side = int(np.sqrt(n_pixels))
            h, w = side, side
        if h * w != n_pixels:
            logger.warning("SEG: cannot reshape %d pixels to %dx%d", n_pixels, h, w)
            return None
        volume = stacked.reshape(stacked.shape[0], h, w)
        class_map = np.argmax(volume, axis=0).astype(np.uint8)
        logger.info("SEG from raw: %d channels, %dx%d, unique=%s",
                     stacked.shape[0], h, w, np.unique(class_map))
        return class_map
    if raw and len(raw) == 1:
        tensor = np.asarray(raw[0]).squeeze()
        if tensor.ndim == 3:
            # (C, H, W) → argmax over channel axis
            return np.argmax(tensor, axis=0).astype(np.uint8)
        if tensor.ndim == 2:
            return tensor.astype(np.uint8)
        if tensor.ndim == 1:
            mi = _current_draw_model_info
            reshaped = _reshape_1d_to_2d(tensor.astype(np.float32),
                                          hint_w=mi.get("input_width", 0),
                                          hint_h=mi.get("input_height", 0))
            if reshaped is not None:
                return reshaped.astype(np.uint8)

    # Fallback: HAL RLE masks (single-class NMS)
    masks = getattr(result, "masks", None)
    if masks:
        for mask in masks:
            rle = mask.mask_rle
            if not rle:
                continue
            binary = _rle_decode(rle, mask.mask_width, mask.mask_height)
            if binary is None:
                continue
            cm = np.zeros((mask.mask_height, mask.mask_width), dtype=np.uint8)
            cm[binary] = mask.class_id + 1  # +1 so background=0
            return cm
    return None


def _seg_lut_colorize(class_map: np.ndarray, target_w: int, target_h: int) -> Optional[Tuple[np.ndarray, Dict[int, int]]]:
    """Vectorised class-map → BGR colorization via LUT (one pass, no per-class loop).

    Returns (colored_mask_bgr, class_pixels_dict) or None on empty result.
    """
    if class_map is None:
        return None

    # Resize to target frame dimensions
    if class_map.shape[1] != target_w or class_map.shape[0] != target_h:
        class_map = cv2.resize(
            class_map, (target_w, target_h), interpolation=cv2.INTER_NEAREST,
        )

    # Build a 256×3 BGR lookup table. Row 0 = background → MUST be black so
    # that the overlay only colours foreground classes (the blend uses
    # max-over-channels > 0 as the foreground mask). The earlier loop started
    # at i=0 and thus set lut[0] = _PALETTE[0] = (48,255,48), which dyed the
    # whole background green; the full-frame addWeighted then darkened it into
    # the dim green-tinted look linknet showed. Foreground class colours are
    # unchanged (class i still maps to _PALETTE[i % len]).
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(1, 256):
        lut[i] = _PALETTE[i % len(_PALETTE)]

    # Vectorised colour lookup via numpy fancy indexing (one shot, no loop)
    colored = lut[class_map.ravel()].reshape(target_h, target_w, 3)

    # Count pixels per class (O(n) with bincount, no sort needed)
    flat = class_map.ravel()
    counts = np.bincount(flat, minlength=256)
    class_pixels: Dict[int, int] = {}
    for cls_id in range(1, 256):  # skip 0 (background)
        c = int(counts[cls_id])
        if c > 0:
            class_pixels[cls_id] = c

    if not class_pixels:
        return None

    return colored, class_pixels


def _draw_segmentation(bgr: np.ndarray, result: Any) -> None:
    h, w = bgr.shape[:2]

    class_map = _seg_class_map_from_result(result)
    if class_map is None:
        return

    colorized = _seg_lut_colorize(class_map, w, h)
    if colorized is None:
        return

    colored_mask, class_pixels = colorized

    # Blend ONLY foreground pixels; the background keeps its original camera
    # colour (same natural look as the detection preview). _seg_lut_colorize
    # sets the background mask row to black, so max-over-channels > 0 marks
    # exactly the foreground. The previous full-frame addWeighted dyed the
    # whole frame via the (non-black) background mask row and darkened it to
    # 45% — linknet looked like a dim, tinted image. This render path is
    # throttled to ~10 Hz upstream (only on new_result && should_render), so a
    # one-off addWeighted-into-temp + masked copy is cheap here.
    alpha = 0.55
    fg_mask = colored_mask.max(axis=2) > 0
    if fg_mask.any():
        blended = cv2.addWeighted(bgr, 1 - alpha, colored_mask, alpha, 0)
        bgr[fg_mask] = blended[fg_mask]

    _overlay_cache["colored_mask"] = colored_mask
    _overlay_cache["class_pixels"] = class_pixels

    # NOTE: the per-class "cls N: X%" stats panel is intentionally NOT drawn
    # here. The browser overlay canvas draws it (drawSegStats in index.html)
    # from the same class_pixels via SSE, so baking a second copy into the JPEG
    # stacked a duplicate panel — the "two cls boxes" seen on linknet. Only the
    # pixel colour overlay is baked server-side (the canvas can't reproduce it
    # cheaply); the textual stats panel stays single-sourced on the canvas.


_FACE_68_GROUPS = [
    ("jaw",        list(range(0, 17)),  (160, 160, 160)),
    ("left_brow",  list(range(17, 22)), (255, 200, 0)),
    ("right_brow", list(range(22, 27)), (255, 200, 0)),
    ("nose",       list(range(27, 36)), (0, 220, 0)),
    ("left_eye",   list(range(36, 42)), (100, 150, 255)),
    ("right_eye",  list(range(42, 48)), (100, 150, 255)),
    ("mouth",      list(range(48, 68)), (80, 80, 255)),
]


def _draw_keypoints(bgr: np.ndarray, result: Any) -> None:
    h, w = bgr.shape[:2]
    for ls in result.landmarks:
        pts = ls.points
        if not pts:
            continue
        n_pts = len(pts)
        px_pts = [(int(p.x * w), int(p.y * h)) for p in pts]
        # Draw feature connections for 68-point models
        if n_pts == 68:
            xs = [p[0] for p in px_pts]
            ys = [p[1] for p in px_pts]
            pad_x = max(20, int((max(xs) - min(xs)) * 0.25))
            pad_y = max(20, int((max(ys) - min(ys)) * 0.25))
            x1, y1 = min(xs) - pad_x, min(ys) - pad_y
            cv2.rectangle(bgr, (x1, y1),
                           (max(xs) + pad_x, max(ys) + pad_y), (0, 200, 0), 2)
            cv2.putText(bgr, f"face ({n_pts} pts)", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)
            for name, indices, color in _FACE_68_GROUPS:
                curve = [px_pts[i] for i in indices]
                if name in ("left_eye", "right_eye", "mouth"):
                    curve.append(curve[0])
                if len(curve) >= 2:
                    cv2.polylines(bgr, [np.array(curve, dtype=np.int32)],
                                  False, color, 1, cv2.LINE_AA)
            for px, py in px_pts:
                cv2.circle(bgr, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)
        elif n_pts > 68:
            # No per-point confidence available — model outputs (x,y,z) only
            left_eye  = [33, 133, 159, 145, 7, 163, 246, 161, 160, 158, 157, 173, 155, 154, 153, 144]
            right_eye = [263, 362, 386, 374, 249, 390, 466, 388, 387, 385, 384, 398, 382, 381, 380, 373]
            nose      = [1, 4, 5, 6, 94, 2, 168, 197, 195, 19]
            lips      = [61, 291, 0, 17, 78, 308, 13, 14]
            brows     = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]

            dot_groups = [
                (left_eye,  (0, 255, 255), 4),
                (right_eye, (0, 255, 255), 4),
                (nose,      (0, 255, 0),   4),
                (lips,      (0, 100, 255), 4),
                (brows,     (0, 200, 255), 3),
            ]
            for indices, color, radius in dot_groups:
                for idx in indices:
                    if idx < n_pts:
                        cv2.circle(bgr, px_pts[idx], radius, color, -1, cv2.LINE_AA)


def _draw_clip(bgr: np.ndarray, result: Any) -> None:
    _draw_classifications(bgr, result)


def _put_text_cjk(bgr: np.ndarray, text: str, pos: tuple,
                  font_scale: float = 0.5, color: tuple = (0, 255, 255),
                  thickness: int = 1) -> None:
    """Render text (including CJK) onto a BGR image using Pillow."""
    has_cjk = any("\u4e00" <= c <= "\u9fff" for c in text)
    if not has_cjk:
        cv2.putText(bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
        return
    font_size = max(14, int(font_scale * 28))
    font = _get_cjk_font(font_size)
    if font is None:
        cv2.putText(bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
        return
    img_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color[::-1])
    bgr[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


_cached_cjk_font = None


def _get_cjk_font(size: int) -> Any:
    global _cached_cjk_font
    if _cached_cjk_font is not None and _cached_cjk_font.size == size:
        return _cached_cjk_font
    paths = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                _cached_cjk_font = ImageFont.truetype(p, size)
                return _cached_cjk_font
            except Exception:
                continue
    _cached_cjk_font = ImageFont.load_default()
    return _cached_cjk_font


def _draw_ocr(bgr: np.ndarray, result: Any) -> None:
    if not getattr(result, "ocr_lines", None):
        return
    h, w = bgr.shape[:2]
    for line in result.ocr_lines:
        bx1, by1 = int(line.bbox.x * w), int(line.bbox.y * h)
        bx2 = int((line.bbox.x + line.bbox.width) * w)
        by2 = int((line.bbox.y + line.bbox.height) * h)
        # Bounding box
        cv2.rectangle(bgr, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
        # Label with dark background pill
        text = f" {line.text} "
        font_scale = 0.65
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_h = th + baseline + 8
        label_y = max(by1 - label_h - 4, 0)
        # Semi-transparent dark background
        overlay = bgr.copy()
        cv2.rectangle(overlay, (bx1, label_y), (bx1 + tw + 4, label_y + label_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, bgr, 0.35, 0, bgr)
        # Colored top border
        cv2.rectangle(bgr, (bx1, label_y), (bx1 + tw + 4, label_y + 2),
                      (0, 255, 255), -1)
        # Text (use CJK-aware renderer for Chinese characters)
        _put_text_cjk(bgr, text, (bx1 + 2, label_y + th + 4),
                      font_scale, (0, 255, 255), thickness)


def _draw_embedding_info(bgr: np.ndarray, result: Any) -> None:
    cv2.putText(bgr, "Embedding mode (no visual overlay)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)


def _extract_depth_data(result: Any) -> Optional[np.ndarray]:
    """Extract depth map float array from inference result.

    Uses ``depth_maps`` from HAL postprocess when available.
    Falls back to raw model output (logits → sigmoid → inverse depth).
    """
    depth_maps = getattr(result, 'depth_maps', [])
    if depth_maps:
        return depth_maps[0].data

    raw = getattr(result, 'raw_outputs', None)
    if raw and len(raw) > 0:
        tensor = raw[0].squeeze().astype(np.float32)
        if tensor.ndim == 1:
            mi = _current_draw_model_info
            reshaped = _reshape_1d_to_2d(
                tensor,
                hint_w=mi.get("input_width", 0),
                hint_h=mi.get("input_height", 0),
            )
            # _reshape_1d_to_2d returns a 2-D ndarray (or None when the tensor is
            # empty). Do NOT write `reshaped or tensor` — a returned ndarray is not
            # a valid bool, so that raises "truth value of an array is ambiguous".
            # That was the repeating [ERROR] Frame processing error on scdepth
            # during model warmup (when depth_maps is empty and we fall back to
            # raw_outputs). Explicit None check is the correct fallback.
            if reshaped is not None:
                tensor = reshaped
        if tensor.ndim == 2:
            sigmoid = 1.0 / (1.0 + np.exp(-tensor))
            return 1.0 / (sigmoid * 10.0 + 0.009)
    return None


def _draw_depth(bgr: np.ndarray, result: Any) -> None:
    """Overlay depth map as a colorized heatmap (server-side, used for RTSP)."""
    depth_data = _extract_depth_data(result)
    if depth_data is None:
        return

    norm = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    resized = cv2.resize(colored, (bgr.shape[1], bgr.shape[0]))
    cv2.addWeighted(bgr, 0.6, resized, 0.4, 0, bgr)
    cv2.putText(bgr, "Depth Estimation", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # Cache for fast re-blend on next camera frame
    _overlay_cache["colored_mask"] = resized


DRAW_HANDLERS: Dict[str, Any] = {
    "detection": _draw_detections,
    "classification": _draw_classifications,
    "segmentation": _draw_segmentation,
    "keypoint": _draw_keypoints,
    "clip": _draw_clip,
    "pipeline_lpr": _draw_ocr,
    "pipeline_ocr": _draw_ocr,
    "embedding": _draw_embedding_info,
    "depth": _draw_depth,
}


# ---------------------------------------------------------------------------
# CTC Greedy Decoder
# ---------------------------------------------------------------------------

def _ctc_greedy_decode(logits: np.ndarray, charset: List[str],
                       blank: int = 0) -> tuple:
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    argmax = np.argmax(logits, axis=-1)
    if not isinstance(argmax, np.ndarray):
        argmax = np.array([argmax])
    chars: List[str] = []
    confidences: List[float] = []
    prev = blank
    for t in range(argmax.shape[0]):
        idx = int(argmax[t])
        if idx != blank and idx != prev:
            if 0 < idx < len(charset):
                chars.append(charset[idx])
                confidences.append(float(logits[t][idx]))
        prev = idx
    text = "".join(chars)
    conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, conf


def _bbox_iou(a: Any, b: Any) -> float:
    """IoU of two bboxes with x, y, w, h attributes."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# FrameBuffer
# ---------------------------------------------------------------------------
class FrameBuffer:
    def __init__(self) -> None:
        self.frame: Optional[bytes] = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.frame_id = 0

    def update(self, jpeg_bytes: bytes) -> None:
        with self.condition:
            self.frame = jpeg_bytes
            self.frame_id += 1
            self.condition.notify_all()

    def wait_for_new(self, last_id: int, timeout: float = 1.0) -> tuple:
        with self.condition:
            if self.condition.wait_for(lambda: self.frame_id > last_id, timeout):
                return self.frame, self.frame_id
            return None, last_id

    def get(self, timeout: float = 2.0) -> Optional[bytes]:
        with self.condition:
            if self.condition.wait_for(lambda: self.frame is not None, timeout):
                return self.frame
            return None


# ---------------------------------------------------------------------------
# ModelShowcase
# ---------------------------------------------------------------------------
class ModelShowcase:
    def __init__(self) -> None:
        self.infer_client: Optional[InferenceClient] = None
        self.running = False

        self.current_model: str = ""
        self.current_model_type: str = "detection"
        self._current_model_info: Dict[str, str] = {}
        self._model_lock = threading.Lock()

        self._latest_result: Optional[Any] = None
        self._result_lock = threading.Lock()

        self.frame_count = 0
        self.infer_count = 0
        self.infer_time_us = 0
        self.current_fps = 0.0

        # Dynamic inference FPS (delta infer_count / delta wall-clock time)
        self._infer_fps: float = 0.0
        self._infer_fps_last_count: int = 0
        self._infer_fps_last_time: float = 0.0

        # Hardware-level inference timing from HAL
        self._hw_infer_time_us: int = 0

        # Prefer a single InferBatch RPC for multi-crop pipelines (OCR/LPR
        # Stage 2). Auto-disabled if the server reports UNIMPLEMENTED, after
        # which Stage 2 falls back to sequential per-crop infer() calls.
        self._batch_enabled: bool = True

        # Cache for server-side overlay: reuse the coloured mask when the
        # inference result hasn't changed between preview frames.  Only the
        # lightweight alpha-blend is re-run on each new camera frame.
        self._cached_colored_mask: Optional[np.ndarray] = None
        self._cached_fg_mask: Optional[np.ndarray] = None
        self._cached_class_pixels: Optional[Dict[int, int]] = None
        self._cached_overlay_infer_count: int = -1
        self._cached_jpeg: Optional[bytes] = None
        # Last successfully decoded preview frame. When the camera "sub" stream
        # hiccups (get_frame returns None — NOT an NPU stall, since inference
        # runs on a separate stream), we re-push this frame so the MJPEG preview
        # keeps flowing instead of cratering to ~0 fps. See run_frame_loop().
        self._last_bgr: Optional[np.ndarray] = None
        # Time-based throttle: for server-side overlay models, only re-render
        # + re-encode at most every _overlay_interval seconds.  Between renders
        # the cached JPEG bytes are pushed directly (near-zero CPU).
        self._overlay_last_render: float = 0.0
        self._overlay_interval: float = 0.1  # 10 FPS target for overlay updates

        self._npu_stats: Dict[str, Any] = {}
        self._npu_stats_lock = threading.Lock()

        self._consecutive_failures = 0
        # Throttle auto re-registration when ai-runtime loses a model (the
        # grpcpp_sync_ser crash restarts ai-runtime and wipes its in-memory
        # model registry → infer returns "Model not found"). We force a
        # re-register at most every _reregister_interval so the loop, which
        # fails ~1-2x/s, doesn't hammer RegisterModel.
        self._last_reregister_ts: float = 0.0
        self._reregister_interval: float = 5.0
        self._degraded = False
        self._degrade_reason: Optional[str] = None
        self._model_switch_cooldown = 1.5  # seconds between model switches

        self._pipeline_stage1_us = 0
        self._pipeline_stage2_us = 0
        self._pipeline_count = 0
        self._ocr_accumulator = _OcrAccumulator(ttl_sec=3.0)
        self._prev_landmarks: Optional[List[Tuple[float, float]]] = None
        self._landmark_smooth_alpha = 0.35

        # GenAI session state
        self._genai_session_id: Optional[str] = None
        self._genai_lock = threading.Lock()

        # Lens control (SDK → device-control gRPC)
        self.device_client = DeviceClient()
        self._lens_lock = threading.Lock()

        self.frame_buffer = FrameBuffer()
        self._sse_subscribers: List = []
        self._sse_lock = threading.Lock()

        self._infer_thread: Optional[threading.Thread] = None
        self._infer_running = False
        self._npu_stats_thread: Optional[threading.Thread] = None
        self._reaper_thread: Optional[threading.Thread] = None
        # Models that must stay resident alongside the current model. The
        # reaper evicts everything else so leaked network groups (left behind
        # by a timed-out infer holding ref_count) get cleaned up. Populated in
        # _verify_current_model (startup) and switch_model via
        # _compute_pinned_models.
        self._pinned_models: set = set()

        self.stream_id = os.environ.get("STREAM_ID", "main")
        self.infer_stream_id = os.environ.get("INFER_STREAM_ID", "sub")
        self._active_infer_stream = self.infer_stream_id
        self.infer_fps_target = int(os.environ.get("INFER_FPS", "10"))
        self._infer_interval = 1.0 / self.infer_fps_target
        self.jpeg_quality = int(os.environ.get("JPEG_QUALITY", "60"))
        self.web_port = int(os.environ.get("WEB_PORT", "8889"))
        # Preview rendering tunables. Downscaling the MJPEG preview and
        # capping its rate keeps the seg/depth overlay loop off the CPU —
        # a full-frame addWeighted + JPEG encode every frame otherwise pegs
        # a core and starves preview FPS.
        self.preview_width = int(os.environ.get("PREVIEW_WIDTH", "640"))
        self.preview_height = int(os.environ.get("PREVIEW_HEIGHT", "360"))
        self.preview_fps = int(os.environ.get("PREVIEW_FPS", "15"))
        # Inference RPC timeout. Normal Hailo infer is ~10-50 ms; a stall
        # past this is treated as NPU contention. Kept well below the old
        # 5 s default so a stalled infer releases its model ref_count (and
        # the frontend unfreezes) promptly instead of hanging ~5 s.
        self.infer_timeout_ms = int(os.environ.get("INFER_TIMEOUT_MS", "1500"))
        # Cold-start grace window. A freshly-loaded HEF's first few inferences
        # hit Hailo context init (1-3 s, vs ~6 ms steady-state), which the
        # tight stall-detection timeout above cuts off — logging a burst of
        # DEADLINE_EXCEEDED right after every model switch. Give the first N
        # inferences a generous timeout so they complete, then revert to the
        # tight timeout for fast stall detection.
        self._infer_warmup_timeout_ms = int(os.environ.get("INFER_WARMUP_TIMEOUT_MS", "6000"))
        self._infer_warmup_count = int(os.environ.get("INFER_WARMUP_COUNT", "3"))

        self.media_preview: Optional[FdMediaClient] = None
        self.media_infer: Optional[FdMediaClient] = None

        # Video source state
        self._video_source: Optional[VideoFrameSource] = None
        self._video_lock = threading.Lock()
        self._video_active = False
        self._video_info: Dict[str, Any] = {}

        default_labels = os.environ.get(
            "DEFAULT_CLIP_LABELS", "a person,a car,a dog,a cat,empty scene",
        )
        self._clip_labels: List[str] = [
            l.strip() for l in default_labels.split(",") if l.strip()
        ]

        gallery_dir = os.environ.get("GALLERY_DIR", "/tmp/neoruntime-gallery")
        self.gallery = GalleryManager(
            data_dir=gallery_dir,
            max_images=int(os.environ.get("GALLERY_MAX_IMAGES", "5000")),
            capture_interval=float(os.environ.get("GALLERY_CAPTURE_INTERVAL", "5.0")),
        )

        # RTSP output mode
        self._output_mode: str = os.environ.get("OUTPUT_MODE", "mjpeg")  # mjpeg | rtsp | both
        self._rtsp_url: str = os.environ.get("RTSP_URL", "rtsp://127.0.0.1:8555/showcase")
        self._rtsp_enabled: bool = self._output_mode in ("rtsp", "both")
        self._rtsp_process: Optional[subprocess.Popen] = None
        self._rtsp_frame_size: Tuple[int, int] = (0, 0)
        self._mediamtx_process: Optional[subprocess.Popen] = None

        # Event Bus client — publishes inference results to camera-daemon's
        # AiOverlaySubscriber so detection boxes are drawn directly on the
        # hardware-encoded RTSP/H264 stream (zero Python CPU cost for drawing).
        # Uses lazy connect: the first _publish_to_event_bus() call connects.
        # Failures are silently swallowed so Event Bus being unavailable never
        # breaks inference.
        self._event_client: Optional[EventClient] = None
        self._event_bus_enabled: bool = os.environ.get("EVENT_BUS_ENABLED", "1") != "0"
        self._event_bus_endpoint: str = os.environ.get(
            "EVENT_BUS_ENDPOINT", "unix:///run/aipc/event-bus.sock"
        )

    def connect(self) -> bool:
        try:
            self.media_preview = FdMediaClient()
            logger.info("Preview media client ready (stream=%s)", self.stream_id)
            self.media_infer = FdMediaClient()
            logger.info("Infer media client ready (stream=%s)", self.infer_stream_id)
            self.infer_client = InferenceClient()
            self.infer_client.connect()
            logger.info("Connected to Inference service")
            self._discover_models()
            self._verify_current_model()
            if AVAILABLE_MODELS:
                self._ensure_model_registered(AVAILABLE_MODELS[0])
            return True
        except Exception as e:
            logger.error("Failed to connect: %s", e)
            return False

    def _discover_models(self) -> None:
        global AVAILABLE_MODELS

        AVAILABLE_MODELS = []
        for m in MODEL_CATALOG:
            if not _entry_available(m):
                continue
            entry = {k: v for k, v in m.items() if k not in ("postprocess_json",)}
            if entry.get("type") == "genai" and not os.path.isfile(entry.get("path", "")):
                # Early signal on fresh devices: the chat entry stays listed,
                # but inference would fail until the weight file is provisioned.
                entry["warning"] = "model file missing on this device"
            AVAILABLE_MODELS.append(entry)

        self._populate_input_shapes()

        ids = {m["id"] for m in AVAILABLE_MODELS}
        if self.current_model not in ids and AVAILABLE_MODELS:
            self.current_model = AVAILABLE_MODELS[0]["id"]
            self.current_model_type = AVAILABLE_MODELS[0]["type"]
            self._current_model_info = AVAILABLE_MODELS[0]
        logger.info("Available models: %s", [m["id"] for m in AVAILABLE_MODELS])

    def refresh_model_paths(self) -> List[str]:
        """Re-resolve catalog paths against the host store, then re-discover.

        POST /api/models/refresh calls this after the operator drops a file
        into /data/aipc/models: catalog paths are evaluated at import time,
        so a freshly provisioned host file is invisible until this runs (or
        the app restarts). Returns the ids that became available.

        Pipeline stages are always image-bundled, so only single-file and
        genai entries need re-resolution.
        """
        before = {m["id"] for m in AVAILABLE_MODELS}
        for m in MODEL_CATALOG:
            if "stages" in m or "path" not in m or not m.get("category"):
                continue
            host = os.path.join(_MODEL_ROOT, m["category"], os.path.basename(m["path"]))
            if os.path.isfile(host):
                m["path"] = host
        self._discover_models()
        return sorted({m["id"] for m in AVAILABLE_MODELS} - before)

        # Seed the reaper pin set with the startup model so the background
        # reaper never evicts the initially-loaded model before the first
        # explicit switch.
        if self._current_model_info is not None:
            self._pinned_models = self._compute_pinned_models(self._current_model_info)

    def _populate_input_shapes(self) -> None:
        try:
            model_infos = self.infer_client.list_models()
            model_map = {m["id"]: m for m in AVAILABLE_MODELS}
            for mi in model_infos:
                entry = model_map.get(mi.model_id)
                if not entry or not mi.inputs:
                    continue
                shape = mi.inputs[0].get("shape", [])
                ih, iw = self._parse_input_shape(shape, entry.get("input_format", "nv12"))
                if ih > 0 and iw > 0:
                    entry["input_height"] = ih
                    entry["input_width"] = iw
                    logger.info("Model %s input: %dx%d (%s)", mi.model_id, iw, ih,
                                entry.get("input_format", "nv12"))
        except Exception as e:
            logger.warning("Failed to query input shapes: %s", e)

    @staticmethod
    def _parse_input_shape(shape: list, input_format: str) -> tuple:
        if not shape or len(shape) < 2:
            return 0, 0
        if len(shape) == 4:
            if shape[3] == 1 and input_format in ("nv12", "nv21"):
                return shape[1] * 2 // 3, shape[2]
            return shape[1], shape[2]
        if len(shape) == 3:
            return shape[0], shape[1]
        if len(shape) == 2:
            return shape[0], shape[1]
        return 0, 0

    @staticmethod
    def _choose_infer_stream(model_info: Dict[str, Any]) -> str:
        """Pick the best inference stream based on model input resolution.

        third (640x384): sufficient for models with input ≤ 640×384
        sub  (1280x720): needed for models with larger input
        """
        dims = []
        if "stages" in model_info:
            for stage in model_info["stages"]:
                dims.append((stage.get("input_width", 0), stage.get("input_height", 0)))
        else:
            dims.append((model_info.get("input_width", 0), model_info.get("input_height", 0)))

        for w, h in dims:
            if w > 640 or h > 384:
                return "sub"
        return "third"

    def _verify_current_model(self) -> None:
        """Register and verify only the first (current) model.

        Other models stay in AVAILABLE_MODELS and are loaded on-demand
        when the user switches to them.  The .hef files were already
        confirmed to exist by _discover_models(), so skipping NPU
        verification here is safe.
        """
        global AVAILABLE_MODELS
        if not AVAILABLE_MODELS:
            return

        model = AVAILABLE_MODELS[0]
        mid = model["id"]
        w = model.get("input_width", 0)
        h = model.get("input_height", 0)

        # genai models don't need NPU verification
        if model.get("type") == "genai":
            logger.info("Model %s (genai) skipped verification", mid)
            return

        if w <= 0 or h <= 0:
            logger.warning("Current model %s has no input dimensions, skipping verification", mid)
            return

        try:
            self._ensure_model_registered(model)
            fmt = model.get("input_format", "nv12")
            dummy_size = w * h * 3 if fmt == "rgb" else w * h * 3 // 2
            dummy = np.zeros(dummy_size, dtype=np.uint8)
            result = self.infer_client.infer(dummy, model_id=mid, timeout_ms=10000)
            if result.infer_time_us > 0:
                model["npu_benchmark_fps"] = round(
                    1_000_000.0 / result.infer_time_us, 1,
                )
                logger.info(
                    "Model %s verified OK — %.1f fps (%d µs)",
                    mid, model["npu_benchmark_fps"], result.infer_time_us,
                )
            else:
                logger.info("Model %s verified OK (no latency data)", mid)
        except Exception as e:
            logger.warning("Model %s verify failed: %s — removed", mid, e)
            AVAILABLE_MODELS = AVAILABLE_MODELS[1:]
            if AVAILABLE_MODELS:
                self.current_model = AVAILABLE_MODELS[0]["id"]
                self.current_model_type = AVAILABLE_MODELS[0]["type"]
                self._current_model_info = AVAILABLE_MODELS[0]

    def _verify_models(self) -> None:
        global AVAILABLE_MODELS
        verified = []
        for model in AVAILABLE_MODELS:
            mid = model["id"]
            if model.get("type") == "genai":
                logger.info("Model %s (genai) skipped verification", mid)
                verified.append(model)
                continue
            w = model.get("input_width", 0)
            h = model.get("input_height", 0)
            fmt = model.get("input_format", "nv12")
            if w <= 0 or h <= 0:
                logger.warning("Skipping %s: no input dimensions", mid)
                continue
            try:
                dummy_size = w * h * 3 if fmt == "rgb" else w * h * 3 // 2
                dummy = np.zeros(dummy_size, dtype=np.uint8)
                result = self.infer_client.infer(dummy, model_id=mid, timeout_ms=10000)
                if result.infer_time_us > 0:
                    model["npu_benchmark_fps"] = round(
                        1_000_000.0 / result.infer_time_us, 1,
                    )
                    logger.info(
                        "Model %s verified OK — %.1f fps (%d µs)",
                        mid, model["npu_benchmark_fps"], result.infer_time_us,
                    )
                else:
                    logger.info("Model %s verified OK (no latency data)", mid)
                verified.append(model)
            except Exception as e:
                logger.warning("Model %s verify failed: %s", mid, e)
        if len(verified) < len(AVAILABLE_MODELS):
            removed = len(AVAILABLE_MODELS) - len(verified)
            logger.warning(
                "%d/%d models verified, %d removed",
                len(verified), len(AVAILABLE_MODELS), removed,
            )
        AVAILABLE_MODELS = verified

    def _start_npu_stats_thread(self) -> None:
        self._npu_stats_thread = threading.Thread(target=self._npu_stats_loop, daemon=True)
        self._npu_stats_thread.start()

    def _npu_stats_loop(self) -> None:
        while self.running:
            try:
                stats = self.infer_client.get_stats()
                with self._npu_stats_lock:
                    self._npu_stats = stats
            except Exception as e:
                logger.debug("NPU stats fetch failed: %s", e)
            time.sleep(5)

    def switch_model(self, model_id: str) -> bool:
        model_info = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
        if not model_info:
            return False
        with self._model_lock:
            old_model = self.current_model
            self.current_model = model_id
            self.current_model_type = model_info["type"]
            self._current_model_info = model_info
            # Update the reaper pin set BEFORE releasing the lock / registering
            # the new model, so the background reaper can never evict the new
            # model or its co-resident dependencies during the switch.
            self._pinned_models = self._compute_pinned_models(model_info)
        if old_model == model_id:
            return True
        logger.info("Switching model: %s -> %s", old_model, model_id)

        # Dynamically adjust inference rate based on model benchmark.
        # Cap at the user-configured INFER_FPS ceiling, but lower it for
        # fast models where Python overhead would saturate the GIL.
        bench_fps = model_info.get("npu_benchmark_fps", 0)
        effective_fps = min(self.infer_fps_target, max(bench_fps * 0.6, 5))
        # Segmentation/depth postprocess (colormap blend / depth decode) runs on
        # the CPU per inference.  Running it faster than the preview can show
        # results wastes a whole core and starves the preview loop.  Cap it so
        # postprocess stays well under one core while the preview still gets a
        # fresh overlay each frame.
        if model_info["type"] in ("segmentation", "depth"):
            seg_depth_cap = int(os.environ.get("SEG_DEPTH_INFER_FPS", "8"))
            effective_fps = min(effective_fps, seg_depth_cap)
        self._infer_interval = 1.0 / effective_fps
        logger.info("Dynamic infer rate: bench=%.1f fps, target=%d, effective=%.1f fps",
                     bench_fps, self.infer_fps_target, effective_fps)

        self._stop_infer_thread()
        self._clear_event_bus_overlay()  # Remove stale boxes from camera-daemon RTSP overlay
        time.sleep(self._model_switch_cooldown)

        # GenAI models don't use the infer loop — they're interactive
        if model_info["type"] == "genai":
            self._pipeline_stage1_us = 0
            self._pipeline_stage2_us = 0
            self._pipeline_count = 0
            self._prev_landmarks = None
            self._ocr_accumulator = _OcrAccumulator(ttl_sec=3.0)
            return True

        if "stages" in model_info:
            self._ensure_pipeline_registered(model_info)
        else:
            self._ensure_model_registered(model_info)
        # face_landmarks needs yolov8n as face detector — keep both loaded
        if model_id == "face_landmarks":
            self._ensure_detector_for_landmarks()
        if model_info["type"] == "clip":
            self._push_clip_labels()
        self._pipeline_stage1_us = 0
        self._pipeline_stage2_us = 0
        self._pipeline_count = 0
        self._prev_landmarks = None
        self._start_infer_thread()
        return True

    def _ensure_model_registered(self, model_info: Dict[str, Any], force: bool = False) -> None:
        mid = model_info["id"]
        try:
            registered_list = self.infer_client.list_models()
            registered = {m.model_id for m in registered_list}
        except Exception:
            registered = set()
            registered_list = []

        if mid in registered and not force:
            logger.info("Model %s already registered", mid)
            return

        catalog = next((m for m in MODEL_CATALOG if m["id"] == mid), None)
        if not catalog:
            logger.error("Model %s not found in catalog", mid)
            return

        # Do NOT synchronously evict other resident models here. ai-runtime's
        # UnregisterModel fails while ref_count > 0 (an infer session holds
        # the model for a few seconds after its loop stops), so a synchronous
        # attempt almost always fails during a switch and would only log a
        # misleading ERROR. The background reaper (_reaper_loop, started in
        # run_frame_loop) is the reliable cleanup path: it evicts every model
        # not in the current pin set within ~2-4 s once the ref clears (verified
        # on-device). Keeping this eviction-free also means a registration
        # failure leaves the previously-resident model in place (graceful
        # fallback), so the restore block below is a no-op safety net.
        evicted: List[str] = []

        try:
            reg_type = catalog.get("register_type", catalog["type"])
            variant = catalog.get("variant")
            self.infer_client.register_model(
                model_path=catalog["path"],
                model_id=catalog["id"],
                owner_id=Config.get_app_id(),
                model_type=reg_type,
                model_variant=variant,
            )
            logger.info("Registered model: %s (type=%s variant=%s)", mid, reg_type or "raw", variant or "")
        except Exception as e:
            import traceback
            logger.error("Failed to register %s: %s\n%s", mid, e, traceback.format_exc())
            # Registration failed — restore evicted models so the NPU
            # pipeline isn't left in a broken state.
            for ev_id in evicted:
                evict_catalog = next((m for m in MODEL_CATALOG if m["id"] == ev_id), None)
                if not evict_catalog:
                    continue
                try:
                    et = evict_catalog.get("register_type", evict_catalog["type"])
                    ev = evict_catalog.get("variant")
                    self.infer_client.register_model(
                        model_path=evict_catalog["path"],
                        model_id=evict_catalog["id"],
                        owner_id=Config.get_app_id(),
                        model_type=et,
                        model_variant=ev,
                    )
                    logger.info("Restored evicted model: %s", ev_id)
                except Exception as re:
                    logger.error("Failed to restore evicted model %s: %s", ev_id, re)
            return

        pp_json = catalog.get("postprocess_json")
        if pp_json:
            try:
                self.infer_client.update_postprocess_config(mid, pp_json)
                logger.info("Pushed postprocess config for %s", mid)
            except Exception as e:
                logger.warning("Failed to push config for %s: %s", mid, e)

    def _push_clip_labels(self) -> None:
        if not self.infer_client or not self._clip_labels:
            return
        try:
            config = json.dumps({"prompts": self._clip_labels})
            self.infer_client.update_postprocess_config(self.current_model, config)
            logger.info("CLIP labels pushed: %s", self._clip_labels)
        except Exception as e:
            logger.warning("Failed to push CLIP labels: %s", e)

    def update_clip_labels(self, labels: List[str]) -> bool:
        self._clip_labels = list(labels)
        if self.current_model_type == "clip":
            self._push_clip_labels()
        return True

    def _ensure_detector_for_landmarks(self) -> None:
        """Ensure yolov8n is registered alongside face_landmarks for face detection."""
        det_id = "yolov8n_detection"
        try:
            registered = {m.model_id for m in self.infer_client.list_models()}
        except Exception:
            return
        if det_id in registered:
            return
        det_info = next((m for m in MODEL_CATALOG if m["id"] == det_id), None)
        if not det_info:
            return
        # Evict something that isn't face_landmarks or yolov8n
        registered_list = self.infer_client.list_models()
        for m in registered_list:
            if m.model_id not in ("face_landmarks", det_id):
                try:
                    self.infer_client.unregister_model(m.model_id)
                    logger.info("Evicted %s to make room for %s", m.model_id, det_id)
                    break
                except Exception:
                    pass
        reg_type = det_info.get("register_type", det_info["type"])
        try:
            self.infer_client.register_model(
                model_path=det_info["path"],
                model_id=det_info["id"],
                owner_id=Config.get_app_id(),
                model_type=reg_type,
            )
            logger.info("Registered %s for face detection", det_id)
        except Exception as e:
            logger.warning("Failed to register %s: %s", det_id, e)

    # ------------------------------------------------------------------
    # Model reaper — self-heals leaked network groups (Bug 1 root cause)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pinned_models(model_info: Dict[str, Any]) -> set:
        """Set of model_ids that must stay resident alongside the current model.

        The reaper evicts everything NOT in this set. We always pin the current
        model itself, plus its co-resident dependencies:
          - face_landmarks needs yolov8n_detection for face detection
          - pipelines keep every stage model co-resident
        """
        pinned = {model_info["id"]}
        if model_info["id"] == "face_landmarks":
            pinned.add("yolov8n_detection")
        for stage in model_info.get("stages", []):
            pinned.add(stage["id"])
        return pinned

    def _reaper_loop(self) -> None:
        """Background self-heal for leaked model network groups.

        When a client infer() times out (DEADLINE_EXCEEDED) the server-side
        session can hold the model's ref_count > 0 for a few seconds after
        the app thread has already moved on. That makes unregister_model fail
        transiently (returns -1), so the network group is never released and
        NPU contexts accumulate across model switches until ai-runtime is
        restarted — which is what starves the ROUND_ROBIN scheduler and
        produces the intermittent DEADLINE_EXCEEDED freezes on detection.

        This loop re-tries unregistering every model that is NOT in the
        current pin set, every ~2s. A model briefly pinned by a timed-out
        session gets cleaned up as soon as the server releases its ref_count.
        Failures are expected (ref_count still > 0) and silently retried.
        """
        logger.info("Model reaper thread started")
        while self.running:
            try:
                time.sleep(2.0)
                if not self.running:
                    break
                pinned = set(self._pinned_models or set())
                if not pinned:
                    continue
                registered = self.infer_client.list_models()
                for m in registered:
                    if m.model_id in pinned:
                        continue
                    try:
                        self.infer_client.unregister_model(m.model_id)
                        logger.info("Reaper evicted leaked model: %s", m.model_id)
                    except Exception:
                        # ref_count still > 0 (timed-out infer still holds it)
                        # — will retry on the next tick once released.
                        pass
            except Exception as e:
                logger.debug("Reaper tick skipped: %s", e)
        logger.info("Model reaper thread stopped")

    def _run_face_landmarks(self, frame: Frame) -> Any:
        """Two-stage face landmarks: yolov8n detects persons → crop face → landmarks."""
        bgr = self.frame_to_bgr(frame)
        return self._run_face_landmarks_impl(bgr)

    def _run_face_landmarks_impl(self, bgr: np.ndarray) -> Any:
        fh, fw = bgr.shape[:2]

        # Stage 1: detect persons with yolov8n
        det_info = next((m for m in MODEL_CATALOG if m["id"] == "yolov8n_detection"), None)
        if not det_info:
            return _PipelineResult()
        det_input = self._prepare_stage_input(bgr, det_info)
        t0 = time.monotonic()
        det_result = self.infer_client.infer(
            det_input, model_id="yolov8n_detection", timeout_ms=3000,
        )
        t1 = time.monotonic()
        self._pipeline_stage1_us = int((t1 - t0) * 1_000_000)

        # Find face or person detection — prefer "face" for precision
        face_det = None
        person_det = None
        face_area = 0
        person_area = 0
        if det_result.objects:
            for obj in det_result.objects:
                area = float(obj.bbox.width) * float(obj.bbox.height)
                label = getattr(obj, "label", "") or ""
                if label == "face" and area > face_area:
                    face_det = obj
                    face_area = area
                elif area > person_area:
                    person_det = obj
                    person_area = area

        if face_det is None and person_det is None:
            return _PipelineResult()

        if face_det:
            # Use yolov8n's face bbox directly — already tight around face
            bx = float(face_det.bbox.x)
            by = float(face_det.bbox.y)
            bw = float(face_det.bbox.width)
            bh = float(face_det.bbox.height)
            margin = 0.15
            face_x1 = max(0.0, bx - bw * margin)
            face_x2 = min(1.0, bx + bw * (1 + margin))
            face_y1 = max(0.0, by - bh * margin)
            face_y2 = min(1.0, by + bh * (1 + margin))
        else:
            # Fallback: estimate face from person bbox
            bx = float(person_det.bbox.x)
            by = float(person_det.bbox.y)
            bw = float(person_det.bbox.width)
            bh = float(person_det.bbox.height)
            margin_x = bw * 0.08
            margin_y = bh * 0.05
            face_x1 = max(0.0, bx - margin_x)
            face_x2 = min(1.0, bx + bw + margin_x)
            face_y1 = max(0.0, by - margin_y)
        face_y2 = min(1.0, by + bh * 0.55)

        px1 = int(face_x1 * fw)
        py1 = int(face_y1 * fh)
        px2 = int(face_x2 * fw)
        py2 = int(face_y2 * fh)
        if px2 <= px1 or py2 <= py1:
            return _PipelineResult()

        crop = bgr[py1:py2, px1:px2]

        # Stage 2: run face_landmarks on the crop
        lm_info = self._current_model_info
        tw, th = lm_info["input_width"], lm_info["input_height"]
        resized = cv2.resize(crop, (tw, th))

        lm_input = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).flatten()
        t2 = time.monotonic()
        lm_result = self.infer_client.infer(
            lm_input, model_id="face_landmarks", timeout_ms=3000,
        )
        t3 = time.monotonic()
        self._pipeline_stage2_us = int((t3 - t2) * 1_000_000)

        # Parse raw output
        raw = getattr(lm_result, "raw_outputs", None)
        if not raw or len(raw) < 1:
            return _PipelineResult()

        landmarks_tensor = np.asarray(raw[0], dtype=np.float32).flatten()
        if landmarks_tensor.max() > 2.0:
            landmarks_tensor = landmarks_tensor / 255.0

        n_pts = landmarks_tensor.size // 3
        if n_pts < 3:
            return _PipelineResult()
        points = landmarks_tensor[:n_pts * 3].reshape(n_pts, 3)

        # Map coordinates: model [0,1] → crop → frame (stretch resize is expected by model)
        _LP = type("_LP", (), {})
        raw_coords = []
        for i in range(n_pts):
            lx = float(points[i, 0])
            ly = float(points[i, 1])
            lx = max(0.0, min(1.0, lx))
            ly = max(0.0, min(1.0, ly))
            fx = face_x1 + lx * (face_x2 - face_x1)
            fy = face_y1 + ly * (face_y2 - face_y1)
            raw_coords.append((fx, fy))

        # Temporal smoothing via exponential moving average
        if self._prev_landmarks is not None and len(self._prev_landmarks) == len(raw_coords):
            alpha = self._landmark_smooth_alpha
            smoothed = [
                (alpha * r[0] + (1 - alpha) * p[0], alpha * r[1] + (1 - alpha) * p[1])
                for r, p in zip(raw_coords, self._prev_landmarks)
            ]
        else:
            smoothed = raw_coords
        self._prev_landmarks = smoothed

        lp_list = []
        for i, (fx, fy) in enumerate(smoothed):
            p = _LP()
            p.x = fx
            p.y = fy
            p.confidence = float(points[i, 2]) if points.shape[1] > 2 else 1.0
            lp_list.append(p)
        # Filter: skip if no face was detected by yolov8n (confidence-based)
        if lp_list and face_det is None:
            avg_conf = sum(p.confidence for p in lp_list) / len(lp_list)
            if avg_conf < 0.30:
                return _PipelineResult()

        _LS = type("_LS", (), {})
        ls = _LS()
        ls.type = "face"
        ls.points = lp_list
        self._pipeline_count = 1
        return type("_FR", (), {"landmarks": [ls],
                                 "objects": [], "classifications": [],
                                 "ocr_lines": [], "masks": [],
                                 "embeddings": []})()

    # ------------------------------------------------------------------
    # Pipeline registration
    # ------------------------------------------------------------------

    def _ensure_pipeline_registered(self, pipeline_info: Dict[str, Any]) -> None:
        stages = pipeline_info.get("stages", [])
        if not stages:
            return
        max_cached = 3
        try:
            registered_list = self.infer_client.list_models()
            registered = {m.model_id for m in registered_list}
        except Exception:
            registered = set()
            registered_list = []
        needed_ids = {s["id"] for s in stages}
        while len([m for m in registered_list if m.model_id not in needed_ids]) > 0 and \
                len(registered_list) >= max_cached:
            for m in registered_list:
                if m.model_id not in needed_ids:
                    try:
                        self.infer_client.unregister_model(m.model_id)
                        logger.info("Evicted model: %s", m.model_id)
                        registered.discard(m.model_id)
                        registered_list = [rm for rm in registered_list
                                           if rm.model_id != m.model_id]
                        break
                    except Exception as e:
                        logger.warning("Failed to evict %s: %s", m.model_id, e)
                        break
        for stage in stages:
            sid = stage["id"]
            if sid in registered:
                logger.info("Pipeline stage %s already registered", sid)
                continue
            try:
                reg_type = stage.get("register_type", stage["type"])
                variant = stage.get("variant")
                self.infer_client.register_model(
                    model_path=stage["path"],
                    model_id=stage["id"],
                    owner_id=Config.get_app_id(),
                    model_type=reg_type,
                    model_variant=variant,
                )
                logger.info("Registered pipeline stage: %s (type=%s variant=%s)", sid, reg_type or "raw", variant or "")
            except Exception as e:
                logger.error("Failed to register pipeline stage %s: %s", sid, e)
                continue
            pp_json = stage.get("postprocess_json")
            if pp_json:
                try:
                    self.infer_client.update_postprocess_config(sid, pp_json)
                    logger.info("Pushed postprocess config for stage %s", sid)
                except Exception as e:
                    logger.warning("Failed to push postprocess for stage %s: %s", sid, e)

    # ------------------------------------------------------------------
    # Letterbox crop for pipeline stage 2
    # ------------------------------------------------------------------

    @staticmethod
    def _letterbox_crop(bgr: np.ndarray, x_norm: float, y_norm: float,
                        w_norm: float, h_norm: float,
                        target_w: int, target_h: int,
                        pad_color: tuple,
                        rotate_vertical: bool = False) -> np.ndarray:
        x_norm, y_norm, w_norm, h_norm = float(x_norm), float(y_norm), float(w_norm), float(h_norm)
        target_w, target_h = int(target_w), int(target_h)
        fh, fw = bgr.shape[:2]
        # Vertical padding is larger than horizontal to preserve descenders
        # and bottom strokes (e.g. "E" bottom bar vs "F").
        mx, my = 0.10, 0.35
        x1 = max(0, int((x_norm - w_norm * mx) * fw))
        y1 = max(0, int((y_norm - h_norm * my) * fh))
        x2 = min(fw, int((x_norm + w_norm + w_norm * mx) * fw))
        y2 = min(fh, int((y_norm + h_norm + h_norm * my) * fh))
        if x2 <= x1 or y2 <= y1:
            return np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
        crop = bgr[y1:y2, x1:x2]
        if rotate_vertical and crop.shape[0] > crop.shape[1]:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ch, cw = crop.shape[:2]
        scale = min(target_w / cw, target_h / ch)
        new_w = int(cw * scale) & ~1
        new_h = int(ch * scale) & ~1
        if new_w < 2 or new_h < 2:
            return np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
        resized = cv2.resize(crop, (new_w, new_h))
        canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def _start_infer_thread(self) -> None:
        self._infer_running = True
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()

    def _stop_infer_thread(self) -> None:
        self._infer_running = False
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=3.0)
        self._infer_thread = None

    def _reconnect_infer_media(self) -> None:
        for attempt in range(3):
            try:
                if self.media_infer:
                    self.media_infer.close()
            except Exception:
                pass
            time.sleep(1.0 * (attempt + 1))
            try:
                self.media_infer = FdMediaClient()
                logger.info("Infer media reconnected (attempt %d)", attempt + 1)
                return
            except Exception as e:
                logger.warning("Infer media reconnect attempt %d failed: %s", attempt + 1, e)
        logger.error("Infer media reconnect failed after 3 attempts")

    def _reconnect_preview_media(self) -> None:
        for attempt in range(3):
            try:
                if self.media_preview:
                    self.media_preview.close()
            except Exception:
                pass
            time.sleep(1.0 * (attempt + 1))
            try:
                self.media_preview = FdMediaClient()
                logger.info("Preview media reconnected (attempt %d)", attempt + 1)
                return
            except Exception as e:
                logger.warning("Preview media reconnect attempt %d failed: %s", attempt + 1, e)
        logger.error("Preview media reconnect failed after 3 attempts")

    @staticmethod
    def _bgr_to_nv12(bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        y_plane = i420[:h, :]
        u_plane = i420[h:h + h // 4, :].reshape(h // 2, w // 2)
        v_plane = i420[h + h // 4:, :].reshape(h // 2, w // 2)
        uv_plane = np.empty((h // 2, w), dtype=np.uint8)
        uv_plane[:, 0::2] = u_plane
        uv_plane[:, 1::2] = v_plane
        return np.vstack([y_plane, uv_plane])

    @staticmethod
    def _nv12_resize(nv12: np.ndarray, src_w: int, src_h: int,
                     dst_w: int, dst_h: int) -> np.ndarray:
        """Resize NV12 frame by resizing Y and UV planes separately.

        Avoids NV12->BGR->resize->BGR->NV12 round-trip.  Operates directly
        on the luma and chroma planes with cv2.resize (INTER_LINEAR).
        """
        y_plane = nv12[:src_h, :]                          # (src_h, src_w)
        uv_plane = nv12[src_h:src_h + src_h // 2, :]      # (src_h/2, src_w)

        y_out = cv2.resize(y_plane, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        # UV plane must stay half the height of Y in NV12 format
        uv_out = cv2.resize(uv_plane, (dst_w, dst_h // 2), interpolation=cv2.INTER_LINEAR)

        return np.vstack([y_out, uv_out])

    def _prepare_nv12_input(self, nv12: np.ndarray, src_w: int, src_h: int,
                             target_w: int, target_h: int,
                             input_fmt: str) -> np.ndarray:
        """Prepare model input directly from NV12 -- avoids BGR round-trip.

        - NV12 model + matching dims  -> just flatten (zero conversion)
        - NV12 model + resize needed  -> resize Y/UV separately, then flatten
        - RGB model                   -> NV12->RGB directly (one conversion)
        """
        if input_fmt == "nv12":
            if src_w == target_w and src_h == target_h:
                return nv12.flatten()
            resized = self._nv12_resize(nv12, src_w, src_h, target_w, target_h)
            return resized.flatten()
        elif input_fmt == "rgb":
            rgb = cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)
            if src_w != target_w or src_h != target_h:
                rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return rgb.flatten()
        else:
            # Fallback: convert via BGR for unusual formats
            logger.warning(
                "Unexpected input format '%s' for NV12 source, using BGR fallback",
                input_fmt,
            )
            bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
            if src_w != target_w or src_h != target_h:
                bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return self._bgr_to_nv12(bgr).flatten()

    def _get_infer_raw(self) -> Optional[Tuple[np.ndarray, int, int, str]]:
        """Get raw frame for inference without forcing BGR conversion.

        Returns (image_data, width, height, format) or None.
        Camera mode returns NV12; video mode returns BGR.
        """
        if self._video_active:
            with self._video_lock:
                if self._video_source:
                    bgr = self._video_source.latest_frame()
                    if bgr is not None:
                        h, w = bgr.shape[:2]
                        return bgr, w, h, "BGR"
            return None
        try:
            infer_stream = self._active_infer_stream
            frame = self.media_infer.get_frame(infer_stream, timeout_ms=3000)
        except Exception as e:
            logger.warning("Infer get_frame error: %s", e)
            self._reconnect_infer_media()
            return None
        if frame is None:
            return None
        return frame.image, frame.width, frame.height, frame.format

    @staticmethod
    def _to_bgr(raw_data: np.ndarray, src_fmt: str) -> np.ndarray:
        """Convert raw frame data to BGR for pipeline/crop operations."""
        if src_fmt == "BGR":
            return raw_data
        elif src_fmt == "NV12":
            return cv2.cvtColor(raw_data, cv2.COLOR_YUV2BGR_NV12)
        elif src_fmt == "RGB":
            return cv2.cvtColor(raw_data, cv2.COLOR_RGB2BGR)
        else:
            return cv2.cvtColor(raw_data, cv2.COLOR_YUV2BGR_NV12)

    def _prepare_input(self, frame: Frame, target_w: int, target_h: int,
                       input_fmt: str) -> np.ndarray:
        # NV12-direct: avoid BGR round-trip when frame is already NV12
        if frame.format == "NV12" and frame.width > 0 and frame.height > 0:
            return self._prepare_nv12_input(
                frame.image, frame.width, frame.height,
                target_w, target_h, input_fmt,
            )
        bgr = self.frame_to_bgr(frame)
        src_h, src_w = bgr.shape[:2]
        if src_w != target_w or src_h != target_h:
            bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if input_fmt == "rgb":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).flatten()
        return self._bgr_to_nv12(bgr).flatten()

    def _run_face_landmarks_bgr(self, bgr: np.ndarray) -> Any:
        return self._run_face_landmarks_impl(bgr)

    def _run_pipeline_bgr(self, bgr: np.ndarray, pipeline_info: Dict[str, Any]) -> Any:
        return self._run_pipeline_impl(bgr, pipeline_info)

    # ------------------------------------------------------------------
    # Two-stage pipeline inference
    # ------------------------------------------------------------------

    def _run_pipeline(self, frame: Frame, pipeline_info: Dict[str, Any]) -> Any:
        bgr = self.frame_to_bgr(frame)
        return self._run_pipeline_impl(bgr, pipeline_info)

    def _run_pipeline_impl(self, bgr: np.ndarray, pipeline_info: Dict[str, Any]) -> Any:
        stages = pipeline_info["stages"]
        det_stage = next(s for s in stages if s["role"] == "detector")
        rec_stage = next(s for s in stages if s["role"] == "recognizer")
        pipeline_type = pipeline_info["type"]
        is_lpr = pipeline_type == "pipeline_lpr"

        # Stage 1: detection
        det_input = self._prepare_stage_input(bgr, det_stage)
        t0 = time.monotonic()
        det_result = self.infer_client.infer(
            det_input, model_id=det_stage["id"], timeout_ms=5000,
        )
        t1 = time.monotonic()
        self._pipeline_stage1_us = int((t1 - t0) * 1_000_000)

        det_obj_count = len(det_result.objects) if det_result.objects else 0
        det_ocr_count = len(det_result.ocr_lines) if getattr(det_result, "ocr_lines", None) else 0
        logger.debug(
            "Pipeline [%s] stage1: %d det, %d ocr, %.1fms",
            pipeline_info["id"], det_obj_count, det_ocr_count, (t1 - t0) * 1000,
        )

        # Handle raw output for detection stages with register_type=""
        if not det_result.objects and not det_ocr_count and getattr(det_result, "raw_outputs", None):
            if det_stage.get("register_type", "") == "":
                if det_stage["id"] in self._YOLOV4_ANCHORS:
                    det_result = self._parse_yolo_grid(det_result, det_stage["id"])
                else:
                    det_result = self._parse_nms_raw_result(det_result, det_stage["id"])

        # Build unified crop list from detection results
        # OCR detection returns ocr_lines, LPR detection returns objects
        crop_list = []
        if det_result.objects:
            for obj in det_result.objects:
                crop_list.append((float(obj.bbox.x), float(obj.bbox.y),
                                  float(obj.bbox.width), float(obj.bbox.height)))
        elif det_ocr_count:
            for line in det_result.ocr_lines:
                crop_list.append((float(line.bbox.x), float(line.bbox.y),
                                  float(line.bbox.width), float(line.bbox.height)))

        if not crop_list:
            self._pipeline_stage2_us = 0
            return _PipelineResult(ocr_lines=[], objects=[])

        # Stage 2: recognition for each detection
        charset = LPR_CHARSET if is_lpr else OCR_CHARSET
        pad_color = (0, 0, 0) if is_lpr else (255, 255, 255)
        rotate_vertical = not is_lpr  # OCR text may be vertical on product labels

        rec_input_w = rec_stage["input_width"]
        rec_input_h = rec_stage["input_height"]
        rec_model_id = rec_stage["id"]
        rec_is_rgb = rec_stage["input_format"] == "rgb"

        def _preprocess_rec(bx: float, by: float, bw: float, bh: float) -> np.ndarray:
            crop_bgr = self._letterbox_crop(
                bgr, bx, by, bw, bh, rec_input_w, rec_input_h,
                pad_color, rotate_vertical=rotate_vertical,
            )
            if rec_is_rgb:
                return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).flatten()
            return self._bgr_to_nv12(crop_bgr).flatten()

        def _recognize_one(bx: float, by: float, bw: float, bh: float) -> Optional[_OcrLineWrap]:
            """Sequential fallback: one infer() RPC per crop."""
            try:
                rec_result = self.infer_client.infer(
                    _preprocess_rec(bx, by, bw, bh),
                    model_id=rec_model_id, timeout_ms=3000,
                )
            except Exception as e:
                logger.debug("Recognition failed for crop: %s", e)
                return None
            text, conf = self._decode_recognition(rec_result, charset)
            if not text:
                return None
            logger.debug(
                "Pipeline [%s] recognized: '%s' conf=%.2f",
                pipeline_info["id"], text, conf,
            )
            return _OcrLineWrap(
                text=text, confidence=conf,
                bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
            )

        all_lines: List[_OcrLineWrap] = []
        t2 = time.monotonic()

        # Prefer a single InferBatch RPC: every crop is independent, so the
        # NPU schedules them as interleaved run_async jobs over one round-robin
        # VDevice instead of N serial round-trips. Single-crop frames also go
        # through here (one-item batch ~= one infer() call).
        batch_ok = False
        if crop_list and self._batch_enabled:
            try:
                rec_inputs = [_preprocess_rec(*crop) for crop in crop_list]
                batch_results = self.infer_client.infer_batch(
                    [BatchInferItem(image=r, model_id=rec_model_id) for r in rec_inputs],
                    timeout_ms=5000,
                )
                for (bx, by, bw, bh), rec_result in zip(crop_list, batch_results):
                    text, conf = self._decode_recognition(rec_result, charset)
                    if text:
                        logger.debug(
                            "Pipeline [%s] recognized: '%s' conf=%.2f",
                            pipeline_info["id"], text, conf,
                        )
                        all_lines.append(_OcrLineWrap(
                            text=text, confidence=conf,
                            bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
                        ))
                batch_ok = True
            except Exception as e:
                if "UNIMPLEMENTED" in str(e):
                    self._batch_enabled = False
                    logger.info(
                        "infer_batch UNIMPLEMENTED on server; "
                        "falling back to sequential recognition"
                    )
                else:
                    logger.debug(
                        "infer_batch failed (%s); falling back to sequential", e
                    )

        # Fallback path (batch disabled, batch failed, or empty crop list):
        # sequential per-crop infer().
        if not batch_ok:
            for bx, by, bw, bh in crop_list:
                line = _recognize_one(bx, by, bw, bh)
                if line:
                    all_lines.append(line)

        t3 = time.monotonic()
        self._pipeline_stage2_us = int((t3 - t2) * 1_000_000)
        self._pipeline_count = len(all_lines)
        return _PipelineResult(ocr_lines=all_lines, objects=[])

    @staticmethod
    def _prepare_stage_input(bgr: np.ndarray, stage: Dict[str, Any]) -> np.ndarray:
        tw, th = stage["input_width"], stage["input_height"]
        if bgr.shape[1] != tw or bgr.shape[0] != th:
            bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
        if stage["input_format"] == "rgb":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).flatten()
        i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        h, w = bgr.shape[:2]
        y_plane = i420[:h, :]
        u_plane = i420[h:h + h // 4, :].reshape(h // 2, w // 2)
        v_plane = i420[h + h // 4:, :].reshape(h // 2, w // 2)
        uv_plane = np.empty((h // 2, w), dtype=np.uint8)
        uv_plane[:, 0::2] = u_plane
        uv_plane[:, 1::2] = v_plane
        return np.vstack([y_plane, uv_plane]).flatten()

    @staticmethod
    def _decode_recognition(rec_result: Any, charset: List[str]) -> tuple:
        if getattr(rec_result, "ocr_lines", None) and rec_result.ocr_lines:
            line = rec_result.ocr_lines[0]
            # HAL softmax over 18385 classes produces near-zero confidence;
            # fall back to raw output CTC decode when confidence is too low.
            if line.text and not all(c == "?" for c in line.text) and line.confidence > 0.01:
                return line.text, line.confidence
        if getattr(rec_result, "raw_outputs", None) and rec_result.raw_outputs:
            raw = rec_result.raw_outputs
            logits = np.asarray(raw[-1], dtype=np.float32)
            # Raw output is flat from HailoRT — reshape to (N, T, C) or (N, C, T)
            if logits.ndim == 1:
                total = logits.size
                # PaddleOCR recognition: (1, 40, 18385) = 735400
                if total == 735400:
                    logits = logits.reshape(1, 40, 18385)
                # LPRNet: (1, 19, 11) = 209
                elif total == 209:
                    logits = logits.reshape(1, 19, 11)
                else:
                    # Try generic reshape from charset size
                    c_len = len(charset)
                    if total % c_len == 0:
                        logits = logits.reshape(1, total // c_len, c_len)
                    else:
                        return "", 0.0
            if logits.ndim == 3:
                logits = logits[0]  # squeeze batch dim → (T, C)
            if logits.ndim != 2:
                return "", 0.0
            # UINT8 → float probability [0,1]
            if logits.max() > 2.0:
                logits = logits / 255.0
            text, conf = _ctc_greedy_decode(logits, charset)
            return text, conf
        return "", 0.0

    # ------------------------------------------------------------------
    # NMS raw output parser for models with built-in NMS (e.g. yolov5m)
    # ------------------------------------------------------------------

    def _parse_nms_raw_result(self, result: Any, model_id: str) -> Any:
        raw = result.raw_outputs
        if not raw or len(raw) == 0:
            return result

        tensor = raw[0]

        # Handle Hailo NMS BY CLASS format (1D uint8 byte buffer that
        # should be reinterpreted as float32).
        # Format: [4 bytes num_detections] + [20 bytes per detection] * max_dets
        # Each detection: [ymin, xmin, ymax, xmax, confidence] as float32
        if tensor.ndim == 1 and tensor.dtype == np.uint8:
            floats = np.frombuffer(tensor.tobytes(), dtype=np.float32)
            if len(floats) < 5:
                return result
            num_dets = int(floats[0])
            if num_dets <= 0 or num_dets > 200:
                return result
            det_data = floats[1:]
            n_fields = 5
            n_available = len(det_data) // n_fields
            n_actual = min(num_dets, n_available)
            dets = det_data[:n_actual * n_fields].reshape(n_actual, n_fields)
            objects = []
            for row in dets:
                ymin, xmin, ymax, xmax, conf = row
                if conf < 0.15:
                    continue
                objects.append(_DetWrap(
                    label="vehicle",
                    score=float(conf),
                    class_id=0,
                    x=float(xmin), y=float(ymin),
                    w=float(xmax - xmin), h=float(ymax - ymin),
                ))
            if objects:
                logger.debug("Parsed %d NMS detections for %s", len(objects), model_id)
                return _NmsResult(objects=objects)
            return result

        # Standard NMS output: 2D+ tensor with [ymin,xmin,ymax,xmax,conf,cls]
        tensor = tensor.astype(np.float32)

        if tensor.ndim >= 3:
            flat = tensor.reshape(-1, tensor.shape[-1])
        elif tensor.ndim == 2:
            flat = tensor
        elif tensor.ndim == 1 and tensor.shape[0] % 6 == 0:
            flat = tensor.reshape(-1, 6)
        else:
            return result

        if flat.shape[-1] < 5:
            return result

        objects = []
        for row in flat:
            vals = row[:6] if row.shape[0] >= 6 else row[:5]
            if vals[4] < 0.01:
                continue
            ymin, xmin, ymax, xmax = vals[0], vals[1], vals[2], vals[3]
            conf = float(vals[4])
            cls_id = int(vals[5]) if len(vals) > 5 else 0
            if conf < 0.15:
                continue
            objects.append(_DetWrap(
                label=_COCO_NAMES.get(cls_id, f"class_{cls_id}"),
                score=conf,
                class_id=cls_id,
                x=xmin, y=ymin,
                w=xmax - xmin, h=ymax - ymin,
            ))
        if objects:
            logger.debug("Parsed %d NMS detections for %s", len(objects), model_id)
            return _NmsResult(objects=objects)
        return result

    def _parse_classification_raw(self, result: Any) -> Any:
        raw = getattr(result, "raw_outputs", None)
        if not raw or len(raw) == 0:
            return result
        tensor = raw[0]
        flat = tensor.flatten().astype(np.float32)
        if flat.size < 2:
            return result
        if flat.max() <= 1.0 and flat.min() >= 0.0:
            probs = flat
        else:
            exp = np.exp(flat - np.max(flat))
            probs = exp / exp.sum()
        top_k = min(5, len(probs))
        indices = np.argsort(probs)[::-1][:top_k]
        _ClsWrap = type("_CW", (), {})
        classifications = []
        for idx in indices:
            c = _ClsWrap()
            c.type = "classification"
            c.class_id = int(idx)
            c.label = _IMAGENET_NAMES.get(int(idx), f"class_{idx}")
            c.confidence = float(probs[idx])
            classifications.append(c)
        return _ClsResult(classifications=classifications)

    def _parse_face_landmarks(self, result: Any) -> Any:
        """Parse face_landmarks_lite raw output: 468 points × 3 (x,y,z) + 1 conf."""
        raw = getattr(result, "raw_outputs", None)
        if not raw or len(raw) < 1:
            return result

        landmarks_tensor = np.asarray(raw[0], dtype=np.float32).flatten()

        # UINT8 → normalized [0,1]
        if landmarks_tensor.max() > 2.0:
            landmarks_tensor = landmarks_tensor / 255.0

        # face_landmarks_lite: 1404 values = 468 points × 3 (x, y, z)
        if landmarks_tensor.size < 6:
            return result

        n_pts = landmarks_tensor.size // 3
        points = landmarks_tensor[:n_pts * 3].reshape(n_pts, 3)

        # Second output (conv25) is face score but often 0 — skip filtering

        # Build landmark set — only use x, y (drop z)
        _LandmarkPoint = type("_LP", (), {})
        lp_list = []
        for i in range(n_pts):
            p = _LandmarkPoint()
            p.x = float(points[i, 0])
            p.y = float(points[i, 1])
            p.confidence = float(points[i, 2]) if points.shape[1] > 2 else 1.0
            lp_list.append(p)

        _LandmarkSet = type("_LS", (), {})
        ls = _LandmarkSet()
        ls.type = "face"
        ls.points = lp_list
        return type("_LandmarkResult", (), {"landmarks": [ls],
                                             "objects": [], "classifications": [],
                                             "ocr_lines": [], "masks": [],
                                             "embeddings": []})()

    # ------------------------------------------------------------------
    # YOLO grid decoder for tiny_yolov4_license_plates (uint16 raw output)
    # ------------------------------------------------------------------

    _YOLOV4_ANCHORS = {
        "license_plate_det": [
            ((13, 13), [(81, 82), (135, 169), (344, 319)]),
            ((26, 26), [(23, 27), (37, 58), (81, 82)]),
        ],
    }

    def _parse_yolo_grid(self, result: Any, model_id: str) -> Any:
        """Parse YOLO grid output (uint16) from Hailo models without NMS."""
        raw = result.raw_outputs
        if not raw or len(raw) < 2:
            return result

        anchors_cfg = self._YOLOV4_ANCHORS.get(model_id)
        if not anchors_cfg:
            return result

        objects = []
        for idx, (grid_info, anchor_list) in enumerate(anchors_cfg):
            grid_h, grid_w = grid_info
            tensor_bytes = raw[idx]
            if tensor_bytes.dtype == np.uint8:
                tensor = tensor_bytes.view(np.uint16).copy()
            elif tensor_bytes.dtype == np.uint16:
                tensor = tensor_bytes
            else:
                continue

            num_anchors = len(anchor_list)
            elements = grid_h * grid_w * num_anchors * 6
            if tensor.size < elements:
                logger.debug("Grid %d: expected %d elements, got %d", idx, elements, tensor.size)
                continue
            grid = tensor[:elements].reshape(grid_h, grid_w, num_anchors, 6).astype(np.float32)

            # UINT16 → float: dequantize assuming linear mapping to [0,1] range
            grid /= 65535.0

            for gy in range(grid_h):
                for gx in range(grid_w):
                    for a in range(num_anchors):
                        tx, ty, tw, th, obj_conf, cls_conf = grid[gy, gx, a]
                        if obj_conf < 0.3:
                            continue
                        aw, ah = anchor_list[a]
                        cx = (gx + tx) / grid_w
                        cy = (gy + ty) / grid_h
                        w = aw * tw / 416.0
                        h = ah * th / 416.0
                        conf = obj_conf * cls_conf
                        if conf < 0.2:
                            continue
                        objects.append(_DetWrap(
                            label="license_plate", score=conf, class_id=0,
                            x=cx - w / 2, y=cy - h / 2, w=w, h=h,
                        ))

        # NMS
        objects.sort(key=lambda o: o.score, reverse=True)
        keep = []
        for obj in objects:
            suppressed = False
            for k in keep:
                iou = _bbox_iou(obj.bbox, k.bbox)
                if iou > 0.45:
                    suppressed = True
                    break
            if not suppressed:
                keep.append(obj)
        keep = keep[:32]

        if keep:
            logger.debug("YOLO grid decoded: %d detections for %s", len(keep), model_id)
            return _NmsResult(objects=keep)
        return result

    def _get_infer_bgr(self) -> Optional[np.ndarray]:
        """Get a BGR frame for inference — from video or camera."""
        if self._video_active:
            with self._video_lock:
                if self._video_source:
                    return self._video_source.latest_frame()
            return None
        try:
            infer_stream = self._active_infer_stream
            frame = self.media_infer.get_frame(infer_stream, timeout_ms=3000)
        except Exception as e:
            logger.warning("Infer get_frame error: %s", e)
            self._reconnect_infer_media()
            return None
        if frame is None:
            return None
        return self.frame_to_bgr(frame)

    def _prepare_input_from_bgr(self, bgr: np.ndarray, target_w: int,
                                 target_h: int, input_fmt: str) -> np.ndarray:
        src_h, src_w = bgr.shape[:2]
        if src_w != target_w or src_h != target_h:
            bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if input_fmt == "rgb":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).flatten()
        return self._bgr_to_nv12(bgr).flatten()

    def _infer_loop(self) -> None:
        while self._infer_running:
            try:
                with self._model_lock:
                    model_id = self.current_model
                    model_info = self._current_model_info
                    is_pipeline = "stages" in (model_info or {})
                    input_fmt = (model_info or {}).get("input_format", "nv12")
                    target_w = (model_info or {}).get("input_width", 0)
                    target_h = (model_info or {}).get("input_height", 0)
                if not self._video_active:
                    infer_stream = self._choose_infer_stream(model_info) if model_info else self.infer_stream_id
                    self._active_infer_stream = infer_stream
                effective_fps = 1.0 / self._infer_interval
                nv12_direct = (not self._video_active
                               and model_id != "face_landmarks"
                               and not is_pipeline)
                logger.info(
                    "Inference loop starting: model=%s, fps=%.1f, source=%s, "
                    "path=%s (fmt=%s)",
                    model_id, effective_fps,
                    "video" if self._video_active else "camera",
                    "NV12-direct" if nv12_direct else "BGR-legacy",
                    input_fmt,
                )
                last_infer_time = 0.0
                # Fresh per switch: this loop runs in a new thread per model
                # load, so this counter covers the cold HEF context-init of
                # the model that was just registered. See INFER_WARMUP_*.
                warmup_left = self._infer_warmup_count

                while self._infer_running:
                    now = time.time()
                    if now - last_infer_time < self._infer_interval:
                        time.sleep(0.01)
                        continue
                    last_infer_time = now

                    # --- NV12-direct path: get raw frame without forcing BGR ---
                    needs_bgr = (model_id == "face_landmarks" or is_pipeline
                                 or self._video_active)
                    raw = None
                    bgr = None  # only set when BGR is actually needed

                    if needs_bgr:
                        # Pipeline / face / video models still use BGR path
                        bgr = self._get_infer_bgr()
                        if bgr is None:
                            if self._video_active:
                                time.sleep(0.05)
                            continue
                    else:
                        # Standard models: get raw frame (NV12 from camera)
                        raw = self._get_infer_raw()
                        if raw is None:
                            if self._video_active:
                                time.sleep(0.05)
                            continue

                    try:
                        if model_id == "face_landmarks":
                            result = self._run_face_landmarks_bgr(bgr)
                            self.infer_count += 1
                            self.infer_time_us = (
                                self._pipeline_stage1_us + self._pipeline_stage2_us
                            )
                        elif is_pipeline:
                            result = self._run_pipeline_bgr(bgr, self._current_model_info)
                            self.infer_count += 1
                            self.infer_time_us = (
                                self._pipeline_stage1_us + self._pipeline_stage2_us
                            )
                        else:
                            # --- NV12-direct: skip BGR conversion entirely ---
                            # target_w, target_h, input_fmt captured above under lock

                            if raw is not None:
                                raw_data, src_w, src_h, src_fmt = raw
                                if src_fmt == "NV12":
                                    # Best case: direct NV12 input, zero conversion
                                    t_prep = time.monotonic()
                                    input_data = self._prepare_nv12_input(
                                        raw_data, src_w, src_h,
                                        target_w, target_h, input_fmt,
                                    )
                                    prep_us = int((time.monotonic() - t_prep) * 1_000_000)
                                else:
                                    # Fallback (video BGR or unusual format)
                                    bgr = self._to_bgr(raw_data, src_fmt)
                                    if target_w > 0 and target_h > 0:
                                        input_data = self._prepare_input_from_bgr(
                                            bgr, target_w, target_h, input_fmt,
                                        )
                                    elif input_fmt == "rgb":
                                        input_data = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).flatten()
                                    else:
                                        input_data = self._bgr_to_nv12(bgr).flatten()
                                    prep_us = -1  # legacy path
                            else:
                                # Should not happen, but safe fallback
                                input_data = np.zeros(1, dtype=np.uint8)
                                prep_us = -1

                            # Use the generous cold-start timeout for the first
                            # few inferences after a switch (Hailo HEF context
                            # init), then revert to the tight stall-detection
                            # timeout. Decrement per attempt so a slow cold
                            # inference still consumes one warmup slot.
                            if warmup_left > 0:
                                infer_timeout = self._infer_warmup_timeout_ms
                                warmup_left -= 1
                            else:
                                infer_timeout = self.infer_timeout_ms
                            result = self.infer_client.infer(
                                input_data, model_id=model_id, timeout_ms=infer_timeout,
                            )
                            self.infer_count += 1
                            self.infer_time_us = result.infer_time_us
                            # Track hardware-level inference time
                            hw_time = getattr(result, "hw_infer_time_us", None)
                            if hw_time:
                                self._hw_infer_time_us = hw_time
                            else:
                                self._hw_infer_time_us = result.infer_time_us

                            if prep_us >= 0:
                                logger.debug(
                                    "NV12-direct prep: %dus (fmt=%s %dx%d->%dx%d)",
                                    prep_us, input_fmt, src_w, src_h,
                                    target_w, target_h,
                                )

                            # Handle models with built-in NMS (no software postprocess)
                            has_raw = getattr(result, "raw_outputs", None)
                            has_masks = bool(getattr(result, "masks", None))
                            has_depth = bool(getattr(result, "depth_maps", None))
                            if (not result.objects and not result.classifications
                                    and not result.ocr_lines
                                    and not has_masks and not has_depth):
                                if has_raw:
                                    if model_id == "face_landmarks":
                                        result = self._parse_face_landmarks(result)
                                    elif model_id == "vit_classification":
                                        result = self._parse_classification_raw(result)
                                    else:
                                        result = self._parse_nms_raw_result(
                                            result, model_id,
                                        )

                        with self._result_lock:
                            # Accumulate OCR lines across frames for multi-line display
                            if (is_pipeline
                                    and self.current_model_type == "pipeline_ocr"
                                    and getattr(result, "ocr_lines", None)):
                                self._ocr_accumulator.add(result.ocr_lines)
                                merged = self._ocr_accumulator.snapshot()
                                if len(merged) > len(result.ocr_lines):
                                    result = _PipelineResult(
                                        ocr_lines=merged, objects=[],
                                    )
                            self._latest_result = result
                        self._push_sse_result(self.infer_count, result)
                        self._publish_to_event_bus(result)

                        # Gallery capture hook for CLIP models
                        if (self.gallery.enabled
                                and self.current_model_type == "clip"
                                and hasattr(result, "embeddings")
                                and result.embeddings):
                            now = time.time()
                            if now - self.gallery._last_capture >= self.gallery.capture_interval:
                                try:
                                    emb = result.embeddings[0].data
                                    # Get BGR for gallery only when needed
                                    if bgr is None and raw is not None:
                                        bgr = self._to_bgr(raw_data, src_fmt)
                                    if bgr is not None:
                                        enc_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                                        _, jpeg = cv2.imencode(".jpg", bgr, enc_params)
                                        h, w = bgr.shape[:2]
                                        self.gallery.add_image(
                                            jpeg.tobytes(), emb,
                                            int(now * 1000), w, h,
                                        )
                                        self.gallery._last_capture = now
                                except Exception as gallery_err:
                                    logger.debug("Gallery capture skipped: %s", gallery_err)

                        prev_failures = self._consecutive_failures
                        self._consecutive_failures = 0
                        if self._degraded:
                            self._degraded = False
                            self._degrade_reason = None
                            logger.info("Inference recovered")
                        elif prev_failures >= _STALL_BURST_INFO_THRESHOLD:
                            logger.info(
                                "Inference recovered after %d transient stalls",
                                prev_failures,
                            )

                        # Compute dynamic infer FPS
                        now_fps = time.monotonic()
                        dt = now_fps - self._infer_fps_last_time
                        if dt >= 1.0 and self._infer_fps_last_time > 0:
                            delta_count = self.infer_count - self._infer_fps_last_count
                            self._infer_fps = delta_count / dt
                            self._infer_fps_last_count = self.infer_count
                            self._infer_fps_last_time = now_fps
                    except Exception as infer_err:
                        # If the loop is shutting down (model switch / app
                        # stop), the in-flight infer failing is expected —
                        # _stop_infer_thread set _infer_running=False before
                        # joining. Don't log it as a failure; just exit.
                        if not self._infer_running:
                            break
                        self._consecutive_failures += 1
                        n = self._consecutive_failures
                        err_str = str(infer_err)

                        # On this Hailo platform the device-infer call
                        # intermittently blocks past the client timeout
                        # (INFER_TIMEOUT_MS, default 1500ms) then recovers on a
                        # later attempt. Crucially, ONE NPU stall lasting >5s
                        # produces SEVERAL consecutive DEADLINE_EXCEEDED
                        # failures (one per ~1.5s timeout cycle) before the
                        # device recovers — so the consecutive-failure count
                        # tracks how long the *current* stall has lasted, not
                        # how many stalls occurred. Warning at any fixed count
                        # would therefore fire on every routine ~5s stall.
                        #
                        # Per the agreed handling: transient recovering RPC
                        # failures stay at DEBUG (quiet steady-state). We warn
                        # ONLY on genuine sustained degradation — the _degraded
                        # block at n>=10, i.e. a true NPU hang (~15-25s of
                        # continuous failure), which never happens in normal
                        # operation. A recovered burst is surfaced at INFO (see
                        # the success path) so it is visible but not an alarm.
                        # Non-RPC (unexpected) errors always warn immediately.
                        is_model_missing = "not found" in err_str.lower()
                        is_transient_rpc = any(
                            tok in err_str
                            for tok in (
                                "DEADLINE_EXCEEDED", "Deadline Exceeded",
                                "Stream removed", "StatusCode.UNAVAILABLE",
                            )
                        )
                        if is_model_missing:
                            # ai-runtime lost this model from its in-memory
                            # registry — the known grpcpp_sync_ser crash
                            # restarts ai-runtime and wipes all registered
                            # models (see memory ai-runtime-crash-core-dumps),
                            # so infer returns "Model not found". Self-heal by
                            # force re-registering (force=True bypasses the
                            # list_models "already registered" check, which can
                            # be stale right after a restart). Throttle to once
                            # per _reregister_interval so the failing loop
                            # (~1-2x/s) doesn't hammer RegisterModel. Keep it at
                            # DEBUG in steady-state so an ai-runtime restart no
                            # longer fills the log with WARNINGs; warn once on
                            # entry for visibility.
                            if n == 1:
                                logger.warning(
                                    "Model %s missing on ai-runtime (likely a "
                                    "service restart); auto re-registering",
                                    model_id,
                                )
                            else:
                                logger.debug(
                                    "Model %s still missing, re-register retry (#%d)",
                                    model_id, n,
                                )
                            now_ts = time.monotonic()
                            if now_ts - self._last_reregister_ts >= self._reregister_interval:
                                self._last_reregister_ts = now_ts
                                try:
                                    self._ensure_model_registered(
                                        self._current_model_info, force=True,
                                    )
                                except Exception as reg_err:
                                    logger.debug(
                                        "Re-register attempt raised: %s", reg_err,
                                    )
                        elif is_transient_rpc:
                            logger.debug(
                                "Transient infer stall, retrying (#%d): %s",
                                n, infer_err,
                            )
                        else:
                            logger.warning(
                                "Infer call failed (#%d): %s",
                                n, infer_err,
                            )

                        if n >= 10 and not self._degraded:
                            self._degraded = True
                            self._degrade_reason = "inference failing"
                            logger.warning("Entering degraded mode")
                        if n >= 5:
                            time.sleep(1.0)
                        elif n <= 3:
                            time.sleep(0.5 * n)
            except Exception as e:
                if self._infer_running:
                    logger.error("Inference error: %s, reconnecting in 2s...", e, exc_info=True)
                    time.sleep(2)

    def _push_sse_result(self, seq: int, result: Any) -> None:
        data = self._build_sse_data(seq, result)
        msg = f"data: {json.dumps(data)}\n\n"
        with self._sse_lock:
            dead = []
            for q in self._sse_subscribers:
                try:
                    q.put(msg)
                except Exception:
                    dead.append(q)
            for q in dead:
                self._sse_subscribers.remove(q)

    # ------------------------------------------------------------------
    # Event Bus — publish inference results for camera-daemon AI overlay
    # ------------------------------------------------------------------

    def _get_event_client(self) -> Optional[EventClient]:
        """Return a lazily-connected EventClient, or None on failure.

        Uses lazy connect so that a missing Event Bus socket at startup
        never prevents the app from running.  Re-connects transparently
        if the previous client was closed or the channel dropped.
        """
        if not self._event_bus_enabled:
            return None
        if self._event_client is None:
            try:
                self._event_client = EventClient(endpoint=self._event_bus_endpoint)
                self._event_client.connect()
                logger.info("Event Bus client connected: %s", self._event_bus_endpoint)
            except Exception as e:
                logger.debug("Event Bus connect failed (will retry): %s", e)
                self._event_client = None
        return self._event_client

    def _build_event_bus_payload(self, result: Any) -> Optional[Dict[str, Any]]:
        """Convert an inference result to the JSON payload expected by
        camera-daemon's AiOverlaySubscriber (parse_json_result).

        Supported types and their wire format:
          detection     → {"num_detections": N, "detections": [...]}
          classification → {"classifications": [...]}   (top-5)
          keypoint      → {"landmarks": [...]}
          pipeline_ocr  → {"ocr_lines": [...]}
          pipeline_lpr  → {"ocr_lines": [...]}

        Types handled entirely via SSE/Canvas (no Event Bus payload):
          segmentation, depth, embedding, clip, genai
        """
        mtype = self.current_model_type

        if mtype == "detection":
            objects = getattr(result, "objects", None) or []
            if not objects:
                return {"num_detections": 0, "detections": []}
            dets = []
            for o in objects:
                dets.append({
                    "class_id": int(getattr(o, "class_id", 0)),
                    "label":    str(getattr(o, "label", "")),
                    "confidence": float(getattr(o, "score", getattr(o, "confidence", 0))),
                    "bbox": [
                        float(o.bbox.x), float(o.bbox.y),
                        float(o.bbox.width), float(o.bbox.height),
                    ],
                })
            return {"num_detections": len(dets), "detections": dets}

        if mtype == "classification":
            clss = getattr(result, "classifications", None) or []
            if not clss:
                return None
            return {
                "classifications": [
                    {
                        "class_id":   int(getattr(c, "class_id", 0)),
                        "label":      str(getattr(c, "label", "")),
                        "confidence": float(getattr(c, "confidence", 0)),
                    }
                    for c in clss
                ]
            }

        if mtype == "keypoint":
            landmark_sets = getattr(result, "landmarks", None) or []
            if not landmark_sets:
                return None
            lm_out = []
            for ls in landmark_sets:
                pts = getattr(ls, "points", []) or []
                lm_out.append({
                    "type":   str(getattr(ls, "type", "face")),
                    "points": [
                        {"x": float(p.x), "y": float(p.y),
                         "confidence": float(getattr(p, "confidence", 1.0))}
                        for p in pts
                    ],
                })
            return {"landmarks": lm_out} if lm_out else None

        if mtype in ("pipeline_ocr", "pipeline_lpr"):
            lines = getattr(result, "ocr_lines", None) or []
            if not lines:
                return {"ocr_lines": []}
            ocr = []
            for line in lines:
                bbox = getattr(line, "bbox", None)
                ocr.append({
                    "text":       str(getattr(line, "text", "")),
                    "confidence": float(getattr(line, "confidence", 0)),
                    "bbox": [
                        float(bbox.x), float(bbox.y),
                        float(bbox.width), float(bbox.height),
                    ] if bbox else [0.0, 0.0, 0.0, 0.0],
                })
            return {"ocr_lines": ocr}

        # segmentation / depth / embedding / clip / genai — no Event Bus payload
        return None

    def _publish_to_event_bus(self, result: Any) -> None:
        """Publish inference result to Event Bus for camera-daemon AI overlay.

        camera-daemon's AiOverlaySubscriber subscribes to "inference/**"
        and calls apply_overlay() to draw boxes on the hardware H264 stream
        before RTSP encoding — zero Python CPU cost for the actual drawing.

        The topic carries the infer stream id in both the path and the
        ``stream_id`` metadata field so camera-daemon can map it to the
        correct display stream via its stream_map config.

        Failures are silently logged at DEBUG level so Event Bus being
        unavailable never interrupts inference.
        """
        try:
            payload = self._build_event_bus_payload(result)
            if payload is None:
                return
            client = self._get_event_client()
            if client is None:
                return
            stream_id = self._active_infer_stream
            client.publish(
                topic=f"inference/{stream_id}",
                payload=payload,
                metadata={"stream_id": stream_id},
            )
        except Exception as e:
            # Reset client so next call will attempt reconnect
            self._event_client = None
            logger.debug("Event Bus publish failed: %s", e)

    def _clear_event_bus_overlay(self) -> None:
        """Publish an empty detection payload to clear stale overlay boxes.

        Called on model switch so camera-daemon stops drawing boxes for the
        old model while the new model's first inference hasn't arrived yet.
        """
        try:
            client = self._get_event_client()
            if client is None:
                return
            stream_id = self._active_infer_stream
            client.publish(
                topic=f"inference/{stream_id}",
                payload={"num_detections": 0, "detections": []},
                metadata={"stream_id": stream_id},
            )
        except Exception as e:
            self._event_client = None
            logger.debug("Event Bus clear overlay failed: %s", e)

    def _downsample_map(self, m: np.ndarray, interp: int = cv2.INTER_NEAREST) -> np.ndarray:
        """Downscale a seg/depth overlay map before base64-encoding for SSE.

        The browser (drawDepth/drawSegStats) upsamples with nearest-neighbour,
        so a small source map renders the same while cutting the per-inference
        tobytes+base64 cost ~16x.  That encode was the dominant scdepth CPU
        sink (python3 pegged one core even with the infer rate capped).
        """
        cap = int(os.environ.get("SSE_MAP_MAX_W", "160"))
        h, w = m.shape[:2]
        if cap <= 0 or w <= cap:
            return m
        new_h = max(1, int(round(h * cap / w)))
        return cv2.resize(m, (cap, new_h), interpolation=interp)

    def _build_sse_data(self, seq: int, result: Any) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "frame_sequence": seq,
            "model_id": self.current_model,
            "model_type": self.current_model_type,
            "infer_count": self.infer_count,
        }
        if self._pipeline_stage1_us > 0:
            data["pipeline"] = {
                "stage1_ms": round(self._pipeline_stage1_us / 1000.0, 1),
                "stage2_ms": round(self._pipeline_stage2_us / 1000.0, 1),
                "results": self._pipeline_count,
            }
        if result.objects:
            data["objects"] = [
                {"label": o.label, "score": round(o.score, 3),
                 "bbox": {"x": o.bbox.x, "y": o.bbox.y, "w": o.bbox.width, "h": o.bbox.height}}
                for o in result.objects
            ]
        if result.classifications:
            data["classifications"] = [
                {"label": c.label, "score": round(c.confidence, 4)}
                for c in result.classifications
            ]
        if result.landmarks:
            data["landmarks"] = [
                {"type": ls.type, "point_count": len(ls.points), "points": [
                    {"x": p.x, "y": p.y, "confidence": round(p.confidence, 3)} for p in ls.points
                ]}
                for ls in result.landmarks
            ]
        if getattr(result, "masks", None):
            data["masks"] = [
                {"class_id": m.class_id, "label": getattr(m, "label", ""),
                 "confidence": round(getattr(m, "confidence", 0), 3)}
                for m in result.masks
            ]
        if getattr(result, "ocr_lines", None):
            data["ocr_lines"] = [
                {"text": getattr(l, "text", ""), "confidence": round(getattr(l, "confidence", 0), 3),
                 "bbox": {"x": l.bbox.x, "y": l.bbox.y, "w": l.bbox.width, "h": l.bbox.height}}
                for l in result.ocr_lines
            ]
        if getattr(result, "embeddings", None):
            data["embedding_dim"] = result.embeddings[0].dim if result.embeddings else 0
            data["embedding_count"] = len(result.embeddings)

        # Send pixel data for browser-side overlay rendering
        if self.current_model_type == "segmentation":
            class_map = _seg_class_map_from_result(result)
            if class_map is not None:
                cm = self._downsample_map(class_map)
                data["seg_class_map"] = {
                    "data": base64.b64encode(cm.tobytes()).decode("ascii"),
                    "h": int(cm.shape[0]),
                    "w": int(cm.shape[1]),
                }
        elif self.current_model_type == "depth":
            depth_data = _extract_depth_data(result)
            if depth_data is not None:
                norm = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                dm = self._downsample_map(norm)
                data["depth_map"] = {
                    "data": base64.b64encode(dm.tobytes()).decode("ascii"),
                    "h": int(dm.shape[0]),
                    "w": int(dm.shape[1]),
                }

        return data

    def frame_to_bgr(self, frame: Frame) -> np.ndarray:
        if frame.format == "NV12":
            return cv2.cvtColor(frame.image, cv2.COLOR_YUV2BGR_NV12)
        elif frame.format == "RGB":
            return cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
        elif frame.format == "BGR":
            return frame.image
        else:
            rgb = frame.to_rgb()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def draw_overlay(self, bgr: np.ndarray, result: Any) -> np.ndarray:
        global _current_draw_model_info
        _current_draw_model_info = self._current_model_info
        handler = DRAW_HANDLERS.get(self.current_model_type)
        if handler:
            handler(bgr, result)
            # Sync module-level overlay cache to instance for frame-loop reuse
            if self.current_model_type in ("segmentation", "depth"):
                self._cached_colored_mask = _overlay_cache.get("colored_mask")
                self._cached_fg_mask = _overlay_cache.get("fg_mask")
                self._cached_class_pixels = _overlay_cache.get("class_pixels")
                self._cached_overlay_infer_count = self.infer_count
        return bgr

    def draw_stats_bar(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        bar_h = 36
        y_start = h - bar_h
        # Darken only the stats-bar strip in place. The old code did a full-frame
        # bgr.copy() + cv2.rectangle + addWeighted just to shade a 36px bar — a
        # wasted full-frame op on every seg/depth render (the linknet bottleneck).
        # addWeighted on the ROI slice (a view into bgr) writes straight back into
        # bgr and skips the copy; 0.4*bar matches the prior 40%-darken exactly.
        bar = bgr[y_start:h]
        cv2.addWeighted(bar, 0.4, bar, 0, 0, bar)

        infer_ms = self.infer_time_us / 1000.0 if self.infer_time_us else 0
        hw_ms = self._hw_infer_time_us / 1000.0 if self._hw_infer_time_us else 0
        # Prefer hardware-level timing (more accurate) over gRPC-level timing
        latency_ms = hw_ms if hw_ms > 0 else infer_ms
        model_name = next(
            (m["name"] for m in AVAILABLE_MODELS if m["id"] == self.current_model),
            self.current_model,
        )
        with self._npu_stats_lock:
            npu = self._npu_stats
        npu_util = f"NPU:{npu.get('device_utilization', 0):.0f}%" if npu else ""
        npu_temp = f"{npu.get('device_temperature', 0):.0f}C" if npu else ""

        text = f"{model_name}  |  FPS:{self.current_fps:.1f}  |  {latency_ms:.1f}ms"
        if self._pipeline_stage1_us > 0:
            s1 = self._pipeline_stage1_us / 1000.0
            s2 = self._pipeline_stage2_us / 1000.0
            text += f" (S1:{s1:.0f}ms S2:{s2:.0f}ms)"
        if npu_util:
            text += f"  |  {npu_util} {npu_temp}"
        text += f"  |  #{self.frame_count}"

        color = (0, 0, 255) if self._degraded else (0, 255, 255)
        cv2.putText(bgr, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def infer_image(self, bgr: np.ndarray) -> dict:
        """Run a single-shot inference on a still BGR image and annotate it.

        Mirrors the capture-then-infer shape of `_infer_loop` but is fully
        synchronous on the calling (request) thread. It reuses
        `_prepare_input_from_bgr` for input prep and `draw_overlay` for rendering
        (which dispatches by `current_model_type`, so every supported model type
        works on a still image for free). Does NOT touch `_infer_loop`,
        `run_frame_loop`, or `frame_buffer` — the live preview is untouched.

        Draws a dedicated one-shot stats bar from the result's own timing rather
        than `draw_stats_bar` (which reads live-loop shared state and would race /
        show stale FPS+frame_count).
        """
        with self._model_lock:
            model_id = self.current_model
            model_info = self._current_model_info
            input_fmt = (model_info or {}).get("input_format", "nv12")
            target_w = (model_info or {}).get("input_width", 0)
            target_h = (model_info or {}).get("input_height", 0)
        if not model_id or not model_info:
            raise RuntimeError("no model loaded; select a model first")
        if target_w <= 0 or target_h <= 0:
            raise RuntimeError(
                f"model {model_id} has no input dimensions; cannot run image inference"
            )

        # Normalize the display canvas so the fixed-scale overlay renders at a
        # readable size (see _IMAGE_RENDER_MAX_SIDE). Inference is unaffected —
        # _prepare_input_from_bgr re-resizes to the model's input dims anyway.
        bgr = _resize_to_side(bgr, _IMAGE_RENDER_MAX_SIDE)

        input_data = self._prepare_input_from_bgr(bgr, target_w, target_h, input_fmt)
        # One-shot may hit a cold HEF context (first infer after switch), so use
        # the warmup timeout (6s default) rather than the steady-state 1.5s.
        result = self.infer_client.infer(
            input_data, model_id=model_id, timeout_ms=self._infer_warmup_timeout_ms,
        )

        # Models with built-in NMS / raw tensor outputs come back as `raw_outputs`,
        # NOT pre-parsed objects/classifications. Without this post-processing the
        # overlay draws nothing (e.g. vit_classification → empty classifications →
        # _draw_classifications early-returns). Mirrors `_infer_loop` lines 3124-3139.
        has_raw = getattr(result, "raw_outputs", None)
        has_masks = bool(getattr(result, "masks", None))
        has_depth = bool(getattr(result, "depth_maps", None))
        if (not result.objects and not result.classifications
                and not result.ocr_lines
                and not has_masks and not has_depth):
            if has_raw:
                if model_id == "face_landmarks":
                    result = self._parse_face_landmarks(result)
                elif model_id == "vit_classification":
                    result = self._parse_classification_raw(result)
                else:
                    result = self._parse_nms_raw_result(result, model_id)

        # Annotate the ORIGINAL-resolution BGR in place (no preview downscale —
        # the user wants the full-res result).
        self.draw_overlay(bgr, result)

        # Dedicated one-shot stats bar (no shared-state read). Scale font /
        # bar height to the rendered size so the text stays readable at 1280+.
        h, w = bgr.shape[:2]
        sf = max(1.0, w / 640.0)  # 1.0 at 640-wide, ~2.0 at 1280
        bar_h = int(36 * sf)
        cv2.addWeighted(bgr[h - bar_h:h], 0.4, bgr[h - bar_h:h], 0, 0, bgr[h - bar_h:h])
        infer_us = getattr(result, "infer_time_us", 0) or 0
        hw_us = getattr(result, "hw_infer_time_us", 0) or 0
        latency_ms = (hw_us / 1000.0) if hw_us > 0 else (infer_us / 1000.0)
        model_name = next(
            (m["name"] for m in AVAILABLE_MODELS if m["id"] == model_id),
            model_id,
        )
        bar_text = f"{model_name}  |  {latency_ms:.1f}ms  |  image"
        cv2.putText(
            bgr, bar_text, (10, h - int(10 * sf)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sf, (0, 255, 255),
            max(1, int(round(sf))),
        )

        ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return {
            "jpeg": jpeg.tobytes(),
            "infer_time_us": infer_us,
            "hw_infer_time_us": hw_us,
            "model": model_id,
            "model_type": self.current_model_type,
            "width": int(w),
            "height": int(h),
        }

    def run_frame_loop(self) -> None:
        self.running = True
        self._start_infer_thread()
        self._start_npu_stats_thread()
        threading.Thread(target=self._push_clip_labels, daemon=True).start()
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop, name="model-reaper", daemon=True,
        )
        self._reaper_thread.start()

        fps_counter = 0
        fps_start = time.time()
        last_frame_time = 0.0
        consecutive_none = 0
        logger.info("Preview frame loop starting: stream_id=%s", self.stream_id)

        while self.running:
            # --- per-frame profiling (DEBUG-gated; see logger.debug below) ---
            _t_gf = _t_cv = _t_rz = _t_ov = _t_enc = _t_fb = 0.0
            _jpeg_sz = 0
            _real_seq = None
            _is_repush = False
            if self._video_active:
                # Video mode: sample the threaded decoder's latest frame. The
                # decode thread advances the file at playback FPS; this loop and
                # inference both just read the newest frame (decoupled from
                # INFER_FPS). Small yield so we don't busy-spin between the
                # preview_fps pacer slots below.
                time.sleep(0.01)
                with self._video_lock:
                    vs = self._video_source
                if vs is None:
                    continue
                bgr = vs.latest_frame()
                if bgr is None:
                    continue
            else:
                try:
                    # Short timeout so the loop keeps spinning (and re-pushing
                    # the last good frame below) even when the camera "sub"
                    # stream hiccups. get_frame blocks up to timeout_ms when no
                    # frame is available — a long timeout pins the whole loop
                    # (1s timeout ≈ 1 fps during a hiccup). 80ms lets the loop
                    # re-push cached frames at ~12 fps instead of cratering.
                    _p_gf = time.perf_counter()
                    frame = self.media_preview.get_frame(self.stream_id, timeout_ms=80)
                    _t_gf = (time.perf_counter() - _p_gf) * 1000.0
                except Exception as e:
                    logger.warning("Preview frame get error: %s", e, exc_info=True)
                    self._reconnect_preview_media()
                    consecutive_none = 0
                    continue
                if frame is None:
                    consecutive_none += 1
                    # With an 80ms get_frame timeout, reconnect only after ~10s
                    # of NO real frame (125 × 80ms) — a transient 1-5s camera
                    # hiccup is bridged by re-pushing the last good frame below,
                    # NOT by tearing down and re-establishing the preview media
                    # (which would thrash at the old threshold of 10).
                    if consecutive_none >= 125:
                        logger.warning("Preview frame: %d consecutive None (~10s), reconnecting", consecutive_none)
                        self._reconnect_preview_media()
                        consecutive_none = 0
                    # Camera preview ("sub") stream hiccup — NOT an NPU stall:
                    # inference runs on a separate stream and model_qps is
                    # unaffected. get_frame returned nothing for timeout_ms
                    # (1s). Rather than bare-continue and freeze the MJPEG
                    # stream (~0 fps crater), re-push the last good frame so
                    # the preview keeps flowing and current_fps stays honest.
                    # Fixes the intermittent linknet/segmentation FPS dips.
                    if self._last_bgr is None:
                        continue
                    bgr = self._last_bgr
                    _is_repush = True
                else:
                    consecutive_none = 0
                    _p_cv = time.perf_counter()
                    bgr = self.frame_to_bgr(frame)
                    _t_cv = (time.perf_counter() - _p_cv) * 1000.0
                    _real_seq = getattr(frame, "frame_sequence", None)
                    # Snapshot (copy) so any downstream in-place overlay draw
                    # on a resized alias never corrupts our cached frame.
                    self._last_bgr = bgr.copy()

            now = time.time()
            if now - last_frame_time < (1.0 / self.preview_fps):
                continue
            last_frame_time = now

            fps_counter += 1
            if now - fps_start >= 1.0:
                self.current_fps = fps_counter / (now - fps_start)
                fps_counter = 0
                fps_start = now

            try:
                with self._result_lock:
                    result = self._latest_result
                if self._video_active:
                    bgr = bgr.copy()

                self.frame_count += 1
                if self.frame_count == 1:
                    logger.info("First frame: %dx%d", bgr.shape[1], bgr.shape[0])

                # RTSP: draw overlay server-side, then write
                if self._rtsp_enabled:
                    rtsp_bgr = bgr.copy()
                    if result:
                        self.draw_overlay(rtsp_bgr, result)
                    self.draw_stats_bar(rtsp_bgr)
                    self._write_rtsp_frame(rtsp_bgr)

                # MJPEG: browser draws overlay via Canvas for detection/classification,
                # but for segmentation/depth we do server-side overlay to show the
                # colour map in the MJPEG stream too.
                if self._output_mode in ("mjpeg", "both"):
                    h, w = bgr.shape[:2]
                    _p_rz = time.perf_counter()
                    if (w, h) != (self.preview_width, self.preview_height):
                        preview_bgr = cv2.resize(
                            bgr, (self.preview_width, self.preview_height),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        preview_bgr = bgr
                    _t_rz = (time.perf_counter() - _p_rz) * 1000.0

                    _t_ov = 0.0
                    _t_enc = 0.0
                    _t_fb = 0.0
                    _jpeg_sz = 0
                    is_overlay_type = self.current_model_type in ("segmentation", "depth")

                    if is_overlay_type:
                        # For seg/depth, only render+encode when a new inference
                        # result has arrived (since infer FPS << preview FPS).
                        # Between new results, re-use the cached coloured mask for
                        # a lightweight foreground-only blend on the fresh camera frame.
                        new_result = (self.infer_count != self._cached_overlay_infer_count)
                        now_render = time.time()
                        # Time-throttle the expensive full render to at most
                        # 1/_overlay_interval Hz (~10 Hz). seg/depth models infer
                        # FASTER than the preview loop, so without this cap,
                        # new_result is True every iteration and we full-render
                        # every frame (draw_overlay + stats bar + imencode at
                        # full preview resolution) — pegging python3 and cratering
                        # stream_fps to ~2 fps (verified: linknet 2.5 vs yolov8n
                        # 12, same camera path). Between renders we re-push the
                        # cached JPEG (cheap); ~10 Hz mask refresh is imperceptible.
                        should_render = (now_render - self._overlay_last_render) >= self._overlay_interval
                        if new_result and result and should_render:
                            # Full render: draw the pixel overlay only. Do NOT bake
                            # draw_stats_bar here — the browser canvas already draws
                            # a stats bar (drawStatsBar in index.html:689) for every
                            # model. Baking a second one into the JPEG stacked a
                            # duplicate bar on seg/depth — the "two layers" seen on
                            # linknet/scdepth (detection never bakes a server bar, so
                            # this also makes seg/depth consistent). RTSP (above)
                            # keeps its bar since it has no overlay canvas.
                            _p = time.perf_counter()
                            self.draw_overlay(preview_bgr, result)
                            _t_ov = (time.perf_counter() - _p) * 1000.0
                            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                            _p = time.perf_counter()
                            _, jpeg = cv2.imencode(".jpg", preview_bgr, encode_params)
                            _t_enc = (time.perf_counter() - _p) * 1000.0
                            _jb = jpeg.tobytes()
                            _jpeg_sz = len(_jb)
                            _p = time.perf_counter()
                            self.frame_buffer.update(_jb)
                            _t_fb = (time.perf_counter() - _p) * 1000.0
                            self._cached_jpeg = _jb
                            self._overlay_last_render = now_render
                        elif self._cached_jpeg:
                            # No new result: the coloured mask is unchanged since
                            # the last full render, so re-push the cached JPEG
                            # instead of re-blending the (static) mask onto every
                            # fresh camera frame.  Skipping the per-frame
                            # addWeighted + full-frame copy + imencode is the main
                            # CPU win that keeps python3 under one core for
                            # seg/depth while still streaming at preview FPS.
                            _jb = self._cached_jpeg
                            _jpeg_sz = len(_jb)
                            _p = time.perf_counter()
                            self.frame_buffer.update(_jb)
                            _t_fb = (time.perf_counter() - _p) * 1000.0
                        else:
                            # First frames, no result yet — show the plain camera
                            # frame so the preview is not blank before first infer.
                            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                            _p = time.perf_counter()
                            _, jpeg = cv2.imencode(".jpg", preview_bgr, encode_params)
                            _t_enc = (time.perf_counter() - _p) * 1000.0
                            _jb = jpeg.tobytes()
                            _jpeg_sz = len(_jb)
                            _p = time.perf_counter()
                            self.frame_buffer.update(_jb)
                            _t_fb = (time.perf_counter() - _p) * 1000.0
                            self._cached_jpeg = _jb
                    else:
                        # Non-overlay types: encode every preview frame (browser
                        # draws overlay via SSE pixel data)
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                        _p = time.perf_counter()
                        _, jpeg = cv2.imencode(".jpg", preview_bgr, encode_params)
                        _t_enc = (time.perf_counter() - _p) * 1000.0
                        _jb = jpeg.tobytes()
                        _jpeg_sz = len(_jb)
                        _p = time.perf_counter()
                        self.frame_buffer.update(_jb)
                        _t_fb = (time.perf_counter() - _p) * 1000.0

                    logger.debug(
                        "PF prof gf=%.1f cv=%.1f rz=%.1f ov=%.1f enc=%.1f fb=%.1f "
                        "jpeg=%dB seq=%s repush=%s mode=%s",
                        _t_gf, _t_cv, _t_rz, _t_ov, _t_enc, _t_fb,
                        _jpeg_sz, _real_seq, _is_repush, self.current_model_type,
                    )
            except Exception as e:
                logger.error("Frame processing error (#%d): %s", self.frame_count, e, exc_info=True)

        logger.info("Frame loop stopped after %d frames", self.frame_count)

    # ------------------------------------------------------------------
    # Video source management
    # ------------------------------------------------------------------

    def start_video(self, path: str) -> Dict[str, Any]:
        self._stop_video()
        vs = VideoFrameSource(path)
        with self._video_lock:
            self._video_source = vs
            self._video_active = True  # set under lock to close the stale-flag window
        self._video_info = {
            "path": path,
            "fps": vs.fps,
            "total_frames": vs.total_frames,
            "width": vs.width,
            "height": vs.height,
            "duration": round(vs.duration, 2),
        }
        logger.info("Video source started: %s (%dx%d, %.1ffps, %.1fs)",
                     path, vs.width, vs.height, vs.fps, vs.duration)
        # Spawn the background decode thread — it advances the file at playback
        # FPS, independent of INFER_FPS. Done after _video_active so consumers
        # are primed, and before the infer thread restart so inference can
        # sample frames as soon as the first one decodes.
        vs.start()
        # Restart infer thread to pick up video source
        self._stop_infer_thread()
        self._start_infer_thread()
        return self._video_info

    def _stop_video(self) -> None:
        was_active = self._video_active
        self._video_active = False
        with self._video_lock:
            if self._video_source:
                self._video_source.close()
                self._video_source = None
        self._video_info = {}
        if was_active:
            logger.info("Video source stopped")

    def stop_video(self) -> None:
        self._stop_video()
        # Restart infer thread for camera source
        self._stop_infer_thread()
        self._start_infer_thread()

    def video_control(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        with self._video_lock:
            vs = self._video_source
        if not vs:
            return {"error": "No video loaded"}

        if action == "play":
            vs.paused = False
        elif action == "pause":
            vs.paused = True
        elif action == "seek":
            vs.seek(kwargs.get("position", 0))
        elif action == "speed":
            vs.speed = max(0.25, min(4.0, kwargs.get("speed", 1.0)))
        elif action == "restart":
            vs.seek(0)
            vs.paused = False
        return self.get_video_status()

    def get_video_status(self) -> Dict[str, Any]:
        with self._video_lock:
            vs = self._video_source
        if not vs:
            return {"active": False}
        return {
            "active": True,
            "paused": vs.paused,
            "speed": vs.speed,
            "position": round(vs.position_sec, 2),
            "duration": round(vs.duration, 2),
            "progress": round(vs.progress, 4),
            "width": vs.width,
            "height": vs.height,
        }

    # ------------------------------------------------------------------
    # RTSP streaming
    # ------------------------------------------------------------------

    def _start_rtsp_server(self) -> None:
        """Start mediamtx RTSP server if RTSP output is enabled."""
        if not self._rtsp_enabled:
            return
        mediamtx_path = "/usr/local/bin/mediamtx"
        if not os.path.isfile(mediamtx_path):
            mediamtx_path = "/app/mediamtx"
        if not os.path.isfile(mediamtx_path):
            logger.warning("mediamtx not found, RTSP server not started")
            self._rtsp_enabled = False
            self._output_mode = "mjpeg"
            return
        try:
            conf_path = "/app/mediamtx.yml"
            self._mediamtx_process = subprocess.Popen(
                [mediamtx_path, conf_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            time.sleep(0.5)
            if self._mediamtx_process.poll() is not None:
                logger.warning("mediamtx exited immediately, RTSP unavailable")
                self._mediamtx_process = None
                self._rtsp_enabled = False
                self._output_mode = "mjpeg"
                return
            logger.info("mediamtx RTSP server started on :8555")
        except Exception as e:
            logger.warning("Failed to start mediamtx: %s", e)
            self._rtsp_enabled = False
            self._output_mode = "mjpeg"

    def _ensure_rtsp_pusher(self, width: int, height: int) -> None:
        """Ensure FFmpeg RTSP pusher is running for the given frame size."""
        if self._rtsp_process is not None and self._rtsp_frame_size == (width, height):
            if self._rtsp_process.poll() is None:
                return
            stderr_out = ""
            try:
                stderr_out = self._rtsp_process.stderr.read().decode(errors="replace")[-500:]
            except Exception:
                pass
            logger.warning("RTSP pusher died (rc=%d): %s", self._rtsp_process.returncode, stderr_out)
            self._stop_rtsp_pusher()

        # Rate-limit restarts to avoid tight loop
        now = time.time()
        if now - getattr(self, "_rtsp_last_start", 0) < 3:
            return
        self._rtsp_last_start = now

        ffmpeg_path = "ffmpeg"
        fps = min(self.infer_fps_target, 15)
        cmd = [
            ffmpeg_path, "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-b:v", "2000k",
            "-maxrate", "2500k",
            "-bufsize", "4000k",
            "-g", "30",
            "-pix_fmt", "yuv420p",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self._rtsp_url,
        ]
        try:
            self._rtsp_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            # Drain stderr in background to prevent pipe blocking
            threading.Thread(target=self._drain_stderr, daemon=True).start()
            self._rtsp_frame_size = (width, height)
            logger.info("RTSP pusher started: %dx%d@%dfps -> %s", width, height, fps, self._rtsp_url)
        except FileNotFoundError:
            logger.warning("ffmpeg not found, RTSP push unavailable")
            self._rtsp_enabled = False
            self._output_mode = "mjpeg"
        except Exception as e:
            logger.warning("RTSP pusher start failed: %s", e)
            self._rtsp_process = None

    def _drain_stderr(self) -> None:
        """Drain FFmpeg stderr to prevent pipe blockage and log errors."""
        if self._rtsp_process and self._rtsp_process.stderr:
            for line in self._rtsp_process.stderr:
                line_s = line.decode(errors="replace").strip()
                if line_s and ("error" in line_s.lower() or "warning" in line_s.lower()):
                    logger.warning("ffmpeg: %s", line_s)

    def _stop_rtsp_pusher(self) -> None:
        if self._rtsp_process is None:
            return
        try:
            self._rtsp_process.stdin.close()
        except Exception:
            pass
        try:
            self._rtsp_process.terminate()
            self._rtsp_process.wait(timeout=3)
        except Exception:
            try:
                self._rtsp_process.kill()
            except Exception:
                pass
        self._rtsp_process = None
        self._rtsp_frame_size = (0, 0)
        logger.info("RTSP pusher stopped")

    def _stop_rtsp_server(self) -> None:
        if self._mediamtx_process is None:
            return
        try:
            self._mediamtx_process.terminate()
            self._mediamtx_process.wait(timeout=3)
        except Exception:
            try:
                self._mediamtx_process.kill()
            except Exception:
                pass
        self._mediamtx_process = None
        logger.info("mediamtx stopped")

    def _write_rtsp_frame(self, bgr: np.ndarray) -> None:
        """Write a BGR frame to the RTSP pusher."""
        if not self._rtsp_enabled:
            return
        h, w = bgr.shape[:2]
        try:
            self._ensure_rtsp_pusher(w, h)
            if self._rtsp_process and self._rtsp_process.poll() is None:
                self._rtsp_process.stdin.write(bgr.tobytes())
        except (BrokenPipeError, OSError):
            logger.warning("RTSP pipe broken, will retry next frame")
            self._stop_rtsp_pusher()
        except Exception as e:
            logger.warning("RTSP write error: %s", e)
            self._stop_rtsp_pusher()

    def set_output_mode(self, mode: str) -> Dict[str, Any]:
        """Switch output mode at runtime."""
        if mode not in ("mjpeg", "rtsp", "both"):
            return {"error": "invalid mode, use: mjpeg, rtsp, both"}
        self._output_mode = mode
        self._rtsp_enabled = mode in ("rtsp", "both")
        os.environ["OUTPUT_MODE"] = mode
        if not self._rtsp_enabled:
            self._stop_rtsp_pusher()
        if self._rtsp_enabled and self._mediamtx_process is None:
            self._start_rtsp_server()
        logger.info("Output mode set to: %s", mode)
        return self.get_output_status()

    def get_output_status(self) -> Dict[str, Any]:
        # camera-daemon's own RTSP/HLS stream (hardware-encoded, AI overlay drawn in C++)
        cam_rtsp_host = os.environ.get("CAMERA_HOST", "127.0.0.1")
        cam_rtsp_port = int(os.environ.get("CAMERA_RTSP_PORT", "8554"))
        cam_rtsp_path = os.environ.get("CAMERA_RTSP_PATH", "main")
        cam_hls_port  = int(os.environ.get("CAMERA_HLS_PORT", "8888"))
        cam_hls_path  = os.environ.get("CAMERA_HLS_PATH", "main")
        cam_webrtc_port = int(os.environ.get("CAMERA_WEBRTC_PORT", "8889"))

        return {
            # model-showcase self-rendered streams
            "mode": self._output_mode,
            "rtsp_enabled": self._rtsp_enabled,
            "rtsp_url": self._rtsp_url if self._rtsp_enabled else None,
            "rtsp_active": self._rtsp_process is not None and self._rtsp_process.poll() is None,
            "rtsp_server_running": (
                self._mediamtx_process is not None
                and self._mediamtx_process.poll() is None
            ),
            # camera-daemon hardware-encoded streams (AI overlay drawn in C++)
            "camera_rtsp_url": (
                f"rtsp://{cam_rtsp_host}:{cam_rtsp_port}/{cam_rtsp_path}"
            ),
            "camera_hls_url": (
                f"http://{cam_rtsp_host}:{cam_hls_port}/{cam_hls_path}/index.m3u8"
            ),
            "camera_webrtc_url": (
                f"http://{cam_rtsp_host}:{cam_webrtc_port}/{cam_hls_path}/whep"
            ),
            "event_bus_enabled": self._event_bus_enabled,
        }

    def stop(self) -> None:
        self.running = False
        if self._reaper_thread and self._reaper_thread.is_alive():
            self._reaper_thread.join(timeout=3.0)
        self._stop_infer_thread()
        self._stop_video()
        self._stop_rtsp_pusher()
        self._stop_rtsp_server()
        if self.media_preview:
            self.media_preview.close()
        if self.media_infer:
            self.media_infer.close()
        if self.infer_client:
            self.infer_client.close()

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "current_model": self.current_model,
            "model_type": self.current_model_type,
            "frame_count": self.frame_count,
            "infer_count": self.infer_count,
            "infer_time_ms": round(self.infer_time_us / 1000.0, 1),
            "hw_infer_time_ms": round(self._hw_infer_time_us / 1000.0, 1) if self._hw_infer_time_us else 0,
            "infer_fps": round(self._infer_fps, 1),
            "stream_fps": round(self.current_fps, 1),
            "preview_stream": self.stream_id,
            "infer_stream": self._active_infer_stream,
            "clip_labels": list(self._clip_labels),
            "output_mode": self._output_mode,
        }
        with self._npu_stats_lock:
            npu = self._npu_stats
        if npu:
            raw_util = npu.get("device_utilization", 0)
            stats["device_utilization"] = round(raw_util * 100, 2)
            stats["device_temperature"] = round(npu.get("device_temperature", 0), 1)
            used = npu.get("used_memory_bytes", 0)
            total = npu.get("total_memory_bytes", 0)
            if total > 0:
                stats["npu_memory_used_mb"] = round(used / (1024 * 1024), 1)
                stats["npu_memory_total_mb"] = round(total / (1024 * 1024), 1)
            # System-level resource metrics
            cpu_util = npu.get("cpu_utilization", -1)
            if cpu_util >= 0:
                stats["cpu_utilization"] = round(cpu_util * 100, 1)
            dsp_util = npu.get("dsp_utilization", -1)
            if dsp_util >= 0:
                stats["dsp_utilization"] = round(dsp_util * 100, 1)
            ram_total = npu.get("ram_total_kib", -1)
            ram_used = npu.get("ram_used_kib", -1)
            if ram_total > 0:
                stats["ram_total_mb"] = round(ram_total / 1024, 0)
                stats["ram_used_mb"] = round(ram_used / 1024, 0)
            for ms in npu.get("model_stats", []):
                if ms["model_id"] == self.current_model:
                    stats["model_qps"] = round(ms.get("current_qps", 0), 1)
                    avg_lat_us = ms.get("avg_latency_us", 0)
                    stats["model_avg_latency_ms"] = round(avg_lat_us / 1000.0, 1)
                    # NPU benchmark FPS: theoretical max from avg inference latency
                    if avg_lat_us > 0:
                        stats["npu_benchmark_fps"] = round(1_000_000.0 / avg_lat_us, 1)
                    # Queue depth and error count
                    stats["queue_depth"] = ms.get("queue_depth", 0)
                    stats["total_errors"] = ms.get("total_errors", 0)
                    break
        # Pre-computed benchmark FPS from model verification (available immediately)
        if "npu_benchmark_fps" not in stats or not stats.get("npu_benchmark_fps"):
            info = self._current_model_info
            precomputed = info.get("npu_benchmark_fps", 0) if info else 0
            if precomputed > 0:
                stats["npu_benchmark_fps"] = precomputed
        return stats

    def add_sse_subscriber(self, queue: Any) -> None:
        with self._sse_lock:
            self._sse_subscribers.append(queue)

    def remove_sse_subscriber(self, queue: Any) -> None:
        with self._sse_lock:
            try:
                self._sse_subscribers.remove(queue)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------
def _build_hd_preview(req) -> dict:
    """Build the HD 1080P preview config injected into the showcase template.

    Reuses the platform-api hardware H.264 stream
    (``wss://<lan-ip>/api/v1/h264/main``) decoded in the browser via MSE,
    so 1080P preview costs zero Python CPU. The host is taken from the request
    the browser used to reach showcase, so it is correct regardless of which LAN
    address was accessed. ``PLATFORM_API_WS_SCHEME``/``PLATFORM_API_PORT`` may
    override the default TLS endpoint for older non-TLS platform-api builds.
    ``PLATFORM_API_TOKEN`` is forwarded as ``?token=`` for platform-api auth.
    """
    host = (req.host or "").rsplit(":", 1)[0] or "127.0.0.1"
    scheme_env = os.environ.get("PLATFORM_API_WS_SCHEME", "").strip().lower()
    scheme = scheme_env
    if scheme in ("http", "https"):
        scheme = "wss" if scheme == "https" else "ws"
    if scheme not in ("ws", "wss"):
        tls_enabled = os.environ.get("PLATFORM_API_TLS", "1").strip().lower()
        scheme = "ws" if tls_enabled in ("0", "false", "no", "off") else "wss"

    default_port = "443" if scheme == "wss" else "8080"
    port = os.environ.get("PLATFORM_API_WS_PORT", "").strip()
    if not port:
        legacy_port = os.environ.get("PLATFORM_API_PORT", "").strip()
        if scheme == "wss" and not scheme_env and legacy_port == "8080":
            port = default_port
        else:
            port = legacy_port or default_port
    token = os.environ.get("PLATFORM_API_TOKEN", "")
    host_port = f"{host}:{port}" if port else host
    ws_url = f"{scheme}://{host_port}/api/v1/h264/main"
    if token:
        ws_url += "?token=" + urllib.parse.quote(token, safe="")
    enabled = os.environ.get("HD_PREVIEW_ENABLED", "1") != "0"
    return {"enabled": enabled, "wsUrl": ws_url, "token": token}


def create_app(showcase: ModelShowcase) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = _MAX_VIDEO_SIZE

    @app.route("/")
    def index():
        return render_template("index.html", hd_preview=_build_hd_preview(request))

    @app.route("/stream")
    def mjpeg_stream():
        def generate():
            last_id = 0
            try:
                while True:
                    frame_bytes, frame_id = showcase.frame_buffer.wait_for_new(last_id, timeout=2.0)
                    if frame_bytes is None:
                        continue
                    last_id = frame_id
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode()
                        + frame_bytes + b"\r\n"
                    )
            except GeneratorExit:
                pass
        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "X-Accel-Buffering": "no"},
        )

    sock = Sock(app)

    @sock.route("/ws/stream")
    def ws_stream(ws):
        logger.info("[ws/stream] client connected")
        sent = 0
        try:
            last_id = 0
            while True:
                frame_bytes, frame_id = showcase.frame_buffer.wait_for_new(last_id, timeout=2.0)
                if frame_bytes is None:
                    continue
                last_id = frame_id
                if frame_bytes and len(frame_bytes) > 0:
                    ws.send(frame_bytes)
                    sent += 1
                    if sent <= 3:
                        logger.info("[ws/stream] sent frame %d, %d bytes", sent, len(frame_bytes))
        except Exception as e:
            logger.info("[ws/stream] disconnected after %d frames: %s", sent, e)

    @app.route("/api/models", methods=["GET"])
    def get_models():
        # Live-filter AVAILABLE_MODELS (computed at startup) so a model file
        # removed at runtime can't be listed as both available and missing.
        models = [m for m in AVAILABLE_MODELS if _entry_available(m)]
        return jsonify({
            "models": models,
            "missing": _missing_models(),
            "current": showcase.current_model,
        })

    @app.route("/api/models/refresh", methods=["POST"])
    def refresh_models():
        """Re-scan the host model store after the operator provisions a file.

        Returns the updated models/missing lists plus the ids that just
        became available, so the UI can confirm the placement worked.
        """
        added = showcase.refresh_model_paths()
        return jsonify({
            "models": AVAILABLE_MODELS,
            "missing": _missing_models(),
            "current": showcase.current_model,
            "added": added,
        })

    @app.route("/api/model", methods=["POST"])
    def switch_model_route():
        data = request.get_json(force=True)
        model_id = data.get("model_id", "")
        if not showcase.switch_model(model_id):
            return jsonify({"error": f"unknown model: {model_id}"}), 400
        resp = {
            "current": showcase.current_model,
            "model_type": showcase.current_model_type,
        }
        benchmark = showcase._current_model_info.get("npu_benchmark_fps", 0)
        if benchmark > 0:
            resp["npu_benchmark_fps"] = benchmark
        return jsonify(resp)

    @app.route("/api/clip/labels", methods=["GET"])
    def get_clip_labels():
        return jsonify({"labels": list(showcase._clip_labels)})

    @app.route("/api/clip/labels", methods=["POST"])
    def update_clip_labels():
        data = request.get_json(force=True)
        labels = data.get("labels", [])
        if not isinstance(labels, list) or len(labels) == 0:
            return jsonify({"error": "labels must be a non-empty list"}), 400
        if len(labels) > 20:
            return jsonify({"error": "maximum 20 labels"}), 400
        cleaned = [l.strip()[:100] for l in labels if isinstance(l, str) and l.strip()]
        if not cleaned:
            return jsonify({"error": "no valid labels"}), 400
        showcase.update_clip_labels(cleaned)
        return jsonify({"labels": list(showcase._clip_labels)})

    @app.route("/api/stats", methods=["GET"])
    def get_stats():
        return jsonify(showcase.get_stats())

    # ── Video source API ──

    @app.route("/api/video/upload", methods=["POST"])
    def upload_video():
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        f = request.files["video"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400

        # Sanitize filename
        safe_name = "".join(c for c in f.filename if c.isalnum() or c in "._-")
        if not safe_name:
            safe_name = "upload.mp4"
        path = os.path.join(_VIDEO_DIR, safe_name)

        # Save with size limit
        f.save(path)
        size = os.path.getsize(path)
        if size > _MAX_VIDEO_SIZE:
            os.remove(path)
            return jsonify({"error": f"File too large ({size // 1024 // 1024}MB, max {_MAX_VIDEO_SIZE // 1024 // 1024}MB)"}), 400

        try:
            info = showcase.start_video(path)
            return jsonify(info)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/image/upload", methods=["POST"])
    def upload_image():
        """One-shot image inference: decode → infer → annotate → return JPEG.

        Returns the annotated image as `image/jpeg` with an `X-Infer-Stats`
        header carrying a JSON stats payload (model, latency, dims). The
        response is a single binary blob the frontend reads via `fetch().blob()`.
        """
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        f = request.files["image"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400

        # Extension allow-list (case-insensitive)
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in _ALLOWED_IMAGE_EXT:
            allowed = ", ".join(_ALLOWED_IMAGE_EXT)
            return jsonify(
                {"error": f"Unsupported image type '{ext}'. Allowed: {allowed}"}
            ), 400

        # Sanitize filename (same rule as video upload)
        safe_name = "".join(c for c in f.filename if c.isalnum() or c in "._-")
        if not safe_name:
            safe_name = "upload" + ext
        path = os.path.join(_IMAGE_DIR, safe_name)

        # Save with size guard
        f.save(path)
        size = os.path.getsize(path)
        if size > _MAX_IMAGE_SIZE:
            os.remove(path)
            return jsonify(
                {"error": f"File too large ({size // 1024 // 1024}MB, max {_MAX_IMAGE_SIZE // 1024 // 1024}MB)"}
            ), 400

        try:
            # imdecode (not imread) so non-ASCII paths don't break. Reads via
            # np.fromfile to honour the actual saved bytes.
            bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                return jsonify({"error": "Invalid or unsupported image file"}), 400
            info = showcase.infer_image(bgr)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.exception("image inference failed")
            return jsonify({"error": f"Inference failed: {e}"}), 500
        finally:
            # Uploaded stills are ephemeral; clean up regardless of outcome.
            try:
                os.remove(path)
            except OSError:
                pass

        stats = json.dumps({
            "model": info["model"],
            "model_type": info["model_type"],
            "infer_time_us": info["infer_time_us"],
            "hw_infer_time_us": info["hw_infer_time_us"],
            "width": info["width"],
            "height": info["height"],
        })
        resp = Response(info["jpeg"], mimetype="image/jpeg")
        resp.headers["X-Infer-Stats"] = stats
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp

    @app.route("/api/video/control", methods=["POST"])
    def video_control():
        data = request.get_json(force=True)
        action = data.get("action", "")
        if action not in ("play", "pause", "seek", "speed", "restart", "stop"):
            return jsonify({"error": "Invalid action"}), 400
        if action == "stop":
            showcase.stop_video()
            return jsonify({"active": False})
        result = showcase.video_control(
            action,
            position=data.get("position", 0),
            speed=data.get("speed", 1.0),
        )
        return jsonify(result)

    @app.route("/api/video/status", methods=["GET"])
    def video_status():
        return jsonify(showcase.get_video_status())

    @app.route("/api/events")
    def sse_events():
        import queue
        q = queue.Queue(maxsize=50)
        showcase.add_sse_subscriber(q)

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
                showcase.remove_sse_subscriber(q)
        return Response(
            generate(), mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Output mode (MJPEG / RTSP) ──────────────────────────────────────────

    @app.route("/api/output-mode", methods=["GET"])
    def get_output_mode():
        return jsonify(showcase.get_output_status())

    @app.route("/api/output-mode", methods=["POST"])
    def set_output_mode():
        data = request.get_json(force=True)
        mode = data.get("mode", "mjpeg")
        result = showcase.set_output_mode(mode)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)

    @app.route("/api/stream_info", methods=["GET"])
    def get_stream_info():
        """Return all available video stream URLs.

        Three stream types:
          mjpeg   — model-showcase MJPEG (AI overlay drawn in Python on CPU)
          rtsp    — model-showcase self-rendered RTSP via mediamtx (same CPU cost)
          camera  — camera-daemon hardware H264 with AI overlay drawn by C++ HAL
                    (zero Python CPU; requires Event Bus to be connected)

        The "camera" streams are always listed even when camera-daemon is not
        available so the frontend knows the expected URLs. The ``event_bus_enabled``
        flag indicates whether model-showcase is currently publishing to Event Bus.
        """
        status = showcase.get_output_status()
        host = request.host.split(":")[0]   # use the same host the browser hit

        # Rewrite localhost/127.0.0.1 URLs to use the actual request host so the
        # browser can reach them from a remote machine.
        def rehost(url: str) -> str:
            if not url:
                return url
            for local in ("127.0.0.1", "localhost"):
                url = url.replace(f"://{local}:", f"://{host}:")
            return url

        streams = {
            "mjpeg": {
                "url":         rehost(f"http://{host}:{showcase.web_port}/video_feed"),
                "type":        "mjpeg",
                "description": "MJPEG stream (AI overlay rendered in Python)",
                "active":      status["mode"] in ("mjpeg", "both"),
            },
            "rtsp_self": {
                "url":         rehost(status.get("rtsp_url") or ""),
                "type":        "rtsp",
                "description": "RTSP stream via mediamtx (AI overlay rendered in Python)",
                "active":      bool(status.get("rtsp_active")),
            },
            "camera_rtsp": {
                "url":         rehost(status["camera_rtsp_url"]),
                "type":        "rtsp",
                "description": "camera-daemon RTSP (H264 hardware encode, AI overlay in C++ HAL)",
                "active":      status["event_bus_enabled"],
            },
            "camera_hls": {
                "url":         rehost(status["camera_hls_url"]),
                "type":        "hls",
                "description": "camera-daemon HLS (low-latency, playable in browser <video> tag)",
                "active":      status["event_bus_enabled"],
            },
            "camera_webrtc": {
                "url":         rehost(status["camera_webrtc_url"]),
                "type":        "webrtc-whep",
                "description": "camera-daemon WebRTC WHEP (sub-500ms latency)",
                "active":      status["event_bus_enabled"],
            },
        }
        return jsonify({
            "streams": streams,
            "event_bus_enabled": status["event_bus_enabled"],
            "current_output_mode": status["mode"],
        })

    # ── GenAI endpoints ──────────────────────────────────────────────────────

    @app.route("/chat")
    def chat_page():
        return render_template("chat.html")

    @app.route("/api/genai/session", methods=["POST"])
    def genai_create_session():
        data = request.get_json(force=True)
        model_id = data.get("model_id", "")
        model_info = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
        if not model_info or model_info["type"] != "genai":
            return jsonify({"error": "not a genai model"}), 400

        with showcase._genai_lock:
            # Stop infer loop and unregister models to free NPU/KV-Cache
            showcase._stop_infer_thread()
            try:
                registered = showcase.infer_client.list_models()
                for m in registered:
                    try:
                        showcase.infer_client.unregister_model(m.model_id)
                        logger.info("Unregistered %s for GenAI session", m.model_id)
                    except Exception:
                        pass
            except Exception:
                pass

            # Destroy existing session if any
            if showcase._genai_session_id:
                try:
                    showcase.infer_client.genai_destroy_session(showcase._genai_session_id)
                except Exception:
                    pass
                showcase._genai_session_id = None

            try:
                sid = showcase.infer_client.genai_create_session(
                    hef_path=model_info["path"],
                    kind=model_info.get("kind", "llm"),
                    optimize_memory=model_info.get("optimize_memory", False),
                )
                showcase._genai_session_id = sid
                with showcase._model_lock:
                    showcase.current_model = model_id
                    showcase.current_model_type = "genai"
                    showcase._current_model_info = model_info
                return jsonify({"session_id": sid})
            except Exception as e:
                logger.error("GenAI session create failed: %s", e)
                return jsonify({"error": str(e)}), 500

    @app.route("/api/genai/stream", methods=["POST"])
    def genai_stream():
        data = request.get_json(force=True)
        messages = data.get("messages", [])
        stop_tokens = data.get("stop_tokens") or ["<|im_end|>", "<|endoftext|>"]
        temperature = data.get("temperature", 0.0)
        max_tokens = data.get("max_tokens", 512)
        image_b64 = data.get("image")
        if image_b64:
            logger.info("Received image_b64: len=%d, prefix=%s",
                        len(image_b64), image_b64[:40])

        if not showcase._genai_session_id:
            return jsonify({"error": "no active genai session"}), 400

        session_id = showcase._genai_session_id

        # Format messages as chat template JSON for HailoRT GenAI API.
        # Frontend sends raw text strings; wrap them as chat messages.
        # For VLM with images, content must use the structured format:
        #   {"role":"user","content":[{"type":"text","text":"..."},{"type":"image"}]}
        chat_messages = []
        has_image = image_b64 is not None
        for msg in messages:
            if isinstance(msg, str):
                if msg.startswith("["):
                    try:
                        chat_messages.extend(json.loads(msg))
                        continue
                    except (json.JSONDecodeError, TypeError):
                        pass
                if has_image:
                    chat_messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": msg or "Describe this image"},
                            {"type": "image"},
                        ],
                    })
                else:
                    chat_messages.append({"role": "user", "content": msg})
            elif isinstance(msg, dict):
                chat_messages.append(msg)
        messages_json = [json.dumps(m) for m in chat_messages]

        # Decode base64 image → RGB bytes for VLM
        # VLM expects frames at the model's input resolution (e.g. 288x512 for Qwen3-VL)
        image_frames = None
        if image_b64:
            import base64
            try:
                raw = base64.b64decode(image_b64.split(",")[-1] if "," in image_b64 else image_b64)
                import numpy as np, cv2
                arr = np.frombuffer(raw, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    # Resize to VLM expected input shape from model manifest
                    vlm_w = showcase._current_model_info.get("vlm_width", 512)
                    vlm_h = showcase._current_model_info.get("vlm_height", 288)
                    rgb = cv2.resize(rgb, (vlm_w, vlm_h), interpolation=cv2.INTER_LINEAR)
                    expected_bytes = vlm_w * vlm_h * 3
                    frame_bytes = rgb.tobytes()
                    logger.info("VLM image: %dx%d -> %dx%d, bytes=%d (expected=%d)",
                                bgr.shape[1], bgr.shape[0], vlm_w, vlm_h,
                                len(frame_bytes), expected_bytes)
                    image_frames = [frame_bytes]
            except Exception as e:
                logger.warning("Failed to decode image: %s", e)
            if not image_frames:
                logger.warning("VLM image decode failed, image_frames is None")

        logger.info("GenAI stream: msgs=%d, has_image=%s, frames=%s",
                     len(messages_json), has_image,
                     len(image_frames) if image_frames else 0)

        _stop_tokens = set(stop_tokens)

        def generate():
            try:
                token_count = 0
                for token in showcase.infer_client.genai_generate(
                    session_id=session_id,
                    messages=messages_json,
                    images=image_frames,
                    stop_tokens=stop_tokens,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    do_sample=temperature > 0,
                ):
                    if token in _stop_tokens:
                        break
                    token_count += 1
                    yield f"data: {json.dumps({'token': token})}\n\n"
                logger.info("GenAI stream done: %d tokens", token_count)
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                logger.error("GenAI stream error: %s", e)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            generate(), mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/genai/abort", methods=["POST"])
    def genai_abort():
        if showcase._genai_session_id:
            try:
                showcase.infer_client.genai_abort(showcase._genai_session_id)
            except Exception:
                pass
        return jsonify({"ok": True})

    @app.route("/api/health", methods=["GET"])
    def health():
        status = "degraded" if showcase._degraded else "ok"
        result: Dict[str, Any] = {"status": status, "app_id": Config.get_app_id()}
        if showcase._degrade_reason:
            result["reason"] = showcase._degrade_reason
        return jsonify(result)

    @app.route("/api/debug/landmarks", methods=["GET"])
    def debug_landmarks():
        """Return raw face landmark data for debugging coordinate alignment."""
        if not showcase.media_client:
            return jsonify({"error": "no media client"}), 503

    @app.route("/api/debug/seg", methods=["GET"])
    def debug_seg():
        """Dump the structure of the latest segmentation inference result."""
        with showcase._result_lock:
            result = showcase._latest_result
        if result is None:
            return jsonify({"error": "no inference result yet"}), 503
        info: Dict[str, Any] = {
            "model_id": showcase.current_model,
            "model_type": showcase.current_model_type,
        }
        # Masks from HAL
        masks = getattr(result, "masks", None)
        info["masks_count"] = len(masks) if masks else 0
        if masks:
            for i, m in enumerate(masks[:5]):
                info[f"mask_{i}"] = {
                    "class_id": m.class_id,
                    "confidence": getattr(m, "confidence", 0),
                    "width": m.mask_width,
                    "height": m.mask_height,
                    "rle_len": len(m.mask_rle) if m.mask_rle else 0,
                }
        # Raw output tensors
        raw = getattr(result, "raw_outputs", None)
        info["raw_outputs_count"] = len(raw) if raw else 0
        if raw:
            for i, t in enumerate(raw[:8]):
                arr = np.asarray(t)
                info[f"raw_{i}"] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
        info["infer_time_us"] = getattr(result, "infer_time_us", 0)
        return jsonify(info)
        frame = showcase.media_client.get_frame(timeout_ms=2000)
        if not frame:
            return jsonify({"error": "no frame"}), 503
        bgr = showcase.frame_to_bgr(frame)
        fh, fw = bgr.shape[:2]

        # Stage 1: detect faces
        det_info = next((m for m in MODEL_CATALOG if m["id"] == "yolov8n_detection"), None)
        if not det_info:
            return jsonify({"error": "no yolov8n model"}), 500
        det_input = showcase._prepare_stage_input(bgr, det_info)
        det_result = showcase.infer_client.infer(det_input, model_id="yolov8n_detection", timeout_ms=3000)

        face_det = None
        if det_result.objects:
            for obj in det_result.objects:
                if getattr(obj, "label", "") == "face":
                    face_det = obj
                    break
            if not face_det:
                for obj in det_result.objects:
                    face_det = obj
                    break

        if not face_det:
            return jsonify({"error": "no face detected", "objects": len(det_result.objects)})

        # Get face crop bounds
        bx = float(face_det.bbox.x)
        by = float(face_det.bbox.y)
        bw = float(face_det.bbox.width)
        bh = float(face_det.bbox.height)
        label = getattr(face_det, "label", "")
        margin = 0.15
        face_x1 = max(0.0, bx - bw * margin)
        face_x2 = min(1.0, bx + bw * (1 + margin))
        face_y1 = max(0.0, by - bh * margin)
        face_y2 = min(1.0, by + bh * (1 + margin))

        px1, py1 = int(face_x1 * fw), int(face_y1 * fh)
        px2, py2 = int(face_x2 * fw), int(face_y2 * fh)
        crop = bgr[py1:py2, px1:px2]

        # Stage 2: face_landmarks
        lm_info = showcase._current_model_info
        tw, th = lm_info["input_width"], lm_info["input_height"]
        resized = cv2.resize(crop, (tw, th))
        lm_input = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).flatten()
        lm_result = showcase.infer_client.infer(lm_input, model_id="face_landmarks", timeout_ms=3000)

        raw = getattr(lm_result, "raw_outputs", None)
        if not raw or len(raw) < 1:
            return jsonify({"error": "no raw outputs"})

        raw_tensor = np.asarray(raw[0], dtype=np.float32).flatten()
        raw_min, raw_max = float(raw_tensor.min()), float(raw_tensor.max())
        raw_sample = [float(v) for v in raw_tensor[:30]]

        if raw_max > 2.0:
            norm_tensor = raw_tensor / 255.0
        else:
            norm_tensor = raw_tensor

        n_pts = norm_tensor.size // 3
        points = norm_tensor[:n_pts * 3].reshape(n_pts, 3)

        # Also try planar format: [x0..xN, y0..yN, z0..zN]
        points_planar = norm_tensor[:n_pts * 3].reshape(3, n_pts).T

        # Key MediaPipe indices
        nose_tip = 4
        left_eye_center = 159
        right_eye_center = 386
        lip_center = 13

        def pt_info(pts, idx):
            if idx < len(pts):
                x, y = float(pts[idx, 0]), float(pts[idx, 1])
                fx = face_x1 + x * (face_x2 - face_x1)
                fy = face_y1 + y * (face_y2 - face_y1)
                return {"raw": [x, y], "frame": [fx, fy], "px": [int(fx * fw), int(fy * fh)]}
            return None

        return jsonify({
            "frame_size": [fw, fh],
            "face_detection": {
                "label": label,
                "bbox": [bx, by, bw, bh],
                "crop_norm": [face_x1, face_y1, face_x2, face_y2],
                "crop_px": [px1, py1, px2, py2],
                "crop_size": [px2 - px1, py2 - py1],
            },
            "model_input": [tw, th],
            "raw_range": [raw_min, raw_max],
            "raw_sample": raw_sample,
            "n_pts": n_pts,
            "interleaved": {
                "nose_tip": pt_info(points, nose_tip),
                "left_eye": pt_info(points, left_eye_center),
                "right_eye": pt_info(points, right_eye_center),
                "lip_center": pt_info(points, lip_center),
                "x_range": [float(points[:, 0].min()), float(points[:, 0].max())],
                "y_range": [float(points[:, 1].min()), float(points[:, 1].max())],
                "z_range": [float(points[:, 2].min()), float(points[:, 2].max())],
            },
            "planar": {
                "nose_tip": pt_info(points_planar, nose_tip),
                "left_eye": pt_info(points_planar, left_eye_center),
                "right_eye": pt_info(points_planar, right_eye_center),
                "lip_center": pt_info(points_planar, lip_center),
                "x_range": [float(points_planar[:, 0].min()), float(points_planar[:, 0].max())],
                "y_range": [float(points_planar[:, 1].min()), float(points_planar[:, 1].max())],
                "z_range": [float(points_planar[:, 2].min()), float(points_planar[:, 2].max())],
            },
        })

    # ── Gallery / CLIP text search ────────────────────────────────────────

    @app.route("/api/search", methods=["POST"])
    def search():
        data = request.get_json(force=True)
        query = data.get("query", "").strip()
        top_k = min(int(data.get("top_k", 20)), 100)
        if not query:
            return jsonify({"error": "empty query"}), 400
        if not showcase.infer_client or not showcase.infer_client.connected:
            return jsonify({"error": "inference client not connected"}), 503
        try:
            embedding = showcase.infer_client.encode_text(query, timeout_ms=10000)
        except Exception as e:
            return jsonify({"error": f"encode_text failed: {e}"}), 500
        results = showcase.gallery.search(embedding, top_k=top_k)
        return jsonify({"query": query, "results": results})

    @app.route("/api/gallery", methods=["GET"])
    def gallery_list():
        page = max(int(request.args.get("page", 1)), 1)
        page_size = min(int(request.args.get("page_size", 50)), 200)
        return jsonify(showcase.gallery.get_page(page, page_size))

    @app.route("/api/gallery/stats", methods=["GET"])
    def gallery_stats():
        return jsonify(showcase.gallery.stats())

    @app.route("/api/gallery/capture", methods=["POST"])
    def gallery_capture():
        data = request.get_json(force=True) if request.is_json else {}
        enabled = data.get("enabled", not showcase.gallery.enabled)
        interval = float(data.get("interval", showcase.gallery.capture_interval))
        showcase.gallery.enabled = enabled
        showcase.gallery.capture_interval = max(interval, 1.0)
        return jsonify({
            "enabled": showcase.gallery.enabled,
            "interval": showcase.gallery.capture_interval,
        })

    @app.route("/api/gallery/<int:img_id>/image", methods=["GET"])
    def gallery_image(img_id: int):
        path = showcase.gallery.get_image_path(img_id)
        if not path or not os.path.exists(path):
            return jsonify({"error": "not found"}), 404
        return Response(open(path, "rb").read(), mimetype="image/jpeg")

    @app.route("/api/gallery", methods=["DELETE"])
    def gallery_delete():
        older_than = int(request.args.get("older_than_seconds", 86400))
        removed = showcase.gallery.delete_older_than(older_than)
        return jsonify({"removed": removed})

    # ── Lens control (SDK → device-control gRPC) ─────────────────────────────

    @app.route("/api/lens/status", methods=["GET"])
    def lens_status():
        try:
            with showcase._lens_lock:
                return jsonify(showcase.device_client.get_lens_status())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/lens/zoom", methods=["POST"])
    def lens_zoom():
        data = request.get_json(force=True)
        level = max(0.0, min(1.0, float(data.get("level", 0))))
        try:
            with showcase._lens_lock:
                showcase.device_client.set_zoom_level(level)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/lens/focus", methods=["POST"])
    def lens_focus():
        data = request.get_json(force=True)
        level = max(0.0, min(1.0, float(data.get("level", 0))))
        try:
            with showcase._lens_lock:
                showcase.device_client.focus_auto(False)
                showcase.device_client.set_focus_level(level)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/lens/zoom-speed", methods=["POST"])
    def lens_zoom_speed():
        data = request.get_json(force=True)
        speed = int(data.get("speed", 0))
        try:
            with showcase._lens_lock:
                showcase.device_client.zoom(speed)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/lens/focus-speed", methods=["POST"])
    def lens_focus_speed():
        data = request.get_json(force=True)
        speed = int(data.get("speed", 0))
        try:
            with showcase._lens_lock:
                showcase.device_client.focus(speed)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/lens/oneshot-af", methods=["POST"])
    def lens_oneshot_af():
        try:
            with showcase._lens_lock:
                showcase.device_client.focus_auto(False)
                showcase.device_client.focus_auto(True)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/lens/reset", methods=["POST"])
    def lens_reset():
        data = request.get_json(force=True) if request.is_json else {}
        zoom = bool(data.get("zoom", True))
        focus = bool(data.get("focus", True))
        try:
            with showcase._lens_lock:
                showcase.device_client.lens_reset_zero(zoom=zoom, focus=focus)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    def _crash_handler(exc_type, exc_value, exc_tb):
        import traceback
        with open("/tmp/crash.log", "w") as f:
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
        logger.error("Unhandled exception: %s", exc_value)
    sys.excepthook = _crash_handler

    app_id = Config.get_app_id()
    logger.info("=" * 60)
    logger.info("  AI Model Showcase v1.1.0")
    logger.info("  App ID: %s", app_id)
    logger.info("  Web Port: %d", int(os.environ.get("WEB_PORT", "8889")))
    logger.info("  Output Mode: %s", os.environ.get("OUTPUT_MODE", "mjpeg"))
    logger.info("  Models: %d in catalog", len(MODEL_CATALOG))
    logger.info("=" * 60)

    showcase = ModelShowcase()

    def _signal_handler(signum: int, frame: Any) -> None:
        logger.info("Received signal %d, shutting down...", signum)
        showcase.stop()
        os._exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if not showcase.connect():
        logger.error("Failed to connect to services, exiting")
        sys.exit(1)

    # Start RTSP server if configured
    showcase._start_rtsp_server()

    # Register web URL with platform so the console shows a "Visit App" button
    try:
        with AppClient() as app_client:
            app_client.register_web_url("/")
        logger.info("Registered web_url with platform")
    except Exception as e:
        logger.warning("Failed to register web_url: %s (non-fatal)", e)

    app = create_app(showcase)
    frame_thread = threading.Thread(target=showcase.run_frame_loop, daemon=True)
    frame_thread.start()

    logger.info("Starting web server on port %d...", showcase.web_port)
    try:
        app.run(host="0.0.0.0", port=showcase.web_port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        showcase.stop()
        logger.info("Model Showcase stopped")


if __name__ == "__main__":
    main()
