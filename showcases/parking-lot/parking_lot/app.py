"""
Main application logic for the Parking Lot pipeline.

Contains ParkingLotApp (media capture, model orchestration, overlay
rendering) and the top-level entry point.
"""

import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.error
import uuid
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from neoruntime_ipc_sdk import (
    BatchInferItem,
    Config,
    DspClient,
    DspError,
    EventClient,
    FdMediaClient,
    Frame,
    InferenceClient,
)

from .config import (
    _COCO_VEHICLE_CLASSES,
    DEGRADED_FAILURE_THRESHOLD,
    DEPTH_SPOOF_THRESHOLD,
    DSP_ENABLED,
    DSP_QUOTA_COOLDOWN_S,
    INFER_TIMEOUT_MS,
    INFER_WARMUP_COUNT,
    INFER_WARMUP_TIMEOUT_MS,
    LPR_CHARSET,
    LPR_CTC_BLANK,
    MODEL_DEFS,
    PLATFORM_API_TOKEN,
    PLATFORM_API_URL,
    STALL_BURST_INFO_THRESHOLD,
    STREAM_ID,
    TARGET_FPS,
    VEHICLE_MODEL,
    WEB_PORT,
)
from .postprocess import (
    PlateDetection,
    PlateSnapshot,
    PipelineResult,
    PlateAccumulator,
    SpoofAlert,
    SpoofResult,
    VehicleDetection,
    analyze_depth_spoof,
    decode_recognition,
    letterbox_crop,
    parse_nms_raw,
    parse_yolo_grid,
    plate_crop_rects,
    prepare_input,
)
from .web import FrameBuffer, create_flask_app, sse_broadcast

logger = logging.getLogger("parking-lot")


# ---------------------------------------------------------------------------
# VideoFrameSource — seekable video frame source with speed/pause control
# ---------------------------------------------------------------------------


class VideoFrameSource:
    """Reads frames from a video file with seek, speed and pause control.

    Borrowed from model-showcase's VideoFrameSource (main.py:233-289).
    """

    def __init__(self, path: str) -> None:
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration: float = self.total_frames / self.fps if self.fps > 0 else 0.0
        self.paused: bool = False
        self.speed: float = 1.0
        self._current_frame: Optional[np.ndarray] = None
        self._frame_idx: int = 0

    def get_bgr_frame(self) -> Optional[np.ndarray]:
        """Return next BGR frame, or ``None`` on EOF."""
        if self.paused:
            return self._current_frame

        # Speed > 1: skip intermediate frames via grab()
        skip = max(1, int(self.speed)) - 1
        for _ in range(skip):
            if not self.cap.grab():
                return None
            self._frame_idx += 1

        ret, frame = self.cap.read()
        if not ret:
            return None
        self._frame_idx += 1
        self._current_frame = frame
        return frame

    def seek(self, position_sec: float) -> None:
        frame_no = int(position_sec * self.fps)
        frame_no = max(0, min(frame_no, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        self._frame_idx = frame_no

    @property
    def position_sec(self) -> float:
        return self._frame_idx / self.fps if self.fps > 0 else 0.0

    @property
    def progress(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return self._frame_idx / self.total_frames

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ---------------------------------------------------------------------------
# ParkingLotApp
# ---------------------------------------------------------------------------


class ParkingLotApp:
    """Orchestrates the multi-model parking lot pipeline."""

    def __init__(self) -> None:
        self.infer_client = InferenceClient()
        self.event_client = EventClient()
        # Dual media clients (ported from model-showcase/main.py:1417-1418):
        # media_preview feeds the MJPEG capture loop on STREAM_ID (e.g. main
        # 1080p); media_infer fetches NV12 frames for the inference loop on
        # _active_infer_stream, independently reconnectable and selectable.
        # media_client kept as a backward-compatible alias == media_preview.
        self.media_preview: Optional[FdMediaClient] = None
        self.media_infer: Optional[FdMediaClient] = None

        self.stream_id = STREAM_ID
        # Stream the inference loop fetches from (chosen by _choose_infer_stream
        # at startup, falls back to STREAM_ID).  Preview always uses stream_id.
        self._active_infer_stream: str = STREAM_ID
        self.target_fps = TARGET_FPS
        self.depth_spoof_threshold = DEPTH_SPOOF_THRESHOLD
        self.plate_accumulator = PlateAccumulator(window_size=5, min_confidence=0.7)

        self._running = False
        self._infer_count = 0

        # Inference timeout tiering + degraded state machine
        # (ported from model-showcase/main.py:1347,1355,2956-3062).
        # _infer_count_per_model tracks warmup per model; _consecutive_failures
        # /_degraded are read/written only by the inference thread (no lock).
        self._infer_count_per_model: Dict[str, int] = {}
        self._consecutive_failures: int = 0
        self._degraded: bool = False

        # InferBatch capability flag — probed once on first frame.  When the
        # ai-runtime binary lacks the InferBatch RPC (returns UNIMPLEMENTED),
        # we permanently switch to sequential infer() calls.  Repeatedly
        # issuing a failing batch RPC every frame wastes a round-trip and can
        # leave the shared gRPC channel in a degraded state that eventually
        # hangs the fallback infer() calls.
        self._batch_enabled: bool = True

        # Web UI
        self.frame_buffer = FrameBuffer()
        self._last_stats: Dict[str, Any] = {}
        self._alert_history: List[Dict[str, Any]] = []
        self._alert_lock = threading.Lock()

        # Plate snapshot gallery (bounded list of cropped images)
        self._plate_snapshots: List[PlateSnapshot] = []
        self._plate_snapshot_lock = threading.Lock()
        self._plate_snapshot_max = 25

        # Upload mode
        self._mode: str = "live"
        self._video_source: Optional[VideoFrameSource] = None
        self._upload_progress: Dict[str, Any] = {
            "current": 0,
            "total": 0,
            "status": "idle",
            "source": "",
            "result_file": None,
        }
        self._upload_lock = threading.Lock()
        self._result_video_path: Optional[str] = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        # Guards _mode / _video_source / _video_writer / _result_video_path
        # across the Flask request threads (switch_*, video_control) and the
        # background _infer_loop / _capture_loop. RLock so a holder may call
        # other locked helpers (e.g. video_control -> switch_to_live) without
        # self-deadlock. Lock order is always _video_lock -> _upload_lock.
        self._video_lock = threading.RLock()

        # Platform stats cache (NPU / model stats from platform API)
        self._platform_stats: Optional[Dict[str, Any]] = None
        self._platform_stats_time: float = 0.0

        # Async pipeline: dual-thread architecture.
        # NOTE: the inference thread fetches its own frames via media_infer
        # (decoupled from the preview client), so there is no longer a
        # capture→infer frame handoff buffer.
        self._latest_result: Optional[PipelineResult] = None
        self._result_lock = threading.Lock()

        self._capture_thread: Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None

        # Per-thread FPS tracking
        self._stream_fps: float = 0.0
        self._stream_fps_ts: float = 0.0
        self._stream_fps_count: int = 0
        self._infer_fps: float = 0.0
        self._infer_fps_ts: float = 0.0
        self._infer_fps_count: int = 0
        # Last Phase-1 NPU-only latency sum (microseconds), per InferResponse.hw_infer_time_us
        self._last_hw_infer_us: int = 0

        # DSP hardware offload (keep_fd + resize_hw/multi_crop_hw).  The
        # client is created lazily on first use (devices without the DSP
        # service disable the path on the first failed job); _dsp_ok is
        # flipped False on fatal errors, _dsp_retry_ts holds a quota
        # cooldown deadline.  Counters feed /api/stats["dsp"] for A/B runs.
        self._dsp_client: Optional[DspClient] = None
        self._dsp_enabled: bool = DSP_ENABLED
        self._dsp_ok: bool = True
        self._dsp_retry_ts: float = 0.0
        self._dsp_stats: Dict[str, Any] = {
            "enabled": DSP_ENABLED, "hw_jobs": 0, "cpu_fallbacks": 0,
            "last_error": "",
        }

    def register_models(self) -> None:
        """Register all required models with the inference service."""
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            registered: set = set()
            try:
                registered = {m.model_id for m in self.infer_client.list_models()}
            except Exception:
                pass

            all_ok = True
            for model_id, mdef in MODEL_DEFS.items():
                variant = mdef.get("variant")
                # app-manager's PreloadModels registers manifest-declared
                # models at app start using variant from platform.db — which
                # is empty for yolov5m_vehicles, so ai-runtime falls back to
                # the default hailo_yolov8n backend (wrong tensor name →
                # postprocess throws). Re-registering with our variant blob
                # makes ai-runtime re-init the postprocess session with the
                # matching backend_function (init_post_process replaces the
                # existing session). Models without a variant have nothing
                # to re-apply, so they keep the cheap skip.
                if model_id in registered and not variant:
                    logger.info("Model %s already registered", model_id)
                    continue
                try:
                    reg_type = mdef.get("register_type", mdef["type"])
                    self.infer_client.register_model(
                        model_path=mdef["path"],
                        model_id=model_id,
                        owner_id=Config.get_app_id(),
                        model_type=reg_type,
                        model_variant=variant,
                    )
                    logger.info(
                        "Registered model: %s (type=%s, variant=%s)",
                        model_id, reg_type or "raw", variant or "-",
                    )
                except Exception as e:
                    logger.error(
                        "Failed to register %s (attempt %d/%d): %s",
                        model_id, attempt, max_retries, e,
                    )
                    all_ok = False

            if all_ok:
                return
            logger.info("Retrying model registration in %ds...", attempt * 2)
            time.sleep(attempt * 2)

        logger.error(
            "Model registration failed after %d attempts, pipeline may not work",
            max_retries,
        )

    @staticmethod
    def frame_to_bgr(frame: Frame) -> np.ndarray:
        """Convert a Frame from the media client to BGR numpy array."""
        if frame.format == "BGR":
            return frame.image
        if frame.format == "RGB":
            return frame.image[:, :, ::-1]
        if frame.format == "NV12":
            return cv2.cvtColor(frame.image, cv2.COLOR_YUV2BGR_NV12)
        if frame.format == "GRAY8":
            return cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
        img = frame.image
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)

    # ------------------------------------------------------------------
    # NV12-direct zero-copy input path (ported from model-showcase/main.py:
    # 2193-2250).  Avoids the NV12->BGR->RGB/NV12 double conversion for
    # whole-frame model inputs.  Plate crops still use BGR (letterbox_crop).
    # ------------------------------------------------------------------

    @staticmethod
    def _bgr_to_nv12(bgr: np.ndarray) -> np.ndarray:
        """Convert BGR to NV12 (semi-planar) via I420 intermediation."""
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
        """Resize NV12 by resizing Y and UV planes separately.

        Avoids NV12->BGR->resize->BGR->NV12 round-trip.
        """
        y_plane = nv12[:src_h, :]
        uv_plane = nv12[src_h:src_h + src_h // 2, :]
        y_out = cv2.resize(y_plane, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
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
            bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
            if src_w != target_w or src_h != target_h:
                bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return self._bgr_to_nv12(bgr).flatten()

    # ------------------------------------------------------------------
    # DSP hardware offload (DspClient).  Whole-frame model-input scaling
    # (resize_hw) and the plate letterbox tiles (multi_crop_hw with
    # native letterbox scaling) run as zero-copy jobs sourced from the
    # keep-fd frame.  Every failure falls back to the CPU paths above —
    # quota errors cool down and retry, everything else disables the
    # path for the run (one warning each).
    # ------------------------------------------------------------------

    def _dsp_active(self) -> bool:
        """True when the DSP path should be attempted for this frame."""
        return (self._dsp_enabled and self._dsp_ok
                and time.monotonic() >= self._dsp_retry_ts)

    def _get_dsp(self) -> Optional[DspClient]:
        """Lazily create the shared DspClient; None when unusable."""
        if not self._dsp_active():
            return None
        if self._dsp_client is None:
            try:
                self._dsp_client = DspClient()
            except Exception as exc:
                self._dsp_fallback(exc, "client-init")
                self._dsp_ok = False
                return None
        return self._dsp_client

    def _dsp_fallback(self, exc: BaseException, scope: str) -> None:
        """Record a DSP failure and choose the CPU-fallback policy.

        Quota errors (daemon code -3) set a 10s cooldown after which the
        DSP path retries; any other error disables it for the run.  Both
        log exactly once per transition.
        """
        self._dsp_stats["cpu_fallbacks"] += 1
        self._dsp_stats["last_error"] = f"{scope}: {exc}"
        if getattr(exc, "code", None) == -3 or "quota" in str(exc).lower():
            self._dsp_retry_ts = time.monotonic() + DSP_QUOTA_COOLDOWN_S
            logger.warning(
                "DSP quota exceeded (%s), CPU fallback for %.0fs then retry",
                scope, DSP_QUOTA_COOLDOWN_S,
            )
        else:
            self._dsp_ok = False
            logger.warning(
                "DSP %s failed (%s); using CPU pipeline for this run", scope, exc,
            )

    def _dsp_model_input(self, src: Any, mdef: Dict[str, Any]) -> Optional[np.ndarray]:
        """Whole-frame model input via DSP resize, or None to stay on CPU.

        Only handles NV12 keep-fd sources whose geometry differs from the
        model input (same-size inputs skip the DSP entirely — a plain
        cvtColor on the CPU is cheaper than a job plus quota).
        """
        if getattr(src, "format", "").upper() != "NV12":
            return None
        tw, th = mdef["input_width"], mdef["input_height"]
        w = getattr(src, "width", 0) or 0
        h = getattr(src, "height", 0) or 0
        if not w or not h or (w == tw and h == th):
            return None
        dsp = self._get_dsp()
        if dsp is None:
            return None
        try:
            # stretch matches the CPU cv2.resize semantics used before
            small = dsp.resize_hw(src, tw, th)
        except DspError as exc:
            self._dsp_fallback(exc, "resize")
            return None
        self._dsp_stats["hw_jobs"] += 1
        if mdef["input_format"] == "nv12":
            return small.flatten()
        code = (cv2.COLOR_YUV2RGB_NV12 if mdef["input_format"] == "rgb"
                else cv2.COLOR_YUV2BGR_NV12)
        return cv2.cvtColor(small, code).flatten()

    def _dsp_plate_tiles(
        self, src: Any, bboxes: List[Tuple[float, float, float, float]],
    ) -> Optional[List[Optional[np.ndarray]]]:
        """Plate letterbox tiles via one DSP multi-crop job, or None for CPU.

        The daemon's letterbox scaling centers the scaled crop (same
        semantics as letterbox_crop's centered paste), so every tile is
        produced by a single multi_crop_hw job in rect order.  Degenerate
        boxes keep their ``None`` slot and the caller substitutes a filled
        canvas, exactly like the CPU path.
        """
        mdef = MODEL_DEFS["plate_recognition"]
        if mdef["input_format"] != "nv12":
            return None
        if getattr(src, "format", "").upper() != "NV12":
            return None
        w = getattr(src, "width", 0) or 0
        h = getattr(src, "height", 0) or 0
        if not w or not h:
            return None
        rects = plate_crop_rects(
            bboxes, w, h, mdef["input_width"], mdef["input_height"],
        )
        idx_map = [i for i, r in enumerate(rects) if r is not None]
        if not idx_map:
            return None
        dsp = self._get_dsp()
        if dsp is None:
            return None
        try:
            tiles = dsp.multi_crop_hw(
                src, [rects[i] for i in idx_map], scaling="letterbox",
            )
        except DspError as exc:
            self._dsp_fallback(exc, "plate_tiles")
            return None
        self._dsp_stats["hw_jobs"] += 1
        scattered: List[Optional[np.ndarray]] = [None] * len(bboxes)
        for slot, tile in zip(idx_map, tiles):
            scattered[slot] = tile
        return scattered

    # ------------------------------------------------------------------
    # Inference timeout tiering + degraded state machine
    # (ported from model-showcase/main.py:1407-1415, 2956-3062).
    # ------------------------------------------------------------------

    def _infer_timeout_ms(self, model_id: str, steady_ms: Optional[int] = None) -> int:
        """Return the infer timeout for this model, widening during warmup.

        First INFER_WARMUP_COUNT calls per model return INFER_WARMUP_TIMEOUT_MS
        (HEF cold-start); subsequent calls return ``steady_ms`` if given else
        INFER_TIMEOUT_MS.  Call-side effect: increments the per-model counter.
        """
        cnt = self._infer_count_per_model.get(model_id, 0)
        self._infer_count_per_model[model_id] = cnt + 1
        if cnt < INFER_WARMUP_COUNT:
            return INFER_WARMUP_TIMEOUT_MS
        return steady_ms if steady_ms is not None else INFER_TIMEOUT_MS

    def _register_infer_success(self) -> None:
        """Reset failure state on a successful inference; log recovery."""
        prev = self._consecutive_failures
        self._consecutive_failures = 0
        if self._degraded:
            self._degraded = False
            logger.info("Inference recovered from degraded state")
        elif prev >= STALL_BURST_INFO_THRESHOLD:
            logger.info("Inference recovered from burst of %d failures", prev)

    def _register_infer_failure(self, exc: BaseException) -> None:
        """Count a failure; mark degraded past DEGRADED_FAILURE_THRESHOLD."""
        self._consecutive_failures += 1
        n = self._consecutive_failures
        if n >= DEGRADED_FAILURE_THRESHOLD and not self._degraded:
            self._degraded = True
            logger.warning("Inference degraded after %d consecutive failures: %s", n, exc)
        elif n == 1:
            logger.warning("Pipeline error: %s", exc)
        else:
            logger.debug("Pipeline error (#%d): %s", n, exc)

    # ------------------------------------------------------------------
    # Media reconnect with exponential backoff
    # (ported from model-showcase/main.py:2161-2191).  3 retries, 1s/2s/3s.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Stream selection by model resolution
    # (ported from model-showcase/main.py:1541-1558, best-effort).
    # ------------------------------------------------------------------

    def _choose_infer_stream(self) -> str:
        """Pick the inference stream based on the largest model input dims.

        Uses media_infer.list_streams() to discover available streams; chooses
        the smallest stream whose frame is large enough for the largest model
        input.  Falls back to STREAM_ID when streams can't be enumerated or no
        stream fits (best-effort, never raises).
        """
        try:
            available = list(self.media_infer.list_streams()) if self.media_infer else []
        except Exception as e:
            logger.info("Cannot list streams (%s); infer stream = %s", e, STREAM_ID)
            return STREAM_ID
        if not available:
            return STREAM_ID

        max_w = max((m.get("input_width", 0) for m in MODEL_DEFS.values()), default=0)
        max_h = max((m.get("input_height", 0) for m in MODEL_DEFS.values()), default=0)

        # Probe each stream's frame size with a short timeout; keep those that
        # fit the largest model input.  Prefer STREAM_ID when it qualifies.
        candidates: List[Tuple[int, str]] = []
        for sid in available:
            try:
                fr = self.media_infer.get_frame(sid, timeout_ms=1000)
            except Exception:
                continue
            if fr is None:
                continue
            fw = int(getattr(fr, "width", 0) or 0)
            fh = int(getattr(fr, "height", 0) or 0)
            if fw >= max_w and fh >= max_h:
                candidates.append((fw * fh, sid))
        if not candidates:
            logger.info(
                "No stream fits largest model input (%dx%d); infer stream = %s",
                max_w, max_h, STREAM_ID,
            )
            return STREAM_ID
        # Prefer STREAM_ID if it qualified; else the smallest fitting stream.
        if STREAM_ID in (s for _, s in candidates):
            chosen = STREAM_ID
        else:
            chosen = min(candidates, key=lambda c: c[0])[1]
        logger.info("Infer stream selected: %s (available=%s)", chosen, available)
        return chosen

    def _detect_vehicles(self, bgr: np.ndarray) -> List[VehicleDetection]:
        model_id = VEHICLE_MODEL
        mdef = MODEL_DEFS[model_id]
        inp = prepare_input(bgr, mdef["input_width"], mdef["input_height"], mdef["input_format"])
        result = self.infer_client.infer(inp, model_id=model_id, timeout_ms=5000)
        if result.objects:
            vehicles = []
            for obj in result.objects:
                cls_name = _COCO_VEHICLE_CLASSES.get(getattr(obj, "class_id", 0), "vehicle")
                vehicles.append(VehicleDetection(
                    bbox=(float(obj.bbox.x), float(obj.bbox.y),
                          float(obj.bbox.width), float(obj.bbox.height)),
                    class_id=getattr(obj, "class_id", 0),
                    class_name=cls_name,
                    confidence=float(obj.score),
                ))
            return vehicles
        if getattr(result, "raw_outputs", None):
            return parse_nms_raw(result)
        return []

    def _estimate_depth(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        mdef = MODEL_DEFS["scdepthv3"]
        inp = prepare_input(bgr, mdef["input_width"], mdef["input_height"], mdef["input_format"])
        result = self.infer_client.infer(inp, model_id="scdepthv3", timeout_ms=5000)
        if getattr(result, "depth_maps", None) and result.depth_maps:
            return result.depth_maps[0].data
        if getattr(result, "raw_outputs", None) and result.raw_outputs:
            raw = np.asarray(result.raw_outputs[0], dtype=np.float32)
            if raw.max() > 2.0:
                raw = raw / 255.0
            h = int(len(raw) / mdef["input_width"])
            return raw.reshape(h, mdef["input_width"])
        return None

    def _detect_plates(self, bgr: np.ndarray) -> List[Tuple[float, float, float, float]]:
        mdef = MODEL_DEFS["license_plate_det"]
        inp = prepare_input(bgr, mdef["input_width"], mdef["input_height"], mdef["input_format"])
        result = self.infer_client.infer(inp, model_id="license_plate_det", timeout_ms=5000)
        if getattr(result, "objects", None) and result.objects:
            return [
                (float(obj.bbox.x), float(obj.bbox.y),
                 float(obj.bbox.width), float(obj.bbox.height))
                for obj in result.objects
            ]
        if getattr(result, "raw_outputs", None):
            return parse_yolo_grid(result, "license_plate_det")
        return []

    def _prepare_plate_input(self, bgr: np.ndarray,
                             bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Crop a plate region and convert to the recognition model's input."""
        x, y, w, h = bbox
        mdef = MODEL_DEFS["plate_recognition"]
        crop = letterbox_crop(bgr, x, y, w, h, mdef["input_width"], mdef["input_height"])
        fmt = mdef.get("input_format", "rgb")
        if fmt == "nv12":
            yuv = cv2.cvtColor(crop, cv2.COLOR_BGR2YUV_I420)
            # Convert I420 (planar YUV) to NV12 (semi-planar)
            h_img = crop.shape[0]
            y_plane = yuv[:h_img, :]
            uv_i420 = yuv[h_img:, :].reshape(2, h_img // 2, -1)
            uv_i420 = uv_i420.transpose(1, 2, 0).reshape(h_img // 2, -1)
            return np.concatenate([y_plane.flatten(), uv_i420.flatten()])
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).flatten()

    def _recognize_plate(self, bgr: np.ndarray,
                         bbox: Tuple[float, float, float, float]) -> PlateDetection:
        rec_input = self._prepare_plate_input(bgr, bbox)
        rec_result = self.infer_client.infer(rec_input, model_id="plate_recognition", timeout_ms=3000)
        text, conf = decode_recognition(rec_result, LPR_CHARSET, blank=LPR_CTC_BLANK)
        text, conf = self.plate_accumulator.update(text, conf)
        return PlateDetection(bbox=bbox, text=text, confidence=conf)

    def _recognize_plates_batch(self, bgr: np.ndarray,
                                bboxes: List[Tuple[float, float, float, float]],
                                dsp_src: Any = None,
                                ) -> List[PlateDetection]:
        """Recognize all detected plates in one batch RPC.

        Plate crops are mutually independent, so N plates collapse from N
        sequential ``infer()`` round-trips to a single ``infer_batch`` (ai-runtime
        runs them in parallel via run_async + the NPU scheduler). Falls back to
        per-plate sequential ``infer()`` when InferBatch is unsupported or fails.

        With a keep-fd NV12 ``dsp_src``, the letterbox tiles come from one DSP
        ``multi_crop_hw`` job (native letterbox scaling); any tile the DSP path
        did not produce falls back to ``_prepare_plate_input`` on BGR.

        Order is preserved (zip with ``bboxes``) so the temporal
        ``plate_accumulator`` sees plates in the same order as before.
        """
        if not bboxes:
            return []

        dsp_tiles: Optional[List[Optional[np.ndarray]]] = None
        if dsp_src is not None and self._dsp_active():
            dsp_tiles = self._dsp_plate_tiles(dsp_src, bboxes)

        def _tile_or_cpu(i: int, bbox: Tuple[float, float, float, float]
                         ) -> np.ndarray:
            if dsp_tiles is not None and dsp_tiles[i] is not None:
                return dsp_tiles[i].flatten()  # type: ignore[union-attr]
            return self._prepare_plate_input(bgr, bbox)

        inputs = [_tile_or_cpu(i, bbox) for i, bbox in enumerate(bboxes)]
        results: List[Optional[Any]] = [None] * len(inputs)

        # Warmup-aware timeout for plate_recognition (steady 3000ms, warmup
        # 6000ms).  Computed once per frame (one counter increment).
        rec_t = self._infer_timeout_ms("plate_recognition", steady_ms=3000)

        # Batch path only pays off for >1 plate; a single plate goes straight
        # to the sequential fill below (no extra RPC overhead).
        if self._batch_enabled and len(inputs) > 1:
            try:
                batch_results = self.infer_client.infer_batch([
                    BatchInferItem(inp, "plate_recognition", timeout_ms=rec_t)
                    for inp in inputs
                ], timeout_ms=5000)
                # Server guarantees one response per request, same order.
                # Pad/truncate defensively so the fallback loop cannot IndexError.
                got = [r for r in batch_results]
                results = (got + [None] * len(inputs))[:len(inputs)]
            except Exception as exc:
                if "UNIMPLEMENTED" in str(exc):
                    logger.warning(
                        "InferBatch unsupported; plate recognition uses sequential infer().",
                    )
                else:
                    logger.warning("Plate batch failed, going sequential: %s", exc)

        # Sequential fill for any unresolved slot (fallback or single plate).
        for i, inp in enumerate(inputs):
            if results[i] is None:
                try:
                    results[i] = self.infer_client.infer(
                        inp, model_id="plate_recognition", timeout_ms=rec_t)
                except Exception as exc:
                    logger.debug("Plate recognition failed: %s", exc)

        plates: List[PlateDetection] = []
        for bbox, rec in zip(bboxes, results):
            if rec is None:
                continue
            text, conf = decode_recognition(rec, LPR_CHARSET, blank=LPR_CTC_BLANK)
            text, conf = self.plate_accumulator.update(text, conf)
            if text:
                plates.append(PlateDetection(bbox=bbox, text=text, confidence=conf))
        return plates

    def run_frame_pipeline(self, bgr: np.ndarray,
                           nv12: Optional[np.ndarray] = None,
                           src_w: int = 0, src_h: int = 0, src_fmt: str = "",
                           timeout_s: float = 10.0,
                           dsp_src: Any = None) -> PipelineResult:
        """Run the full detection pipeline on a single frame.

        Phase 1: Vehicle detection, depth estimation, and plate detection
        are submitted as a single batch RPC (``infer_batch``). ai-runtime
        runs them in parallel on the NPU via shared VDevice ROUND_ROBIN
        scheduling, reducing 3 gRPC round-trips to 1.

        Phase 2: Plate recognition runs sequentially because it depends on
        plate detection bounding boxes from Phase 1.

        When ``nv12`` + dims + ``src_fmt=="NV12"`` are supplied (live mode),
        Phase-1 whole-frame inputs are prepared via ``_prepare_nv12_input``
        (zero-copy / single-conversion) instead of the BGR round-trip.  With a
        keep-fd ``dsp_src`` frame, whole-frame scaling goes through DSP
        ``resize_hw`` first (zero-copy source) and plate crops become one DSP
        ``multi_crop_hw`` letterbox job; every DSP miss falls back to the
        CPU paths above.
        """
        t0 = time.monotonic()
        fh, fw = bgr.shape[:2]

        # -- Phase 1: batch inference (3 models in parallel) ----------------
        vehicle_mdef = MODEL_DEFS[VEHICLE_MODEL]
        depth_mdef = MODEL_DEFS["scdepthv3"]
        plate_mdef = MODEL_DEFS["license_plate_det"]

        # NV12-direct path when the live frame is NV12 (avoids BGR round-trip).
        nv12_ok = (nv12 is not None and src_fmt == "NV12" and src_w > 0 and src_h > 0)

        def _prep(mdef: Dict[str, Any]) -> np.ndarray:
            if dsp_src is not None and self._dsp_active():
                hw = self._dsp_model_input(dsp_src, mdef)
                if hw is not None:
                    return hw
            if nv12_ok:
                return self._prepare_nv12_input(
                    nv12, src_w, src_h,
                    mdef["input_width"], mdef["input_height"], mdef["input_format"])
            return prepare_input(
                bgr, mdef["input_width"],
                mdef["input_height"], mdef["input_format"])

        vehicle_input = _prep(vehicle_mdef)
        depth_input = _prep(depth_mdef)
        plate_input = _prep(plate_mdef)

        # Per-model warmup-aware timeouts (one increment per model per frame).
        v_t = self._infer_timeout_ms(VEHICLE_MODEL)
        d_t = self._infer_timeout_ms("scdepthv3")
        p_t = self._infer_timeout_ms("license_plate_det")

        vehicle_result = depth_result = plate_result = None

        # Try batch inference once; on UNIMPLEMENTED, disable permanently so
        # subsequent frames go straight to sequential (no per-frame failed RPC).
        if self._batch_enabled:
            try:
                batch_results = self.infer_client.infer_batch([
                    BatchInferItem(vehicle_input, VEHICLE_MODEL, timeout_ms=v_t),
                    BatchInferItem(depth_input, "scdepthv3", timeout_ms=d_t),
                    BatchInferItem(plate_input, "license_plate_det", timeout_ms=p_t),
                ], timeout_ms=int(timeout_s * 1000))
                vehicle_result, depth_result, plate_result = batch_results
            except Exception as exc:
                msg = str(exc)
                if "UNIMPLEMENTED" in msg:
                    self._batch_enabled = False
                    logger.warning(
                        "InferBatch unsupported by ai-runtime; switching to "
                        "sequential infer() permanently (batch disabled).",
                    )
                else:
                    logger.warning(
                        "Batch inference failed, using sequential this frame: %s", exc,
                    )

        # Sequential fallback: individual infer() calls when InferBatch is
        # unavailable or failed.  Skipped if batch already populated results.
        if vehicle_result is None:
            try:
                vehicle_result = self.infer_client.infer(
                    vehicle_input, model_id=VEHICLE_MODEL, timeout_ms=v_t)
                depth_result = self.infer_client.infer(
                    depth_input, model_id="scdepthv3", timeout_ms=d_t)
                plate_result = self.infer_client.infer(
                    plate_input, model_id="license_plate_det", timeout_ms=p_t)
            except Exception as exc2:
                logger.error("Sequential inference failed: %s", exc2)

        # -- Capture NPU-only latency (pure hardware time, excludes IPC/queue) --
        phase1 = [r for r in (vehicle_result, depth_result, plate_result) if r is not None]
        if phase1:
            hw_sum = sum(getattr(r, "hw_infer_time_us", 0) or 0 for r in phase1)
            if hw_sum > 0:
                self._last_hw_infer_us = hw_sum

        # -- Parse vehicle detections --------------------------------------
        vehicles: List[VehicleDetection] = []
        if vehicle_result is not None:
            try:
                if vehicle_result.objects:
                    for obj in vehicle_result.objects:
                        cls_name = _COCO_VEHICLE_CLASSES.get(
                            getattr(obj, "class_id", 0), "vehicle")
                        vehicles.append(VehicleDetection(
                            bbox=(float(obj.bbox.x), float(obj.bbox.y),
                                  float(obj.bbox.width), float(obj.bbox.height)),
                            class_id=getattr(obj, "class_id", 0),
                            class_name=cls_name,
                            confidence=float(obj.score),
                        ))
                elif getattr(vehicle_result, "raw_outputs", None):
                    vehicles = parse_nms_raw(vehicle_result)
            except Exception as exc:
                logger.warning("Vehicle parse failed: %s", exc)

        # -- Parse depth map -----------------------------------------------
        depth_map: Optional[np.ndarray] = None
        if depth_result is not None:
            try:
                if getattr(depth_result, "depth_maps", None) and depth_result.depth_maps:
                    depth_map = depth_result.depth_maps[0].data
                elif getattr(depth_result, "raw_outputs", None) and depth_result.raw_outputs:
                    raw = np.asarray(depth_result.raw_outputs[0], dtype=np.float32)
                    if raw.max() > 2.0:
                        raw = raw / 255.0
                    # Depth model output may differ from input dims;
                    # infer shape from actual data size.
                    h = int(len(raw) / depth_mdef["input_width"])
                    depth_map = raw.reshape(h, depth_mdef["input_width"])
            except Exception as exc:
                logger.warning("Depth parse failed: %s", exc)

        # -- Parse plate detections ----------------------------------------
        plate_dets: List[Tuple[float, ...]] = []
        if plate_result is not None:
            try:
                if getattr(plate_result, "objects", None) and plate_result.objects:
                    plate_dets = [
                        (float(obj.bbox.x), float(obj.bbox.y),
                         float(obj.bbox.width), float(obj.bbox.height))
                        for obj in plate_result.objects
                    ]
                elif getattr(plate_result, "raw_outputs", None):
                    plate_dets = parse_yolo_grid(plate_result, "license_plate_det")
            except Exception as exc:
                logger.warning("Plate parse failed: %s", exc)

        # -- Phase 2: plate recognition (batched) --------------------------
        # N plates → 1 InferBatch RPC; falls back to sequential internally.
        # dsp_src lets the tiles come from one DSP multi_crop letterbox job.
        plates = self._recognize_plates_batch(bgr, plate_dets, dsp_src=dsp_src)

        # -- Depth spoof analysis ------------------------------------------
        spoof_alerts: List[SpoofAlert] = []
        if depth_map is not None:
            for v in vehicles:
                spoof = analyze_depth_spoof(
                    depth_map, v.bbox, fh, fw,
                    threshold=self.depth_spoof_threshold,
                )
                if spoof.is_spoof:
                    spoof_alerts.append(SpoofAlert(
                        vehicle=v, reason="low_depth_variance",
                        score=spoof.confidence,
                    ))

        t1 = time.monotonic()
        return PipelineResult(
            vehicles=vehicles,
            plates=plates,
            depth_map=depth_map,
            spoof_alerts=spoof_alerts,
            infer_time_ms=(t1 - t0) * 1000.0,
        )

    def _draw_overlay(self, bgr: np.ndarray, result: PipelineResult) -> np.ndarray:
        """Draw overlay on a copy — used by video-file path."""
        frame = bgr.copy()
        self._draw_on(frame, result)
        return frame

    def _draw_overlay_inplace(self, bgr: np.ndarray, result: PipelineResult) -> None:
        """Draw overlay directly on bgr — no copy, used by live capture."""
        self._draw_on(bgr, result)

    def _draw_on(self, frame: np.ndarray, result: PipelineResult) -> None:
        """Shared drawing logic."""
        fh, fw = frame.shape[:2]

        fps_text = f"{self._last_stats.get('fps', '--')} FPS"
        cv2.rectangle(frame, (0, 0), (fw, 36), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Parking Lot  |  {fps_text}  |  Vehicles: {len(result.vehicles)}  Plates: {len(result.plates)}",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
        )

        for v in result.vehicles:
            x, y, w, h = v.bbox
            pt1 = (int(x * fw), int(y * fh))
            pt2 = (int((x + w) * fw), int((y + h) * fh))
            cv2.rectangle(frame, pt1, pt2, (0, 200, 0), 2)
            label = f"{v.class_name} {v.confidence:.0%}"
            cv2.putText(frame, label, (pt1[0], pt1[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        # NOTE: Plate boxes are intentionally NOT drawn on the preview frame.
        # Plate crop + recognized text are surfaced via the "License Plates"
        # gallery (add_plate_snapshot / /api/plates), which crops the original
        # bgr (not this overlay) and renders text in the browser where CJK
        # plates display correctly. cv2 Hershey fonts are ASCII-only, so
        # drawing Chinese plate text here would render as "???".

        for alert in result.spoof_alerts:
            v = alert.vehicle
            x, y, w, h = v.bbox
            pt1 = (int(x * fw), int(y * fh))
            pt2 = (int((x + w) * fw), int((y + h) * fh))
            cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 3)
            cv2.putText(frame, "SPOOF", (pt1[0], pt1[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ------------------------------------------------------------------
    # Plate snapshot gallery
    # ------------------------------------------------------------------

    def add_plate_snapshot(self, bgr: np.ndarray, plate: PlateDetection) -> str:
        """Crop the plate region from *bgr*, store as JPEG, return snapshot_id."""
        fh, fw = bgr.shape[:2]
        x, y, w, h = plate.bbox
        # Add margin for visual context
        mx, my = 0.15, 0.30
        px1 = max(0, int((x - w * mx) * fw))
        py1 = max(0, int((y - h * my) * fh))
        px2 = min(fw, int((x + w + w * mx) * fw))
        py2 = min(fh, int((y + h + h * my) * fh))
        if px2 <= px1 or py2 <= py1:
            return ""
        crop = bgr[py1:py2, px1:px2]
        _, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])

        snap_id = uuid.uuid4().hex[:12]
        snapshot = PlateSnapshot(
            snapshot_id=snap_id,
            plate_text=plate.text,
            confidence=plate.confidence,
            timestamp=time.strftime("%H:%M:%S"),
            image_jpeg=jpeg.tobytes(),
        )

        with self._plate_snapshot_lock:
            # Dedup: replace existing entry with same plate text
            self._plate_snapshots = [
                s for s in self._plate_snapshots if s.plate_text != plate.text
            ]
            self._plate_snapshots.insert(0, snapshot)
            self._plate_snapshots = self._plate_snapshots[: self._plate_snapshot_max]

        return snap_id

    def get_plate_snapshots(self) -> List[Dict[str, Any]]:
        """Return snapshot metadata (no image bytes)."""
        with self._plate_snapshot_lock:
            return [
                {
                    "id": s.snapshot_id,
                    "plate": s.plate_text,
                    "confidence": round(float(s.confidence), 3),
                    "time": s.timestamp,
                }
                for s in self._plate_snapshots
            ]

    def get_plate_snapshot_image(self, snapshot_id: str) -> Optional[bytes]:
        """Return JPEG bytes for a specific snapshot, or None."""
        with self._plate_snapshot_lock:
            for s in self._plate_snapshots:
                if s.snapshot_id == snapshot_id:
                    return s.image_jpeg
        return None

    def _fetch_platform_stats(self) -> Optional[Dict[str, Any]]:
        """Fetch NPU / model stats from the platform API with TTL caching.

        Returns a dict with npu_util, cpu_util, ram_used_mb, ram_total_mb,
        and models list, or None on failure.
        """
        now = time.monotonic()
        if self._platform_stats is not None and (now - self._platform_stats_time) < 3.0:
            return self._platform_stats

        try:
            url = f"{PLATFORM_API_URL}/api/v1/ai/stats"
            req = urllib.request.Request(url, headers={
                "Authorization": f"Bearer {PLATFORM_API_TOKEN}",
            })
            with urllib.request.urlopen(req, timeout=2) as resp:
                body = json.loads(resp.read())
            data = body.get("data") if isinstance(body, dict) else None
            if not data:
                return self._platform_stats

            models = []
            for ms in data.get("model_stats", []):
                raw_qps = ms.get("current_qps") or ms.get("qps")
                raw_lat_us = ms.get("avg_latency_us") or ms.get("latency_ms")
                lat_ms = raw_lat_us / 1000.0 if raw_lat_us and raw_lat_us > 0 else None
                models.append({
                    "id": ms.get("model_id", "unknown"),
                    "qps": round(raw_qps, 1) if raw_qps and raw_qps > 0 else None,
                    "latency_ms": round(lat_ms, 1) if lat_ms else None,
                })

            self._platform_stats = {
                "npu_util": data.get("device_utilization", 0),
                "cpu_util": data.get("cpu_utilization", 0),
                "ram_used_mb": round(data.get("ram_used_kib", 0) / 1024),
                "ram_total_mb": round(data.get("ram_total_kib", 0) / 1024),
                # Hardware FPS from HAL (InferenceStats.hw_fps); absent if the
                # running ai-runtime predates the field — left as None then.
                "hw_fps": data.get("hw_fps"),
                "models": models,
            }
            self._platform_stats_time = now
        except Exception as exc:
            logger.debug("Platform stats fetch failed: %s", exc)

        return self._platform_stats

    def get_stats(self) -> Dict[str, Any]:
        platform = self._fetch_platform_stats()
        return {
            "fps": self._stream_fps or self._last_stats.get("fps"),
            "infer_ms": self._last_stats.get("infer_ms"),
            # Pure NPU latency for the last Phase-1 batch (vehicle+depth+plate det),
            # summed across the 3 models. None until the first successful frame.
            "hw_infer_ms": round(self._last_hw_infer_us / 1000.0, 2) if self._last_hw_infer_us else None,
            "infer_fps": self._infer_fps,
            # NPU hardware FPS reported by the platform (HAL), when available.
            "hw_fps": (platform or {}).get("hw_fps"),
            "frame_count": self._infer_count,
            "vehicles": self._last_stats.get("vehicles", 0),
            "plates": self._last_stats.get("plates", 0),
            "alerts": len(self._alert_history),
            "platform": platform,
            # --- Observability (ported from model-showcase get_stats) ---
            # Degraded flag + consecutive-failure count drive the front-end
            # health dot (green=ok / red=degraded).
            "degraded": self._degraded,
            "consecutive_failures": self._consecutive_failures,
            # Raw NPU latency in microseconds (hw_infer_ms is the rounded ms).
            "hw_infer_us": self._last_hw_infer_us,
            # Current steady-state inference timeout (ms); warmup window uses
            # INFER_WARMUP_TIMEOUT_MS for the first INFER_WARMUP_COUNT calls.
            "infer_timeout_ms": INFER_TIMEOUT_MS,
            # DSP offload counters for A/B runs: hw_jobs = successful DSP
            # jobs, cpu_fallbacks = jobs that went back to CPU (quota hits
            # + permanent disables), last_error = most recent failure.
            "dsp": dict(self._dsp_stats),
        }

    def get_alerts(self) -> List[Dict[str, Any]]:
        with self._alert_lock:
            return list(self._alert_history[:50])

    def get_mode(self) -> Dict[str, Any]:
        with self._upload_lock:
            progress = dict(self._upload_progress)
        return {"mode": self._mode, "progress": progress}

    def switch_to_live(self) -> None:
        with self._video_lock:
            self._finalize_video()
            if self._video_source is not None:
                self._video_source.close()
                self._video_source = None
            self._mode = "live"
        with self._upload_lock:
            self._upload_progress["status"] = "idle"
        logger.info("Switched to live stream mode")

    def switch_to_upload(self, video_path: str) -> None:
        with self._video_lock:
            self._finalize_video()
            if self._video_source is not None:
                self._video_source.close()
                self._video_source = None
            try:
                self._video_source = VideoFrameSource(video_path)
            except ValueError as e:
                logger.error("Cannot open video: %s", e)
                # Restore a consistent state: no source, live mode.
                self._mode = "live"
                with self._upload_lock:
                    self._upload_progress["status"] = "error"
                    self._upload_progress["error"] = str(e)
                return
            self._mode = "upload"
            with self._upload_lock:
                self._upload_progress = {
                    "current": 0,
                    "total": self._video_source.total_frames,
                    "status": "processing",
                    "source": os.path.basename(video_path),
                    "result_file": None,
                }
            self._result_video_path = None
            self._video_writer = None
        logger.info("Switched to upload mode: %s", video_path)

    def video_control(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        """Pause / play / seek / speed control for uploaded video."""
        with self._video_lock:
            vs = self._video_source
            if vs is None:
                return {"error": "No video loaded"}
            if action == "pause":
                vs.paused = True
            elif action == "play":
                vs.paused = False
            elif action == "seek":
                vs.seek(float(kwargs.get("position_sec", 0)))
            elif action == "speed":
                vs.speed = float(kwargs.get("speed", 1.0))
            elif action == "stop":
                # RLock: re-entering _video_lock via switch_to_live() is safe.
                self.switch_to_live()
            else:
                return {"error": f"Unknown action: {action}"}
            return self.get_video_status()

    def get_video_status(self) -> Dict[str, Any]:
        with self._video_lock:
            vs = self._video_source
            if vs is None:
                return {"active": False}
            return {
                "active": True,
                "paused": vs.paused,
                "speed": vs.speed,
                "position_sec": round(vs.position_sec, 2),
                "duration": round(vs.duration, 2),
                "progress": round(vs.progress, 4),
                "total_frames": vs.total_frames,
                "fps": vs.fps,
            }

    def _finalize_video(self) -> None:
        """Release VideoWriter and mark upload progress as done."""
        with self._video_lock:
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
            if self._mode == "upload" and self._upload_progress.get("status") == "processing":
                result_file = (
                    os.path.basename(self._result_video_path)
                    if self._result_video_path else None
                )
                with self._upload_lock:
                    self._upload_progress["status"] = "done"
                    self._upload_progress["result_file"] = result_file
                logger.info("Upload video done: %d frames processed",
                            self._upload_progress.get("current", 0))
                sse_broadcast({
                    "upload_done": True,
                    "frames": self._upload_progress.get("current", 0),
                })

    def publish_results(self, result: PipelineResult) -> None:
        if result.vehicles:
            payload = [
                {"bbox": [float(c) for c in v.bbox], "class": v.class_name,
                 "confidence": round(float(v.confidence), 3)}
                for v in result.vehicles
            ]
            self.event_client.publish("parking/vehicles", payload)

        if result.plates:
            payload = [
                {"bbox": [float(c) for c in p.bbox], "plate": p.text,
                 "confidence": round(float(p.confidence), 3)}
                for p in result.plates
            ]
            self.event_client.publish("parking/plates", payload)

        for alert in result.spoof_alerts:
            self.event_client.publish("parking/alerts", {
                "type": "spoof_detected",
                "bbox": [float(c) for c in alert.vehicle.bbox],
                "vehicle_class": alert.vehicle.class_name,
                "reason": alert.reason,
                "score": round(float(alert.score), 3),
            })

    # ------------------------------------------------------------------
    # Async dual-thread architecture
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Thread: capture frames at camera FPS, draw overlay, push MJPEG.

        Runs independently from inference.  Grabs every frame the camera
        daemon provides, overlays the *latest* detection result, and
        pushes the JPEG to the frame buffer.  Stream FPS is therefore
        decoupled from inference latency.
        """
        self._stream_fps_ts = time.monotonic()

        # Per-step timing accumulators (logged every 5 s)
        _pt = {"get": 0.0, "convert": 0.0, "overlay": 0.0, "encode": 0.0, "n": 0, "q": 0}
        _pt_log_ts = time.monotonic()

        while self._running:
            if self._mode != "live":
                time.sleep(0.5)
                continue

            try:
                t0 = time.monotonic()
                frame = self.media_preview.get_frame(
                    self.stream_id, timeout_ms=3000,
                )
                t1 = time.monotonic()
            except Exception as e:
                logger.warning("Frame acquisition error: %s", e)
                # Reconnect with exponential backoff (1s/2s/3s) instead of a
                # bare sleep — a transient camera-daemon hiccup should not
                # silently drop frames forever.
                self._reconnect_preview_media()
                continue

            if frame is None:
                continue

            bgr = self.frame_to_bgr(frame)
            t2 = time.monotonic()

            t3 = time.monotonic()  # overlay = 0 (no overlay)

            # Encode to JPEG — 720p direct encode (no overlay, no resize)
            _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
            t4 = time.monotonic()

            self.frame_buffer.put(jpeg.tobytes())

            # Accumulate per-step timing
            _pt["get"] += t1 - t0
            _pt["convert"] += t2 - t1
            _pt["overlay"] += t3 - t2
            _pt["encode"] += t4 - t3
            _pt["n"] += 1
            _pt["q"] += len(jpeg.tobytes()) // 1024  # average JPEG size in KB

            # Stream FPS tracking (windowed over 1 s)
            self._stream_fps_count += 1
            now = time.monotonic()
            dt = now - self._stream_fps_ts
            if dt >= 1.0:
                self._stream_fps = round(self._stream_fps_count / dt, 1)
                self._stream_fps_count = 0
                self._stream_fps_ts = now

            # Log per-step averages every 5 s
            if now - _pt_log_ts >= 5.0 and _pt["n"] > 0:
                n = _pt["n"]
                logger.info(
                    "[capture-timing] get=%.1fms  convert=%.1fms  encode=%.1fms  "
                    "total=%.1fms  jpeg=%dKB  (n=%d, stream_fps=%.1f)",
                    _pt["get"] / n * 1000,
                    _pt["convert"] / n * 1000,
                    _pt["encode"] / n * 1000,
                    (_pt["get"] + _pt["convert"] + _pt["encode"]) / n * 1000,
                    _pt["q"] // n,
                    n,
                    self._stream_fps,
                )
                _pt = {"get": 0.0, "convert": 0.0, "overlay": 0.0, "encode": 0.0, "n": 0, "q": 0}
                _pt_log_ts = now

    def _infer_loop(self) -> None:
        """Thread: unified inference loop for both live and upload modes.

        Live mode: picks the latest frame from the capture thread (skip stale).
        Upload mode: reads frames sequentially from VideoFrameSource.

        Both paths share the same post-processing via _process_result().

        The entire per-frame body is wrapped so that no exception — including
        one from _process_result()'s VideoWriter path or a transient source
        race — can terminate the loop. Previously a single raised exception
        killed the inference thread silently, freezing the stream with no
        recovery; now it logs a traceback and serves the next frame.
        """
        self._infer_fps_ts = time.monotonic()

        while self._running:
            try:
                bgr, nv12, src_w, src_h, src_fmt, src_frame = self._get_next_frame()
                if bgr is None:
                    if src_frame is not None:
                        src_frame.release()
                    time.sleep(0.01)
                    continue

                t0 = time.monotonic()

                # Run the full detection pipeline (~80 ms).  Pass the raw NV12
                # buffer + geometry through so the pipeline can take the
                # zero-copy NV12 direct path when the source is NV12.  A keep-fd
                # src_frame additionally enables the DSP resize/multi-crop path;
                # it is released in the finally below (covers the `continue`
                # error path too — all DSP use happens inside the pipeline).
                try:
                    result = self.run_frame_pipeline(
                        bgr,
                        nv12=nv12,
                        src_w=src_w,
                        src_h=src_h,
                        src_fmt=src_fmt,
                        timeout_s=8.0,
                        dsp_src=src_frame,
                    )
                    self._register_infer_success()
                except Exception as e:
                    # Transient NPU stall / pipeline error — register toward
                    # the degraded state machine but never kill the loop.
                    self._register_infer_failure(e)
                    logger.warning("Pipeline error: %s", e)
                    continue

                try:
                    self._process_result(bgr, result, t0)
                finally:
                    if src_frame is not None:
                        src_frame.release()
            except Exception as e:
                # Watchdog: never let one frame's post-processing kill the
                # whole inference pipeline. Log full traceback, back off,
                # and keep serving frames.
                logger.exception("Inference loop frame failed (recovered): %s", e)
                time.sleep(0.1)

    def _get_next_frame(self):
        """Return the next frame depending on the current mode.

        Returns a tuple ``(bgr, nv12, w, h, fmt, src_frame)``:

        * ``bgr``  — BGR ndarray (always populated; used for overlay + plate
          crop letter-boxing).
        * ``nv12`` — raw NV12 ndarray when the live source is NV12, else
          ``None``.  When present, the inference pipeline can skip the
          BGR→RGB/NV12 double conversion and feed the model directly.
        * ``w``/``h``/``fmt`` — source frame geometry/format; 0/"" when
          unavailable (upload path or attribute lookup failed).
        * ``src_frame`` — the keep-fd Frame when the DSP path is active
          (zero-copy source for resize_hw/multi_crop_hw; the caller must
          release it), else ``None``.

        Live: fetch via the dedicated ``media_infer`` client on the chosen
        infer stream (decoupled from the preview client), with exponential
        backoff reconnect on transient errors.  When the DSP path is active
        the frame is acquired with ``keep_fd=True`` and materialized via
        ``to_array()`` (keep-fd frames carry a dma-buf handle, not pixels,
        until first materialization) — the copy cost equals a normal
        receive, so keep_fd is strictly additive.
        Upload: read next frame from VideoFrameSource (sequential, BGR only).

        The upload branch snapshots ``_mode``/``_video_source`` and performs
        EOF cleanup atomically under ``_video_lock``, so a concurrent
        switch_*/video_control cannot close the source between the None-check
        and the read (the prior use-after-close AttributeError).
        """
        with self._video_lock:
            mode = self._mode
            vs = self._video_source
            if mode == "upload" and vs is not None:
                frame = vs.get_bgr_frame()
                if frame is None:
                    # Video finished — auto-switch to live (atomic w/ switch paths)
                    logger.info("Video EOF, auto-switching to live mode")
                    self._finalize_video()
                    vs.close()
                    self._video_source = None
                    self._mode = "live"
                    with self._upload_lock:
                        self._upload_progress["status"] = "idle"
                # Upload frames are BGR; no NV12/DSP source available.
                return (frame, None, 0, 0, "", None)

        # Live mode: fetch directly from the inference media client (NOT the
        # capture/preview client) so inference latency/stream are decoupled
        # from preview.  This matches model-showcase's dual-client design.
        keep = self._dsp_active()
        try:
            frame = self.media_infer.get_frame(
                self._active_infer_stream, timeout_ms=3000, keep_fd=keep,
            )
        except Exception as e:
            logger.warning("Infer media get_frame error: %s", e)
            self._reconnect_infer_media()
            return (None, None, 0, 0, "", None)

        if frame is None:
            return (None, None, 0, 0, "", None)

        # Frame attributes (image/format/width/height) are exposed by the
        # pybind wrapper but hidden from dir(); use getattr with defaults.
        fmt = getattr(frame, "format", "") or ""
        w = getattr(frame, "width", 0) or 0
        h = getattr(frame, "height", 0) or 0

        if keep:
            # Materialize the dma-buf into .image (cached) — keep-fd frames
            # start with image=None and every downstream consumer reads
            # .image (frame_to_bgr, NV12 direct path).
            frame.to_array()

        # NV12 direct path: only valid when the source actually delivered NV12
        # and geometry is known.  Otherwise nv12 stays None and the pipeline
        # falls back to BGR→RGB/NV12 conversion.
        nv12 = frame.image if fmt.upper() == "NV12" else None
        bgr = self.frame_to_bgr(frame)
        return (bgr, nv12, w, h, fmt, frame if keep else None)

    def _process_result(
        self, bgr: np.ndarray, result: PipelineResult, t0: float,
    ) -> None:
        """Shared post-processing for both live and upload inference results.

        Handles: overlay, FrameBuffer, stats, event bus, alerts, plate
        snapshots, SSE broadcast, and upload-only extras (VideoWriter,
        progress, frame skipping).
        """
        elapsed = time.monotonic() - t0

        # --- Overlay + FrameBuffer ---
        overlay = self._draw_overlay(bgr, result)
        _, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.frame_buffer.put(jpeg.tobytes())

        # --- Upload-only: VideoWriter + progress ---
        # Snapshot source/writer under _video_lock so a concurrent switch_*
        # cannot release the writer mid-write (write-after-release) or null
        # the source between the mode check and the _frame_idx read.
        with self._video_lock:
            is_upload = self._mode == "upload" and self._video_source is not None
            vs = self._video_source
            writer = self._video_writer
            if is_upload and writer is None and self._result_video_path is None:
                # Lazy-init VideoWriter on first frame
                h, w = bgr.shape[:2]
                result_dir = "/app/uploads"
                os.makedirs(result_dir, exist_ok=True)
                self._result_video_path = os.path.join(
                    result_dir, f"result_{int(time.time())}.mp4",
                )
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    self._result_video_path, fourcc, vs.fps, (w, h),
                )
                self._video_writer = writer

            if is_upload and writer is not None:
                # Write under the lock: switches are rare, and writing to a
                # writer that another thread just released is the bug we're
                # fixing. Holding for one frame encode (~ms) is acceptable.
                writer.write(overlay)

            frame_idx = vs._frame_idx if vs is not None else 0

        if is_upload:
            with self._upload_lock:
                self._upload_progress["current"] = frame_idx

        # --- Publish detection result for capture-thread overlay ---
        with self._result_lock:
            self._latest_result = result

        # --- Stats ---
        fps_val = 1.0 / elapsed if elapsed > 0 else 0
        self._last_stats = {
            "fps": round(fps_val, 1) if is_upload else self._stream_fps,
            "infer_ms": round(result.infer_time_ms, 1),
            "vehicles": len(result.vehicles),
            "plates": len(result.plates),
        }

        # --- Event bus ---
        self.publish_results(result)

        # --- Alerts ---
        for alert in result.spoof_alerts:
            entry = {
                "type": "spoof_detected",
                "detail": f"{alert.vehicle.class_name} score={alert.score:.2f}",
                "bbox": list(alert.vehicle.bbox),
                "time": time.strftime("%H:%M:%S"),
            }
            with self._alert_lock:
                self._alert_history.insert(0, entry)
                self._alert_history = self._alert_history[:50]

        # --- Plate snapshots + SSE payload ---
        # vehicles/plates carry normalized [x, y, w, h] bboxes so the web UI
        # can draw detection boxes on the HD preview via a client-side canvas
        # (cv2 Hershey fonts can't render CJK plate text in-frame). Counts are
        # exposed separately as vehicle_count/plate_count for badges/stats.
        plate_entries = []
        for p in result.plates:
            snap_id = self.add_plate_snapshot(bgr, p) if p.text else ""
            plate_entries.append({
                "bbox": [float(c) for c in p.bbox],
                "plate": p.text,
                "confidence": round(float(p.confidence), 3),
                "snapshot_id": snap_id,
            })

        # spoof_alerts wrap the exact vehicle objects (SpoofAlert(vehicle=v)
        # at run_frame_pipeline), so identity marks spoofed vehicles reliably.
        spoof_vehicle_ids = {id(a.vehicle) for a in result.spoof_alerts}
        vehicle_entries = [
            {
                "bbox": [float(c) for c in v.bbox],
                "class": v.class_name,
                "conf": round(float(v.confidence), 3),
                "spoof": id(v) in spoof_vehicle_ids,
            }
            for v in result.vehicles
        ]

        sse_payload: Dict[str, Any] = {
            "vehicles": vehicle_entries,
            "vehicle_count": len(result.vehicles),
            "plates": plate_entries,
            "plate_count": len(result.plates),
            "spoof_alerts": [
                {"type": "spoof_detected",
                 "detail": f"{a.vehicle.class_name} score={a.score:.2f}"}
                for a in result.spoof_alerts
            ],
        }
        if is_upload:
            # Reuse the vs snapshot taken under _video_lock above (1052);
            # its plain attrs (_frame_idx/total_frames) survive close().
            sse_payload["upload_progress"] = {
                "current": vs._frame_idx,
                "total": vs.total_frames,
            }
        sse_broadcast(sse_payload)

        # --- Inference FPS tracking ---
        self._infer_count += 1
        if self._infer_count % 100 == 0:
            logger.info(
                "Frame %d: %d veh, %d plates, %d alerts, %.1fms "
                "(stream %.1f fps, infer %.1f fps)",
                self._infer_count,
                len(result.vehicles),
                len(result.plates),
                len(result.spoof_alerts),
                result.infer_time_ms,
                self._stream_fps,
                self._infer_fps,
            )

        self._infer_fps_count += 1
        now = time.monotonic()
        dt = now - self._infer_fps_ts
        if dt >= 1.0:
            self._infer_fps = round(self._infer_fps_count / dt, 1)
            self._infer_fps_count = 0
            self._infer_fps_ts = now

        # --- Upload-only: pace to source FPS ---
        if is_upload:
            target_interval = 1.0 / min(vs.fps, self.target_fps)
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Safety: skip frames if a single frame took > 5s
            if elapsed > 5.0:
                skip = min(int(elapsed * vs.fps) - 1, 100)
                if skip > 0:
                    logger.info("Slow frame (%.1fs), skipping %d frames", elapsed, skip)
                    # Use VideoFrameSource speed for frame skipping
                    vs.speed = float(skip + 1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True
        self.register_models()

        # Dual media clients (ported from model-showcase): the preview/capture
        # thread and the inference thread each own an independent FdMediaClient
        # so their timeouts, reconnect backoff, and stream choice don't
        # interfere.  media_client is kept as a backward-compat alias.
        try:
            self.media_preview = FdMediaClient()
        except Exception as e:
            logger.error("Failed to connect preview media client: %s", e)
            return
        try:
            self.media_infer = FdMediaClient()
        except Exception as e:
            logger.error("Failed to connect infer media client: %s", e)
            return
        self.media_client = self.media_preview

        # Pick the infer stream by model input resolution (best-effort; falls
        # back to STREAM_ID when probing fails or only one stream exists).
        self._active_infer_stream = self._choose_infer_stream()
        logger.info(
            "Infer stream selected: %r (default=%s)", self._active_infer_stream, STREAM_ID,
        )

        logger.info(
            "Parking lot pipeline started (async dual-thread): "
            "target_fps=%d, spoof_threshold=%.3f",
            self.target_fps, self.depth_spoof_threshold,
        )

        # Capture thread: frame acquisition + MJPEG streaming at camera FPS
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="capture", daemon=True,
        )
        self._capture_thread.start()

        # Inference thread: detection pipeline at its own pace
        self._infer_thread = threading.Thread(
            target=self._infer_loop, name="inference", daemon=True,
        )
        self._infer_thread.start()

        # Main thread: wait for shutdown signal
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3)
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=3)
        self._finalize_video()
        if self._video_source is not None:
            self._video_source.close()
            self._video_source = None
        for client in (self.media_preview, self.media_infer):
            if client:
                try:
                    client.close()
                except Exception:
                    pass
        self.media_preview = None
        self.media_infer = None
        self.media_client = None
        if self._dsp_client is not None:
            try:
                self._dsp_client.close()
            except Exception:
                pass
            self._dsp_client = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point — sets up signal handlers and starts the pipeline."""
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper()),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app = ParkingLotApp()

    def _signal_handler(sig: int, frame: Any) -> None:
        logger.info("Received signal %d, shutting down", sig)
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Start Flask via waitress in background thread
    flask_app = create_flask_app(app)

    try:
        from waitress import serve as waitress_serve
        flask_thread = threading.Thread(
            target=lambda: waitress_serve(
                flask_app, host="0.0.0.0", port=WEB_PORT, threads=8,
            ),
            daemon=True,
        )
        logger.info("Using waitress production server on http://0.0.0.0:%d", WEB_PORT)
    except ImportError:
        flask_thread = threading.Thread(
            target=lambda: flask_app.run(host="0.0.0.0", port=WEB_PORT, threaded=True),
            daemon=True,
        )
        logger.warning("waitress not installed, falling back to Flask dev server")

    flask_thread.start()
    logger.info("Web UI started on http://0.0.0.0:%d", WEB_PORT)

    app.run()
