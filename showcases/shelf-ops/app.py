"""shelf-ops — 货架槽位识别 demo, application entry point.

Flask web server + background shelf-scan loop, running TWO evidence chains
per tick on the same 4K frame (NE503 / Hailo-15H):

  A. per-item goods detection — yolo_world v5.4.0 two-input HEF
     (docs/device-gate.md §9): letterboxed RGB + 80-prompt vocab embeddings
     -> NMS-by-class stream -> boxes + ITEMS count, drawn on HD canvas and
     baked into the MJPEG overlay (detector.py).
  B. slot-anchored CLIP classification (§7):
     register CLIP ViT-B/32 encoder (identity variant, flaky-register retry)
     -> build vocabulary text matrix via EncodeText (cached, vocab.py)
     -> per-slot polygon crop (~10) -> 224x224 NV12
     -> infer_with_tensors -> 512 uint8 -> mean-center + unit-norm
     -> cosine vs text matrix -> per-slot code/EMPTY + confidence
     -> slots.apply() occupancy -> analytics buckets + stockout/restock events

  -> HD preview: platform-api hardware H.264 (browser MSE) with a canvas
     overlay; MJPEG /stream kept as the fallback (overlay.py baked frames);
     SSE fan-out (alerts.py).

A broken/missing detector degrades the app to chain B only
(/api/health reports detector:unavailable); DETECT_ENABLED=0 turns it off.

Without a device the app runs in simulation status (synthetic per-slot
results + synthetic detection boxes) so the shell page + analytics render
offline.
"""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import urllib.parse

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from hailo_ipc_sdk.app import AppClient
from hailo_ipc_sdk.config import Config
from hailo_ipc_sdk.inference import InferenceClient
from hailo_ipc_sdk.media import FdMediaClient

from alerts import AlertBroker
from analytics import ShelfAnalytics
from clip_classify import (
    SlotClassifier,
    crop_box,
    polygon_bbox,
    register_clip_model,
)
from config import (
    DEGRADED_FAILURE_THRESHOLD, INFER_WARMUP_COUNT, INFER_WARMUP_TIMEOUT_MS,
    load_config,
)
from detector import (
    GoodsDetector,
    bundled_vocab_path,
    count_items,
    items_by_category,
    load_vocab,
    register_detector_model,
)
from media_import import (
    decode_image_bytes,
    jpeg_b64,
    sample_positions,
    validate_image_upload,
    validate_video_upload,
)
from overlay import draw_detections, draw_slots
from slots import SlotManager, build_grid_slots, goods_summary, items_by_code
from vocab import Vocabulary, load_or_build_vocab_embeddings

logger = logging.getLogger("shelf-ops")

_STATUS_SIMULATION = "simulation"      # no device -> synthetic frames
_STATUS_LIVE = "live"                  # device inference running
_STATUS_DEGRADED = "degraded"          # consecutive infer failures
_STATUS_ERROR = "error"                # model register/load failure

# Detector health states surfaced via /api/health.
_DETECT_OK = "ok"
_DETECT_DEGRADED = "degraded"          # infer failing, holding last boxes
_DETECT_UNAVAILABLE = "unavailable"    # registration/load failed at boot
_DETECT_OFF = "off"                    # disabled via DETECT_ENABLED=0
_DETECT_HOLD_TICKS = 3                 # failed ticks before boxes blank

# Synthetic per-slot pattern for simulation mode (cycles codes + vacancies so
# occupancy/analytics see state changes).
_SIM_PATTERN = ("A", "B", "C", "D", "E", "EMPTY", "A", "C", "E", "EMPTY")
# Synthetic detection labels (head of the bundled goods vocabulary) + the
# stray negative that rides the bottom edge. Categories mirror the vocab
# sidecar mapping so simulation exercises the real import-chip rendering.
_SIM_DET_LABELS = ("bottle", "can", "box", "carton", "package")
_SIM_DET_CATEGORIES = ("bottle", "can", "box", "carton", "box")


class ShelfOpsApp:
    """Flask app + background shelf-scan loop."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.app = Flask(__name__, template_folder="templates",
                         static_folder="static")
        self._running = threading.Event()
        self._running.set()
        self._status = _STATUS_SIMULATION
        self._model_ready = False
        self._clip_model_id = ""
        self._last_scan = None          # dict: slot states + counts snapshot
        self._latest_jpeg = None
        self._jpeg_lock = threading.Lock()
        self._frame_seq = 0             # successful scan frame counter
        self._consecutive_failures = 0
        self._warmup_left = INFER_WARMUP_COUNT

        # HD-live preview: overlay snapshot shared by scan->render threads;
        # idle counter so the preview loop stops encoding when nothing watches
        self._overlay_lock = threading.Lock()
        self._last_snaps: list = []
        self._stream_clients = 0
        self._stream_clients_lock = threading.Lock()

        # device/session clients (created lazily inside the worker so a
        # missing socket in file/simulation modes never blocks Flask startup).
        # Two FdMediaClients mirror gym-ops' media / media_infer split: the
        # preview loop and the scan loop must not race one UDS socket.
        self.infer: InferenceClient | None = None
        self.media: FdMediaClient | None = None        # MJPEG preview source
        self.media_infer: FdMediaClient | None = None  # scan frame source

        # analytics + state layers (device-independent)
        db_dir = os.path.dirname(cfg.analytics_db)
        if db_dir and not os.path.isdir(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except OSError:
                pass
        self.analytics = ShelfAnalytics(
            db_path=cfg.analytics_db,
            bucket_seconds=cfg.bucket_seconds,
            retention_hours=cfg.retention_hours,
            stockout_seconds=cfg.stockout_seconds,
            alert_cooldown_seconds=cfg.alert_cooldown_seconds,
        )
        self.broker = AlertBroker(cooldown_seconds=cfg.alert_cooldown_seconds)
        self.vocab = Vocabulary(cfg.vocabulary)
        self.slots_mgr = SlotManager(cfg.slots)
        self.classifier: SlotClassifier | None = None

        # per-item detection state (detector.py); _last_dets is written only
        # by the worker thread and read by the preview thread under
        # _overlay_lock — lists are replaced wholesale, never mutated.
        self.detector: GoodsDetector | None = None
        self._detect_state = _DETECT_OFF
        self._last_dets: list = []
        self._detect_fail_ticks = 0

        # SSE client queues (observable via /api/events)
        self._sse_lock = threading.Lock()
        self._sse_clients: list[queue.Queue] = []

        # Offline photo import shares the NPU clients with the scan thread:
        # _npu_lock serializes their RPC sections (one UDS socket), and
        # _import_lock fails a second concurrent import fast with 409.
        self._npu_lock = threading.Lock()
        self._import_lock = threading.Lock()

        # Offline video import job (single slot): None when never run, else
        # the dict written by _import_video and mutated by its worker thread
        # (_run_video_import). Replaced wholesale by the next import.
        self._video_job: dict | None = None

        # Built-in demo video mode (demo.py): loops the bundled
        # demo.mp4 through this SAME pipeline for customer demos.
        # Runs its own d-prefixed SlotManager so analytics/heatmap rows stay
        # separate from live cells; analytics + broker timers swap to the
        # fast demo values while on and are restored on toggle-off.
        self._demo = False
        self._demo_lock = threading.Lock()
        self._demo_src = None                      # DemoVideoSource when on
        # which video demo mode plays: "builtin" (bundled asset) or
        # "imported" (the last successful /api/import/video upload)
        self._demo_source = "builtin"
        self._demo_slot_dicts = self._build_demo_slots()
        self._demo_slots_mgr = SlotManager(self._demo_slot_dicts)
        self._live_stockout_seconds = cfg.stockout_seconds
        self._live_alert_cooldown = cfg.alert_cooldown_seconds

        self.app.config["MAX_CONTENT_LENGTH"] = (
            (cfg.import_max_image_mb + 1) * 1024 * 1024)

        self._worker = threading.Thread(target=self._loop, name="shelf-scan",
                                        daemon=True)

    # ---- background loop ---------------------------------------------------
    def _loop(self) -> None:
        # Model registration + vocabulary embedding build happen once up front.
        try:
            if os.environ.get("SHELF_SIMULATE", "").lower() in ("1", "true", "yes"):
                logger.info("SHELF_SIMULATE set: forcing simulation mode")
            else:
                self._setup_device()
        except Exception as e:  # noqa: BLE001
            # failing setup must not leave half-initialized clients: subsequent
            # ticks would hammer the missing socket every iteration
            self.infer = None
            self.media = None
            self.media_infer = None
            self._model_ready = False
            logger.warning("device setup failed (%s); running in simulation", e)

        try:
            while self._running.is_set():
                self._tick()
        except Exception:  # pragma: no cover - worker must not die silently
            logger.exception("shelf-scan loop crashed")

    def _setup_device(self) -> None:
        """Connect SDK, register CLIP, build the vocabulary text matrix."""
        self.infer = InferenceClient()
        self.media = FdMediaClient()          # preview stream (MJPEG)
        self.media_infer = FdMediaClient()    # scan/inference stream

        cdef = self.cfg.clip_def()
        register_clip_model(
            self.infer,
            model_path=self.cfg.clip_full_path(),
            model_id=cdef.name,
            owner_id=Config.get_app_id(),
            retries=self.cfg.clip_register_retries,
            inputs=cdef.inputs,
        )
        self._clip_model_id = cdef.name
        self._model_ready = True

        def _encode(text: str) -> np.ndarray:
            assert self.infer is not None
            return np.asarray(self.infer.encode_text(text, timeout_ms=60000),
                              dtype=np.float32)

        def _missing_emb(_text: str) -> np.ndarray:
            raise RuntimeError(
                "embedding_mode=file but no cached matrix matched the "
                f"vocabulary at {self.cfg.embedding_path}")

        txt = load_or_build_vocab_embeddings(
            path=self.cfg.embedding_path,
            entries=self.vocab.sorted_entries(),
            encode=_encode if self.cfg.embedding_mode == "device"
            else _missing_emb,
            template=self.cfg.encode_template,
        )
        self.classifier = SlotClassifier(txt, timeout_ms=self.cfg.clip_timeout_ms)
        self.classifier.model_id = self._clip_model_id
        self._setup_detector()
        self._status = _STATUS_LIVE
        logger.info("shelf-ops live: clip=%s vocab=%dx%d slots=%d "
                    "detector=%s", self._clip_model_id, txt.shape[0],
                    txt.shape[1], len(self.slots_mgr.slots),
                    self._detect_state)

    def _setup_detector(self) -> None:
        """Register yolo_world + load the vocab; failure is non-fatal.

        A missing HEF or broken asset must not take the CLIP chain down —
        the app degrades to cell counting only and reports
        detector:unavailable via /api/health.
        """
        if not self.cfg.detect_enabled:
            self._detect_state = _DETECT_OFF
            logger.info("detector disabled (DETECT_ENABLED=0)")
            return
        try:
            ddef = self.cfg.detect_def()
            register_detector_model(
                self.infer,
                model_path=self.cfg.detect_full_path(),
                model_id=ddef.name,
                owner_id=Config.get_app_id(),
                retries=self.cfg.detect_register_retries,
            )
            if self.cfg.detect_emb_path:      # explicit .npy + sibling .json
                npy = self.cfg.detect_emb_path
                sidecar = os.path.splitext(npy)[0] + ".json"
            else:                             # bundled pair
                base = bundled_vocab_path()
                npy = os.path.join(base, "yolo_world_vocab.npy")
                sidecar = os.path.join(base, "yolo_world_vocab.json")
            emb, labels, n_goods, categories = load_vocab(npy, sidecar)
            self.detector = GoodsDetector(
                emb, labels, n_goods,
                categories=categories,
                threshold=self.cfg.detect_threshold,
                nms_iou=self.cfg.detect_nms_iou,
                timeout_ms=self.cfg.detect_timeout_ms,
            )
            self.detector.model_id = ddef.name
            self._detect_state = _DETECT_OK
            logger.info("detector live: %s vocab=%d rows (goods=%d) "
                        "threshold=%.2f", ddef.name, len(labels), n_goods,
                        self.cfg.detect_threshold)
        except Exception as e:  # noqa: BLE001 - detector is additive
            self.detector = None
            self._detect_state = _DETECT_UNAVAILABLE
            logger.warning("detector setup failed (%s); CLIP-only mode", e)

    def _items_count(self, dets: list) -> int | None:
        """ITEMS m only while the detection chain is active (ok/degraded);
        None hides the ITEMS suffix and blanks the topbar Items stat to '-',
        keeping "detector off" distinguishable from "detector saw 0 items"."""
        if self._detect_state in (_DETECT_OK, _DETECT_DEGRADED):
            return count_items(dets)
        return None

    def _items_breakdown(self, dets: list, snaps: list) -> dict | None:
        """Per-category item counts (each detection joined to its cell's CLIP
        code); same activity guard as _items_count so the panel can tell
        "detector off" (None) from "no items" ({})."""
        if self._detect_state in (_DETECT_OK, _DETECT_DEGRADED):
            return items_by_code(dets, snaps)
        return None

    def _tick(self) -> None:
        """One shelf scan: frame -> per-slot CLIP classify -> slots/analytics."""
        now = time.time()
        demo = self._demo
        if demo and self.infer is not None and self._model_ready:
            live = True
            per_slot = self._demo_tick(now)        # also refreshes _last_dets
        elif self.infer is not None and self._model_ready:
            live = True
            per_slot = self._infer_once()          # also refreshes _last_dets
        else:
            live = False
            per_slot = self._synthetic_per_slot(
                now, slots=(self._demo_slot_dicts if demo else None))
            if self.cfg.detect_enabled:
                self._last_dets = self._synthetic_detections(now, demo=demo)
                self._detect_state = _DETECT_OK
            else:  # sim parity with the device path: DETECT_ENABLED=0 -> off
                self._last_dets = []
                self._detect_state = _DETECT_OFF

        # demo scans drive the d-prefixed slot set so heatmap/analytics rows
        # never collide with live g-cells (and vice versa)
        mgr = self._demo_slots_mgr if demo else self.slots_mgr
        snaps, _transitions = mgr.apply(per_slot)
        events = self.analytics.record(snaps, now=now)

        # stockout/restock fan out via the broker to /api/events subscribers
        self.broker.fanout(events)

        # HD-live: hand the scan overlay to the preview thread instead of
        # baking a JPEG here — the video cadence must not wait for the slow
        # slot-classify RPCs (~0.5 s on device). Preview thread draws these
        # cached snaps onto fresh frames at its own fps.
        dets = self._last_dets
        with self._overlay_lock:
            self._last_snaps = snaps

        scan = {
            "status": self._status,
            "model_ready": self._model_ready and live,
            "frame_seq": self._frame_seq,
            "mode": self.cfg.slot_mode,
            "demo": demo,
            "goods": goods_summary(snaps),
            "items": self._items_count(dets),
            "items_by_code": self._items_breakdown(dets, snaps),
            "detections": dets,
            "slots": [s.__dict__ for s in snaps],
            "events": events[-8:],
            "ts": now,
        }
        self._last_scan = scan
        self._broadcast_sse(scan)
        time.sleep(1.0 / max(self.cfg.infer_fps, 0.5))

    def _infer_once(self) -> dict[str, dict]:
        """Pull one frame and classify every slot crop; per-slot results.

        A failed crop is simply absent from the result dict — SlotManager
        holds the previous observation for that slot, so one bad frame cannot
        flap occupancy. Only an all-slots-failure scan counts as a failure
        toward the degraded status.
        """
        assert self.infer and self.media_infer and self.classifier
        try:
            frame = self._get_latest_frame(self.media_infer,
                                           self.cfg.stream_id)
        except Exception as e:  # noqa: BLE001
            self._nudge_degraded(e)
            return {}
        if frame is None:
            return {}

        rgb_frame = np.asarray(frame.to_rgb())
        codes = self.vocab.codes()

        timeout = INFER_WARMUP_TIMEOUT_MS if self._warmup_left > 0 else None
        if self._warmup_left > 0:
            self._warmup_left -= 1

        # chain A on the same frame the slot crops come from (adds ~0.25 s);
        # both RPC sections share the NPU socket with import analysis
        # (_analyze_import), so they serialize against request threads.
        per_slot: dict[str, dict] = {}
        first_error: Exception | None = None
        with self._npu_lock:
            self._detect_frame(rgb_frame, timeout)
            for slot in self.slots_mgr.slots:
                try:
                    box = polygon_bbox(slot.polygon, rgb_frame.shape,
                                       self.cfg.slot_crop_pad)
                    vec = self.classifier.embed(
                        self.infer, crop_box(rgb_frame, box), timeout_ms=timeout)
                    row, score = self.classifier.best(vec)
                except Exception as e:  # noqa: BLE001 - per-slot isolation
                    if first_error is None:
                        first_error = e
                    logger.debug("slot %s classify failed: %s", slot.id, e)
                    continue
                if score < self.cfg.clip_min_cos:
                    continue  # uncertain crop -> hold previous observation
                per_slot[slot.id] = {"code": codes[row], "score": score}

        if not per_slot and first_error is not None:
            self._nudge_degraded(first_error)
            return {}
        self._consecutive_failures = 0
        self._status = _STATUS_LIVE
        self._frame_seq += 1
        return per_slot

    def _nudge_degraded(self, err: Exception) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= DEGRADED_FAILURE_THRESHOLD:
            self._status = _STATUS_DEGRADED
        logger.debug("scan tick failed (%d): %s",
                     self._consecutive_failures, err)

    def _detect_frame(self, rgb_frame: np.ndarray,
                      timeout_ms: int | None, demo: bool = False) -> None:
        """Chain A: per-item detection on the scan frame, in place on
        self._last_dets. A failed call holds the last boxes for a few ticks
        (SlotManager holds slot observations the same way) and must NOT feed
        the CLIP degraded counter — the two chains fail independently.
        """
        if self.detector is None:
            return
        try:
            if demo:
                # demo grid is not the live grid: its own region/cell size,
                # plus import-grade tiles/threshold/cap so ITEMS visibly
                # climbs while the restock hands fill the shelf
                region = self.cfg.demo_grid_region
                rows, cols = self.cfg.demo_grid_rows, self.cfg.demo_grid_cols
                tiles = self.cfg.demo_detect_tiles
                threshold = self.cfg.demo_detect_threshold
                cap = self.cfg.demo_detect_max_items
            else:
                region = (self.cfg.grid_region
                          if self.cfg.slot_mode == "grid" else None)
                rows, cols = self.cfg.grid_rows, self.cfg.grid_cols
                # tiles=1 delegates to detect(); DETECT_TILES=4 opts the live
                # scan into the same 2x2 path imports use (at ~0.5 fps cost)
                tiles = self.cfg.detect_tiles
                threshold = None
                cap = self.cfg.detect_max_items
            self._last_dets = self.detector.detect_tiled(
                self.infer, rgb_frame, tiles=tiles,
                threshold=threshold, timeout_ms=timeout_ms, region=region,
                rows=rows, cols=cols, max_detections=cap,
            )
            if demo:
                # detect_tiled labels cells with its g-prefix; demo analytics
                # rows use d-ids, so rewrite before the items_by_code join
                for det in self._last_dets:
                    cell = det.get("cell") or ""
                    if cell.startswith("g"):
                        det["cell"] = "d" + cell[1:]
            self._detect_fail_ticks = 0
            self._detect_state = _DETECT_OK
        except Exception as e:  # noqa: BLE001 - detection must not kill CLIP
            self._detect_fail_ticks += 1
            self._detect_state = _DETECT_DEGRADED
            if self._detect_fail_ticks > _DETECT_HOLD_TICKS:
                self._last_dets = []
            logger.debug("detect tick failed (%d): %s",
                         self._detect_fail_ticks, e)

    # ---- built-in demo video mode --------------------------------------------
    def _build_demo_slots(self) -> list[dict]:
        """Demo grid slots: the same generator as the live grid but with
        d-prefixed ids, so demo analytics/heatmap rows never collide with
        live g-cells (both write into the same sqlite db)."""
        cfg = self.cfg
        slots = build_grid_slots(cfg.demo_grid_region, cfg.demo_grid_rows,
                                 cfg.demo_grid_cols)
        for s in slots:
            sid = str(s.get("id", ""))
            if sid.startswith("g"):
                s["id"] = "d" + sid[1:]
        return slots

    def _demo_video_abs(self, source: str | None = None) -> str:
        """Resolve the demo video path: the bundled asset, or the persisted
        copy of the last video import. Relative builtin paths resolve against
        the app dir."""
        if (source or self._demo_source) == "imported":
            return self._imported_video_path()
        path = self.cfg.demo_video_path
        if os.path.isabs(path):
            return path
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    def _imported_video_path(self) -> str:
        """Persistent slot for the last imported video: one file, overwritten
        by each successful import. Defaults to a media/ folder next to the
        analytics db — on device that is the config volume, so the upload
        survives container updates."""
        if self.cfg.import_media_dir:
            base = self.cfg.import_media_dir
        else:
            base = os.path.join(os.path.dirname(self.cfg.analytics_db),
                                "media")
        return os.path.join(base, "imported.mp4")

    def _demo_available(self) -> bool:
        return os.path.isfile(self._demo_video_abs("builtin"))

    def _demo_payload(self) -> dict:
        """Demo state block shared by /api/demo, /api/config, /api/health."""
        return {
            "enabled": self._demo,
            "source": self._demo_source,
            "available": self._demo_available(),
            "imported_available": os.path.isfile(self._imported_video_path()),
            "grid": {
                "region": [float(v) for v in self.cfg.demo_grid_region],
                "rows": self.cfg.demo_grid_rows,
                "cols": self.cfg.demo_grid_cols,
                "show_cells": True,
            },
            "video": self._demo_src.info if self._demo_src is not None else None,
        }

    def _set_demo(self, enabled: bool,
                  source: str | None = None) -> tuple[bool, str]:
        """Toggle demo mode (idempotent); source selects which video plays
        ("builtin" bundled asset or "imported" last upload). Switching the
        source while demo is on swaps the decoder in place — grid and demo
        timers are shared and stay as they are."""
        from demo import DemoVideoSource  # lazy: cv2 cost only when used

        with self._demo_lock:
            target = source or self._demo_source
            if enabled and self._demo and target == self._demo_source:
                return True, "demo already on"
            if not enabled and not self._demo:
                return True, "demo already off"
            if enabled:
                path = self._demo_video_abs(target)
                if not os.path.isfile(path):
                    missing = ("no imported video yet — import one first"
                               if target == "imported"
                               else f"demo video missing: {path}")
                    return False, missing
                try:
                    src = DemoVideoSource(path, speed=self.cfg.demo_speed)
                except (OSError, ValueError) as e:
                    return False, f"demo video unusable: {e}"
                swapped = self._demo and target != self._demo_source
                if self._demo_src is not None:
                    self._demo_src.close()
                self._demo_src = src
                self._demo_source = target
                if not self._demo:
                    self._demo_slot_dicts = self._build_demo_slots()
                    self._demo_slots_mgr = SlotManager(self._demo_slot_dicts)
                    # fast demo timers: the live 120 s stockout threshold and
                    # 30 s alert cooldown would swallow the ~21 s loop entirely
                    self.analytics.stockout_seconds = (
                        self.cfg.demo_stockout_seconds)
                    self.broker.cooldown_seconds = (
                        self.cfg.demo_alert_cooldown_seconds)
                    self._demo = True
                logger.info("demo mode ON (%s, speed=%.2fx, loop %.1f s)",
                            src.path, src.speed, src.info["loop_seconds"])
                return True, ("demo source swapped" if swapped else "demo on")
            # off: restore live timers, drop the decoder, reset demo slots
            self.analytics.stockout_seconds = self._live_stockout_seconds
            self.broker.cooldown_seconds = self._live_alert_cooldown
            if self._demo_src is not None:
                self._demo_src.close()
                self._demo_src = None
            self._demo = False
            self._demo_slot_dicts = self._build_demo_slots()
            self._demo_slots_mgr = SlotManager(self._demo_slot_dicts)
            logger.info("demo mode OFF (live timers restored)")
            return True, "demo off"

    def _demo_route(self):
        """GET /api/demo -> state; POST /api/demo {"enabled": bool,
        "source"?: "builtin"|"imported"} (source defaults to the current
        one — the covert toggle button keeps working unchanged)."""
        if request.method == "GET":
            return jsonify({"ok": True, "demo": self._demo_payload()})
        body = request.get_json(silent=True)
        if (not isinstance(body, dict)
                or not isinstance(body.get("enabled"), bool)):
            return jsonify({"ok": False,
                            "error": 'body must be {"enabled": true|false}'}), 400
        source = body.get("source")
        if source is not None and source not in ("builtin", "imported"):
            return jsonify({"ok": False,
                            "error": 'source must be "builtin" or "imported"'
                            }), 400
        ok, msg = self._set_demo(body["enabled"], source)
        return jsonify({"ok": ok, "message": msg,
                        "demo": self._demo_payload()}), (200 if ok else 409)

    def _demo_tick(self, now: float) -> dict[str, dict]:
        """One scan over the demo video: paced frame -> demo-grid detect +
        per-slot CLIP classify (device path; simulation uses the synthetics).
        Reads hold the previous observation exactly like the live path."""
        src = self._demo_src
        if src is None or self.infer is None or self.classifier is None:
            return {}
        rgb = src.read_rgb()
        codes = self.vocab.codes()

        timeout = INFER_WARMUP_TIMEOUT_MS if self._warmup_left > 0 else None
        if self._warmup_left > 0:
            self._warmup_left -= 1

        per_slot: dict[str, dict] = {}
        with self._npu_lock:
            self._detect_frame(rgb, timeout, demo=True)
            for slot in self._demo_slots_mgr.slots:
                try:
                    box = polygon_bbox(slot.polygon, rgb.shape,
                                       self.cfg.slot_crop_pad)
                    vec = self.classifier.embed(
                        self.infer, crop_box(rgb, box), timeout_ms=timeout)
                    row, score = self.classifier.best(vec)
                except Exception:  # noqa: BLE001 - per-slot isolation
                    continue
                if score < self.cfg.clip_min_cos:
                    continue  # uncertain crop -> hold previous observation
                per_slot[slot.id] = {"code": codes[row], "score": score}

        self._consecutive_failures = 0
        self._status = _STATUS_LIVE
        self._frame_seq += 1
        return per_slot

    def _synthetic_detections(self, now: float, demo: bool = False) -> list[dict]:
        """Simulation boxes: 3-5 goods cycling inside the grid + one stray
        negative drifting along the bottom edge (same schema as detect()).
        demo=True renders the demo grid geometry with d-prefixed cells."""
        if demo:
            rx1, ry1, rx2, ry2 = self.cfg.demo_grid_region
            rows, cols = self.cfg.demo_grid_rows, self.cfg.demo_grid_cols
            prefix = "d"
        else:
            rx1, ry1, rx2, ry2 = self.cfg.grid_region
            rows, cols = self.cfg.grid_rows, self.cfg.grid_cols
            prefix = "g"
        cw, ch = (rx2 - rx1) / cols, (ry2 - ry1) / rows
        t = now * 0.05
        n = 3 + (int(now // 15) % 3)
        out: list[dict] = []
        for i in range(n):
            r, c = divmod(i, cols)
            cx = rx1 + (c + 0.5) * cw + 0.01 * math.sin(t + i)
            cy = ry1 + (r + 0.5) * ch + 0.01 * math.cos(t * 1.3 + i)
            w, h = cw * 0.5, ch * 0.6
            out.append({
                "label": _SIM_DET_LABELS[i % len(_SIM_DET_LABELS)],
                "score": round(0.55 + 0.2 * (0.5 + 0.5 * math.sin(t * 2 + i)),
                               4),
                "box": [round(v, 5) for v in
                        (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)],
                "goods": True,
                "category": _SIM_DET_CATEGORIES[i % len(_SIM_DET_CATEGORIES)],
                "cell": f"{prefix}{r + 1}-{c + 1}",
            })
        hx = min(max(rx1 + (rx2 - rx1) * (0.5 + 0.4 * math.sin(t * 0.7)),
                     rx1 + 0.05), rx2 - 0.05)
        col = min(int((hx - rx1) / (rx2 - rx1) * cols), cols - 1)
        out.append({
            "label": "hand", "score": 0.41,
            "box": [round(v, 5) for v in (hx - 0.04, ry2 - 0.12,
                                          hx + 0.04, ry2 - 0.02)],
            "goods": False,
            "category": "negative",
            "cell": f"{prefix}{rows}-{col + 1}",
        })
        return out

    def _synthetic_per_slot(self, now: float,
                            slots: list | None = None) -> dict[str, dict]:
        """Simulation results: cycle each slot through codes + vacancies."""
        out: dict[str, dict] = {}
        for i, slot in enumerate(slots if slots is not None
                                 else (self.cfg.slots or [])):
            sid = str(slot.get("id", ""))
            if not sid:
                continue
            phase = (int(now // 20) + i) % len(_SIM_PATTERN)
            code = _SIM_PATTERN[phase]
            out[sid] = {"code": code,
                        "score": 0.88 if code != "EMPTY" else 0.76}
        return out

    def _get_latest_frame(self, media: FdMediaClient, stream_id: str,
                          timeout_ms: int = 3000):
        """Pull one frame then drain queued frames to keep the scan fresh."""
        frame = media.get_frame(stream_id, timeout_ms=timeout_ms)
        if frame is None:
            return None
        deadline = time.monotonic() + 0.12
        drained = 0
        while drained < 120 and time.monotonic() < deadline:
            nxt = media.get_frame(stream_id, timeout_ms=1)
            if nxt is None:
                break
            frame = nxt
            drained += 1
        return frame

    def _preview_loop(self) -> None:
        """Decoupled MJPEG preview (gym-ops HD-live pattern).

        A dedicated thread on its own FdMediaClient renders video at
        cfg.preview_fps while the scan loop runs at the slow infer_fps
        (~2 FPS on device). The most recent scan's overlay (slot tints +
        code chips) is drawn onto each fresh frame, downscaled to
        cfg.preview_width, and JPEG-encoded into _latest_jpeg. Since no
        browser is watching, the loop idles instead of wasting CPU.
        """
        target_dt = 1.0 / max(self.cfg.preview_fps, 5)
        while self._running.is_set():
            with self._stream_clients_lock:
                has_clients = self._stream_clients > 0
            if not has_clients:
                time.sleep(0.2)
                continue
            t0 = time.time()
            frame = self._preview_frame()
            if frame is None:
                time.sleep(target_dt)
                continue
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._jpeg_lock:
                    self._latest_jpeg = buf.tobytes()
            dt = time.time() - t0
            if dt < target_dt:
                time.sleep(target_dt - dt)

    def _preview_frame(self) -> np.ndarray | None:
        """One BGR frame for MJPEG: camera frame + last scan overlay.

        Falls back to a synthetic dark canvas when no device is attached so
        the stream + slot overlay still render in simulation mode.
        """
        with self._overlay_lock:
            snaps = self._last_snaps
            dets = self._last_dets
        frame = None
        if self._demo_src is not None:  # demo video replaces the camera
            try:
                frame = self._demo_src.read_bgr()
            except Exception:  # noqa: BLE001
                frame = None
        elif self.media is not None:
            try:
                f = self._get_latest_frame(self.media, self.cfg.stream_id,
                                           timeout_ms=500)
                if f is not None:
                    frame = cv2.cvtColor(f.to_rgb(), cv2.COLOR_RGB2BGR)
            except Exception:  # noqa: BLE001
                frame = None
        if frame is None:  # simulation fallback
            w, h = 640, 480
            frame = np.full((h, w, 3), 18, dtype=np.uint8)  # dark canvas
            cv2.line(frame, (0, h // 2), (w, h // 2), (34, 34, 34), 1)  # shelf
        return self._draw_overlay(frame, snaps, dets)

    def _draw_overlay(self, frame: np.ndarray, snaps: list,
                      dets: list | None = None) -> np.ndarray:
        """Draw slot tints + code chips + detection boxes; downscale first.

        Downscaling FIRST keeps the JPEG cheap at preview fps — overlay.py
        draws in normalized coords so polygons/chips/boxes scale after
        resize.
        """
        h, w = frame.shape[:2]
        pw = max(self.cfg.preview_width, 320)
        if pw < w:
            frame = cv2.resize(frame, (pw, max(1, int(h * pw / w))),
                               interpolation=cv2.INTER_LINEAR)
        dets = dets or []
        demo_items_only = self._demo   # demo video: clean view, boxes only
        try:
            if snaps and not demo_items_only:
                if self.cfg.slot_mode == "grid":
                    grid = self.cfg.grid_region
                else:
                    grid = None
                draw_slots(frame, snaps, self.slots_mgr.slots,
                           grid_region=grid, items=self._items_count(dets),
                           show_cells=self.cfg.grid_show_cells)
        except Exception as e:  # noqa: BLE001
            logger.warning("overlay draw failed: %s", e)
        try:
            if dets:
                # demo hides shelf/price-tag boxes and the grid/chips layer;
                # the rack panel + events feed carry the occupancy story
                draw_detections(frame, dets, goods_only=demo_items_only)
        except Exception as e:  # noqa: BLE001
            logger.warning("detection overlay failed: %s", e)
        return frame

    # ---- SSE ---------------------------------------------------------------
    def _broadcast_sse(self, snap: dict) -> None:
        data = json.dumps(snap, default=str)
        with self._sse_lock:
            dead: list[queue.Queue] = []
            for q in self._sse_clients:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    dead.append(q)
            for d in dead:
                self._sse_clients.remove(d)

    # ---- media import (offline photo analysis) ------------------------------

    def _import_image(self):
        """POST /api/import/image — one shelf photo through the same two
        chains (CLIP cell classify + per-item detect) on a transient
        SlotManager; returns annotated JPEG + counts. Live occupancy,
        heatmap and events are never touched by an import. """
        f = request.files.get("file")
        if f is not None:
            raw, filename = f.read(), f.filename or ""
        else:  # raw-body POST (scripting); name via ?filename=
            raw, filename = request.get_data(), request.args.get("name", "")
        try:
            validate_image_upload(raw, filename, self.cfg.import_max_image_mb)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 415
        try:
            rgb = decode_image_bytes(raw)
        except Exception:  # noqa: BLE001 - PIL rejected the bytes
            return jsonify({"ok": False,
                            "error": "could not decode image"}), 415
        if not self._import_lock.acquire(blocking=False):
            return jsonify({"ok": False,
                            "error": "another import is still analyzing"}), 409
        try:
            result = self._analyze_import(rgb)
        finally:
            self._import_lock.release()
        return jsonify({"ok": True, "result": result})

    def _analyze_import(self, rgb: np.ndarray) -> dict:
        """One-shot analysis of an uploaded frame, mirroring _infer_once /
        _detect_frame / the sim branch of _tick — but on a transient
        SlotManager so imported photos never pollute live analytics. """
        h, w = rgb.shape[:2]
        dets: list = []
        if self.infer is not None and self._model_ready:
            one_shot = SlotManager(self.cfg.slots)
            if self.detector is not None:
                region = (self.cfg.grid_region
                          if self.cfg.slot_mode == "grid" else None)
                with self._npu_lock:
                    try:
                        # accurate path: 2x2 tiles + probe53 threshold —
                        # small goods survive that a full-frame 640
                        # letterbox starves (~30px cans shrink to ~16px)
                        dets = self.detector.detect_tiled(
                            self.infer, rgb,
                            tiles=self.cfg.import_detect_tiles,
                            threshold=self.cfg.import_detect_threshold,
                            max_detections=self.cfg.import_detect_max_items,
                            timeout_ms=None, region=region,
                            rows=self.cfg.grid_rows,
                            cols=self.cfg.grid_cols)
                    except Exception as e:  # noqa: BLE001
                        # detection failure must not kill the CLIP pass
                        logger.warning("import detect failed: %s", e)
            codes = self.vocab.codes()
            per_slot: dict[str, dict] = {}
            with self._npu_lock:
                for slot in one_shot.slots:
                    try:
                        box = polygon_bbox(slot.polygon, rgb.shape,
                                           self.cfg.slot_crop_pad)
                        vec = self.classifier.embed(
                            self.infer, crop_box(rgb, box), timeout_ms=None)
                        row, score = self.classifier.best(vec)
                    except Exception as e:  # noqa: BLE001 - per-slot isolation
                        logger.debug("import slot %s failed: %s", slot.id, e)
                        continue
                    if score < self.cfg.clip_min_cos:
                        continue  # uncertain crop -> no observation
                    per_slot[slot.id] = {"code": codes[row], "score": score}
            snaps, _ = one_shot.apply(per_slot)
        else:  # sim parity with _tick: synthetic results, same shapes
            now = time.time()
            one_shot = SlotManager(self.cfg.slots)
            snaps, _ = one_shot.apply(self._synthetic_per_slot(now))
            if self.cfg.detect_enabled:
                dets = self._synthetic_detections(now)

        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frame = self._draw_overlay(frame, snaps, dets)
        return {
            "width": w,
            "height": h,
            "goods": goods_summary(snaps),
            "items": count_items(dets),
            "items_by_code": items_by_code(dets, snaps),
            "items_by_category": items_by_category(dets),
            "detections": dets,
            "annotated": jpeg_b64(frame),
            "ts": time.time(),
        }

    # ---- media import (offline video analysis) ------------------------------

    def _import_video(self):
        """POST /api/import/video — accept an MP4/MOV, sample it evenly, and
        run the per-item detect chain on every sampled frame in a background
        thread. Returns 202 + a job view the client polls on
        /api/import/video/status. The _import_lock is held from here until
        the worker finishes (one import at a time, 409 otherwise); the
        _npu_lock is taken per frame so live scanning keeps running between
        frames. """
        f = request.files.get("file")
        if f is not None:
            raw, filename = f.read(), f.filename or ""
        else:  # raw-body POST (scripting); name via ?name=
            raw, filename = request.get_data(), request.args.get("name", "")
        try:
            ext = validate_video_upload(raw, filename,
                                        self.cfg.import_max_video_mb)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 415
        if not self._import_lock.acquire(blocking=False):
            return jsonify({"ok": False,
                            "error": "another import is still analyzing"}), 409
        fd, path = tempfile.mkstemp(suffix=ext)
        job: dict | None = None
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(raw)
            # Synchronous probe: a container cv2 cannot open must fail now
            # (422), not inside the worker the client cannot influence.
            cap = cv2.VideoCapture(path)
            try:
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            finally:
                cap.release()
            if frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
                return jsonify({"ok": False,
                                "error": "could not open video stream"}), 422
            positions = sample_positions(frames,
                                         self.cfg.import_video_max_frames)
            job = {
                "state": "running",
                "done": 0,
                "total": len(positions),
                "error": None,
                "result": None,
                "video": {
                    "width": width,
                    "height": height,
                    "fps": round(fps, 2),
                    "frames": frames,
                    "duration_seconds": round(frames / fps, 1),
                },
            }
            self._video_job = job
            threading.Thread(target=self._run_video_import,
                             args=(path, job), daemon=True).start()
            path = None  # ownership moved to the worker
        finally:
            if job is None:  # failed before handoff: clean up here
                os.unlink(path)
                self._import_lock.release()
        return jsonify({"ok": True, "job": self._video_view(self._video_job)}), 202

    def _import_video_status(self):
        """GET /api/import/video/status — single-slot job view (None if no
        video was ever imported); includes result once state is done. """
        return jsonify({"ok": True, "job": self._video_view(self._video_job)})

    def _video_view(self, job: dict | None) -> dict | None:
        if job is None:
            return None
        view = {"state": job["state"], "done": job["done"],
                "total": job["total"], "error": job["error"],
                "video": job["video"]}
        if job["state"] == "done" and job["result"] is not None:
            view["result"] = job["result"]
        return view

    def _run_video_import(self, path: str, job: dict) -> None:
        """Worker wrapper: analyze, record outcome, keep the upload for
        demo playback when it analyzed cleanly, and always release the
        import lock and the temp file. """
        try:
            job["result"] = self._analyze_video(path, job)
            job["state"] = "done"
            self._keep_imported_video(path)
        except Exception as e:  # noqa: BLE001 - surfaced via status
            logger.warning("video import failed: %s", e)
            job["state"] = "error"
            job["error"] = str(e)
        finally:
            try:
                os.unlink(path)   # no-op when os.replace already moved it
            except OSError:
                pass
            self._import_lock.release()

    def _keep_imported_video(self, path: str) -> bool:
        """Move the analyzed upload to the persistent media slot so demo
        mode can play it back as the live feed (one file; the next import
        overwrites it). Best-effort: an unwritable destination logs a
        warning and leaves analysis unaffected. On device the config volume
        is a different filesystem from the temp dir, so os.replace fails
        EXDEV and we fall back to copy + unlink of the original."""
        dest = self._imported_video_path()
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                os.replace(path, dest)   # same filesystem: atomic + free
            except OSError:              # cross-device temp dir: copy instead
                shutil.copyfile(path, dest)
            return True
        except Exception as e:  # noqa: BLE001 - playback is optional
            logger.warning("keeping imported video failed: %s", e)
            return False

    def _analyze_video(self, path: str, job: dict) -> dict:
        """Per-frame analysis: seek each sampled position, run the per-item
        detect chain (import-grade tiles/threshold, no region/cell join — a
        video is not the camera view), aggregate the timeline, and annotate
        up to import_video_keyframes stills. """
        cap = cv2.VideoCapture(path)
        try:
            fps = job["video"]["fps"] or 1.0
            positions = sample_positions(job["video"]["frames"],
                                         self.cfg.import_video_max_frames)
            keyframe_at = set(sample_positions(len(positions),
                                              self.cfg.import_video_keyframes))
            timeline: list[dict] = []
            keyframes: list[dict] = []
            peak_items, peak_t, peak_dets = -1, 0.0, []
            for i, pos in enumerate(positions):
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, frame = cap.read()
                if ok and frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    dets = self._detect_import_frame(rgb)
                    items = count_items(dets)
                    t = round(pos / fps, 1)
                    timeline.append({"t": t, "items": items})
                    if items > peak_items:
                        peak_items, peak_t, peak_dets = items, t, dets
                    if i in keyframe_at:
                        keyframes.append({
                            "t": t, "items": items,
                            "image": jpeg_b64(
                                self._annotated_still(frame, dets)),
                        })
                job["done"] += 1
            return {
                "video": job["video"],
                "sampled": len(positions),
                "items_peak": max(peak_items, 0),
                "peak_t": peak_t,
                "items_mean": (round(sum(s["items"] for s in timeline)
                                     / len(timeline), 1)
                               if timeline else 0),
                "items_by_category": items_by_category(peak_dets),
                "timeline": timeline,
                "keyframes": keyframes,
                "ts": time.time(),
            }
        finally:
            cap.release()

    def _detect_import_frame(self, rgb: np.ndarray) -> list:
        """Per-item detections for one imported video frame — the accurate
        import path (tiles + lowered threshold), _npu_lock held only around
        the RPC so live scans interleave between frames. """
        if self.infer is not None and self._model_ready \
                and self.detector is not None:
            with self._npu_lock:
                try:
                    return self.detector.detect_tiled(
                        self.infer, rgb,
                        tiles=self.cfg.import_detect_tiles,
                        threshold=self.cfg.import_detect_threshold,
                        max_detections=self.cfg.import_detect_max_items,
                        timeout_ms=None)
                except Exception as e:  # noqa: BLE001 - frame isolation
                    logger.warning("video frame detect failed: %s", e)
                    return []
        if self.cfg.detect_enabled:  # sim parity with _analyze_import
            return self._synthetic_detections(time.time())
        return []

    @staticmethod
    def _annotated_still(frame_bgr: np.ndarray, dets: list) -> np.ndarray:
        """Annotated keyframe: boxes painted on a copy, capped at 640px wide
        so the base64 payload stays small. """
        out = frame_bgr.copy()
        draw_detections(out, dets)
        h, w = out.shape[:2]
        if w > 640:
            scale = 640 / w
            out = cv2.resize(out, (640, max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        return out

    # ---- routes ------------------------------------------------------------
    def _register_routes(self) -> None:
        self.app.add_url_rule("/api/health", "health", self._health)
        self.app.add_url_rule("/api/config", "config", self._config)
        self.app.add_url_rule("/api/state", "state", self._state)
        self.app.add_url_rule("/api/heatmap", "heatmap", self._heatmap)
        self.app.add_url_rule("/api/events", "events", self._events)
        self.app.add_url_rule("/api/preview", "preview", self._preview)
        self.app.add_url_rule("/api/import/image", "import_image",
                              self._import_image, methods=["POST"])
        self.app.add_url_rule("/api/import/video", "import_video",
                              self._import_video, methods=["POST"])
        self.app.add_url_rule("/api/import/video/status", "import_video_status",
                              self._import_video_status, methods=["GET"])
        self.app.add_url_rule("/api/demo", "demo", self._demo_route,
                              methods=["GET", "POST"])
        self.app.add_url_rule("/", "index", self._index)
        self.app.add_url_rule("/stream", "stream", self._stream)

    def _health(self):
        return jsonify({
            "ok": True,
            "app": "shelf-ops",
            "status": self._status,
            "model": self._clip_model_id or self.cfg.clip_model,
            "model_ready": self._model_ready,
            "detector": self._detect_state,
            "detector_model": self.detector.model_id if self.detector else None,
            "mode": self.cfg.embedding_mode,
            "slots": len(self.cfg.slots),
            "demo": self._demo,
        })

    def _config(self):
        """Slots + vocabulary for the UI (names + polygons + prompt groups).

        When the demo video is on, the slot/grid blocks swap to the demo view
        (d-prefixed cells + demo grid with forced cell borders) so the rack
        panel and any canvas overlay follow the demo instead of the live
        planogram; the frontend re-fetches this after toggling /api/demo.
        """
        vocab = []
        for v in self.vocab.sorted_entries():
            vocab.append({
                "code": v.code,
                "label": v.label,
                "label_cn": v.label_cn,
                "prompt": v.prompt,
            })
        demo = self._demo_payload()
        return jsonify({
            "slots": self._demo_slot_dicts if self._demo else self.cfg.slots,
            "mode": "grid" if self._demo else self.cfg.slot_mode,
            "grid": (demo["grid"] if self._demo else
                     (self.cfg.grid_dict() if self.cfg.slot_mode == "grid"
                      else None)),
            "vocabulary": vocab,
            "infer_fps": self.cfg.infer_fps,
            "preview_fps": self.cfg.preview_fps,
            "clip_min_cos": self.cfg.clip_min_cos,
            "slot_crop_pad": self.cfg.slot_crop_pad,
            "demo": demo,
        })

    def _state(self):
        """Latest shelf snapshot for polling clients."""
        if self._last_scan is None:
            return jsonify({"status": self._status, "scan": None})
        scan = dict(self._last_scan)
        scan["status"] = self._status
        return jsonify(scan)

    def _heatmap(self):
        """Time-bucket aggregation: ?span=1h|6h|1d|7d & ?slot=& ?code="""
        span = request.args.get("span", "6h")
        buckets = {"1h": 6, "6h": 36, "1d": 144, "7d": 1008}.get(span, 36)
        rows = self.analytics.heatmap(buckets=buckets)
        slot = request.args.get("slot")
        code = request.args.get("code")
        if slot:
            rows = [r for r in rows if r["slot_id"] == slot]
        if code:
            rows = [r for r in rows if r["code"] == code]
        return jsonify({
            "span": span,
            "bucket_seconds": self.cfg.bucket_seconds,
            "rows": rows,
        })

    def _events(self):
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._sse_lock:
            self._sse_clients.append(q)
        # stockout/restock events fan out through the broker; scans broadcast
        # via _broadcast_sse. Both land on the same per-client queue: scan
        # snapshots at tick rate, alert events when analytics confirms them.
        self.broker.subscribe(q)

        def gen():
            try:
                yield "data: connected\n\n"
                while self._running.is_set():
                    try:
                        data = q.get(timeout=15)
                        yield f"data: {data}\n\n"
                    except queue.Empty:
                        yield ": keepalive\n\n"
            finally:
                with self._sse_lock:
                    if q in self._sse_clients:
                        self._sse_clients.remove(q)
                self.broker.unsubscribe(q)
        return Response(gen(), mimetype="text/event-stream")

    def _preview(self):
        """Platform-api H.264 preview info: browser plays it natively via MSE
        (ported from gym-ops `_preview` / model-showcase main.py).

        app-manager does NOT expand `${VAR}` references in the manifest, so
        PLATFORM_API_TOKEN can arrive as the literal `${AIPC_TOKEN_KEY}`; that
        must report enabled=false so the UI falls back to the MJPEG stream
        instead of opening a doomed WebSocket.
        """
        scheme = self.cfg.platform_api_ws_scheme
        port = self.cfg.platform_api_port
        token = self.cfg.platform_api_token
        token_ok = bool(token) and "${" not in token
        host = request.host.split(":")[0] if request.host else "localhost"
        ws_url = ""
        if token_ok:
            ws_url = f"{scheme}://{host}:{port}/api/v1/h264/{self.cfg.stream_id}"
            ws_url += "?token=" + urllib.parse.quote(token, safe="")
        return jsonify({"enabled": token_ok, "wsUrl": ws_url,
                        "stream_id": self.cfg.stream_id})

    def _index(self):
        return render_template("index.html", cfg=self.cfg, status=self._status)

    # host/ip-literal only — anything else must not leak into a security header
    _HOST_RE = re.compile(r"^[A-Za-z0-9.\-:\[\]]+$")

    def _csp(self) -> str:
        """Per-request CSP. connect-src must also admit the platform-api H.264
        WebSocket (wss://<same-host>:443) — a different origin from both the
        direct :8891 page and the app-manager /apps reverse proxy, so a static
        'self' would block the HD preview. media-src admits blob: because MSE
        feeds the <video> a URL.createObjectURL(MediaSource) blob."""
        host = request.host.split(":")[0] if request.host else ""
        ws_src = (f" wss://{host}:{self.cfg.platform_api_port}"
                  if host and self._HOST_RE.match(host) else "")
        return ("default-src 'self'; img-src 'self' data:; "
                "script-src 'self'; style-src 'self' 'unsafe-inline'; "
                f"connect-src 'self'{ws_src}; media-src 'self' blob:; "
                "form-action 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; object-src 'none'")

    def _decorate_headers(self, resp: Response) -> Response:
        """Minimal security headers on every response.

        The page is a LAN ops console with no state-changing endpoints; the CSP
        (script-src 'self', connect-src 'self' for SSE/fetch, frame-ancestors
        'none') is defense-in-depth against any future injection. style-src
        allows 'unsafe-inline' for the rack meter/heat tints set via innerHTML.
        """
        resp.headers.setdefault("Content-Security-Policy", self._csp())
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return resp

    def _stream(self):
        def gen():
            with self._stream_clients_lock:
                self._stream_clients += 1
            try:
                boundary = b"--frame\r\n"
                while self._running.is_set():
                    with self._jpeg_lock:
                        jpg = self._latest_jpeg
                    if jpg is None:
                        time.sleep(0.2)
                        continue
                    yield (boundary + b"Content-Type: image/jpeg\r\n\r\n"
                           + jpg + b"\r\n")
                    time.sleep(1.0 / max(self.cfg.preview_fps, 1))
            finally:
                with self._stream_clients_lock:
                    self._stream_clients = max(0, self._stream_clients - 1)
        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    # ---- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        # Register web URL with the platform so the app-manager console
        # shows a "Visit App" button on the shelf-ops card (mirrors gym-ops).
        try:
            AppClient().register_web_url("/")
        except Exception as e:  # noqa: BLE001
            logger.warning("register_web_url failed: %s (non-fatal)", e)
        self._register_routes()
        self.app.after_request(self._decorate_headers)
        self._worker.start()
        self._preview_thread = threading.Thread(target=self._preview_loop,
                                                name="shelf-preview",
                                                daemon=True)
        self._preview_thread.start()

    def shutdown(self) -> None:
        self._running.clear()
        for t in (self._worker, getattr(self, "_preview_thread", None)):
            if t is not None:
                t.join(timeout=2)
        if self._demo_src is not None:
            self._demo_src.close()
            self._demo_src = None
        try:
            self.analytics.close()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    cfg = load_config()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("shelf-ops starting (clip=%s port=%d)",
                cfg.clip_model, cfg.web_port)

    app = ShelfOpsApp(cfg)
    app.start()
    try:
        app.app.run(host="0.0.0.0", port=cfg.web_port, threaded=True,
                    use_reloader=False)
    except KeyboardInterrupt:
        app.shutdown()


if __name__ == "__main__":
    main()
