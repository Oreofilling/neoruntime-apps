"""Pure helpers for the teaching app: constants, model/frame plumbing,
event gating, buffers, rolling-window stats and the settings store.

Everything here is offline-testable: no SDK object is constructed at
import time (tests stub neoruntime_ipc_sdk before loading). Device-
learned contracts are documented on the functions that enforce them.
"""

import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2  # draw_badge only (headless wheel in the image)

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

MODEL_FILENAME = "hailo_yolov8n_384_640.hef"
MODEL_ID = "yolov8n_384_640"
HOST_MODEL_DIRS = ("/data/aipc/models/detection",)
BUNDLED_MODEL_DIR = "/opt/aipc/bundled-models"

ALL_LABELS = ("person", "vehicle", "face", "license_plate")

STREAM_ID = "third"                 # geometry == model input (640x384)
OVERLAY_STREAM = "main"             # console stream, same FOV
OVERLAY_TTL_MS = 1000
ZONE_REFRESH_S = 0.8                # zone lease refresh cadence (B loop)
ZONE_POLYGON = {
    "points": [[0.08, 0.08], [0.55, 0.08], [0.55, 0.55], [0.08, 0.55]],
    "label": "teaching zone",
    "closed": True,
}

EVENT_COOLDOWN_S = 5.0
JPEG_QUALITY = 82
WINDOW_S = 5.0

POLICIES = ("software_only", "prefer_hardware", "hardware_only")
OVERLAY_PATHS = ("off", "platform")
BOOL_SETTINGS = ("bform_on", "draw_on", "zone_on", "aform_on")
MIN_SCORE_MIN, MIN_SCORE_MAX = 0.05, 0.9

REFUSAL_OP = "teaching.no_such_leg"  # unregistered op: routing refusal demo

AFORM_STATUS_FIELDS = (
    "running", "results_seen", "results_annotated", "annotate_errors",
    "result_queue_drops", "subscribe_dropped", "last_error",
    "last_latency_ms", "avg_latency_ms", "last_skew_us", "avg_skew_us",
)


def log(message):
    app_id = os.environ.get("APP_ID", "sdk-teaching-demo")
    print(f"[{app_id}] {message}", flush=True)


def _sdk():
    """Late import so tests can stub neoruntime_ipc_sdk first."""
    import neoruntime_ipc_sdk
    return neoruntime_ipc_sdk


# --------------------------------------------------------------------------
# model + frame helpers (contracts identical to the SDK examples)
# --------------------------------------------------------------------------

def resolve_model_path():
    """Host-first resolution; the flat /data/aipc/models/ root is a trap.

    Some devices keep a same-named file there that is a DIFFERENT compile
    of the network — only the detection/ copy matches the vendored sha256.
    """
    for directory in HOST_MODEL_DIRS:
        candidate = Path(directory) / MODEL_FILENAME
        if candidate.is_file():
            return candidate, "host"
    bundled = Path(BUNDLED_MODEL_DIR) / MODEL_FILENAME
    if bundled.is_file():
        return bundled, "bundled"
    raise FileNotFoundError(
        f"{MODEL_FILENAME} not found in {HOST_MODEL_DIRS} (host mount) "
        f"or {BUNDLED_MODEL_DIR} (image); expected at least one source"
    )


def ensure_model_registered(inference):
    """Register once per device; True if this call did the registering.

    model_type="detection" enables server-side NMS decode (objects
    populated); model_variant MUST be None — a bare string gets wrapped
    as a backend_function and infer fails HAILO_INVALID_OPERATION.
    Re-registering an existing id breaks the postprocess link, hence the
    list_models() guard.
    """
    for info in inference.list_models():
        if info.model_id == MODEL_ID:
            log(f"model '{MODEL_ID}' already registered, reusing")
            return False
    path, source = resolve_model_path()
    log(f"registering model '{MODEL_ID}' from {source} path {path}")
    inference.register_model(
        str(path), model_id=MODEL_ID, model_type="detection", model_variant=None
    )
    return True


def to_input_box(obj, meta):
    """Source-frame coords -> model-input coords (dst = src*scale+origin)."""
    x1 = obj.bbox.x * meta.scale[0] + meta.origin[0]
    y1 = obj.bbox.y * meta.scale[1] + meta.origin[1]
    x2 = (obj.bbox.x + obj.bbox.width) * meta.scale[0] + meta.origin[0]
    y2 = (obj.bbox.y + obj.bbox.height) * meta.scale[1] + meta.origin[1]
    return _sdk().DetectedObject(
        label=obj.label, score=obj.score, class_id=obj.class_id,
        bbox=_sdk().BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
    )


def to_numpy(tensor):
    """Materialize whatever the pipeline hands back into a numpy array.

    Three shapes exist: a plain numpy array (passthrough), a Frame
    (to_array), and a retained input ref — a DspBufferRef, whose
    read-back is read() (passing the ref itself to draw_detections dies
    on `image.copy()` one stack frame later).
    """
    if hasattr(tensor, "to_array"):  # Frame
        return tensor.to_array()
    if hasattr(tensor, "read"):  # DspBufferRef (retained input)
        return tensor.read()
    return tensor


def as_frame(array):
    """Wrap a drawn tensor back into a Frame for MJPEG encoding.

    2D arrays are NV12 in the SDK layout (h*3//2, w) — the zero-copy
    passthrough case, and drawing preserves the input format. 3D arrays
    are RGB. Mislabelling NV12 as RGB crashes JPEG encoding three stack
    frames later, so the dimension check matters.
    """
    if array.ndim == 2:
        height = array.shape[0] * 2 // 3
        width = array.shape[1]
        fmt = "NV12"
    else:
        height, width = array.shape[:2]
        fmt = "RGB"
    return _sdk().Frame(sequence=0, timestamp_ns=0, width=width,
                        height=height, format=fmt, image=array)


def draw_badge(array, text):
    """Policy/objects badge on the pixel path (luma plane for NV12).

    Drawing on pixels the app owns is the whole point of the B form —
    the cheapest possible demonstration of that ownership.
    """
    if array.ndim == 2:
        height = array.shape[0] * 2 // 3
        plane = array[:height, :]
    else:
        plane = array
    cv2.putText(plane, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                0, 3, cv2.LINE_AA)  # black shadow
    cv2.putText(plane, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                255, 1, cv2.LINE_AA)  # white pass
    return array


# --------------------------------------------------------------------------
# events (pure decision logic, offline-testable)
# --------------------------------------------------------------------------

def payload_from_objects(sequence, objects):
    """(frame_sequence, filtered objects) -> JSON-safe detection dict."""
    return {
        "frame_sequence": sequence,
        "count": len(objects),
        "objects": [
            {
                "label": obj.label,
                "score": round(float(obj.score), 3),
                "bbox": [obj.bbox.x, obj.bbox.y,
                         obj.bbox.width, obj.bbox.height],
            }
            for obj in objects
        ],
    }


class CooldownGate:
    """Allow at most one publish per cooldown, only with hits.

    Injected clock keeps the event path testable offline. First
    qualifying call publishes (last=None).
    """

    def __init__(self, cooldown_s=EVENT_COOLDOWN_S):
        self.cooldown_s = cooldown_s
        self.last_publish_ts = None

    def should_publish(self, now, has_objects):
        if not has_objects:
            return False
        if self.last_publish_ts is None:
            return True
        return (now - self.last_publish_ts) >= self.cooldown_s

    def record(self, now):
        self.last_publish_ts = now


class DetectionState:
    """Thread-safe latest-result snapshot + lifetime counters.

    Written by the engine/consumer threads, read by the HTTP thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None
        self.results_consumed = 0
        self.events_published = 0

    def update(self, sequence, objects):
        with self._lock:
            self.results_consumed += 1
            self._latest = payload_from_objects(sequence, objects)

    def note_event(self):
        with self._lock:
            self.events_published += 1

    def snapshot(self):
        with self._lock:
            return {
                "latest": self._latest,
                "results_consumed": self.results_consumed,
                "events_published": self.events_published,
            }


# --------------------------------------------------------------------------
# MJPEG source + rolling window counters
# --------------------------------------------------------------------------

class FrameBuffer:
    """Latest-JPEG-wins buffer (Condition) feeding /stream.mjpg.

    Slow clients simply see fewer frames; there is no backlog.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = None
        self._frame_id = 0

    def update(self, jpeg_bytes):
        with self._cond:
            self._jpeg = jpeg_bytes
            self._frame_id += 1
            self._cond.notify_all()

    def wait_for_new(self, after_id, timeout=2.0):
        with self._cond:
            if self._frame_id <= after_id:
                self._cond.wait(timeout)
            return self._jpeg, self._frame_id

    @property
    def frame_id(self):
        return self._frame_id


class WindowStats:
    """Rolling window (WINDOW_S) + cumulative counters for the B loop.

    infer_ms is the server-reported counter (result.infer_time_us);
    pipeline_ms is the SDK's client-side prepare+infer+decode timing;
    draw_ms and overlay_ms are measured with perf_counter.
    """

    def __init__(self, window_s=WINDOW_S):
        self._lock = threading.Lock()
        self._window_s = window_s
        self._start = time.monotonic()
        self._current = self._blank()
        self.totals = self._blank()
        self.last_window = None

    @staticmethod
    def _blank():
        return {"frames": 0, "objects": 0, "infer_ms": 0.0,
                "pipeline_ms": 0.0, "draw_ms": 0.0,
                "overlay_ms": 0.0, "overlay_calls": 0}

    def note(self, *, objects=0, infer_ms=0.0, pipeline_ms=0.0,
             draw_ms=0.0, overlay_ms=0.0):
        with self._lock:
            for bucket in (self._current, self.totals):
                bucket["frames"] += 1
                bucket["objects"] += objects
                bucket["infer_ms"] += infer_ms
                bucket["pipeline_ms"] += pipeline_ms
                bucket["draw_ms"] += draw_ms
                if overlay_ms:
                    bucket["overlay_ms"] += overlay_ms
                    bucket["overlay_calls"] += 1

    def poll(self, now=None):
        """Close the window if due; returns the last closed window."""
        now = time.monotonic() if now is None else now
        with self._lock:
            elapsed = now - self._start
            if elapsed < self._window_s or not self._current["frames"]:
                return self.last_window
            window, self._current = self._current, self._blank()
            self._start = now
        frames = window["frames"]
        self.last_window = {
            "fps": round(frames / elapsed, 1),
            "objects_per_frame": round(window["objects"] / frames, 1),
            "infer_ms": round(window["infer_ms"] / frames, 1),
            "pipeline_ms": round(window["pipeline_ms"] / frames, 1),
            "draw_ms": round(window["draw_ms"] / frames, 1),
            "overlay_ms": (round(window["overlay_ms"] / window["overlay_calls"], 1)
                           if window["overlay_calls"] else None),
            "frames": frames,
        }
        return self.last_window

    def status(self):
        with self._lock:
            totals = dict(self.totals)
        return {"last_window": self.last_window, "totals": totals}


def health_deltas(prev, cur):
    """Per-op counter deltas between two router.health() snapshots."""
    prev_ops = (prev or {}).get("ops") or {}
    cur_ops = (cur or {}).get("ops") or {}
    deltas = {}
    for op, health in cur_ops.items():
        before = prev_ops.get(op) or {}
        deltas[op] = {
            "backend": health.get("backend"),
            "hw": (health.get("hardware_calls", 0)
                   - before.get("hardware_calls", 0)),
            "sw": (health.get("software_calls", 0)
                   - before.get("software_calls", 0)),
            "fallbacks": (health.get("fallbacks", 0)
                          - before.get("fallbacks", 0)),
        }
    return deltas


# --------------------------------------------------------------------------
# settings (the page's control surface)
# --------------------------------------------------------------------------

@dataclass
class TeachingSettings:
    bform_on: bool = True
    draw_on: bool = True
    min_score: float = 0.4
    labels: tuple = ("person", "vehicle")
    overlay_path: str = "off"      # off | platform
    zone_on: bool = False
    policy: str = "prefer_hardware"
    aform_on: bool = False


def validate_partial(partial):
    """Validate a partial settings update; ValueError -> HTTP 400."""
    if not isinstance(partial, dict):
        raise ValueError("body must be a JSON object")
    out = {}
    for key, value in partial.items():
        if key == "min_score":
            score = float(value)
            out[key] = min(max(score, MIN_SCORE_MIN), MIN_SCORE_MAX)
        elif key == "labels":
            if not isinstance(value, list) or not value:
                raise ValueError("labels must be a non-empty list")
            unknown = [label for label in value if label not in ALL_LABELS]
            if unknown:
                raise ValueError(
                    f"unknown labels {unknown}; valid: {list(ALL_LABELS)}")
            out[key] = tuple(value)
        elif key == "policy":
            if value not in POLICIES:
                raise ValueError(f"policy must be one of {list(POLICIES)}")
            out[key] = value
        elif key == "overlay_path":
            if value not in OVERLAY_PATHS:
                raise ValueError(
                    f"overlay_path must be one of {list(OVERLAY_PATHS)}")
            out[key] = value
        elif key in BOOL_SETTINGS:
            out[key] = bool(value)
        else:
            raise ValueError(f"unknown setting: {key!r}")
    return out


class SettingsStore:
    """Lock-guarded settings; snapshots are fresh dicts (JSON-safe)."""

    def __init__(self, base):
        self._lock = threading.Lock()
        self._settings = base

    def snapshot(self):
        with self._lock:
            return asdict(self._settings)

    def apply(self, validated):
        with self._lock:
            for key, value in validated.items():
                setattr(self._settings, key, value)
            return asdict(self._settings)
