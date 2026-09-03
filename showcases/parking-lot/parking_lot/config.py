"""
Configuration constants and model definitions for the Parking Lot pipeline.

All environment-variable-driven configuration lives here so the rest of
the codebase can import stable Python values rather than calling
``os.environ`` inline.
"""

import json
import os
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MODEL_ROOT = "/data/aipc/models"

# Models bundled in the image (flat layout) — used when the host has not
# provisioned /data/aipc/models itself, so installs are plug-and-play.
_BUNDLED_MODEL_ROOT = "/opt/aipc/models"


def _model_path(*parts: str) -> str:
    """Resolve a model file: host-provisioned copy first, bundled otherwise.

    Devices that pre-provision /data/aipc/models (subdir layout) keep using
    their own copy; everything else falls back to the image-bundled file
    under /opt/aipc/models (flat, basename only).
    """
    host = os.path.join(_MODEL_ROOT, *parts)
    if os.path.isfile(host):
        return host
    return os.path.join(_BUNDLED_MODEL_ROOT, os.path.basename(parts[-1]))


# ---------------------------------------------------------------------------
# DSP hardware offload (keep_fd + DspClient resize/multi-crop)
# ---------------------------------------------------------------------------

# Offloads whole-frame model-input scaling and the plate letterbox tiles to
# the camera-daemon DSP service via zero-copy keep-fd frames.  A/B switch:
# PARKING_LOT_DSP=0 forces the pure-CPU pipeline (identical frame flow,
# plain get_frame, cv2 resize/letterbox) for on-device comparisons.
DSP_ENABLED = os.environ.get("PARKING_LOT_DSP", "1").strip().lower() not in (
    "0", "false", "off",
)

# Quota errors (daemon code -3) fall back to CPU for this long before the
# DSP path retries — MPix/s budget is shared, so sustained pipelines may
# oscillate if the retry is immediate.
DSP_QUOTA_COOLDOWN_S = float(os.environ.get("PARKING_LOT_DSP_COOLDOWN", "10"))

# ---------------------------------------------------------------------------
# Character set for license plate OCR (CTC decoder)
# PaddleOCR v5 dictionary — loaded from ppocrv5_dict.txt bundled with the app.
# Layout: dict_chars + space + CTC_blank.  Blank is at the LAST index.
# ---------------------------------------------------------------------------

_PPOCRV5_DICT_PATH = os.path.join(os.path.dirname(__file__), "ppocrv5_dict.txt")


def _load_ppocrv5_charset() -> List[str]:
    """Load PaddleOCR v5 dictionary as a list of character tokens.

    PaddleOCR uses: dict characters (N) + optional space + CTC blank at end.
    With use_space_char=True (default), the layout is:
        index 0 .. N-1    : dict chars
        index N           : space character (" ")
        index N+1         : CTC blank  (not included in charset;
                          passed separately to ctc_greedy_decode)

    IMPORTANT: We return a **list** (not a string) because some dict entries
    are multi-codepoint characters (e.g. flag emojis like 🇩🇪).  Using a
    string would break the 1:1 index mapping that CTC decoding relies on,
    since Python string indexing operates on code points, not graphemes.
    """
    if not os.path.isfile(_PPOCRV5_DICT_PATH):
        return list("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ") + [" "]
    with open(_PPOCRV5_DICT_PATH, "r", encoding="utf-8") as f:
        # NOTE: use rstrip("\n") only — Python's strip() removes U+3000
        # (full-width space) which is a valid dict character, causing a
        # charset-length mismatch vs the model's output dimensionality.
        chars = [line.rstrip("\n") for line in f if line.rstrip("\n")]
    return chars + [" "]


LPR_CHARSET: List[str] = _load_ppocrv5_charset()

# PaddleOCR v5: CTC blank token is at the LAST index (len(charset)).
LPR_CTC_BLANK = len(LPR_CHARSET)

# ---------------------------------------------------------------------------
# COCO vehicle class mapping (used by yolov5m_vehicles)
# ---------------------------------------------------------------------------

_COCO_VEHICLE_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODEL_DEFS = {
    "yolov5m_vehicles": {
        "path": _model_path("detection", "yolov5m_vehicles.hef"),
        "type": "detection",
        "input_format": "rgb",
        "input_width": 1920,
        "input_height": 1080,
        # register_type MUST be non-empty ("detection") so ai-runtime's
        # init_post_process runs (grpc_service.cpp gate) and applies the
        # variant config_json below. An empty register_type skips postprocess
        # entirely (raw outputs, no detected objects).
        "register_type": "detection",
        # Full postprocess config_json blob (NOT just a backend name). The
        # Hailo libyolo_hailortpp_post.so exports a "yolov5m_vehicles" backend
        # whose nms tensor name (yolov5m_vehicles/yolov5_nms_postprocess)
        # matches this HEF. ai-runtime (model_manager.cpp) uses the blob
        # verbatim as config_json when it starts with '{'. HAL strips the
        # backend_function loader key and passes the rest to the YOLO plugin,
        # which validates it against its schema — so the blob MUST carry the
        # full YOLO config (iou/detection thresholds, labels, ...), mirroring
        # /home/root/apps/shared/resources/configs/yolov8.json. A bare
        # {"backend_function":"yolov5m_vehicles"} is schema-invalid ({} after
        # strip) and falls back to hailo_yolov8n -> tensor mismatch.
        # HEF nms: 1 class, 80 boxes/class.
        "variant": json.dumps({
            "backend_function": "yolov5m_vehicles",
            "iou_threshold": 0.45,
            "detection_threshold": 0.30,
            "output_activation": "none",
            "label_offset": 0,
            "max_boxes": 80,
            "labels": ["vehicle"],
        }),
    },
    "scdepthv3": {
        "path": _model_path("depth", "scdepthv3.hef"),
        "type": "depth",
        "input_format": "rgb",
        "input_width": 320,
        "input_height": 256,
    },
    "license_plate_det": {
        "path": _model_path("detection", "tiny_yolov4_license_plates.hef"),
        "type": "detection",
        "input_format": "rgb",
        "input_width": 416,
        "input_height": 416,
        "register_type": "",
    },
    "plate_recognition": {
        "path": _model_path("ocr", "paddle_ocr_v5_mobile_recognition_nv12.hef"),
        "type": "ocr_recognition",
        "input_format": "nv12",
        "input_width": 320,
        "input_height": 48,
    },
}

# ---------------------------------------------------------------------------
# YOLOv4 anchors for license plate detection grid decoder
# ---------------------------------------------------------------------------

_YOLOV4_ANCHORS: Dict[str, List[Tuple[Tuple[int, int], List[Tuple[int, int]]]]] = {
    "license_plate_det": [
        ((13, 13), [(81, 82), (135, 169), (344, 319)]),
        ((26, 26), [(23, 27), (37, 58), (81, 82)]),
    ],
}

# ---------------------------------------------------------------------------
# Runtime configuration from environment variables
# ---------------------------------------------------------------------------

WEB_PORT = int(os.environ.get("WEB_PORT", "8090"))
STREAM_ID = os.environ.get("STREAM_ID", "main")
TARGET_FPS = int(os.environ.get("TARGET_FPS", "20"))
VEHICLE_MODEL = os.environ.get("VEHICLE_MODEL", "yolov5m_vehicles")
DEPTH_SPOOF_THRESHOLD = float(os.environ.get("DEPTH_SPOOF_THRESHOLD", "0.02"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Platform API (NPU / model stats)
PLATFORM_API_URL = os.environ.get("PLATFORM_API_URL", "http://127.0.0.1:8080")
PLATFORM_API_TOKEN = os.environ.get("PLATFORM_API_TOKEN", "")

# ---------------------------------------------------------------------------
# Inference timeout tiering + warmup + degraded state machine
# (ported from model-showcase/main.py:1407-1415).  HEF cold-start takes 1-3s
# for NPU context init; a tight steady-state timeout would false-alarm during
# warmup, so the first INFER_WARMUP_COUNT inferences per model use the wider
# INFER_WARMUP_TIMEOUT_MS, then drop to INFER_TIMEOUT_MS (fail-fast).
# ---------------------------------------------------------------------------
INFER_TIMEOUT_MS = int(os.environ.get("INFER_TIMEOUT_MS", "1500"))
INFER_WARMUP_TIMEOUT_MS = int(os.environ.get("INFER_WARMUP_TIMEOUT_MS", "6000"))
INFER_WARMUP_COUNT = int(os.environ.get("INFER_WARMUP_COUNT", "3"))
# Log a recovery INFO only after a burst of >= this many consecutive failures
STALL_BURST_INFO_THRESHOLD = int(os.environ.get("STALL_BURST_INFO_THRESHOLD", "3"))
# Consecutive failures at/above this mark the pipeline degraded (health dot red)
DEGRADED_FAILURE_THRESHOLD = int(os.environ.get("DEGRADED_FAILURE_THRESHOLD", "10"))

# Upload limits
UPLOAD_DIR = "/app/uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"}
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB
