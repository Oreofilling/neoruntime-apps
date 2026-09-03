"""
Post-processing utilities for the Parking Lot pipeline.

Contains data types, NMS/grid decoders, CTC OCR decoder, anti-spoofing
depth analysis, and image preparation helpers.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import (
    LPR_CHARSET,
    LPR_CTC_BLANK,
    _COCO_VEHICLE_CLASSES,
    _YOLOV4_ANCHORS,
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VehicleDetection:
    bbox: Tuple[float, float, float, float]  # x, y, w, h (normalized)
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class PlateDetection:
    bbox: Tuple[float, float, float, float]  # x, y, w, h (normalized)
    text: str
    confidence: float


@dataclass(frozen=True)
class PlateSnapshot:
    """A cropped plate image stored for the gallery view."""
    snapshot_id: str       # UUID
    plate_text: str
    confidence: float
    timestamp: str         # HH:MM:SS
    image_jpeg: bytes      # Cropped plate JPEG


@dataclass(frozen=True)
class SpoofResult:
    is_spoof: bool
    confidence: float
    depth_variance: float
    gradient_variance: float


@dataclass(frozen=True)
class SpoofAlert:
    vehicle: VehicleDetection
    reason: str
    score: float


@dataclass
class PipelineResult:
    vehicles: List[VehicleDetection] = field(default_factory=list)
    plates: List[PlateDetection] = field(default_factory=list)
    depth_map: Optional[np.ndarray] = None
    spoof_alerts: List[SpoofAlert] = field(default_factory=list)
    infer_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# CTC Greedy Decoder
# ---------------------------------------------------------------------------


def ctc_greedy_decode(
    logits: np.ndarray, charset: List[str], blank: int = 0,
) -> Tuple[str, float]:
    """Decode CTC logits into text using greedy strategy.

    blank: index of the CTC blank token. For PaddleOCR v5 this is the
           last index (len(charset)), for older models it was 0.
    charset: a list of character tokens (one per model class, excluding blank).
    """
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    argmax = np.argmax(logits, axis=-1)
    if not isinstance(argmax, np.ndarray):
        argmax = np.array([argmax])
    decoded: List[str] = []
    confidences: List[float] = []
    prev = blank
    for t in range(argmax.shape[0]):
        idx = int(argmax[t])
        if idx != blank and idx != prev:
            if 0 <= idx < len(charset):
                decoded.append(charset[idx])
                confidences.append(float(logits[t][idx]))
        prev = idx
    text = "".join(decoded)
    conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, conf


# Keep backward-compatible alias for tests
_ctc_greedy_decode = ctc_greedy_decode


# ---------------------------------------------------------------------------
# OCR Accumulator — temporal smoothing for plate text
# ---------------------------------------------------------------------------


class PlateAccumulator:
    """Majority-vote temporal smoother for license plate readings."""

    def __init__(self, window_size: int = 5, min_confidence: float = 0.7) -> None:
        self._window_size = window_size
        self._min_confidence = min_confidence
        self._history: List[Tuple[str, float]] = []
        self._lock = threading.Lock()

    def update(self, text: str, confidence: float) -> Tuple[str, float]:
        with self._lock:
            if text and confidence >= self._min_confidence:
                self._history.append((text, confidence))
                if len(self._history) > self._window_size:
                    self._history = self._history[-self._window_size:]
            if not self._history:
                return text, confidence
            counts: Dict[str, List[float]] = {}
            for t, c in self._history:
                counts.setdefault(t, []).append(c)
            best_text = max(counts, key=lambda k: len(counts[k]))
            avg_conf = sum(counts[best_text]) / len(counts[best_text])
            return best_text, avg_conf


# Backward-compatible alias
_PlateAccumulator = PlateAccumulator


# ---------------------------------------------------------------------------
# NMS output parser (yolov5m uint8 -> float32 NMS format)
# ---------------------------------------------------------------------------


def parse_nms_raw(result: Any) -> List[VehicleDetection]:
    """Parse yolov5 NMS raw output into VehicleDetection list."""
    raw = getattr(result, "raw_outputs", None)
    if not raw or len(raw) == 0:
        return []

    tensor = raw[0]
    if tensor.ndim == 1 and tensor.dtype == np.uint8:
        floats = np.frombuffer(tensor.tobytes(), dtype=np.float32)
        if len(floats) < 5:
            return []
        num_dets = int(floats[0])
        if num_dets <= 0 or num_dets > 200:
            return []
        det_data = floats[1:]
        n_fields = 5
        n_actual = min(num_dets, len(det_data) // n_fields)
        dets = det_data[: n_actual * n_fields].reshape(n_actual, n_fields)
        vehicles = []
        for row in dets:
            ymin, xmin, ymax, xmax, conf = row
            if conf < 0.15:
                continue
            vehicles.append(
                VehicleDetection(
                    bbox=(
                        float(xmin),
                        float(ymin),
                        float(xmax - xmin),
                        float(ymax - ymin),
                    ),
                    class_id=0,
                    class_name="vehicle",
                    confidence=float(conf),
                )
            )
        return vehicles

    tensor = tensor.astype(np.float32)
    if tensor.ndim >= 3:
        flat = tensor.reshape(-1, tensor.shape[-1])
    elif tensor.ndim == 2:
        flat = tensor
    elif tensor.ndim == 1 and tensor.shape[0] % 6 == 0:
        flat = tensor.reshape(-1, 6)
    else:
        return []

    if flat.shape[-1] < 5:
        return []

    vehicles = []
    for row in flat:
        vals = row[:6] if row.shape[0] >= 6 else row[:5]
        if vals[4] < 0.15:
            continue
        ymin, xmin, ymax, xmax = vals[0], vals[1], vals[2], vals[3]
        conf = float(vals[4])
        cls_id = int(vals[5]) if len(vals) > 5 else 0
        cls_name = _COCO_VEHICLE_CLASSES.get(cls_id, "vehicle")
        vehicles.append(
            VehicleDetection(
                bbox=(
                    float(xmin),
                    float(ymin),
                    float(xmax - xmin),
                    float(ymax - ymin),
                ),
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
            )
        )
    return vehicles


# Backward-compatible alias
_parse_nms_raw = parse_nms_raw


# ---------------------------------------------------------------------------
# YOLO grid decoder (tiny_yolov4 uint16 grid output)
# ---------------------------------------------------------------------------


def parse_yolo_grid(
    result: Any, model_id: str,
) -> List[Tuple[float, float, float, float]]:
    """Decode tiny_yolov4 uint16 grid output with NMS suppression."""
    raw = getattr(result, "raw_outputs", None)
    if not raw or len(raw) < 2:
        return []

    anchors_cfg = _YOLOV4_ANCHORS.get(model_id)
    if not anchors_cfg:
        return []

    detections: List[Tuple[float, float, float, float, float]] = []
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
            continue
        grid = (
            tensor[:elements]
            .reshape(grid_h, grid_w, num_anchors, 6)
            .astype(np.float32)
        )
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
                    # Gate calibrated against the HEF's measured output
                    # distribution (93.72, 2026-09-03): on ground-truth
                    # plates — isolated renders AND a real photo — obj runs
                    # 0.35-0.45 (background < 0.3) but cls saturates ~0.5,
                    # so obj*cls tops out ~0.20. A 0.2 gate rejects every
                    # plate, synthetic or real; 0.12 keeps ~30% margin under
                    # the observed plate floor (0.15) while the obj>=0.3
                    # pre-gate still suppresses background.
                    if conf < 0.12:
                        continue
                    detections.append((cx - w / 2, cy - h / 2, w, h, conf))

    detections.sort(key=lambda d: d[4], reverse=True)
    keep: List[Tuple[float, float, float, float]] = []
    for det in detections:
        bx, by, bw, bh, _ = det
        suppressed = False
        for kx, ky, kw, kh in keep:
            if iou(bx, by, bw, bh, kx, ky, kw, kh) > 0.45:
                suppressed = True
                break
        if not suppressed:
            keep.append((bx, by, bw, bh))
    return keep[:32]


# Backward-compatible alias
_parse_yolo_grid = parse_yolo_grid


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def iou(
    x1: float, y1: float, w1: float, h1: float,
    x2: float, y2: float, w2: float, h2: float,
) -> float:
    """Compute intersection-over-union of two axis-aligned boxes."""
    ax1, ay1 = max(x1, x2), max(y1, y2)
    ax2, ay2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


# Alias
_iou = iou


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def prepare_input(
    bgr: np.ndarray, target_w: int, target_h: int, input_fmt: str,
) -> np.ndarray:
    """Resize and convert a BGR frame to the model's expected input format."""
    src_h, src_w = bgr.shape[:2]
    if src_w != target_w or src_h != target_h:
        bgr = cv2.resize(
            bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR,
        )
    if input_fmt == "rgb":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).flatten()
    h, w = bgr.shape[:2]
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    y_plane = i420[:h, :]
    u_plane = i420[h : h + h // 4, :].reshape(h // 2, w // 2)
    v_plane = i420[h + h // 4 :, :].reshape(h // 2, w // 2)
    uv_plane = np.empty((h // 2, w), dtype=np.uint8)
    uv_plane[:, 0::2] = u_plane
    uv_plane[:, 1::2] = v_plane
    return np.vstack([y_plane, uv_plane]).flatten()


# Alias
_prepare_input = prepare_input


# Plate-crop margins shared by the CPU letterbox path and the DSP
# multi-crop path: horizontal 10%, vertical 35% of the box size (the OCR
# model needs context above/below the plate line).
PLATE_MARGIN_X = 0.10
PLATE_MARGIN_Y = 0.35


def plate_crop_rects(
    bboxes: List[Tuple[float, float, float, float]],
    frame_w: int,
    frame_h: int,
    target_w: int,
    target_h: int,
) -> List[Optional[Tuple[int, int, int, int, int, int]]]:
    """Margin-expand normalized plate boxes into DSP multi-crop rects.

    Mirrors letterbox_crop()'s margin + clamp math, then even-aligns every
    coordinate and size (the DSP service requires even NV12 crop geometry —
    at most a 1-pixel difference from the CPU path). Degenerate boxes map
    to ``None`` so the caller can substitute a filled canvas. Output order
    matches ``bboxes``.
    """
    rects: List[Optional[Tuple[int, int, int, int, int, int]]] = []
    for x, y, w, h in bboxes:
        x1 = max(0, int((x - w * PLATE_MARGIN_X) * frame_w)) & ~1
        y1 = max(0, int((y - h * PLATE_MARGIN_Y) * frame_h)) & ~1
        x2 = min(frame_w, int((x + w + w * PLATE_MARGIN_X) * frame_w)) & ~1
        y2 = min(frame_h, int((y + h + h * PLATE_MARGIN_Y) * frame_h)) & ~1
        if x2 <= x1 or y2 <= y1:
            rects.append(None)
            continue
        rects.append((x1, y1, x2 - x1, y2 - y1, target_w, target_h))
    return rects


# Alias
_plate_crop_rects = plate_crop_rects


def letterbox_crop(
    bgr: np.ndarray,
    x: float, y: float, w: float, h: float,
    target_w: int, target_h: int,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Extract and letterbox-resize a crop from a BGR image."""
    fh, fw = bgr.shape[:2]
    mx, my = PLATE_MARGIN_X, PLATE_MARGIN_Y
    x1 = max(0, int((x - w * mx) * fw))
    y1 = max(0, int((y - h * my) * fh))
    x2 = min(fw, int((x + w + w * mx) * fw))
    y2 = min(fh, int((y + h + h * my) * fh))
    if x2 <= x1 or y2 <= y1:
        return np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    crop = bgr[y1:y2, x1:x2]
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
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


# Alias
_letterbox_crop = letterbox_crop


# ---------------------------------------------------------------------------
# Recognition decoder
# ---------------------------------------------------------------------------


def decode_recognition(
    rec_result: Any, charset: List[str], blank: int = 0,
) -> Tuple[str, float]:
    """Decode a recognition model result into (text, confidence).

    Supports PaddleOCR v5 output shape (1, 40, N) where N = len(charset).
    The model outputs uint8 quantized logits normalised to [0, 1].
    """
    if getattr(rec_result, "ocr_lines", None) and rec_result.ocr_lines:
        line = rec_result.ocr_lines[0]
        if (
            line.text
            and not all(c == "?" for c in line.text)
            and line.confidence > 0.01
        ):
            return line.text, line.confidence
    raw = getattr(rec_result, "raw_outputs", None)
    if not raw:
        return "", 0.0
    logits = np.asarray(raw[-1], dtype=np.float32)
    if logits.ndim == 1:
        total = logits.size
        num_classes = blank + 1  # model output dim includes blank token
        if total % num_classes == 0:
            seq_len = total // num_classes
            logits = logits.reshape(1, seq_len, num_classes)
        else:
            return "", 0.0
    if logits.ndim == 3:
        logits = logits[0]
    if logits.ndim != 2:
        return "", 0.0
    # Dequantise uint8 logits to [0, 1]
    if logits.max() > 2.0:
        logits = logits / 255.0
    return ctc_greedy_decode(logits, charset, blank=blank)


# Alias
_decode_recognition = decode_recognition


# ---------------------------------------------------------------------------
# Anti-spoofing via depth analysis
# ---------------------------------------------------------------------------


def analyze_depth_spoof(
    depth_map: np.ndarray,
    vehicle_bbox: Tuple[float, float, float, float],
    frame_h: int,
    frame_w: int,
    threshold: float = 0.02,
    downsample: int = 16,
) -> SpoofResult:
    """Detect printed photos/screens by analyzing depth variance in vehicle bbox.

    A real 3D vehicle has significant depth variance across its body.
    A printed photo/screen has near-uniform depth, producing low variance.

    Limitation: heuristic approach — works for printed photos vs real vehicles
    at moderate distance. Does NOT detect 3D-printed models or replay attacks.
    """
    x, y, w, h = vehicle_bbox
    y1 = max(0, int(y * frame_h))
    y2 = min(frame_h, int((y + h) * frame_h))
    x1 = max(0, int(x * frame_w))
    x2 = min(frame_w, int((x + w) * frame_w))

    if depth_map.shape[0] != frame_h or depth_map.shape[1] != frame_w:
        dy1 = int(y1 * depth_map.shape[0] / frame_h)
        dy2 = int(y2 * depth_map.shape[0] / frame_h)
        dx1 = int(x1 * depth_map.shape[1] / frame_w)
        dx2 = int(x2 * depth_map.shape[1] / frame_w)
        roi = depth_map[dy1:dy2, dx1:dx2]
    else:
        roi = depth_map[y1:y2, x1:x2]

    if roi.size == 0:
        return SpoofResult(
            is_spoof=False,
            confidence=0.0,
            depth_variance=0.0,
            gradient_variance=0.0,
        )

    small = cv2.resize(roi, (downsample, downsample))
    var = float(np.var(small))

    grad_x = cv2.Sobel(small, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(small, cv2.CV_32F, 0, 1)
    grad_var = float(np.var(np.sqrt(grad_x**2 + grad_y**2)))

    score = 0.7 * (1.0 - min(var / threshold, 1.0)) + 0.3 * (
        1.0 - min(grad_var / 0.01, 1.0)
    )

    return SpoofResult(
        is_spoof=score > 0.6,
        confidence=score,
        depth_variance=var,
        gradient_variance=grad_var,
    )
