"""Slot-anchored CLIP classification on the device runtime.

No on-device detector can see shelf products (docs/device-gate.md §6-§7:
yolo_world_v2s class outputs are constants; the platform yolov8n head only
knows person/vehicle/face), so recognition anchors on the configured slot
polygons instead of detected boxes:

    polygon -> padded pixel bbox -> crop -> RGB 224x224 -> NV12
    -> clip_vit_b_32_image_encoder_nv12.hef -> 512 x uint8
    -> mean-center + unit-normalize -> cosine vs unit-norm float text
    embeddings -> (vocabulary row, cosine)

Device contract (probes 19-21/32, docs/device-gate.md §6):
  - register with model_type "embedding", variant
    {"backend_function": "identity"}, input spec [1,224,224,3] uint8.
    Registration is FLAKY on a busy runtime (~1-in-3) -> retries.
  - input tensor is flat NV12 (Y plane then interleaved U/V), 75264 bytes
    at 224x224.
  - output is 512 uint8 with a large per-vector offset: decode = subtract
    mean, then unit-normalize (the centering is load-bearing, probe20).
  - measured ~48 ms per crop; 10 slots ≈ 0.5 s per scan.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

CLIP_INPUT_SIZE = 224
CLIP_EMBED_DIM = 512
_REGISTER_RETRY_DELAY_S = 2.0


def rgb_to_nv12(rgb: np.ndarray) -> np.ndarray:
    """RGB (h,w,3) uint8 -> flat NV12 (h*1.5,w) semi-planar bytes."""
    h, w = rgb.shape[:2]
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
    y_plane = yuv[:h]
    u_plane = yuv[h:h + h // 4]
    v_plane = yuv[h + h // 4:]
    uv = np.empty((h // 2 * w), dtype=np.uint8)
    uv[0::2] = u_plane.flatten()
    uv[1::2] = v_plane.flatten()
    return np.concatenate([y_plane.flatten(), uv])


def crop_box(frame_rgb: np.ndarray, xyxy) -> np.ndarray:
    """Clip a pixel xyxy box to frame bounds and return the crop (min 1px)."""
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = int(np.clip(round(x1), 0, w - 1))
    y1 = int(np.clip(round(y1), 0, h - 1))
    x2 = int(np.clip(round(x2), 0, w))
    y2 = int(np.clip(round(y2), 0, h))
    if x2 <= x1 or y2 <= y1:
        x2, y2 = x1 + 1, y1 + 1
    return frame_rgb[y1:y2, x1:x2]


def polygon_bbox(polygon, frame_shape, pad: float = 0.0) -> np.ndarray:
    """Normalized polygon -> pixel xyxy bbox, padded by ``pad`` of its size.

    ``pad`` is relative to the bbox width/height (0.05 = 5% each side) so a
    slot crop keeps a little shelf context around the marked region; the
    result stays clamped to the frame.
    """
    if len(polygon) < 3:
        raise ValueError("polygon needs at least 3 vertices")
    h, w = frame_shape[:2]
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    dx, dy = (x2 - x1) * float(pad), (y2 - y1) * float(pad)
    x1, y1 = max(0.0, x1 - dx), max(0.0, y1 - dy)
    x2, y2 = min(1.0, x2 + dx), min(1.0, y2 + dy)
    return np.array([x1 * (w - 1), y1 * (h - 1), x2 * (w - 1), y2 * (h - 1)],
                    dtype=np.float32)


def decode_clip_vector(raw: np.ndarray) -> np.ndarray:
    """Device uint8 CLIP output -> mean-centered unit-norm float32 vector."""
    v = np.asarray(raw, dtype=np.float32).reshape(-1)
    if v.size < CLIP_EMBED_DIM:
        raise ValueError(f"CLIP output too short: {v.size} < {CLIP_EMBED_DIM}")
    v = v[:CLIP_EMBED_DIM]
    v = v - v.mean()
    norm = float(np.linalg.norm(v))
    if norm < 1e-6:
        raise ValueError("CLIP output is constant (zero variance)")
    return v / norm


def register_clip_model(
    client,
    model_path: str,
    model_id: str,
    owner_id: str,
    retries: int = 3,
    inputs: list | None = None,
) -> None:
    """Register the CLIP encoder; retry — the runtime is flaky under load."""
    spec = inputs or [{"name": "input_layer1", "shape": [1, 224, 224, 3],
                       "dtype": "uint8"}]
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            client.register_model(
                model_path=model_path,
                model_id=model_id,
                owner_id=owner_id,
                model_type="embedding",
                model_variant='{"backend_function": "identity"}',
                inputs=spec,
            )
            return
        except Exception as e:  # noqa: BLE001 - runtime raises opaque errors
            msg = str(e)
            if "already" in msg.lower():
                return  # benign: registered by a previous run
            last_error = e
            print(f"[clip] register attempt {attempt + 1} failed: {msg}")
            time.sleep(_REGISTER_RETRY_DELAY_S)
    raise RuntimeError(f"CLIP registration failed after {retries}: {last_error}")


class SlotClassifier:
    """Cosine classifier of slot crops against vocabulary text embeddings."""

    model_id: str = ""  # bound to the registered CLIP model by the app

    def __init__(self, txt_unit: np.ndarray, timeout_ms: int = 3000) -> None:
        t = np.asarray(txt_unit, dtype=np.float32)
        if t.ndim != 2 or t.shape[1] != CLIP_EMBED_DIM:
            raise ValueError(
                f"txt_unit must be (C, {CLIP_EMBED_DIM}), got {t.shape}")
        self.txt = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        self.timeout_ms = int(timeout_ms)

    def embed(self, client, crop_rgb: np.ndarray,
              timeout_ms: int | None = None) -> np.ndarray:
        """Encode one RGB crop (any size) with the device CLIP model."""
        crop = cv2.resize(crop_rgb, (CLIP_INPUT_SIZE, CLIP_INPUT_SIZE),
                          interpolation=cv2.INTER_AREA)
        out = client.infer_with_tensors(
            self.model_id,
            inputs=[rgb_to_nv12(crop).flatten()],
            input_names=["input_layer1"],
            timeout_ms=int(timeout_ms or self.timeout_ms),
        )
        return decode_clip_vector(out[0])

    def best(self, vec: np.ndarray) -> tuple[int, float]:
        """Top vocabulary row for a decoded image vector, with cosine."""
        cos = self.txt @ np.asarray(vec, dtype=np.float32)
        row = int(np.argmax(cos))
        return row, float(cos[row])

    def classify(self, client, frame_rgb: np.ndarray, boxes,
                 timeout_ms: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Classify each pixel xyxy box: -> (vocab row idx, cosine) arrays."""
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        rows = np.zeros(len(boxes), dtype=np.int64)
        scores = np.zeros(len(boxes), dtype=np.float32)
        for i, box in enumerate(boxes):
            vec = self.embed(client, crop_box(frame_rgb, box), timeout_ms)
            rows[i], scores[i] = self.best(vec)
        return rows, scores
