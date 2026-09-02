"""Per-item goods detection via yolo_world_v2s v5.4.0 (two-input HEF).

Device contract (probe-validated 2026-08-25; docs/device-gate.md §9):
  - HEF: /data/aipc/models/yolo_world_v2s_540.hef, ~0.24 s/infer.
  - input_layer1: uint8 [1,640,640,3] RGB, letterboxed centered, black bars.
  - input_layer2: uint16 [1,80,512] vocabulary embeddings,
    u16 = clip(rint(emb / 2.7e-5 + 9778), 0, 65535)   (qp_scale / qp_zp).
  - Output: ONE NMS-by-class float32 stream — per class 0..79:
    [f32 count][count x 5 x f32 (y1, x1, y2, x2, score)]  — step is 5, not 1.
  - On-chip score threshold is ~0: the host thresholds and then runs a
    greedy cross-class NMS (score-descending) over the survivors.

Vocabulary is fixed at 80 rows: `n_goods` goods prompts first, then
negative/environment prompts (person/hand/shelf/...). Negative hits are
reported (gray boxes, "the detector sees people") but not counted as items.
Embeddings ship as assets/yolo_world_vocab.npy + .json sidecar.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

DET_INPUT_SIZE = 640
DET_EMB_ROWS = 80
DET_EMB_DIM = 512
DET_QP_SCALE = 2.7e-5
DET_QP_ZP = 9778.0
DET_INPUT_IMAGE = "yolo_world_v2s/input_layer1"
DET_INPUT_EMB = "yolo_world_v2s/input_layer2"
DET_MAX_PER_CLASS = 300          # parser sanity bound on the class count
DET_MAX_DETECTIONS = 64          # scan-dict payload cap
DET_REGISTER_RETRY_DELAY_S = 2.0

# Grid mapping used for the per-detection "cell" annotation; mirrors the
# app's grid mode geometry (build_grid_slots ids are g{r}-{c}, 1-based).
DEFAULT_GRID_ROWS = 3
DEFAULT_GRID_COLS = 4


def quantize_u16(emb_f32: np.ndarray) -> np.ndarray:
    """f32 embeddings -> the HEF's uint16 layer2 encoding."""
    v = np.asarray(emb_f32, dtype=np.float32) / DET_QP_SCALE + DET_QP_ZP
    return np.clip(np.rint(v), 0, 65535).astype(np.uint16)


def parse_nms(out0) -> tuple[list, str | None]:
    """Decode the NMS-by-class stream; returns (dets, error).

    dets entries: (cls:int, score:float, y1, x1, y2, x2) with box coords
    normalized to the 640x640 letterboxed frame (y1..x2 order).
    """
    arr = np.asarray(out0)
    if arr.dtype == np.float32:
        f32 = np.ascontiguousarray(arr, np.float32).reshape(-1)
    else:
        flat = np.ascontiguousarray(arr, np.uint8).reshape(-1)
        f32 = np.frombuffer(flat.tobytes(), dtype="<f4")
    dets: list = []
    o = 0
    for c in range(DET_EMB_ROWS):
        if o >= f32.size:
            return dets, f"class {c}: buffer exhausted at word {o}"
        cnt = float(f32[o])
        o += 1
        if abs(cnt - round(cnt)) > 1e-3 or not (0 <= cnt <= DET_MAX_PER_CLASS):
            return dets, f"class {c}: bad count {cnt!r} at word {o - 1}"
        for _ in range(int(round(cnt))):
            if o + 5 > f32.size:
                return dets, f"class {c}: buffer exhausted at word {o}"
            y1, x1, y2, x2, s = f32[o:o + 5]
            o += 5
            dets.append((int(c), float(s), float(y1), float(x1),
                         float(y2), float(x2)))
    return dets, None


def iou(a, b) -> float:
    """IoU of two 6-tuples (cls, score, y1, x1, y2, x2)."""
    ay1, ax1, ay2, ax2 = a[2], a[3], a[4], a[5]
    by1, bx1, by2, bx2 = b[2], b[3], b[4], b[5]
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter = iy * ix
    ua = (ay2 - ay1) * (ax2 - ax1) + (by2 - by1) * (bx2 - bx1) - inter
    return inter / ua if ua > 0 else 0.0


def cross_class_nms(dets, thr: float = 0.45) -> list:
    """Greedy score-descending suppression across all classes."""
    keep: list = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if all(iou(d, k) <= thr for k in keep):
            keep.append(d)
    return keep


def letterbox(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Resize into 640x640 keeping aspect, centered black padding.

    Ported verbatim from the probe pipeline (gate_cls_probe12.letter_box);
    the un-letterbox meta carries the scale + centered pads in 640-space.
    """
    h, w = rgb.shape[:2]
    r = min(DET_INPUT_SIZE / h, DET_INPUT_SIZE / w)
    new_w, new_h = (int(round(w * r)), int(round(h * r)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (DET_INPUT_SIZE - new_w) / 2.0, (DET_INPUT_SIZE - new_h) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    meta = {"scale": float(r), "pad_top": (DET_INPUT_SIZE - new_h) / 2.0,
            "pad_left": (DET_INPUT_SIZE - new_w) / 2.0,
            "orig_h": int(h), "orig_w": int(w)}
    return out, meta


def unletterbox_point(y640: float, x640: float, meta: dict) -> tuple[float, float]:
    """Map a point in 640-frame normalized coords to original-frame
    normalized coords (either may land outside [0,1) in a letterbox bar)."""
    y = (y640 * DET_INPUT_SIZE - meta["pad_top"]) / meta["scale"] / meta["orig_h"]
    x = (x640 * DET_INPUT_SIZE - meta["pad_left"]) / meta["scale"] / meta["orig_w"]
    return y, x


def register_detector_model(client, model_path: str, model_id: str,
                            owner_id: str, retries: int = 3) -> None:
    """Register the yolo_world detection HEF; retry like the CLIP path."""
    spec = [
        {"name": DET_INPUT_IMAGE,
         "shape": [1, DET_INPUT_SIZE, DET_INPUT_SIZE, 3], "dtype": "uint8"},
        {"name": DET_INPUT_EMB,
         "shape": [1, DET_EMB_ROWS, DET_EMB_DIM], "dtype": "uint16"},
    ]
    variant = json.dumps({"labels": [f"c{i}" for i in range(DET_EMB_ROWS)],
                          "backend_function": "identity"})
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            client.register_model(
                model_path=model_path,
                model_id=model_id,
                owner_id=owner_id,
                model_type="detection",
                model_variant=variant,
                inputs=spec,
            )
            return
        except Exception as e:  # noqa: BLE001 - runtime raises opaque errors
            msg = str(e)
            if "already" in msg.lower():
                return  # benign: registered by a previous run
            last_error = e
            print(f"[detect] register attempt {attempt + 1} failed: {msg}")
            time.sleep(DET_REGISTER_RETRY_DELAY_S)
    raise RuntimeError(f"detector registration failed after {retries}: "
                       f"{last_error}")


def load_vocab(npy_path: str, json_path: str) -> tuple[np.ndarray, list, int, list]:
    """Load the bundled (emb, labels, n_goods, categories) vocabulary pair."""
    emb = np.load(npy_path, allow_pickle=False).astype(np.float32, copy=False)
    if emb.size != DET_EMB_ROWS * DET_EMB_DIM:
        raise ValueError(f"vocab emb must hold {DET_EMB_ROWS}x{DET_EMB_DIM} "
                         f"f32, got {emb.size} elements")
    emb = emb.reshape(DET_EMB_ROWS, DET_EMB_DIM)
    with open(json_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)
    labels = [str(n).strip() for n in sidecar.get("labels", [])]
    if len(labels) != emb.shape[0]:
        raise ValueError(f"vocab sidecar has {len(labels)} labels for "
                         f"{emb.shape[0]} embedding rows")
    n_goods = int(sidecar.get("n_goods", 0))
    if not (0 < n_goods <= len(labels)):
        raise ValueError(f"vocab n_goods out of range: {n_goods}")
    categories = [str(c).strip() or "other"
                  for c in sidecar.get("categories", [])]
    if len(categories) != len(labels):    # legacy sidecar: neutral fallback
        categories = ["other"] * len(labels)
    return emb, labels, n_goods, categories


def bundled_vocab_path() -> str:
    """assets/ directory shipped next to this module (inside the image)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


class GoodsDetector:
    """One detect() call per scan tick against the shared 4K frame."""

    model_id: str = ""  # bound to the registered model by the app

    def __init__(self, emb_f32: np.ndarray, labels: list[str], n_goods: int,
                 threshold: float = 0.25, nms_iou: float = 0.45,
                 timeout_ms: int = 15000,
                 categories: list[str] | None = None) -> None:
        emb = np.asarray(emb_f32, dtype=np.float32)
        if emb.size != DET_EMB_ROWS * DET_EMB_DIM:
            raise ValueError(f"emb must hold {DET_EMB_ROWS}x{DET_EMB_DIM} "
                             f"f32, got {emb.size} elements")
        self.emb_u16 = quantize_u16(emb.reshape(DET_EMB_ROWS, DET_EMB_DIM))
        if len(labels) != DET_EMB_ROWS:
            raise ValueError(f"need {DET_EMB_ROWS} labels, got {len(labels)}")
        self.labels = [str(n).strip() for n in labels]
        self.n_goods = int(n_goods)
        self.threshold = float(threshold)
        self.nms_iou = float(nms_iou)
        self.timeout_ms = int(timeout_ms)
        cats = ([str(c).strip() or "other" for c in categories]
                if categories is not None else [])
        if len(cats) != len(self.labels):
            cats = ["other"] * len(self.labels)
        self.categories = cats

    def detect(self, client, frame_rgb: np.ndarray,
               timeout_ms: int | None = None,
               region: tuple[float, float, float, float] | None = None,
               rows: int = DEFAULT_GRID_ROWS,
               cols: int = DEFAULT_GRID_COLS,
               max_detections: int | None = None) -> list[dict]:
        """Detect goods on one RGB frame; json-serializable output.

        region is the grid region in original-frame normalized coords
        (x1, y1, x2, y2); detections whose center falls outside get
        cell=None (and are excluded from items()).
        """
        img640, meta = letterbox(frame_rgb)
        out = client.infer_with_tensors(
            self.model_id,
            inputs=[img640.flatten(), self.emb_u16.flatten()],
            input_names=[DET_INPUT_IMAGE, DET_INPUT_EMB],
            timeout_ms=int(timeout_ms or self.timeout_ms),
        )
        dets, err = parse_nms(out[0])
        if err:
            print(f"[detect] parse failed: {err}")
            return []
        keep = [d for d in dets if d[1] >= self.threshold]
        cap = int(max_detections) if max_detections else DET_MAX_DETECTIONS
        final = cross_class_nms(keep, self.nms_iou)[:cap]
        full: list[tuple] = []
        for cls, score, y1, x1, y2, x2 in final:
            fy1, fx1 = unletterbox_point(y1, x1, meta)
            fy2, fx2 = unletterbox_point(y2, x2, meta)
            full.append((cls, score, fy1, fx1, fy2, fx2))
        return self._annotate(full, region, rows, cols)

    def detect_tiled(self, client, frame_rgb: np.ndarray,
                     tiles: int = 4, overlap: float = 0.08,
                     threshold: float | None = None,
                     max_detections: int = 200,
                     region: tuple[float, float, float, float] | None = None,
                     rows: int = DEFAULT_GRID_ROWS,
                     cols: int = DEFAULT_GRID_COLS,
                     timeout_ms: int | None = None) -> list[dict]:
        """2x2 tiled detect — ~2x linear resolution for one-shot analysis.

        Full-frame 640 letterbox starves small goods (~30px cans shrink to
        ~16px); four overlapping quadrants each letterbox near-native. The
        quadrant owning each center (midline partition) contributes the
        box, then one global cross-class NMS merges across tiles. ~4x the
        latency of detect() — for imports, not the live scan loop.
        Pinned by probe53 on stocked_shelf.jpg: tiles=4/overlap=.08/
        threshold=.15 recovers all 4 shelf rows (~63 items vs 22 before).
        """
        if tiles <= 1:
            return self.detect(client, frame_rgb, timeout_ms=timeout_ms,
                               region=region, rows=rows, cols=cols,
                               max_detections=max_detections)
        if tiles != 4:
            raise ValueError(f"tiles must be 1 or 4, got {tiles}")
        thr = self.threshold if threshold is None else float(threshold)
        oh, ow = frame_rgb.shape[:2]
        oy, ox = int(oh * overlap), int(ow * overlap)
        spans_y = ((0, oh // 2 + oy, True), (oh // 2 - oy, oh, False))
        spans_x = ((0, ow // 2 + ox, True), (ow // 2 - ox, ow, False))
        merged: list[tuple] = []
        for y0, y1, top in spans_y:
            for x0, x1, left in spans_x:
                img640, meta = letterbox(frame_rgb[y0:y1, x0:x1])
                out = client.infer_with_tensors(
                    self.model_id,
                    inputs=[img640.flatten(), self.emb_u16.flatten()],
                    input_names=[DET_INPUT_IMAGE, DET_INPUT_EMB],
                    timeout_ms=int(timeout_ms or self.timeout_ms),
                )
                dets, err = parse_nms(out[0])
                if err:
                    print(f"[detect] tile parse failed: {err}")
                    continue
                th, tw = y1 - y0, x1 - x0
                for cls, score, ty1, tx1, ty2, tx2 in dets:
                    if score < thr:
                        continue
                    fy1, fx1 = unletterbox_point(ty1, tx1, meta)
                    fy2, fx2 = unletterbox_point(ty2, tx2, meta)
                    gy1 = (y0 + fy1 * th) / oh
                    gx1 = (x0 + fx1 * tw) / ow
                    gy2 = (y0 + fy2 * th) / oh
                    gx2 = (x0 + fx2 * tw) / ow
                    cy, cx = (gy1 + gy2) / 2.0, (gx1 + gx2) / 2.0
                    if (cy < 0.5) != top or (cx < 0.5) != left:
                        continue        # box owned by the neighbor quadrant
                    merged.append((cls, score, gy1, gx1, gy2, gx2))
        final = cross_class_nms(merged, self.nms_iou)[:max(1, int(max_detections))]
        return self._annotate(final, region, rows, cols)

    def _annotate(self, dets_full: list[tuple],
                  region: tuple[float, float, float, float] | None,
                  rows: int, cols: int) -> list[dict]:
        """(cls, score, y1, x1, y2, x2) in full-frame normalized coords ->
        json-serializable item dicts."""
        rx1, ry1, rx2, ry2 = region or (0.0, 0.0, 1.0, 1.0)
        items: list[dict] = []
        for cls, score, y1, x1, y2, x2 in sorted(dets_full,
                                                 key=lambda d: -d[1]):
            cy, cx = (y1 + y2) / 2.0, (x1 + x2) / 2.0
            if not (0.0 <= cy < 1.0 and 0.0 <= cx < 1.0):
                continue                # center fell inside a letterbox bar
            items.append({
                "label": self.labels[cls] if cls < len(self.labels) else f"c{cls}",
                "score": round(float(score), 4),
                "box": [round(v, 5) for v in (x1, y1, x2, y2)],
                "goods": cls < self.n_goods,
                "category": (self.categories[cls]
                             if cls < len(self.categories) else "other"),
                "cell": _grid_cell(cx, cy, rx1, ry1, rx2, ry2, rows, cols),
            })
        return items

    def items(self, dets: list[dict]) -> int:
        """ITEMS m: deduped goods boxes with center inside the grid region."""
        return count_items(dets)


def count_items(dets: list[dict]) -> int:
    """ITEMS m: deduped goods boxes with center inside the grid region."""
    return sum(1 for d in dets if d.get("goods") and d.get("cell"))


def items_by_category(dets: list[dict]) -> dict[str, int]:
    """Per-category tally of counted goods (bottle/can/bag/...). Imports
    surface this instead of the cell-position join items_by_code uses —
    it is the true per-item identity from the detector."""
    out: dict[str, int] = {}
    for d in dets:
        if not (d.get("goods") and d.get("cell")):
            continue
        cat = str(d.get("category") or "other")
        out[cat] = out.get(cat, 0) + 1
    return out


def _grid_cell(cx: float, cy: float, rx1: float, ry1: float,
               rx2: float, ry2: float, rows: int, cols: int) -> str | None:
    """Grid cell id (g{r}-{c}, 1-based) for a center inside the region."""
    if not (rx1 <= cx < rx2 and ry1 <= cy < ry2):
        return None
    r = min(int((cy - ry1) / (ry2 - ry1) * rows), rows - 1)
    c = min(int((cx - rx1) / (rx2 - rx1) * cols), cols - 1)
    return f"g{r + 1}-{c + 1}"
