"""Unit tests for detector.py (yolo_world v5.4.0 per-item detection)."""
import json
import os

import numpy as np
import pytest

import detector
from detector import (
    DET_EMB_DIM,
    DET_EMB_ROWS,
    DET_INPUT_EMB,
    DET_INPUT_IMAGE,
    DET_INPUT_SIZE,
    DET_MAX_DETECTIONS,
    DET_QP_SCALE,
    DET_QP_ZP,
    GoodsDetector,
    bundled_vocab_path,
    count_items,
    cross_class_nms,
    iou,
    items_by_category,
    letterbox,
    load_vocab,
    parse_nms,
    quantize_u16,
    register_detector_model,
    unletterbox_point,
)

N_LABELS = [f"l{i}" for i in range(DET_EMB_ROWS)]
N_LABELS[0], N_LABELS[45] = "bottle", "person"


def _emb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(DET_EMB_ROWS, DET_EMB_DIM)).astype(np.float32)


def _nms_buffer(per_class: dict) -> np.ndarray:
    """Build the NMS-by-class f32 stream: per class [count][count x 5]."""
    words: list[float] = []
    for c in range(DET_EMB_ROWS):
        boxes = per_class.get(c, [])
        words.append(float(len(boxes)))
        for y1, x1, y2, x2, s in boxes:
            words.extend(float(v) for v in (y1, x1, y2, x2, s))
    return np.array(words, dtype=np.float32)


# ------------------------------- quantization --------------------------------

def test_quantize_u16_range_clamp_and_roundtrip():
    emb = np.array([[0.0, -0.259, 0.610, 100.0, -100.0]], dtype=np.float32)
    u16 = quantize_u16(emb)
    assert u16.dtype == np.uint16
    assert u16.min() >= 0 and u16.max() <= 65535
    assert u16[0, 3] == 65535 and u16[0, 4] == 0     # out-of-range clamped
    assert u16[0, 0] == int(round(DET_QP_ZP))        # zero -> zp offset
    # unclipped values round-trip within 1 LSB of the qp scale
    ok = np.array([[0.0, -0.259, 0.610]], dtype=np.float32)
    back = (quantize_u16(ok).astype(np.float32) - DET_QP_ZP) * DET_QP_SCALE
    assert np.max(np.abs(back - ok)) <= DET_QP_SCALE * 1.01


# ---------------------------------- parser -----------------------------------

def test_parse_nms_round_trip_multi_box_multi_class():
    buf = _nms_buffer({
        0: [(0.1, 0.1, 0.3, 0.3, 0.9), (0.6, 0.6, 0.8, 0.8, 0.4)],
        45: [(0.7, 0.1, 0.9, 0.3, 0.5)],
    })
    dets, err = parse_nms(buf)
    assert err is None
    # o += 5 trap: both class-0 boxes survive a count>1 class (f32 precision)
    expected = [(0, 0.9, 0.1, 0.1, 0.3, 0.3), (0, 0.4, 0.6, 0.6, 0.8, 0.8),
                (45, 0.5, 0.7, 0.1, 0.9, 0.3)]
    assert len(dets) == 3
    for got, exp in zip(dets, expected):
        assert got[0] == exp[0]
        assert list(got[1:]) == pytest.approx(list(exp[1:]), rel=1e-6)


def test_parse_nms_accepts_uint8_view():
    buf = _nms_buffer({3: [(0.2, 0.2, 0.4, 0.4, 0.7)]})
    dets, err = parse_nms(np.ascontiguousarray(buf).view(np.uint8))
    assert err is None
    assert dets[0][0] == 3
    assert list(dets[0][1:]) == pytest.approx([0.7, 0.2, 0.2, 0.4, 0.4],
                                              rel=1e-6)


def test_parse_nms_rejects_bad_count():
    buf = np.array([1.5] + [0.0] * 5, dtype=np.float32)  # fractional count
    dets, err = parse_nms(buf)
    assert dets == [] and err and "bad count" in err
    over = np.array([301.0], dtype=np.float32)           # above sanity bound
    _, err = parse_nms(over)
    assert err and "bad count" in err


def test_parse_nms_rejects_truncated():
    buf = _nms_buffer({0: [(0.1, 0.1, 0.3, 0.3, 0.9)]})[:3]  # cut mid-box
    dets, err = parse_nms(buf)
    assert dets == [] and err and "exhausted" in err


# --------------------------------- nms / iou ---------------------------------

def _det(cls, score, y1, x1, y2, x2):
    return (cls, score, y1, x1, y2, x2)


def test_iou_symmetry_and_disjoint():
    a = _det(0, 0.9, 0.1, 0.1, 0.5, 0.5)
    b = _det(1, 0.8, 0.2, 0.2, 0.6, 0.6)
    assert iou(a, b) == pytest.approx(iou(b, a))
    assert 0.0 < iou(a, b) < 1.0
    far = _det(2, 0.7, 0.7, 0.7, 0.9, 0.9)
    assert iou(a, far) == 0.0


def test_cross_class_nms_suppresses_and_orders():
    hi = _det(0, 0.9, 0.1, 0.1, 0.5, 0.5)
    lo = _det(1, 0.8, 0.12, 0.12, 0.52, 0.52)   # heavy overlap, lower score
    keep = cross_class_nms([lo, hi], 0.45)
    assert keep == [hi]                          # cross-class suppression
    far = _det(2, 0.5, 0.7, 0.7, 0.9, 0.9)
    out = cross_class_nms([far, lo, hi], 0.45)
    assert out == [hi, far]                      # score-descending


# ---------------------------- letterbox geometry ------------------------------

def test_letterbox_landscape_and_portrait():
    rgb = np.full((2160, 3840, 3), 90, dtype=np.uint8)
    img, meta = letterbox(rgb)
    assert img.shape == (DET_INPUT_SIZE, DET_INPUT_SIZE, 3)
    assert meta["scale"] == pytest.approx(640 / 3840)
    assert meta["pad_top"] == pytest.approx(140.0) and meta["pad_left"] == 0.0
    assert np.all(img[:140] == 0) and np.all(img[-140:] == 0)  # black bars
    assert np.all(img[200:400] != 0)                           # image band

    rgb_t = np.full((3840, 2160, 3), 90, dtype=np.uint8)
    img_t, meta_t = letterbox(rgb_t)
    assert meta_t["pad_left"] == pytest.approx(140.0) and meta_t["pad_top"] == 0.0
    assert np.all(img_t[:, :140] == 0) and np.all(img_t[:, -140:] == 0)


def test_unletterbox_point_center_and_bar():
    rgb = np.full((2160, 3840, 3), 90, dtype=np.uint8)
    _, meta = letterbox(rgb)
    y, x = unletterbox_point(0.5, 0.5, meta)      # frame center maps to center
    assert y == pytest.approx(0.5, abs=1e-4) and x == pytest.approx(0.5, abs=1e-4)
    y, _ = unletterbox_point(0.1, 0.5, meta)      # inside the top black bar
    assert y < 0.0                                 # lands outside the frame


def test_grid_cell_mapping():
    from detector import _grid_cell
    region = (0.1, 0.2, 0.9, 0.8)
    assert _grid_cell(0.1, 0.2, *region, 3, 4) == "g1-1"
    assert _grid_cell(0.5, 0.5, *region, 3, 4) == "g2-3"
    assert _grid_cell(0.89, 0.79, *region, 3, 4) == "g3-4"
    assert _grid_cell(0.95, 0.5, *region, 3, 4) is None  # outside region
    assert _grid_cell(0.5, 0.9, *region, 3, 4) is None


# -------------------------------- GoodsDetector -------------------------------

class _FakeClient:
    """Returns canned NMS buffers per call; records request shapes."""

    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def infer_with_tensors(self, model_id, inputs, input_names=None,
                           timeout_ms=None):
        self.calls.append({"model_id": model_id, "inputs": inputs,
                           "names": list(input_names or []),
                           "timeout_ms": timeout_ms})
        return [self.outputs.pop(0)]


def _square_detector() -> GoodsDetector:
    return GoodsDetector(_emb(), list(N_LABELS), n_goods=40,
                         threshold=0.25, nms_iou=0.45)


def test_detector_init_validates():
    with pytest.raises(ValueError):
        GoodsDetector(np.zeros((3, 7), dtype=np.float32), N_LABELS, 2)
    with pytest.raises(ValueError):
        GoodsDetector(_emb(), N_LABELS[:-1], 40)


def test_detect_end_to_end_filters_dedups_and_maps():
    det = _square_detector()
    client = _FakeClient([_nms_buffer({
        0: [(0.1, 0.1, 0.3, 0.3, 0.90),        # A goods, kept
            (0.12, 0.12, 0.32, 0.32, 0.85),    # overlaps A -> suppressed
            (0.6, 0.6, 0.8, 0.8, 0.30),        # C goods, kept
            (0.4, 0.1, 0.5, 0.2, 0.10)],       # below threshold
        45: [(0.1, 0.6, 0.3, 0.8, 0.50)],      # E person, kept (not counted)
    })])
    frame = np.full((DET_INPUT_SIZE, DET_INPUT_SIZE, 3), 90, dtype=np.uint8)
    dets = det.detect(client, frame, region=(0.0, 0.0, 1.0, 1.0),
                      rows=3, cols=4)

    # request contract: flattened uint8 image + uint16 vocab, named inputs
    (call,) = client.calls
    assert call["names"] == [DET_INPUT_IMAGE, DET_INPUT_EMB]
    assert call["inputs"][0].size == DET_INPUT_SIZE ** 2 * 3
    assert call["inputs"][0].dtype == np.uint8
    assert call["inputs"][1].size == DET_EMB_ROWS * DET_EMB_DIM
    assert call["inputs"][1].dtype == np.uint16

    assert [d["label"] for d in dets] == ["bottle", "person", "bottle"]
    assert [d["score"] for d in dets] == [0.9, 0.5, 0.3]      # score-desc
    a, e, c = dets
    assert a["goods"] is True and a["cell"] == "g1-1"          # center (0.2,0.2)
    assert c["goods"] is True and c["cell"] == "g3-3"          # center (0.7,0.7)
    assert e["goods"] is False                                  # neg not counted
    assert a["box"] == [pytest.approx(v, abs=1e-3)
                        for v in (0.1, 0.1, 0.3, 0.3)]
    json.dumps(dets)                                            # SSE-serializable
    assert det.items(dets) == 2
    assert count_items(dets) == 2


def test_detect_drops_letterbox_bar_centers():
    det = _square_detector()
    client = _FakeClient([_nms_buffer({
        0: [(0.05, 0.3, 0.25, 0.5, 0.90),       # center y640 = 0.15 -> top bar
            (0.40, 0.3, 0.60, 0.5, 0.80)],      # center mid-frame -> kept
    })])
    frame = np.full((640, 1280, 3), 90, dtype=np.uint8)  # wide: r=0.5, pad 160
    dets = det.detect(client, frame)
    assert len(dets) == 1
    assert 0.0 <= dets[0]["box"][1] < 1.0        # survivor inside the frame


def test_detect_parse_failure_returns_empty():
    det = _square_detector()
    client = _FakeClient([np.array([1.5], dtype=np.float32)])  # bad count
    frame = np.full((64, 64, 3), 90, dtype=np.uint8)
    assert det.detect(client, frame) == []


def test_detect_caps_payload():
    boxes = []
    for i in range(70):                          # 10x7 disjoint tiles
        r, c = divmod(i, 10)
        y1, x1 = 0.02 + r * 0.13, 0.02 + c * 0.09
        boxes.append((y1, x1, y1 + 0.06, x1 + 0.06, 0.9 - i * 0.001))
    det = _square_detector()
    client = _FakeClient([_nms_buffer({0: boxes})])
    frame = np.full((DET_INPUT_SIZE, DET_INPUT_SIZE, 3), 90, dtype=np.uint8)
    assert len(det.detect(client, frame)) == DET_MAX_DETECTIONS


def test_detect_max_detections_param():
    boxes = []
    for i in range(70):                          # 10x7 disjoint tiles
        r, c = divmod(i, 10)
        y1, x1 = 0.02 + r * 0.13, 0.02 + c * 0.09
        boxes.append((y1, x1, y1 + 0.06, x1 + 0.06, 0.9 - i * 0.001))
    det = _square_detector()
    client = _FakeClient([_nms_buffer({0: boxes})])
    frame = np.full((DET_INPUT_SIZE, DET_INPUT_SIZE, 3), 90, dtype=np.uint8)
    assert len(det.detect(client, frame, max_detections=10)) == 10


def test_detect_annotates_category():
    det = GoodsDetector(_emb(), list(N_LABELS), n_goods=40,
                        categories=["can"] * DET_EMB_ROWS)
    assert set(det.categories) == {"can"}
    client = _FakeClient([_nms_buffer({
        0: [(0.1, 0.1, 0.3, 0.3, 0.9)],
        45: [(0.5, 0.5, 0.6, 0.6, 0.5)],          # negative row keeps a category
    })])
    dets = det.detect(client, np.full((64, 64, 3), 90, dtype=np.uint8))
    assert [d["category"] for d in dets] == ["can", "can"]

    assert set(_square_detector().categories) == {"other"}  # no sidecar


# -------------------------------- detect_tiled --------------------------------

TILE_OVERLAP = 0.08


def _tiled_buffers(frame, entries):
    """Build the 4 per-tile NMS buffers in detect_tiled's TL/TR/BL/BR call
    order. entries: (cls, score, full_box, quads) with full_box =
    (fy1, fx1, fy2, fx2) full-frame normalized and quads = set of (top,
    left) tiles that see the object — a straddling object appears in both
    so the midline partition must drop the non-owning copy."""
    oh, ow = frame.shape[:2]
    oy, ox = int(oh * TILE_OVERLAP), int(ow * TILE_OVERLAP)
    spans_y = ((0, oh // 2 + oy, True), (oh // 2 - oy, oh, False))
    spans_x = ((0, ow // 2 + ox, True), (ow // 2 - ox, ow, False))
    bufs = []
    for y0, y1, top in spans_y:
        for x0, x1, left in spans_x:
            _, meta = letterbox(frame[y0:y1, x0:x1])
            per_class: dict[int, list] = {}
            for cls, score, (fy1, fx1, fy2, fx2), quads in entries:
                if (top, left) not in quads:
                    continue
                ty1 = ((fy1 * oh - y0) * meta["scale"]
                       + meta["pad_top"]) / DET_INPUT_SIZE
                tx1 = ((fx1 * ow - x0) * meta["scale"]
                       + meta["pad_left"]) / DET_INPUT_SIZE
                ty2 = ((fy2 * oh - y0) * meta["scale"]
                       + meta["pad_top"]) / DET_INPUT_SIZE
                tx2 = ((fx2 * ow - x0) * meta["scale"]
                       + meta["pad_left"]) / DET_INPUT_SIZE
                per_class.setdefault(cls, []).append((ty1, tx1, ty2, tx2,
                                                      score))
            bufs.append(_nms_buffer(per_class))
    return bufs


def test_detect_tiled_offsets_and_partition_dedup():
    frame = np.full((DET_INPUT_SIZE, DET_INPUT_SIZE, 3), 90, dtype=np.uint8)
    entries = [
        (0, 0.90, (0.15, 0.15, 0.25, 0.25), {(True, True)}),      # TL only
        # straddles the x midline: both TL and TR see it, TR owns the center
        (0, 0.80, (0.25, 0.45, 0.35, 0.55), {(True, True), (True, False)}),
        (0, 0.70, (0.65, 0.65, 0.75, 0.75), {(False, False)}),    # BR only
    ]
    det = _square_detector()
    client = _FakeClient(_tiled_buffers(frame, entries))
    dets = det.detect_tiled(client, frame)

    assert len(client.calls) == 4                  # one infer per quadrant
    assert [d["score"] for d in dets] == [0.9, 0.8, 0.7]  # straddler once
    a, b, c = dets                                 # box is [x1, y1, x2, y2]
    assert a["box"] == [pytest.approx(v, abs=1e-3)
                        for v in (0.15, 0.15, 0.25, 0.25)]
    assert b["box"] == [pytest.approx(v, abs=1e-3)
                        for v in (0.45, 0.25, 0.55, 0.35)]
    assert c["box"] == [pytest.approx(v, abs=1e-3)
                        for v in (0.65, 0.65, 0.75, 0.75)]
    assert det.items(dets) == 3


def test_detect_tiled_threshold_override():
    frame = np.full((DET_INPUT_SIZE, DET_INPUT_SIZE, 3), 90, dtype=np.uint8)
    entries = [(0, 0.12, (0.15, 0.15, 0.25, 0.25), {(True, True)})]
    det = _square_detector()                       # constructor threshold 0.25
    kept = det.detect_tiled(_FakeClient(_tiled_buffers(frame, entries)),
                            frame, threshold=0.10)
    assert [d["score"] for d in kept] == [0.12]
    dropped = det.detect_tiled(_FakeClient(_tiled_buffers(frame, entries)),
                               frame)
    assert dropped == []


def test_detect_tiled_caps_and_annotates():
    frame = np.full((DET_INPUT_SIZE, DET_INPUT_SIZE, 3), 90, dtype=np.uint8)
    entries = []
    for i in range(6):                             # 6 disjoint boxes in TL quad
        y1, x1 = 0.03 + (i % 3) * 0.13, 0.03 + (i // 3) * 0.13
        entries.append((0, 0.9 - i * 0.05, (y1, x1, y1 + 0.06, x1 + 0.06),
                        {(True, True)}))
    det = GoodsDetector(_emb(), list(N_LABELS), n_goods=40,
                        categories=["can"] * DET_EMB_ROWS)
    dets = det.detect_tiled(_FakeClient(_tiled_buffers(frame, entries)),
                            frame, max_detections=3)
    assert [d["score"] for d in dets] == [0.9, 0.85, 0.8]
    assert all(d["category"] == "can" and d["goods"] for d in dets)


def test_detect_tiled_tiles1_delegates_and_rejects_others():
    det = _square_detector()
    client = _FakeClient([_nms_buffer({})])
    dets = det.detect_tiled(client, np.full((64, 64, 3), 90, np.uint8),
                            tiles=1)
    assert dets == [] and len(client.calls) == 1   # single full-frame infer
    with pytest.raises(ValueError):
        det.detect_tiled(_FakeClient([]),
                         np.full((64, 64, 3), 90, np.uint8), tiles=3)


def test_count_items_mixed():
    dets = [
        {"label": "bottle", "goods": True, "cell": "g1-1"},
        {"label": "person", "goods": False, "cell": "g2-2"},  # neg: no count
        {"label": "bottle", "goods": True, "cell": None},     # out of region
    ]
    assert count_items(dets) == 1


def test_items_by_category_tallies_goods_only():
    dets = [
        {"label": "soda can", "goods": True, "cell": "g1-1",
         "category": "can"},
        {"label": "beverage can", "goods": True, "cell": "g1-2",
         "category": "can"},
        {"label": "bottle", "goods": True, "cell": "g2-1",
         "category": "bottle"},
        {"label": "person", "goods": False, "cell": "g3-3",
         "category": "negative"},                # negatives never counted
        {"label": "snack bag", "goods": True, "cell": None,
         "category": "bag"},                     # outside region
        {"label": "legacy row", "goods": True, "cell": "g4-4"},  # no cat
    ]
    assert items_by_category(dets) == {"can": 2, "bottle": 1, "other": 1}


# -------------------------------- registration --------------------------------

class _FlakyRegister:
    def __init__(self, failures: list[str]) -> None:
        self.failures = list(failures)
        self.calls = 0
        self.last_kwargs: dict = {}

    def register_model(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.failures:
            raise RuntimeError(self.failures.pop(0))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(detector.time, "sleep", lambda s: None)


def test_register_retries_then_succeeds():
    client = _FlakyRegister(["rpc timeout", "device busy"])
    register_detector_model(client, "/m.hef", "yw", "shelf-ops", retries=3)
    assert client.calls == 3
    assert client.last_kwargs["model_type"] == "detection"
    spec_names = [i["name"] for i in client.last_kwargs["inputs"]]
    assert spec_names == [DET_INPUT_IMAGE, DET_INPUT_EMB]


def test_register_already_registered_is_ok():
    client = _FlakyRegister(["model id already exists"])
    register_detector_model(client, "/m.hef", "yw", "shelf-ops", retries=3)
    assert client.calls == 1


def test_register_exhausts_retries():
    client = _FlakyRegister(["boom", "boom", "boom"])
    with pytest.raises(RuntimeError):
        register_detector_model(client, "/m.hef", "yw", "shelf-ops", retries=3)
    assert client.calls == 3


# --------------------------------- vocabulary ---------------------------------

def test_bundled_vocab_loads():
    d = bundled_vocab_path()
    emb, labels, n_goods, categories = load_vocab(
        os.path.join(d, "yolo_world_vocab.npy"),
        os.path.join(d, "yolo_world_vocab.json"))
    assert emb.shape == (DET_EMB_ROWS, DET_EMB_DIM)
    assert emb.dtype == np.float32
    assert len(labels) == DET_EMB_ROWS and labels[0] == "bottle"
    assert n_goods == 24                            # probe53 curated list
    assert len(categories) == DET_EMB_ROWS
    assert categories[0] == "bottle"                # goods rows carry families
    assert "can" in categories and "bag" in categories
    assert set(categories[n_goods:]) == {"negative"}  # pad rows are negatives


def test_load_vocab_rejects_mismatched_sidecar(tmp_path):
    npy = tmp_path / "v.npy"
    np.save(npy, _emb())
    good = {"labels": N_LABELS, "n_goods": 40}
    json_path = tmp_path / "v.json"

    json_path.write_text(json.dumps({**good, "labels": N_LABELS[:-1]}))
    with pytest.raises(ValueError):
        load_vocab(str(npy), str(json_path))

    json_path.write_text(json.dumps({**good, "n_goods": 0}))
    with pytest.raises(ValueError):
        load_vocab(str(npy), str(json_path))

    json_path.write_text(json.dumps({**good, "categories": ["bottle"]}))
    _, _, _, cats = load_vocab(str(npy), str(json_path))
    assert set(cats) == {"other"}                 # wrong length -> fallback

    json_path.write_text(json.dumps(good))        # legacy sidecar, no key
    emb, labels, n, cats = load_vocab(str(npy), str(json_path))
    assert emb.shape == (DET_EMB_ROWS, DET_EMB_DIM) and n == 40
    assert set(cats) == {"other"}
