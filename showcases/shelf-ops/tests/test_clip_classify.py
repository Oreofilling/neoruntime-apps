"""Unit tests for clip_classify.py (slot-anchored CLIP helpers)."""
import numpy as np
import pytest

import clip_classify
from clip_classify import (
    CLIP_EMBED_DIM,
    CLIP_INPUT_SIZE,
    SlotClassifier,
    crop_box,
    decode_clip_vector,
    polygon_bbox,
    register_clip_model,
    rgb_to_nv12,
)

NV12_LEN = CLIP_INPUT_SIZE * CLIP_INPUT_SIZE * 3 // 2  # 75264


# ------------------------------- pixel helpers -------------------------------

def test_rgb_to_nv12_layout():
    # cv2's YUV_I420 is studio-swing (BT.601 limited range): 16..235
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    white = np.full((224, 224, 3), 255, dtype=np.uint8)
    for rgb, y_val in ((black, 16), (white, 235)):
        nv12 = rgb_to_nv12(rgb)
        assert nv12.dtype == np.uint8
        assert nv12.size == NV12_LEN
        y_plane = nv12[:CLIP_INPUT_SIZE * CLIP_INPUT_SIZE]
        assert np.allclose(y_plane, y_val, atol=1)
        # neutral chroma
        assert np.all(nv12[CLIP_INPUT_SIZE * CLIP_INPUT_SIZE:] == 128)


def test_crop_box_clamps_and_min_size():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :] = (0, 255, 0)
    # out-of-frame box gets clipped
    crop = crop_box(frame, (-10, -10, 300, 500))
    assert crop.shape == (100, 200, 3)
    # degenerate box yields at least 1 px
    tiny = crop_box(frame, (50, 50, 50, 50))
    assert tiny.shape[0] >= 1 and tiny.shape[1] >= 1


def test_polygon_bbox_pad_and_clamp():
    frame_shape = (1080, 1920, 3)
    poly = [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]]
    box = polygon_bbox(poly, frame_shape, pad=0.0)
    assert np.allclose(box, [0.1 * 1919, 0.2 * 1079,
                             0.3 * 1919, 0.4 * 1079], atol=1.0)
    padded = polygon_bbox(poly, frame_shape, pad=0.5)
    assert padded[0] < box[0] and padded[1] < box[1]
    assert padded[2] > box[2] and padded[3] > box[3]
    # pad cannot escape the frame
    edge = polygon_bbox([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]],
                        frame_shape, pad=10.0)
    assert edge[0] == 0.0 and edge[1] == 0.0
    with pytest.raises(ValueError):
        polygon_bbox([[0.0, 0.0], [1.0, 1.0]], frame_shape)


# ------------------------------- decode + math -------------------------------

def test_decode_clip_vector_centers_and_normalizes():
    direction = np.zeros(CLIP_EMBED_DIM, dtype=np.float32)
    direction[7] = 3.0
    direction[100] = -4.0
    raw = np.clip(120.0 + direction, 0, 255).astype(np.uint8)
    v = decode_clip_vector(raw)
    assert v.size == CLIP_EMBED_DIM
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5
    assert abs(float(v.mean())) < 1e-6
    # direction preserved (sign of dominant components)
    assert v[7] > 0 and v[100] < 0
    assert v[7] > v[100]


def test_decode_clip_vector_rejects_bad_input():
    with pytest.raises(ValueError):
        decode_clip_vector(np.zeros(100, dtype=np.uint8))
    with pytest.raises(ValueError):
        decode_clip_vector(np.full(CLIP_EMBED_DIM, 88, dtype=np.uint8))


# ------------------------------ SlotClassifier -------------------------------

def _txt_matrix() -> np.ndarray:
    """3 canonical directions (scaled up to prove re-normalization)."""
    m = np.zeros((3, CLIP_EMBED_DIM), dtype=np.float32)
    for i, comp in enumerate((0, 11, 250)):
        m[i, comp] = 5.0
    return m


def test_classifier_init_validates_and_normalizes():
    with pytest.raises(ValueError):
        SlotClassifier(np.zeros((3, 128), dtype=np.float32))
    clf = SlotClassifier(_txt_matrix())
    assert np.allclose(np.linalg.norm(clf.txt, axis=1), 1.0, atol=1e-5)


def test_classifier_best_argmax():
    clf = SlotClassifier(_txt_matrix())
    vec = np.zeros(CLIP_EMBED_DIM, dtype=np.float32)
    vec[250] = 1.0
    row, cos = clf.best(vec)
    assert row == 2 and cos == pytest.approx(1.0, abs=1e-5)


class _FakeClient:
    """Returns canned uint8 CLIP outputs per call; records request sizes."""

    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = list(outputs)
        self.input_sizes: list[int] = []

    def infer_with_tensors(self, model_id, inputs, input_names=None,
                           timeout_ms=None):
        self.input_sizes.append(len(inputs[0]))
        out = self.outputs.pop(0)
        return [out]


def _uint8_vector(component: int, offset: float = 120.0) -> np.ndarray:
    v = np.full(CLIP_EMBED_DIM, offset, dtype=np.float32)
    v[component] = offset + 60.0
    return np.clip(v, 0, 255).astype(np.uint8)


def test_classifier_embed_and_classify_round_trip():
    clf = SlotClassifier(_txt_matrix())
    client = _FakeClient([_uint8_vector(0), _uint8_vector(0), _uint8_vector(11)])
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)

    vec = clf.embed(client, frame[:100, :80])
    assert client.input_sizes == [NV12_LEN]        # flat NV12 224x224x1.5
    row, cos = clf.best(vec)
    assert row == 0 and cos > 0.9

    boxes = [[10, 10, 200, 200], [300, 100, 600, 400]]
    rows, scores = clf.classify(client, frame, boxes)
    assert rows.tolist() == [0, 1]
    assert np.all(scores > 0.9)


# ------------------------------ registration ---------------------------------

class _FlakyRegister:
    def __init__(self, failures: list[str]) -> None:
        self.failures = list(failures)
        self.calls = 0

    def register_model(self, **kwargs):
        self.calls += 1
        if self.failures:
            raise RuntimeError(self.failures.pop(0))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(clip_classify.time, "sleep", lambda s: None)


def test_register_retries_then_succeeds():
    client = _FlakyRegister(["rpc timeout", "device busy"])
    register_clip_model(client, "/m.hef", "clip", "shelf-ops", retries=3)
    assert client.calls == 3


def test_register_already_registered_is_ok():
    client = _FlakyRegister(["model id already exists"])
    register_clip_model(client, "/m.hef", "clip", "shelf-ops", retries=3)
    assert client.calls == 1


def test_register_exhausts_retries():
    client = _FlakyRegister(["boom", "boom", "boom"])
    with pytest.raises(RuntimeError):
        register_clip_model(client, "/m.hef", "clip", "shelf-ops", retries=3)
    assert client.calls == 3
