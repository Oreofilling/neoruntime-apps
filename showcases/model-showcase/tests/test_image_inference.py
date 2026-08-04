"""Tests for one-shot image upload inference.

Covers:
- ``ModelShowcase.infer_image``: single-shot infer + annotate + JPEG encode with a
  mocked ``infer_client``.
- ``POST /api/image/upload``: happy path (annotated JPEG + X-Infer-Stats header)
  and negative cases (missing file, bad extension, decode failure, no model loaded).

Container-only imports (hailo_ipc_sdk, flask_sock) are stubbed so ``import main``
works in a plain unit-test environment.
"""

from __future__ import annotations

import io
import sys
import threading
import types
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from PIL import Image

# --- stub container-only / unavailable imports so `import main` succeeds ----
for _name in ("hailo_ipc_sdk", "flask_sock"):
    if _name not in sys.modules:
        _stub = types.ModuleType(_name)
        _stub.__getattr__ = lambda attr: MagicMock()  # any name -> MagicMock
        sys.modules[_name] = _stub

import main  # noqa: E402  (must follow the stubs above)


# --- helpers ---------------------------------------------------------------

def _png_bytes(w: int = 64, h: int = 48) -> bytes:
    """A small distinct RGB PNG (red gradient) as bytes."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 2] = np.linspace(0, 255, w, dtype=np.uint8)  # BGR red ramp
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
    return buf.getvalue()


def _minimal_showcase(model_id: str = "yolov8n_detection",
                      model_type: str = "detection",
                      input_fmt: str = "rgb",
                      input_w: int = 640,
                      input_h: int = 640) -> main.ModelShowcase:
    """A ModelShowcase with only the fields infer_image() touches populated.

    Built via __new__ to bypass the heavy __init__ (media clients, threads).
    """
    sc = main.ModelShowcase.__new__(main.ModelShowcase)
    sc._model_lock = threading.Lock()
    sc.current_model = model_id
    sc.current_model_type = model_type
    sc._current_model_info = {
        "input_format": input_fmt,
        "input_width": input_w,
        "input_height": input_h,
    }
    sc._infer_warmup_timeout_ms = 6000
    # Fake result: empty detections, real-ish timing fields.
    fake_result = MagicMock()
    fake_result.objects = []
    fake_result.infer_time_us = 1234
    fake_result.hw_infer_time_us = 2000
    sc.infer_client = MagicMock()
    sc.infer_client.infer.return_value = fake_result
    return sc


# --- infer_image unit tests ------------------------------------------------

def test_infer_image_returns_annotated_jpeg_and_stats():
    # Arrange
    sc = _minimal_showcase()
    bgr = np.full((96, 128, 3), 80, dtype=np.uint8)

    # Act
    info = sc.infer_image(bgr)

    # Assert
    assert isinstance(info["jpeg"], bytes) and len(info["jpeg"]) > 0
    decoded = cv2.imdecode(np.frombuffer(info["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None, "jpeg must decode back to an image"
    assert info["model"] == "yolov8n_detection"
    assert info["model_type"] == "detection"
    # Input 128x96 is normalized to the 1280 render side (longest side → 1280).
    assert info["width"] == 1280 and info["height"] == 960
    assert info["infer_time_us"] == 1234
    assert info["hw_infer_time_us"] == 2000
    sc.infer_client.infer.assert_called_once()


def test_infer_image_raises_when_no_model_loaded():
    # Arrange
    sc = _minimal_showcase()
    sc.current_model = ""
    sc._current_model_info = None
    bgr = np.full((48, 48, 3), 10, dtype=np.uint8)

    # Act / Assert
    with pytest.raises(RuntimeError, match="no model loaded"):
        sc.infer_image(bgr)


def test_infer_image_raises_when_model_has_no_input_dims():
    # Arrange
    sc = _minimal_showcase()
    sc._current_model_info = {"input_format": "rgb", "input_width": 0, "input_height": 0}
    bgr = np.full((48, 48, 3), 10, dtype=np.uint8)

    # Act / Assert
    with pytest.raises(RuntimeError, match="input dimensions"):
        sc.infer_image(bgr)


def test_infer_image_passes_correct_model_id_and_timeout():
    # Arrange
    sc = _minimal_showcase(model_id="linknet_seg")
    bgr = np.full((48, 48, 3), 30, dtype=np.uint8)

    # Act
    sc.infer_image(bgr)

    # Assert — infer called with the captured model_id and warmup timeout
    _args, kwargs = sc.infer_client.infer.call_args
    assert kwargs["model_id"] == "linknet_seg"
    assert kwargs["timeout_ms"] == sc._infer_warmup_timeout_ms


def test_infer_image_parses_raw_outputs_for_vit_classification():
    # Arrange — vit_classification returns a raw logits tensor, NOT pre-parsed
    # classifications. Without the raw-output dispatch (mirroring _infer_loop),
    # classifications stay empty and _draw_classifications draws nothing.
    sc = _minimal_showcase(model_id="vit_classification", model_type="classification")
    raw_result = MagicMock()
    raw_result.objects = []
    raw_result.classifications = []
    raw_result.ocr_lines = []
    raw_result.masks = None
    raw_result.depth_maps = None
    raw_result.infer_time_us = 3000
    raw_result.hw_infer_time_us = 1500
    # raw_outputs[0] = logits tensor; _parse_classification_raw softmaxes → top-5.
    raw_result.raw_outputs = [np.array([[2.0, 0.1, 5.0, 0.2, 3.0, 1.0]], dtype=np.float32)]
    sc.infer_client.infer.return_value = raw_result
    bgr = np.full((96, 128, 3), 80, dtype=np.uint8)

    # Act
    info = sc.infer_image(bgr)

    # Assert — overlay was drawn: the classification bar darkens the top strip
    # away from the uniform 80 input. (No-parse path leaves top == 80.)
    decoded = cv2.imdecode(np.frombuffer(info["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert not np.allclose(decoded[5:20, 5:20], 80, atol=8), \
        "classification overlay must be drawn (top bar region must differ from input)"
    assert info["model"] == "vit_classification"
    assert info["model_type"] == "classification"



# --- /api/image/upload HTTP tests ------------------------------------------

def _app_with_mock_showcase(infer_image_ret=None, infer_image_exc=None):
    """Build a Flask app around a MagicMock showcase for route tests."""
    showcase = MagicMock()
    if infer_image_exc is not None:
        showcase.infer_image.side_effect = infer_image_exc
    else:
        showcase.infer_image.return_value = infer_image_ret or {
            "jpeg": b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9",
            "infer_time_us": 1500,
            "hw_infer_time_us": 900,
            "model": "yolov8n_detection",
            "model_type": "detection",
            "width": 128,
            "height": 96,
        }
    return main.create_app(showcase)


def _post_image(app, filename="shot.png", content=b"", field="image"):
    data = {field: (io.BytesIO(content), filename)}
    return app.test_client().post(
        "/api/image/upload",
        data=data,
        content_type="multipart/form-data",
    )


def test_upload_image_happy_path_returns_jpeg_and_stats_header():
    # Arrange
    app = _app_with_mock_showcase()
    png = _png_bytes()

    # Act
    resp = _post_image(app, filename="shot.png", content=png)

    # Assert
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert resp.headers.get("X-Infer-Stats"), "stats header must be present"
    import json as _json
    stats = _json.loads(resp.headers["X-Infer-Stats"])
    assert stats["model"] == "yolov8n_detection"
    assert stats["width"] == 128 and stats["height"] == 96
    assert stats["hw_infer_time_us"] == 900
    assert resp.data, "body must contain JPEG bytes"


def test_upload_image_missing_file_returns_400():
    # Arrange
    app = _app_with_mock_showcase()

    # Act
    resp = app.test_client().post("/api/image/upload", data={}, content_type="multipart/form-data")

    # Assert
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_upload_image_bad_extension_returns_400():
    # Arrange
    app = _app_with_mock_showcase()

    # Act
    resp = _post_image(app, filename="movie.mp4", content=b"not an image")

    # Assert
    assert resp.status_code == 400
    assert "Unsupported" in resp.get_json()["error"]


def test_upload_image_undecodable_returns_400():
    # Arrange — valid extension, garbage bytes -> cv2.imdecode returns None
    app = _app_with_mock_showcase()

    # Act
    resp = _post_image(app, filename="broken.png", content=b"definitely not a png")

    # Assert
    assert resp.status_code == 400
    assert "Invalid" in resp.get_json()["error"]


def test_upload_image_no_model_loaded_returns_400():
    # Arrange
    app = _app_with_mock_showcase(infer_image_exc=RuntimeError("no model loaded; select a model first"))
    png = _png_bytes()

    # Act
    resp = _post_image(app, filename="shot.png", content=png)

    # Assert
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "no model loaded; select a model first"
