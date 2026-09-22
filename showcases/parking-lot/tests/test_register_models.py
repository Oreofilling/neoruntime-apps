"""Unit tests for startup model registration vs an incumbent preload.

app-manager preloads manifest-declared models at app start with a
platform-composed variant that differs textually from the app's blob, and
ai-runtime refuses a same-id re-registration with a different config. The
1.2.1 fix classifies that refusal instead of retrying it into a 28s stall:

- refusal shows the incumbent already routes through our backend_function
  (``expected_backend`` in MODEL_DEFS) -> keep the incumbent, no eviction;
- incumbent backend differs (the pre-profile platform-row trap: default
  hailo_yolov8n backend, wrong NMS tensor name) -> force-drop via
  ``unregister_model`` (empty owner = system-level unload) and re-register;
- any other failure -> propagate to the existing retry loop.

Only the app-side decision logic is tested; the inference client is a mock,
no SDK daemon is required.
"""

import sys
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import parking_lot.app as app_module
from parking_lot.app import ParkingLotApp, _incumbent_uses_backend
from parking_lot.config import MODEL_DEFS


def _refusal(backend: str) -> str:
    """ai-runtime refusal with the incumbent variant embedded, mirroring
    the live text captured on device 93.72 (advisory fields differ from the
    app blob: label_offset 1, composed 5-label table)."""
    return (
        "model id 'yolov5m_vehicles' is already registered with a different "
        "configuration (type='detection' variant='{\"backend_function\": "
        f"\"{backend}\", \"iou_threshold\": 0.5, \"detection_threshold\": 0.3, "
        "\"output_activation\": \"none\", \"label_offset\": 1, \"max_boxes\": 80, "
        "\"labels\": [\"1\", \"2\", \"3\", \"4\", \"5\"]}')"
    )


def make_app() -> ParkingLotApp:
    """ParkingLotApp with only the registration fields set (no __init__)."""
    app = ParkingLotApp.__new__(ParkingLotApp)
    app.infer_client = MagicMock()
    return app


@pytest.fixture(autouse=True)
def _fake_app_id(monkeypatch):
    """Config.get_app_id() reads the SDK runtime config; swap the module's
    Config reference for a stub so no daemon/env is needed."""
    monkeypatch.setattr(
        app_module, "Config", SimpleNamespace(get_app_id=lambda: "parking_lot"),
    )


# --- _incumbent_uses_backend ------------------------------------------------

def test_incumbent_uses_backend_matches_expected_backend():
    # Arrange / Act / Assert — incumbent routing through our backend reads as ours
    assert _incumbent_uses_backend(_refusal("yolov5m_vehicles"), "yolov5m_vehicles")


def test_incumbent_uses_backend_rejects_different_backend():
    # Arrange / Act / Assert — the pre-profile trap (default yolov8 backend)
    assert not _incumbent_uses_backend(_refusal("hailo_yolov8n_384_640"), "yolov5m_vehicles")


def test_incumbent_uses_backend_returns_false_without_variant_json():
    # Arrange — refusal without an embedded variant (older runtime text)
    msg = "model id 'x' is already registered with a different configuration"

    # Act / Assert — parse failure reads as "not ours", caller replaces
    assert not _incumbent_uses_backend(msg, "yolov5m_vehicles")


def test_incumbent_uses_backend_returns_false_on_malformed_json():
    # Arrange — variant captured but not valid JSON
    msg = "already registered (variant='{oops}')"

    # Act / Assert
    assert not _incumbent_uses_backend(msg, "yolov5m_vehicles")


# --- _register_one conflict handling ----------------------------------------

def test_register_one_keeps_correct_incumbent_without_eviction():
    # Arrange — preload already registered yolov5m_vehicles via the custom
    # profile; the textual variant difference triggers a refusal
    app = make_app()
    app.infer_client.register_model.side_effect = Exception(_refusal("yolov5m_vehicles"))

    # Act
    app._register_one("yolov5m_vehicles", MODEL_DEFS["yolov5m_vehicles"])

    # Assert — kept as-is: exactly one attempt, nothing unregistered
    assert app.infer_client.register_model.call_count == 1
    app.infer_client.unregister_model.assert_not_called()


def test_register_one_replaces_misconfigured_incumbent():
    # Arrange — platform row predates the custom profile, so the incumbent
    # routes through the default yolov8 backend (tensor mismatch trap)
    app = make_app()
    app.infer_client.register_model.side_effect = [
        Exception(_refusal("hailo_yolov8n_384_640")),
        None,  # re-register after the force-drop succeeds
    ]

    # Act
    app._register_one("yolov5m_vehicles", MODEL_DEFS["yolov5m_vehicles"])

    # Assert — force-dropped once, then re-applied with our variant blob
    app.infer_client.unregister_model.assert_called_once_with(model_id="yolov5m_vehicles")
    assert app.infer_client.register_model.call_count == 2
    rerun = app.infer_client.register_model.call_args_list[1]
    assert rerun.kwargs["model_variant"] == MODEL_DEFS["yolov5m_vehicles"]["variant"]


def test_register_one_propagates_non_conflict_failures():
    # Arrange — a transport-level error is not an incumbent decision
    app = make_app()
    app.infer_client.register_model.side_effect = Exception("connection refused")

    # Act / Assert — surfaces to the retry loop, no unregister attempted
    with pytest.raises(Exception, match="connection refused"):
        app._register_one("yolov5m_vehicles", MODEL_DEFS["yolov5m_vehicles"])
    app.infer_client.unregister_model.assert_not_called()


# --- register_models orchestration -------------------------------------------

def test_register_models_skips_registered_models_without_variant():
    # Arrange — everything except the variant-carrying model is resident
    resident = [SimpleNamespace(model_id=m) for m in
                ("scdepthv3", "license_plate_det", "plate_recognition")]
    app = make_app()
    app.infer_client.list_models.return_value = resident
    app.infer_client.register_model.side_effect = Exception(_refusal("yolov5m_vehicles"))

    # Act
    app.register_models()

    # Assert — only the variant model talked to the runtime, incumbent kept
    registered_ids = [c.kwargs["model_id"]
                      for c in app.infer_client.register_model.call_args_list]
    assert registered_ids == ["yolov5m_vehicles"]
    app.infer_client.unregister_model.assert_not_called()


def test_register_models_registers_all_on_clean_device():
    # Arrange — fresh runtime: nothing resident, every register succeeds
    app = make_app()
    app.infer_client.list_models.return_value = []

    # Act
    app.register_models()

    # Assert — all four models registered exactly once, single pass
    registered_ids = sorted(
        c.kwargs["model_id"] for c in app.infer_client.register_model.call_args_list
    )
    assert registered_ids == sorted(MODEL_DEFS.keys())
    app.infer_client.unregister_model.assert_not_called()
