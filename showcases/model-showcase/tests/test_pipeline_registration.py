"""Tests for pipeline-vs-single-file registration routing (1.4.7 fix).

``_ensure_model_registered`` used to assume every catalog entry has a
top-level ``path``. Pipeline entries (``lpr_pipeline`` / ``ocr_pipeline``)
only have per-stage paths, so callers that reached it with a pipeline —
startup verify (``connect``/``_verify_current_model``) and the infer-loop
auto-recovery — crashed with ``KeyError: 'path'`` instead of registering
the stages.

Container-only imports (neoruntime_ipc_sdk, flask_sock) are stubbed so
``import main`` works in a plain unit-test environment.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

for _name in ("neoruntime_ipc_sdk", "flask_sock"):
    if _name not in sys.modules:
        _stub = types.ModuleType(_name)
        _stub.__getattr__ = lambda attr: MagicMock()
        sys.modules[_name] = _stub

import main  # noqa: E402


def _pipeline_entry() -> dict:
    return {
        "id": "lpr_pipeline",
        "type": "pipeline_lpr",
        "name": "License Plate Recognition",
        "stages": [
            {"id": "license_plate_det", "path": "/det.hef", "type": "detection"},
            {"id": "lprnet", "path": "/ocr.hef", "type": "ocr_recognition"},
        ],
    }


def _minimal_showcase() -> main.ModelShowcase:
    sc = main.ModelShowcase.__new__(main.ModelShowcase)
    sc.current_model = ""
    sc.current_model_type = ""
    sc._current_model_info = None
    sc.infer_client = MagicMock()
    return sc


def test_ensure_model_registered_delegates_pipeline_to_stage_registration(monkeypatch):
    # Arrange — the recovery/startup callers hand a pipeline entry to
    # _ensure_model_registered
    pipeline = _pipeline_entry()
    monkeypatch.setattr(main, "MODEL_CATALOG", [pipeline])
    sc = _minimal_showcase()
    sc.infer_client.list_models.return_value = []

    # Act — previously KeyError: 'path' at the single-model register call
    sc._ensure_model_registered(pipeline)

    # Assert — each stage was registered individually, no top-level register
    registered_ids = [
        call.kwargs["model_id"]
        for call in sc.infer_client.register_model.call_args_list
    ]
    assert registered_ids == ["license_plate_det", "lprnet"]


def test_ensure_model_registered_skips_already_registered_stages(monkeypatch):
    # Arrange — both stages already resident (e.g. after a transient infer
    # failure that triggered auto-recovery)
    pipeline = _pipeline_entry()
    monkeypatch.setattr(main, "MODEL_CATALOG", [pipeline])
    sc = _minimal_showcase()

    class _Registered:
        def __init__(self, model_id):
            self.model_id = model_id

    sc.infer_client.list_models.return_value = [
        _Registered("license_plate_det"), _Registered("lprnet"),
    ]

    # Act
    sc._ensure_model_registered(pipeline)

    # Assert — delegation checked presence per stage; nothing re-registered
    sc.infer_client.register_model.assert_not_called()


def test_ensure_model_registered_force_reregisters_stages_despite_stale_list(monkeypatch):
    # Arrange — list_models() is stale (right after a service restart) and
    # claims the stages are registered when they are not; the infer-loop
    # auto-recovery passes force=True precisely for this case
    pipeline = _pipeline_entry()
    monkeypatch.setattr(main, "MODEL_CATALOG", [pipeline])
    sc = _minimal_showcase()

    class _Registered:
        def __init__(self, model_id):
            self.model_id = model_id

    sc.infer_client.list_models.return_value = [
        _Registered("license_plate_det"), _Registered("lprnet"),
    ]

    # Act
    sc._ensure_model_registered(pipeline, force=True)

    # Assert — force survives the delegation: both stages re-registered
    registered_ids = [
        call.kwargs["model_id"]
        for call in sc.infer_client.register_model.call_args_list
    ]
    assert registered_ids == ["license_plate_det", "lprnet"]


def test_ensure_model_registered_still_registers_single_file_entries(monkeypatch):
    # Arrange — plain single-file entry must keep its direct-register path
    entry = {
        "id": "yolov8n_detection", "type": "detection",
        "path": "/det.hef", "category": "detection",
    }
    monkeypatch.setattr(main, "MODEL_CATALOG", [entry])
    sc = _minimal_showcase()
    sc.infer_client.list_models.return_value = []

    # Act
    sc._ensure_model_registered(entry)

    # Assert
    call = sc.infer_client.register_model.call_args
    assert call.kwargs["model_id"] == "yolov8n_detection"
    assert call.kwargs["model_path"] == "/det.hef"
