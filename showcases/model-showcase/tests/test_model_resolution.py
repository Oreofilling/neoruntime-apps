"""Tests for host-first / bundled-fallback model path resolution.

Covers the plug-and-play change:
- ``_model_path``: host-provisioned copy wins; image-bundled flat-layout copy
  is the fallback (gym-ops ``_resolve_model_path`` pattern).
- ``_discover_models``/``_is_available``: bundled-only entries are available,
  both-missing entries are hidden, pipeline entries need every stage present,
  genai entries are always available.

Container-only imports (neoruntime_ipc_sdk, flask_sock) are stubbed so ``import main``
works in a plain unit-test environment.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

# --- stub container-only / unavailable imports so `import main` succeeds ----
for _name in ("neoruntime_ipc_sdk", "flask_sock"):
    if _name not in sys.modules:
        _stub = types.ModuleType(_name)
        _stub.__getattr__ = lambda attr: MagicMock()  # any name -> MagicMock
        sys.modules[_name] = _stub

import main  # noqa: E402  (must follow the stubs above)


# --- helpers ---------------------------------------------------------------

def _minimal_showcase() -> "main.ModelShowcase":
    """A ModelShowcase with only the fields _discover_models() touches.

    Built via __new__ to bypass the heavy __init__ (media clients, threads).
    """
    sc = main.ModelShowcase.__new__(main.ModelShowcase)
    sc.current_model = ""
    sc.current_model_type = ""
    sc._current_model_info = None
    sc.infer_client = MagicMock()
    return sc


# --- _model_path unit tests ------------------------------------------------

def test_model_path_prefers_host_copy_over_bundled(tmp_path, monkeypatch):
    # Arrange
    host_root = tmp_path / "host"
    bundled_root = tmp_path / "bundled"
    host_file = host_root / "detection" / "det.hef"
    host_file.parent.mkdir(parents=True)
    host_file.write_bytes(b"host-copy")
    bundled_root.mkdir()
    (bundled_root / "det.hef").write_bytes(b"bundled-copy")
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))

    # Act
    resolved = main._model_path("detection", "det.hef")

    # Assert — device that pre-provisioned its store keeps using its own copy
    assert resolved == str(host_file)


def test_model_path_falls_back_to_bundled_when_host_missing(tmp_path, monkeypatch):
    # Arrange — fresh device: no host store entry, bundled copy in the image
    host_root = tmp_path / "host"
    host_root.mkdir()
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    bundled_file = bundled_root / "det.hef"
    bundled_file.write_bytes(b"bundled-copy")
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))

    # Act
    resolved = main._model_path("detection", "det.hef")

    # Assert — flat layout: filename directly under the bundled root
    assert resolved == str(bundled_file)


def test_model_path_bundled_layout_has_no_category_subdir(tmp_path, monkeypatch):
    # Arrange — the Dockerfile COPYs models/ flat; category must not leak in
    host_root = tmp_path / "host"
    host_root.mkdir()
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))

    # Act
    resolved = main._model_path("keypoint", "face_landmarks_lite.hef")

    # Assert
    assert resolved == str(bundled_root / "face_landmarks_lite.hef")


def test_model_path_returns_bundled_path_even_when_file_absent(tmp_path, monkeypatch):
    # Arrange — resolution itself never fails; availability is gated later by
    # _is_available (vit_large / unbundled models rely on this contract)
    host_root = tmp_path / "host"
    host_root.mkdir()
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))

    # Act
    resolved = main._model_path("classification", "vit_large.hef")

    # Assert — deterministic bundled path even though neither copy exists
    assert resolved == str(bundled_root / "vit_large.hef")


# --- _discover_models / _is_available tests --------------------------------

def _catalog_entry(entry_id: str, mtype: str, path=None, stages=None) -> dict:
    entry = {"id": entry_id, "type": mtype, "name": entry_id}
    if path is not None:
        entry["path"] = str(path)
    if stages is not None:
        entry["stages"] = [
            {"id": s_id, "path": str(s_path)} for s_id, s_path in stages
        ]
    return entry


def test_discover_models_marks_bundled_only_entries_available(tmp_path, monkeypatch):
    # Arrange — fresh device: only the bundled copies exist
    host_root = tmp_path / "host"
    host_root.mkdir()
    bundled_root = tmp_path / "bundled"
    bundled_file = bundled_root / "det.hef"
    bundled_root.mkdir()
    bundled_file.write_bytes(b"bundled")
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))
    monkeypatch.setattr(main, "MODEL_CATALOG", [
        _catalog_entry("bundled_det", "detection", path=bundled_file),
    ])
    sc = _minimal_showcase()

    # Act
    sc._discover_models()

    # Assert
    assert [m["id"] for m in main.AVAILABLE_MODELS] == ["bundled_det"]
    assert sc.current_model == "bundled_det"
    assert sc._current_model_info["id"] == "bundled_det"


def test_discover_models_hides_entry_missing_from_both_stores(tmp_path, monkeypatch):
    # Arrange — vit_large-style: nowhere on disk
    host_root = tmp_path / "host"
    host_root.mkdir()
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))
    missing = tmp_path / "nowhere.hef"
    monkeypatch.setattr(main, "MODEL_CATALOG", [
        _catalog_entry("ghost", "classification", path=missing),
    ])
    sc = _minimal_showcase()

    # Act
    sc._discover_models()

    # Assert
    assert main.AVAILABLE_MODELS == []
    assert sc.current_model == ""


def test_discover_models_pipeline_requires_every_stage_present(tmp_path, monkeypatch):
    # Arrange — LPR-style pipeline: det stage bundled, OCR stage missing
    host_root = tmp_path / "host"
    host_root.mkdir()
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    det = bundled_root / "det.hef"
    det.write_bytes(b"det")
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))
    monkeypatch.setattr(main, "MODEL_CATALOG", [
        _catalog_entry("lpr_pipeline", "pipeline_lpr",
                       stages=[("plate_det", det), ("plate_ocr", tmp_path / "missing.hef")]),
    ])
    sc = _minimal_showcase()

    # Act
    sc._discover_models()

    # Assert — one missing stage hides the whole pipeline
    assert main.AVAILABLE_MODELS == []


def test_discover_models_genai_always_available_without_file(tmp_path, monkeypatch):
    # Arrange — Qwen3-VL ships with the device runtime; no path check applies
    host_root = tmp_path / "host"
    host_root.mkdir()
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))
    monkeypatch.setattr(main, "MODEL_CATALOG", [
        _catalog_entry("qwen_vl", "genai"),  # no path key at all
    ])
    sc = _minimal_showcase()

    # Act
    sc._discover_models()

    # Assert
    assert [m["id"] for m in main.AVAILABLE_MODELS] == ["qwen_vl"]


def test_discover_models_host_copy_beats_bundled_in_real_catalog(tmp_path, monkeypatch):
    # Arrange — full real catalog against a host store that provisioned one
    # model: that entry must resolve via the host path even though a bundled
    # copy also exists.
    host_root = tmp_path / "host"
    host_det = host_root / "detection" / "hailo_yolov8n_384_640.hef"
    host_det.parent.mkdir(parents=True)
    host_det.write_bytes(b"host")
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    (bundled_root / "hailo_yolov8n_384_640.hef").write_bytes(b"bundled")
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))
    # Re-resolve the baked catalog paths against the patched roots, exactly
    # as import-time evaluation would on that device.
    monkeypatch.setattr(main, "MODEL_CATALOG", [
        {**entry, "path": main._model_path("detection", "hailo_yolov8n_384_640.hef")}
        if entry["id"] == "yolov8n_detection" and "path" in entry else entry
        for entry in main.MODEL_CATALOG
    ])
    sc = _minimal_showcase()

    # Act
    sc._discover_models()

    # Assert
    entry = next(m for m in main.AVAILABLE_MODELS if m["id"] == "yolov8n_detection")
    assert entry["path"] == str(host_det)
