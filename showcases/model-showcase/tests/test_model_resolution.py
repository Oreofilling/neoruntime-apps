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


# --- missing-list / acquisition-hint tests ----------------------------------

def test_missing_models_lists_unavailable_entry_with_hints(tmp_path, monkeypatch):
    # Arrange — vit_large-style entry: file nowhere on the device, but the
    # catalog carries copy-paste acquisition hints for the UI card.
    host_root = tmp_path / "host"
    host_root.mkdir()
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))
    acq = {
        "url": "https://example.com/vit_large.hef",
        "sha256": "ab" * 32,
        "size_bytes": 280158208,
        "target": "/data/aipc/models/classification/vit_large.hef",
    }
    monkeypatch.setattr(main, "MODEL_CATALOG", [
        {**_catalog_entry("vit", "classification",
                          path=bundled_root / "vit_large.hef"),
         "acquisition": acq},
    ])

    # Act
    missing = main._missing_models()

    # Assert — surfaced with hints instead of silently vanishing
    assert [m["id"] for m in missing] == ["vit"]
    assert missing[0]["acquisition"] == acq
    assert "postprocess_json" not in missing[0]


def test_missing_models_empty_when_everything_available(tmp_path, monkeypatch):
    # Arrange — bundled copy exists, so nothing is missing
    host_root = tmp_path / "host"
    host_root.mkdir()
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    (bundled_root / "det.hef").write_bytes(b"bundled")
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))
    monkeypatch.setattr(main, "MODEL_CATALOG", [
        _catalog_entry("det", "detection", path=bundled_root / "det.hef"),
    ])

    # Act / Assert
    assert main._missing_models() == []


def test_discover_models_genai_missing_file_carries_warning(tmp_path, monkeypatch):
    # Arrange — Qwen3-VL-style: weight file absent, entry must STAY listed
    # (interactive chat entry) but carry an early warning for the UI badge.
    host_root = tmp_path / "host"
    host_root.mkdir()
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))
    monkeypatch.setattr(main, "MODEL_CATALOG", [
        _catalog_entry("qwen", "genai", path=bundled_root / "Qwen3-VL-2B-Instruct.hef"),
    ])
    sc = _minimal_showcase()

    # Act
    sc._discover_models()

    # Assert
    assert [m["id"] for m in main.AVAILABLE_MODELS] == ["qwen"]
    assert main.AVAILABLE_MODELS[0].get("warning")


def test_discover_models_genai_provisioned_has_no_warning(tmp_path, monkeypatch):
    # Arrange — device provisioned the weight file in its host store
    host_root = tmp_path / "host"
    genai_file = host_root / "genai" / "Qwen3-VL-2B-Instruct.hef"
    genai_file.parent.mkdir(parents=True)
    genai_file.write_bytes(b"weights")
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))
    monkeypatch.setattr(main, "MODEL_CATALOG", [
        _catalog_entry("qwen", "genai", path=genai_file),
    ])
    sc = _minimal_showcase()

    # Act
    sc._discover_models()

    # Assert — listed, no warning
    assert [m["id"] for m in main.AVAILABLE_MODELS] == ["qwen"]
    assert "warning" not in main.AVAILABLE_MODELS[0]


# --- refresh (POST /api/models/refresh) tests -------------------------------

def test_refresh_model_paths_detects_newly_placed_host_file(tmp_path, monkeypatch):
    # Arrange — import-time resolution landed on the bundled fallback because
    # the host file didn't exist yet; the operator provisions it afterwards.
    host_root = tmp_path / "host"
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))
    monkeypatch.setattr(main, "MODEL_CATALOG", [
        {**_catalog_entry("vit", "classification",
                          path=main._model_path("classification", "vit_large.hef")),
         "category": "classification"},
    ])
    sc = _minimal_showcase()
    sc._discover_models()
    assert main.AVAILABLE_MODELS == []  # nothing runnable pre-provisioning
    assert [m["id"] for m in main._missing_models()] == ["vit"]

    # Act — operator drops the file into the host store, UI hits refresh
    host_file = host_root / "classification" / "vit_large.hef"
    host_file.parent.mkdir(parents=True)
    host_file.write_bytes(b"provisioned")
    added = sc.refresh_model_paths()

    # Assert — entry became available via the host path, no restart needed
    assert added == ["vit"]
    assert [m["id"] for m in main.AVAILABLE_MODELS] == ["vit"]
    assert main.AVAILABLE_MODELS[0]["path"] == str(host_file)
    assert main._missing_models() == []


def test_refresh_model_paths_reports_nothing_new_when_still_missing(tmp_path, monkeypatch):
    # Arrange — refresh pressed but the operator hasn't actually placed anything
    host_root = tmp_path / "host"
    host_root.mkdir()
    bundled_root = tmp_path / "bundled"
    bundled_root.mkdir()
    monkeypatch.setattr(main, "_MODEL_ROOT", str(host_root))
    monkeypatch.setattr(main, "_BUNDLED_MODEL_ROOT", str(bundled_root))
    monkeypatch.setattr(main, "MODEL_CATALOG", [
        {**_catalog_entry("vit", "classification",
                          path=main._model_path("classification", "vit_large.hef")),
         "category": "classification"},
    ])
    sc = _minimal_showcase()
    sc._discover_models()

    # Act / Assert — no false "added" report
    assert sc.refresh_model_paths() == []
    assert [m["id"] for m in main.AVAILABLE_MODELS] == []
