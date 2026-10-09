"""Offline tests for 04-register-model (examples ladder rung 4).

SDK stub keeps the import alive on machines without the SDK. The
clients the app constructs (InferenceClient, FdMediaClient) and the
pipeline classes it uses are replaced per-test with fakes that record
call order — the lifecycle sequence is the thing under test.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

APP_DIR = Path(__file__).resolve().parent.parent

if "neoruntime_ipc_sdk" not in sys.modules:
    _stub = types.ModuleType("neoruntime_ipc_sdk")
    _stub.__getattr__ = lambda attr: MagicMock(
        name=f"neoruntime_ipc_sdk.{attr}")
    sys.modules["neoruntime_ipc_sdk"] = _stub


@pytest.fixture(scope="session")
def app_mod():
    spec = importlib.util.spec_from_file_location(
        "ladder04_app", APP_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ladder04_app"] = module
    spec.loader.exec_module(module)
    return module


class FakeInference:
    """Records register/unregister calls and current model ids."""

    def __init__(self, models=()):
        self.models = list(models)
        self.calls = []

    def list_models(self):
        self.calls.append(("list", tuple(self.models)))
        return [SimpleNamespace(model_id=m) for m in self.models]

    def register_model(self, model_path, model_id, model_type,
                       owner_id, model_variant):
        self.calls.append(("register", model_id, model_path, model_type))
        self.models.append(model_id)

    def unregister_model(self, model_id):
        self.calls.append(("unregister", model_id))
        self.models.remove(model_id)

    def close(self):
        self.calls.append(("close",))


class FakeFrame:
    def __init__(self):
        self.sequence = 42

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def to_array(self):
        raise NotImplementedError("not needed on the pipeline fake")


class FakeMedia:
    def __init__(self, frame="default"):
        self.frame = FakeFrame() if frame == "default" else frame
        self.closed = False

    def get_frame(self, _stream_id, timeout_ms=5000):
        return self.frame

    def close(self):
        self.closed = True


@pytest.fixture
def wire(app_mod, monkeypatch, tmp_path):
    """Install fakes plus a bundled model file; returns the fakes."""
    bundled = tmp_path / "yolov8n.hef"
    bundled.write_bytes(b"hef-bytes")
    monkeypatch.setattr(app_mod, "CANDIDATE_PATHS", [str(bundled)])

    inference = FakeInference()
    media = FakeMedia()
    pipeline = MagicMock()
    pipeline.run.return_value = SimpleNamespace(
        objects=[SimpleNamespace(label="person", score=0.9)],
        release=lambda: None)

    monkeypatch.setattr(app_mod, "InferenceClient", lambda: inference)
    monkeypatch.setattr(app_mod, "FdMediaClient", lambda: media)
    monkeypatch.setattr(app_mod, "InferencePipeline",
                        lambda **_kwargs: pipeline)
    monkeypatch.setattr(app_mod, "Preprocessor",
                        SimpleNamespace(from_model=lambda *_a, **_k: object()))
    return SimpleNamespace(inference=inference, media=media,
                           pipeline=pipeline, bundled=bundled)
