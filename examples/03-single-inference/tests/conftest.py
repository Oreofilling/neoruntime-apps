"""Offline tests for 03-single-inference (examples ladder rung 3).

SDK stub keeps the import alive on machines without the SDK; the
InferenceClient the app constructs is replaced per-test with a fake
whose subscribe() yields one scripted result.
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
        "ladder03_app", APP_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ladder03_app"] = module
    spec.loader.exec_module(module)
    return module


def make_object(label, score, x=0.1, y=0.2, width=0.3, height=0.4):
    return SimpleNamespace(
        label=label, score=score,
        bbox=SimpleNamespace(x=x, y=y, width=width, height=height))


def make_result(*objects):
    return SimpleNamespace(objects=list(objects))


class FakeInference:
    """InferenceClient stand-in with one scripted subscribe result."""

    def __init__(self, models=("yolov8n", "other"), result=None,
                 error=None):
        self.models = list(models)
        self.result = result if result is not None else make_result()
        self.error = error
        self.closed = False
        self.subscribed_with = None

    def list_models(self):
        return [SimpleNamespace(model_id=m) for m in self.models]

    def subscribe(self, stream, model, fps):
        self.subscribed_with = (stream, model, fps)
        if self.error is not None:
            raise self.error
        yield (118, self.result)

    def close(self):
        self.closed = True


@pytest.fixture
def inference_factory(app_mod, monkeypatch):
    """Replace the module's InferenceClient; installs the given fake."""
    def install(inference):
        monkeypatch.setattr(app_mod, "InferenceClient", lambda: inference)

    return install


# Exposed as fixtures (not "from conftest import ...") so several rungs
# can run in ONE pytest invocation: a single conftest module name is
# shared across dirs, but fixtures stay per-directory.
@pytest.fixture
def fake_inference():
    return FakeInference


@pytest.fixture
def object_maker():
    return make_object


@pytest.fixture
def result_maker():
    return make_result
