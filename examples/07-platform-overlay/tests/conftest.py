"""Offline tests for 07-platform-overlay (examples ladder rung 7).

SDK stub keeps the import alive on machines without the SDK. The
pieces under test are the annotate semantics (ttl passed along, empty
results still sent) and one full pass of the annotator loop over
scripted results — no device, no daemon.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

APP_DIR = Path(__file__).resolve().parent.parent

if "neoruntime_ipc_sdk" not in sys.modules:
    _stub = types.ModuleType("neoruntime_ipc_sdk")
    _stub.__getattr__ = lambda attr: SimpleNamespace(
        __name__=f"neoruntime_ipc_sdk.{attr}")
    sys.modules["neoruntime_ipc_sdk"] = _stub


@pytest.fixture(scope="session")
def app_mod():
    spec = importlib.util.spec_from_file_location(
        "ladder07_app", APP_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ladder07_app"] = module
    spec.loader.exec_module(module)
    return module


class FakeOverlayClient:
    """OverlayClient stand-in recording every call."""

    def __init__(self):
        self.enabled_with = None
        self.annotated = []  # (stream_id, object_count, ttl_ms)
        self.disabled = False
        self.closed = False

    def enable(self, **kwargs):
        self.enabled_with = kwargs

    def disable(self):
        self.disabled = True

    def annotate_result(self, stream_id, result, ttl_ms=None):
        self.annotated.append((stream_id, len(result.objects), ttl_ms))

    def close(self):
        self.closed = True


class FakeInferenceClient:
    """InferenceClient stand-in yielding scripted results."""

    def __init__(self, results):
        self._results = list(results)
        self.subscribe_kwargs = None
        self.closed = False

    def subscribe(self, **kwargs):
        self.subscribe_kwargs = kwargs
        yield from self._results

    def close(self):
        self.closed = True


@pytest.fixture
def make_object_():
    return lambda label, score: SimpleNamespace(label=label, score=score)


@pytest.fixture
def make_result_():
    """Exposed as fixtures (not "from conftest import ...") so several
    rungs can run in ONE pytest invocation."""
    return lambda *objects: SimpleNamespace(objects=list(objects))


@pytest.fixture
def overlay():
    return FakeOverlayClient()


@pytest.fixture
def wire(app_mod, monkeypatch, make_object_, make_result_):
    """Install annotator fakes; returns a recorder for inspection."""
    results = [
        (1, make_result_(make_object_("person", 0.9),
                         make_object_("dog", 0.4))),
        (2, make_result_()),  # empty: must still annotate (clearing)
    ]
    inference = FakeInferenceClient(results)
    overlay_client = FakeOverlayClient()
    monkeypatch.setattr(app_mod, "InferenceClient", lambda: inference)
    monkeypatch.setattr(app_mod, "OverlayClient", lambda: overlay_client)
    return SimpleNamespace(inference=inference, overlay=overlay_client)
