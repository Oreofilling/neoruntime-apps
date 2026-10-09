"""Offline tests for 06-event-bridge (examples ladder rung 6).

SDK stub keeps the import alive on machines without the SDK. The
pieces under test are the cooldown gate, the payload shaping, the
publish decision, and the tap loop over scripted bus events — no
device, no daemon.
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
        "ladder06_app", APP_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ladder06_app"] = module
    spec.loader.exec_module(module)
    return module


class FakeEventClient:
    """EventClient stand-in: records publishes, replays incoming."""

    def __init__(self, incoming=()):
        self.published = []  # (topic, payload)
        self.subscribed_topic = None
        self.closed = False
        self._incoming = list(incoming)

    def publish(self, topic, payload, **_kwargs):
        self.published.append((topic, dict(payload)))

    def subscribe(self, topic):
        self.subscribed_topic = topic
        yield from self._incoming

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
def events():
    return FakeEventClient()


@pytest.fixture
def bus_event():
    def _make(topic="app/event-bridge/detections",
              payload=None, source="someone-else"):
        return SimpleNamespace(topic=topic,
                               payload=payload or {}, source=source)
    return _make
