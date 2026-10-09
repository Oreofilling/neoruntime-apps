"""Offline tests for 02-first-frame (examples ladder rung 2).

SDK stub keeps the import alive on machines without the SDK; the
FdMediaClient the app constructs is replaced per-test with a fake, so
the grab/write/exit-code paths run without any device.
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
        "ladder02_app", APP_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ladder02_app"] = module
    spec.loader.exec_module(module)
    return module


class FakeFrame:
    """Minimal Frame: context manager + JPEG encoding."""

    def __init__(self, jpeg=b"fake-jpeg-bytes"):
        self._jpeg = jpeg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def to_jpeg_bytes(self, quality=85):
        return self._jpeg


class FakeMedia:
    """FdMediaClient stand-in with scripted behavior."""

    def __init__(self, info="default", frame="default", streams=None):
        self.info = (SimpleNamespace(width=1920, height=1080, fps=25,
                                     format="NV12")
                     if info == "default" else info)
        self.frame = FakeFrame() if frame == "default" else frame
        self.streams = streams if streams is not None else ["main", "sub"]
        self.closed = False

    def get_stream_info(self, _stream_id):
        return self.info

    def list_streams(self):
        return self.streams

    def get_frame(self, _stream_id, timeout_ms=5000):
        return self.frame

    def close(self):
        self.closed = True


@pytest.fixture
def media_factory(app_mod, monkeypatch):
    """Replace the module's FdMediaClient; installs the given fake."""
    def install(media):
        monkeypatch.setattr(app_mod, "FdMediaClient", lambda: media)

    return install


# Exposed as fixtures (not "from conftest import ...") so several rungs
# can run in ONE pytest invocation: a single conftest module name is
# shared across dirs, but fixtures stay per-directory.
@pytest.fixture
def fake_frame():
    return FakeFrame


@pytest.fixture
def fake_media():
    return FakeMedia
