"""Offline tests for 05-live-detection (examples ladder rung 5).

SDK stub keeps the import alive on machines without the SDK. The loop's
SDK touchpoints are replaced with fakes; the pieces under test are the
one-slot buffer, the NV12/RGB frame wrapping, the MJPEG part framing,
and one full pass of the loop body over scripted frames.
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
        "ladder05_app", APP_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ladder05_app"] = module
    spec.loader.exec_module(module)
    return module


class FakeArray:
    """ndarray stand-in: only ndim/shape are exercised on this path."""

    def __init__(self, ndim, shape):
        self.ndim = ndim
        self.shape = shape


class FakeSdkFrame:
    """Frame(...) constructed by as_frame; only JPEG encoding used."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to_jpeg_bytes(self, quality=85):
        return b"jpeg-for-" + str(self.kwargs["format"]).encode()


class FakeMediaFrame:
    """A frame yielded by media.subscribe()."""

    def __init__(self, sequence):
        self.sequence = sequence
        self.array = FakeArray(3, (1080, 1920, 3))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def to_array(self):
        return self.array


class FakeMedia:
    """FdMediaClient stand-in yielding a fixed frame list."""

    def __init__(self, frames):
        self._frames = frames
        self.closed = False

    def subscribe(self, _stream_id):
        yield from self._frames

    def close(self):
        self.closed = True


@pytest.fixture
def wire(app_mod, monkeypatch):
    """Install loop fakes; returns a recorder for inspection."""
    frames = [FakeMediaFrame(10), FakeMediaFrame(11)]
    drawn = []
    released = []

    def fake_draw(array, objects):
        drawn.append((array, list(objects)))
        return array

    pipeline = MagicMock()
    pipeline.run.side_effect = lambda _frame: SimpleNamespace(
        objects=[SimpleNamespace(label="person", score=0.9),
                 SimpleNamespace(label="cat", score=0.1)],
        release=lambda: released.append(1))

    inference = MagicMock()
    monkeypatch.setattr(app_mod, "InferenceClient", lambda: inference)
    monkeypatch.setattr(app_mod, "Preprocessor",
                        SimpleNamespace(from_model=lambda *_a, **_k: object()))
    monkeypatch.setattr(app_mod, "InferencePipeline",
                        lambda **_kwargs: pipeline)
    monkeypatch.setattr(app_mod, "FdMediaClient",
                        lambda: FakeMedia(frames))
    monkeypatch.setattr(app_mod, "draw_detections", fake_draw)

    sdk_frames = []
    real_frame = FakeSdkFrame

    def frame_ctor(**kwargs):
        frame = real_frame(**kwargs)
        sdk_frames.append(frame)
        return frame

    monkeypatch.setattr(app_mod, "Frame", frame_ctor)
    return SimpleNamespace(frames=frames, drawn=drawn, released=released,
                           pipeline=pipeline, sdk_frames=sdk_frames)
