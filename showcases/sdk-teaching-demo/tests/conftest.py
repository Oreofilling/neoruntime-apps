"""Shared fixtures for the sdk-teaching-demo test suite.

Pattern (borrowed from showcases/model-showcase): inject a stub module
for neoruntime_ipc_sdk so the app imports cleanly on machines without
the SDK installed (dev laptops, CI). All SDK constructions in the app
go through sdk_helpers._sdk(), which fixtures monkeypatch with a fake
namespace of small hand-rolled classes — real exceptions, functional
config/inference clients, MagicMocks for the rest.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SHOWCASE_DIR = Path(__file__).resolve().parent.parent

if "neoruntime_ipc_sdk" not in sys.modules:
    _stub = types.ModuleType("neoruntime_ipc_sdk")
    _stub.__getattr__ = lambda attr: MagicMock(
        name=f"neoruntime_ipc_sdk.{attr}")
    sys.modules["neoruntime_ipc_sdk"] = _stub

# teaching_app.py does `from sdk_helpers import ...` — make that resolve
# to the same file the tests load below.
sys.path.insert(0, str(SHOWCASE_DIR))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, SHOWCASE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeBoundingBox:
    def __init__(self, x, y, width, height):
        self.x, self.y = x, y
        self.width, self.height = width, height


class FakeDetectedObject:
    def __init__(self, label, score, class_id, bbox):
        self.label = label
        self.score = score
        self.class_id = class_id
        self.bbox = bbox


class FakeFrame:
    """as_frame() output: only to_jpeg_bytes is exercised in tests."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_jpeg_bytes(self, quality=None):
        return b"fake-jpeg"


def make_fake_sdk():
    """Functional stand-in namespace for neoruntime_ipc_sdk."""

    class HardwareUnavailable(Exception):
        pass

    class FakeAppClient:
        registered = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def register_web_url(self, url):
            FakeAppClient.registered.append(url)

    class FakeInferenceClient:
        def __init__(self):
            self.registered = []
            self.closed = False

        def list_models(self):
            return []

        def register_model(self, path, **kwargs):
            self.registered.append((path, kwargs))

        def close(self):
            self.closed = True

    policy_log = []

    def _refuse(op, *args, **kwargs):
        raise HardwareUnavailable(f"no hardware leg for {op!r}")

    fake = SimpleNamespace(
        Config=SimpleNamespace(get_app_id=lambda: "sdk-teaching-demo"),
        AppClient=FakeAppClient,
        HardwareUnavailable=HardwareUnavailable,
        InferenceClient=FakeInferenceClient,
        Preprocessor=SimpleNamespace(
            from_model=lambda client, model_id: SimpleNamespace()),
        InferencePipeline=lambda **kwargs: MagicMock(),
        FdMediaClient=MagicMock(),
        OverlayClient=MagicMock,
        EventClient=MagicMock,
        StreamPipeline=MagicMock(),
        OverlayConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        draw_detections=lambda image, boxes: image,
        get_default_router=lambda: SimpleNamespace(
            health=lambda: {
                "policy": policy_log[-1] if policy_log else "prefer_hardware",
                "ops": {},
                "recent_degradations": [],
            },
            run=_refuse,
        ),
        set_route_policy=lambda policy: policy_log.append(policy),
        Frame=FakeFrame,
        DetectedObject=FakeDetectedObject,
        BoundingBox=FakeBoundingBox,
        policy_log=policy_log,
    )
    return fake


@pytest.fixture(scope="session")
def helpers():
    return _load("sdk_helpers")


@pytest.fixture(scope="session")
def teaching(helpers):
    return _load("teaching_app")


@pytest.fixture(scope="session")
def main_mod(teaching):
    return _load("main")


@pytest.fixture
def fake_sdk():
    return make_fake_sdk()


@pytest.fixture
def app(monkeypatch, fake_sdk, helpers, teaching):
    """A TeachingApp wired to the fake SDK namespace."""
    monkeypatch.setattr(helpers, "_sdk", lambda: fake_sdk)
    monkeypatch.setattr(teaching, "_sdk", lambda: fake_sdk)
    # connect() must succeed offline: no host model store on dev machines
    monkeypatch.setattr(helpers, "resolve_model_path",
                        lambda: (Path("/tmp/dummy.hef"), "test"))
    instance = teaching.TeachingApp()
    instance.overlay = MagicMock()
    instance.events = MagicMock()
    instance.router = fake_sdk.get_default_router()
    return instance
