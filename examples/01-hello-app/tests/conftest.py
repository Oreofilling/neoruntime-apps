"""Offline tests for 01-hello-app (examples ladder rung 1).

The app imports neoruntime_ipc_sdk only for Config, which is env-based
and socket-free. On machines without the SDK installed (dev laptops),
a stub module keeps the import alive; the lifecycle behavior under test
needs no SDK at all.
"""

import importlib.util
import sys
import types
from pathlib import Path
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
    """Load the rung's app.py under a unique module name."""
    spec = importlib.util.spec_from_file_location(
        "ladder01_app", APP_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ladder01_app"] = module
    spec.loader.exec_module(module)
    return module
