"""Guard the SDK compatibility contract: the repo sdk.lock stays within
the versions this demo's contracts have been verified against.

Failing here is deliberate friction: bump the SDK → re-verify the demo
(offline suite + device e2e) → then extend SUPPORTED_SDK_VERSIONS.
"""

import importlib.metadata
from pathlib import Path

import pytest

from sdk_helpers import SUPPORTED_SDK_VERSIONS

REPO_ROOT = Path(__file__).resolve().parents[3]  # tests → showcase → showcases → repo
LOCK = REPO_ROOT / "sdk.lock"


def _major_minor(version):
    return ".".join(version.split(".")[:2])


def test_sdk_lock_within_supported_versions():
    pinned = LOCK.read_text(encoding="utf-8").strip()
    mm = _major_minor(pinned)
    assert mm in SUPPORTED_SDK_VERSIONS, (
        f"sdk.lock pins {pinned} but the demo contracts target "
        f"{SUPPORTED_SDK_VERSIONS}. Re-verify against the new SDK "
        f"(offline suite + device e2e), then add {mm!r} to "
        f"SUPPORTED_SDK_VERSIONS in sdk_helpers.py — or roll sdk.lock back."
    )


def test_installed_real_sdk_matches_supported_versions():
    """Reads dist metadata, so it works even though conftest installs a
    neoruntime_ipc_sdk stub into sys.modules first."""
    try:
        installed = importlib.metadata.version("neoruntime-ipc-sdk")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("neoruntime-ipc-sdk not installed in this env")
    assert _major_minor(installed) in SUPPORTED_SDK_VERSIONS, (
        f"installed neoruntime-ipc-sdk {installed} is outside the demo's "
        f"verified set {SUPPORTED_SDK_VERSIONS}"
    )
