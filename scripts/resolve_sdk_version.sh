#!/usr/bin/env bash
# Resolve the neoruntime-ipc-sdk version to pin in Docker builds:
# AIPC_SDK_VERSION if set, else the latest release published on PyPI.
set -euo pipefail

if [ -n "${AIPC_SDK_VERSION:-}" ]; then
    echo "$AIPC_SDK_VERSION"
    exit 0
fi

VERSION="$(curl -fsSL https://pypi.org/pypi/neoruntime-ipc-sdk/json 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])' \
    2>/dev/null || true)"

if [ -z "$VERSION" ]; then
    echo "Error: cannot resolve neoruntime-ipc-sdk version (PyPI unreachable); set AIPC_SDK_VERSION" >&2
    exit 1
fi
echo "$VERSION"
