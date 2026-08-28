#!/usr/bin/env bash
# Resolve the neoruntime-ipc-sdk version to pin in Docker builds.
# Resolution order:
#   1. AIPC_SDK_VERSION=<version>  explicit pin (CI canaries, local overrides)
#   2. AIPC_SDK_VERSION=latest     newest release on PyPI
#   3. sdk.lock (repo root)        default — keeps clone→build reproducible
#                                  and works offline; bumped only after a
#                                  canary build via .github/workflows/sdk-probe.yml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="$SCRIPT_DIR/../sdk.lock"

pypi_latest() {
    curl -fsSL https://pypi.org/pypi/neoruntime-ipc-sdk/json 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])' \
        2>/dev/null || true
}

case "${AIPC_SDK_VERSION:-}" in
    latest)
        VERSION="$(pypi_latest)"
        if [ -z "$VERSION" ]; then
            echo "Error: AIPC_SDK_VERSION=latest but PyPI is unreachable" >&2
            exit 1
        fi
        echo "$VERSION"
        exit 0
        ;;
    ?*) echo "$AIPC_SDK_VERSION"; exit 0 ;;
esac

if [ ! -f "$LOCK_FILE" ]; then
    echo "Error: $LOCK_FILE not found — it pins the SDK for reproducible builds." >&2
    echo "Restore it from git, or set AIPC_SDK_VERSION to build anyway." >&2
    exit 1
fi

VERSION="$(tr -d '[:space:]' < "$LOCK_FILE")"
if ! [[ "$VERSION" =~ ^[0-9]+(\.[0-9]+)+ ]]; then
    echo "Error: sdk.lock contains '$VERSION' — expected a bare version like 0.5.0" >&2
    exit 1
fi
echo "$VERSION"
