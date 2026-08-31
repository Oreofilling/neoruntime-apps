#!/usr/bin/env bash
# Minimal in-image smoke test: proves a freshly built image can start Python,
# import the pinned neoruntime-ipc-sdk, byte-compile the app sources, and
# import the entry module behind CMD — without a device or camera.
# Catches the common "build succeeded, app broken" cases: missing deps,
# bad imports, syntax errors, wrong file layout.
#
# Usage: scripts/smoke_image.sh <image-tag>
#        scripts/smoke_image.sh <app-dir>   # tag derived from app.yaml
set -euo pipefail

IMAGE="${1:-}"
if [ -z "$IMAGE" ]; then
    echo "Usage: $0 <image-tag | app-dir>" >&2
    exit 1
fi

# App directory → derive the tag the same way build_app.sh does.
if [ -d "$IMAGE" ]; then
    APP_YAML="$IMAGE/app.yaml"
    if [ ! -f "$APP_YAML" ]; then
        echo "Error: $APP_YAML not found" >&2
        exit 1
    fi
    APP_DIR="$(cd "$IMAGE" && pwd)"
    IMAGE="$(grep -m1 '^\s*image:' "$APP_YAML" | awk '{print $2}' | tr -d '"')"
    if [ -z "$IMAGE" ]; then
        VERSION="$(grep -m1 '^\s*version:' "$APP_YAML" | awk '{print $2}' | tr -d '"')"
        IMAGE="neoruntime/$(basename "$APP_DIR"):${VERSION:-1.0.0}"
    fi
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Error: image not found locally: $IMAGE" >&2
    exit 1
fi

echo "==> smoke [$IMAGE] SDK import"
docker run --rm --entrypoint python3 "$IMAGE" -c \
    'import neoruntime_ipc_sdk, importlib.metadata as m; \
     print("neoruntime-ipc-sdk", m.version("neoruntime-ipc-sdk"))'

echo "==> smoke [$IMAGE] byte-compile /app"
docker run --rm --entrypoint python3 "$IMAGE" -m compileall -q /app

# Import the entry module: last CMD word ending in .py (e.g. main.py → main).
MODULE="$(docker inspect -f '{{json .Config.Cmd}}' "$IMAGE" | python3 -c '
import json, os, sys
cmd = json.load(sys.stdin)
py_args = [a for a in cmd if a.endswith(".py")]
print(os.path.splitext(os.path.basename(py_args[-1]))[0] if py_args else "")
')"
if [ -n "$MODULE" ]; then
    echo "==> smoke [$IMAGE] import entry module: $MODULE"
    docker run --rm --entrypoint python3 "$IMAGE" -c "import $MODULE"
else
    echo "==> smoke [$IMAGE] no .py in CMD — skipped entry-module import"
fi

echo "==> smoke [$IMAGE] OK"
