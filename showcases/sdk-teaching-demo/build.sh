#!/usr/bin/env bash
# Build the teaching app via the repo-wide showcase artifact builder
# (same delegation as model-showcase). CI runs this for every showcase
# root with a Dockerfile; locally: ./build.sh [arm64|amd64].
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARCH="${1:-arm64}"

exec "$PROJECT_ROOT/scripts/build_showcase_artifacts.sh" \
    --arch "$ARCH" \
    --output "$PROJECT_ROOT/dist/showcases" \
    "$(basename "$SCRIPT_DIR")"
