#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARCH="${1:-arm64}"

exec "$PROJECT_ROOT/scripts/build_showcase_artifacts.sh" \
    --arch "$ARCH" \
    --output "$PROJECT_ROOT/dist/showcases" \
    "$(basename "$SCRIPT_DIR")"
