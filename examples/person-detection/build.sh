#!/bin/bash
# Build script for Person Detection Application
# Uses the shared build_app.sh helper
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${1:-arm64}"
"${SCRIPT_DIR}/../../scripts/build_app.sh" "$SCRIPT_DIR" --arch "$ARCH"
