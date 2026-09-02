#!/bin/bash
# Build Visualizer Application Image

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="visualizer"
APP_VERSION="1.0.0"
OUTPUT_DIR="${SCRIPT_DIR}/dist"

# Default: ARM64 for Hailo-15
PLATFORM="linux/arm64"

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

log() { echo -e "${GREEN}[build]${NC} $*"; }

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --x86|--x86_64|--amd64) PLATFORM="linux/amd64"; shift ;;
        --arm64|--aarch64) PLATFORM="linux/arm64"; shift ;;
        *) shift ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

log "Building ${APP_NAME}:${APP_VERSION} for ${PLATFORM}"

# SDK comes from PyPI inside the Dockerfile (pinned via SDK_VERSION build-arg)
SDK_VERSION="$("${SCRIPT_DIR}/../../scripts/resolve_sdk_version.sh")"

TARBALL="${OUTPUT_DIR}/${APP_NAME}-${APP_VERSION}.tar"

if [[ "$PLATFORM" == "linux/arm64" ]] && [[ "$(uname -m)" != "aarch64" ]]; then
    log "Cross-compiling for ARM64..."
    docker buildx create --name multiarch --use 2>/dev/null || docker buildx use multiarch 2>/dev/null || true
    docker buildx build \
        --platform "${PLATFORM}" \
        --build-arg SDK_VERSION="${SDK_VERSION}" \
        -t "${APP_NAME}:${APP_VERSION}" \
        --output "type=docker,dest=${TARBALL}" \
        "$SCRIPT_DIR"
else
    docker build --build-arg SDK_VERSION="${SDK_VERSION}" -t "${APP_NAME}:${APP_VERSION}" "$SCRIPT_DIR"
    docker save -o "$TARBALL" "${APP_NAME}:${APP_VERSION}"
fi

log "Build complete: $TARBALL"
log "Size: $(du -h "$TARBALL" | cut -f1)"
