#!/bin/bash
#
# Build Face Cascade Application Image (ARM64)
#
# Output: face-cascade-1.0.0.tar (OCI image tarball for aipc-cli)
#
# Usage:
#   ./build.sh              # Build for ARM64 (default, for Hailo-15)
#   ./build.sh --x86        # Build for x86_64 (for local testing)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="face-cascade"
APP_VERSION="1.0.0"
OUTPUT_DIR="${SCRIPT_DIR}/dist"
SDK_DIR="${SCRIPT_DIR}/../../sdk/python/hailo_ipc_sdk"

# Default: ARM64 for Hailo-15
PLATFORM="linux/arm64"
ARCH_SUFFIX="arm64"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[build]${NC} $*"; }
warn() { echo -e "${YELLOW}[build]${NC} $*"; }
err()  { echo -e "${RED}[build]${NC} $*" >&2; }

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --x86|--x86_64|--amd64)
            PLATFORM="linux/amd64"
            ARCH_SUFFIX="x86"
            shift
            ;;
        --arm64|--aarch64)
            PLATFORM="linux/arm64"
            ARCH_SUFFIX="arm64"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--x86|--arm64]"
            echo ""
            echo "Options:"
            echo "  --arm64    Build for ARM64 (default, for Hailo-15)"
            echo "  --x86      Build for x86_64 (for local testing)"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Check Docker
if ! command -v docker &>/dev/null; then
    err "Docker is required to build the application image"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

log "============================================"
log "  Building ${APP_NAME}:${APP_VERSION}"
log "  Platform: ${PLATFORM}"
log "============================================"

# Copy SDK to build context
if [[ -d "$SDK_DIR" ]]; then
    log "Copying AIPC SDK to build context..."
    rm -rf "$SCRIPT_DIR/hailo_ipc_sdk"
    cp -r "$SDK_DIR" "$SCRIPT_DIR/hailo_ipc_sdk"
else
    err "SDK not found at: $SDK_DIR"
    exit 1
fi

# Setup buildx for cross-platform builds
TARBALL="${OUTPUT_DIR}/${APP_NAME}-${APP_VERSION}.tar"

if [[ "$PLATFORM" == "linux/arm64" ]] && [[ "$(uname -m)" != "aarch64" ]]; then
    # Cross-compile: use buildx
    log "Cross-compiling for ARM64 using buildx..."

    # Ensure buildx builder exists
    docker buildx create --name multiarch --use 2>/dev/null || docker buildx use multiarch 2>/dev/null || true

    # Build and export
    docker buildx build \
        --platform "${PLATFORM}" \
        -t "${APP_NAME}:${APP_VERSION}" \
        --output "type=docker,dest=${TARBALL}" \
        "$SCRIPT_DIR"
else
    # Native build
    log "Building Docker image..."
    docker build -t "${APP_NAME}:${APP_VERSION}" "$SCRIPT_DIR"

    log "Saving image to: $TARBALL"
    docker save -o "$TARBALL" "${APP_NAME}:${APP_VERSION}"
fi

# Clean up SDK copy
rm -rf "$SCRIPT_DIR/hailo_ipc_sdk"

# Summary
TARBALL_SIZE=$(du -h "$TARBALL" | cut -f1)
log ""
log "============================================"
log "  Build Complete"
log "============================================"
log "  Image:    ${APP_NAME}:${APP_VERSION}"
log "  Tarball:  $TARBALL"
log "  Size:     $TARBALL_SIZE"
log ""
log "  Deploy to device:"
log "    scp $TARBALL root@<device>:/opt/aipc/images/"
log "    scp ${SCRIPT_DIR}/app.yaml root@<device>:/opt/aipc/images/"
log ""
log "  Install on device:"
log "    aipc-cli app install app.yaml ${APP_NAME}-${APP_VERSION}.tar"
log "    aipc-cli app start face_cascade"
log "============================================"
