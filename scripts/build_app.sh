#!/bin/bash
# Unified app build script for AIPC platform
# Usage: ./scripts/build_app.sh <app-dir> [--arch arm64|amd64] [--output ./dist]
#
# Automates: SDK install → docker build → save image → package .nrt → cleanup
#
# The SDK is installed from PyPI: neoruntime-ipc-sdk==$SDK_VERSION, where the
# version resolves via scripts/resolve_sdk_version.sh — sdk.lock by default,
# AIPC_SDK_VERSION to override (or =latest to float).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
ARCH="arm64"
OUTPUT_DIR=""
APP_DIR=""

usage() {
    echo "Usage: $0 <app-directory> [--arch arm64|amd64] [--output <dir>]"
    echo ""
    echo "  app-directory   Path to app directory containing Dockerfile and app.yaml"
    echo "  --arch          Target architecture (default: arm64)"
    echo "  --output        Output directory for .nrt package (default: app directory)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch) ARCH="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --help|-h) usage ;;
        -*) echo "Unknown option: $1"; usage ;;
        *)
            if [ -z "$APP_DIR" ]; then APP_DIR="$1"; shift
            else echo "Unknown argument: $1"; usage; fi ;;
    esac
done

if [ -z "$APP_DIR" ] || [ ! -d "$APP_DIR" ]; then
    echo "Error: App directory not found: ${APP_DIR:-<none>}"
    exit 1
fi

APP_DIR="$(cd "$APP_DIR" && pwd)"
APP_NAME="$(basename "$APP_DIR")"

# SDK version: sdk.lock by default; AIPC_SDK_VERSION to override.
SDK_VERSION="$("$SCRIPT_DIR/resolve_sdk_version.sh")"

APP_YAML="$APP_DIR/app.yaml"
if [ ! -f "$APP_YAML" ]; then
    echo "Error: app.yaml not found in $APP_DIR"
    exit 1
fi

VERSION=$(grep -m1 '^\s*version:' "$APP_YAML" | awk '{print $2}' | tr -d '"')
VERSION="${VERSION:-1.0.0}"

IMAGE_TAG=$(grep -m1 '^\s*image:' "$APP_YAML" | awk '{print $2}' | tr -d '"')
IMAGE_TAG="${IMAGE_TAG:-aipc/${APP_NAME}:${VERSION}}"

OUTPUT_DIR="${OUTPUT_DIR:-$APP_DIR}"
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Building ${APP_NAME}:${VERSION} for ${ARCH}"
echo "  Image: ${IMAGE_TAG}"
echo "  SDK: neoruntime-ipc-sdk ${SDK_VERSION} ($(if [ -n "${AIPC_SDK_VERSION:-}" ]; then echo AIPC_SDK_VERSION; else echo sdk.lock; fi))"
echo "============================================"

# Stage models (zoo fetch, sha256-pinned via models.manifest; no-op otherwise)
if [ -f "$APP_DIR/models.manifest" ]; then
    echo "Staging models..."
    "$SCRIPT_DIR/fetch_models.sh" "$APP_DIR"
fi

# Build
echo "Building Docker image..."
if [ "$ARCH" = "arm64" ]; then
    docker buildx build --platform linux/arm64 --load \
        --build-arg SDK_VERSION="$SDK_VERSION" -t "$IMAGE_TAG" "$APP_DIR"
else
    docker build --build-arg SDK_VERSION="$SDK_VERSION" -t "$IMAGE_TAG" "$APP_DIR"
fi

# Export
echo "Exporting image..."
IMAGE_TAR="$APP_DIR/image.tar"
docker save "$IMAGE_TAG" -o "$IMAGE_TAR"

# Package (.nrt = tar.gz bundle with the same layout as showcase bundles:
# <app>-<version>-<arch>/{app.yaml, image.tar, SHA256SUMS} — one package
# format for web import and CLI install).
echo "Creating .nrt package..."
PACKAGE_DIR="${APP_NAME}-${VERSION}-${ARCH}"
NRT_PACKAGE="$OUTPUT_DIR/${PACKAGE_DIR}.nrt"
STAGING_ROOT="$OUTPUT_DIR/.tmp-nrt-staging"
STAGING="$STAGING_ROOT/$PACKAGE_DIR"
rm -rf "$STAGING_ROOT"
mkdir -p "$STAGING"
cp "$APP_YAML" "$STAGING/app.yaml"
mv "$IMAGE_TAR" "$STAGING/image.tar"
(cd "$STAGING" && sha256sum * > SHA256SUMS)
tar -C "$STAGING_ROOT" -czf "$NRT_PACKAGE" "$PACKAGE_DIR"
rm -rf "$STAGING_ROOT"

echo ""
echo "============================================"
echo "  Build complete!"
echo "  Package: $NRT_PACKAGE"
echo "  Size: $(du -h "$NRT_PACKAGE" | cut -f1)"
echo "============================================"
echo ""
echo "To install on device:"
echo "  tar xzf ${PACKAGE_DIR}.nrt && aipc-cli app install <app-id> ${PACKAGE_DIR}/app.yaml ${PACKAGE_DIR}/image.tar"
