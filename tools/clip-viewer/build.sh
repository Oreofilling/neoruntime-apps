#!/bin/bash
# Build script for CLIP Zero-Shot Viewer Application
# Usage: ./build.sh [arm64|amd64]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="clip-viewer"
VERSION="2.1.0"
ARCH="${1:-arm64}"

echo "============================================"
echo "  Building ${APP_NAME}:${VERSION} for ${ARCH}"
echo "============================================"

# SDK comes from PyPI inside the Dockerfile (pinned via SDK_VERSION build-arg)
SDK_VERSION="$("${SCRIPT_DIR}/../../scripts/resolve_sdk_version.sh")"

# Build Docker image
echo "Building Docker image..."
if [ "$ARCH" = "arm64" ]; then
    docker buildx build --platform linux/arm64 --load --build-arg SDK_VERSION="${SDK_VERSION}" -t "aipc/${APP_NAME}:${VERSION}" .
else
    docker build --build-arg SDK_VERSION="${SDK_VERSION}" -t "aipc/${APP_NAME}:${VERSION}" .
fi

# Export image
echo "Exporting image..."
docker save "aipc/${APP_NAME}:${VERSION}" -o image.tar

# Create .nrt package (tar.gz, same layout as build_app.sh bundles)
echo "Creating .nrt package..."
PACKAGE_DIR="${APP_NAME}-${VERSION}-${ARCH}"
NRT_PACKAGE="${PACKAGE_DIR}.nrt"
rm -rf .tmp-nrt-staging "${NRT_PACKAGE}"
mkdir -p ".tmp-nrt-staging/${PACKAGE_DIR}"
cp app.yaml ".tmp-nrt-staging/${PACKAGE_DIR}/app.yaml"
mv image.tar ".tmp-nrt-staging/${PACKAGE_DIR}/image.tar"
(cd ".tmp-nrt-staging/${PACKAGE_DIR}" && sha256sum * > SHA256SUMS)
tar -C .tmp-nrt-staging -czf "${NRT_PACKAGE}" "$PACKAGE_DIR"
rm -rf .tmp-nrt-staging

echo ""
echo "============================================"
echo "  Build complete!"
echo "  Package: ${NRT_PACKAGE}"
echo "  Size: $(du -h ${NRT_PACKAGE} | cut -f1)"
echo "============================================"
echo ""
echo "To install on device:"
echo "  1. Web UI: upload ${NRT_PACKAGE} in the app import dialog"
echo "  2. CLI: tar xzf ${NRT_PACKAGE} && aipc-cli app install <app-id> ${PACKAGE_DIR}/app.yaml ${PACKAGE_DIR}/image.tar"
