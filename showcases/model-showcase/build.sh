#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="model-showcase"
VERSION="1.2.1"
ARCH="${1:-arm64}"

echo "============================================"
echo "  Building ${APP_NAME}:${VERSION} for ${ARCH}"
echo "============================================"

echo "Copying SDK wheel..."
cp "${SCRIPT_DIR}/../../dist/hailo_ipc_sdk-0.3.0-py3-none-any.whl" "${SCRIPT_DIR}/"

echo "Building Docker image..."
if [ "$ARCH" = "arm64" ]; then
    docker buildx build --platform linux/arm64 --load -t "aipc/${APP_NAME}:${VERSION}" .
else
    docker build -t "aipc/${APP_NAME}:${VERSION}" .
fi

echo "Exporting image to tar..."
docker save "aipc/${APP_NAME}:${VERSION}" -o "${APP_NAME}-image.tar"

echo ""
echo "============================================"
echo "  Build complete!"
echo "  Image: aipc/${APP_NAME}:${VERSION}"
echo "  Tar:   ${APP_NAME}-image.tar"
echo "  Size:  $(du -h ${APP_NAME}-image.tar | cut -f1)"
echo "============================================"
echo ""
echo "Import on device:"
echo "  ctr -n aipc images import /tmp/${APP_NAME}-image.tar"
echo ""

rm -f "${SCRIPT_DIR}/hailo_ipc_sdk-0.3.0-py3-none-any.whl"
