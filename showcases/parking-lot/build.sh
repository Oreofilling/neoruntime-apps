#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="parking-lot"
VERSION="1.0.0"
ARCH="${1:-arm64}"

echo "============================================"
echo "  Building ${APP_NAME}:${VERSION} for ${ARCH}"
echo "============================================"

echo "Copying SDK wheel..."
cp "${SCRIPT_DIR}/../../dist/hailo_ipc_sdk-0.3.0-py3-none-any.whl" "${SCRIPT_DIR}/"

echo "Building Docker image..."
if [ "$ARCH" = "arm64" ]; then
    docker buildx build --platform linux/arm64 --load -t "${APP_NAME}:${VERSION}" .
else
    docker build -t "${APP_NAME}:${VERSION}" .
fi

echo "Exporting image to tar..."
docker save "${APP_NAME}:${VERSION}" -o "${APP_NAME}-image.tar"

echo ""
echo "============================================"
echo "  Build complete!"
echo "  Image: ${APP_NAME}:${VERSION}"
echo "  Tar:   ${APP_NAME}-image.tar"
echo "  Size:  $(du -h ${APP_NAME}-image.tar | cut -f1)"
echo "============================================"
echo ""
echo "Deploy to device:"
echo "  scp ${APP_NAME}-image.tar root@192.0.2.72:/tmp/"
echo "  scp app.yaml root@192.0.2.72:/data/apps/manifests/${APP_NAME}/"
echo "  ssh root@192.0.2.72 'ctr -n aipc images import /tmp/${APP_NAME}-image.tar'"
echo ""

rm -f "${SCRIPT_DIR}/hailo_ipc_sdk-0.3.0-py3-none-any.whl"
