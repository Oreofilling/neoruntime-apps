#!/bin/bash
# Build script for Person Detection Application
# Usage: ./build.sh [arm64|amd64]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="person-detection"
VERSION="1.0.0"
ARCH="${1:-arm64}"

echo "============================================"
echo "  Building ${APP_NAME}:${VERSION} for ${ARCH}"
echo "============================================"

# Copy SDK
echo "Copying SDK..."
cp -r "${SCRIPT_DIR}/../../sdk/python/hailo_ipc_sdk" "${SCRIPT_DIR}/"
cp "${SCRIPT_DIR}/../../sdk/python/setup.py" "${SCRIPT_DIR}/"
cp "${SCRIPT_DIR}/../../sdk/python/README.md" "${SCRIPT_DIR}/"

# Build Docker image
echo "Building Docker image..."
if [ "$ARCH" = "arm64" ]; then
    docker buildx build --platform linux/arm64 --load -t "aipc/${APP_NAME}:${VERSION}" .
else
    docker build -t "aipc/${APP_NAME}:${VERSION}" .
fi

# Export image
echo "Exporting image..."
docker save "aipc/${APP_NAME}:${VERSION}" -o image.tar

# Create .aipc package
echo "Creating .aipc package..."
rm -f "${APP_NAME}.aipc"
zip -r "${APP_NAME}.aipc" app.yaml image.tar

# Cleanup
rm -rf hailo_ipc_sdk setup.py README.md image.tar

echo ""
echo "============================================"
echo "  Build complete!"
echo "  Package: ${APP_NAME}.aipc"
echo "  Size: $(du -h ${APP_NAME}.aipc | cut -f1)"
echo "============================================"
echo ""
echo "To install on device:"
echo "  1. Web UI: Upload ${APP_NAME}.aipc"
echo "  2. API: curl -X POST http://<device>:8080/api/v1/apps -F 'app=@${APP_NAME}.aipc'"