#!/usr/bin/env bash
# Build downloadable showcase bundles containing a Docker image tarball and
# the app YAML files required for installation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ARCH="arm64"
OUTPUT_ROOT="$PROJECT_ROOT/dist/showcases"
WHEEL_PATH=""
SHOWCASES=()

usage() {
    cat <<'USAGE'
Usage: scripts/build_showcase_artifacts.sh [options] [showcase...]

Options:
  --arch arm64|amd64     Target container platform architecture (default: arm64)
  --output <dir>         Output directory for bundles (default: dist/showcases)
  --wheel <path>         Python SDK wheel to stage into each Docker build context
                         (default resolution: third_party/ > dist/ > sibling SDK repos)
  -h, --help             Show this help

Examples:
  scripts/build_showcase_artifacts.sh --wheel ../neoruntime-sdks/python/dist/hailo_ipc_sdk-0.3.0-py3-none-any.whl
  scripts/build_showcase_artifacts.sh model-showcase parking-lot --arch arm64 --output dist/showcases
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)
            ARCH="${2:-}"
            shift 2
            ;;
        --output)
            OUTPUT_ROOT="${2:-}"
            shift 2
            ;;
        --wheel)
            WHEEL_PATH="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            SHOWCASES+=("$1")
            shift
            ;;
    esac
done

case "$ARCH" in
    arm64|aarch64) ARCH="arm64" ;;
    amd64|x86_64) ARCH="amd64" ;;
    *)
        echo "Unsupported architecture: $ARCH" >&2
        exit 1
        ;;
esac

if [ -z "$OUTPUT_ROOT" ]; then
    echo "--output cannot be empty" >&2
    exit 1
fi

find_default_wheel() {
    # Priority order: vendored wheel first so clean clones build with no
    # sibling SDK repos; dist/ and sibling repos remain as fallbacks.
    local bases=(
        "$PROJECT_ROOT/third_party"
        "$PROJECT_ROOT/dist"
        "$PROJECT_ROOT/../neoruntime-sdks/python/dist"
        "$PROJECT_ROOT/../ne503-aipc-sdks/python/dist"
    )

    local base wheels=()
    shopt -s nullglob
    for base in "${bases[@]}"; do
        wheels=("$base"/hailo_ipc_sdk-*.whl)
        if [ "${#wheels[@]}" -gt 0 ]; then
            printf '%s\n' "${wheels[@]}" | sort -V | tail -n 1
            shopt -u nullglob
            return
        fi
    done
    shopt -u nullglob

    echo "No SDK wheel found. Vendor one into third_party/ or pass --wheel <path>." >&2
    exit 1
}

if [ -z "$WHEEL_PATH" ]; then
    WHEEL_PATH="$(find_default_wheel)"
fi

if [ ! -f "$WHEEL_PATH" ]; then
    echo "SDK wheel not found: $WHEEL_PATH" >&2
    exit 1
fi

if [ "${#SHOWCASES[@]}" -eq 0 ]; then
    for app_dir in "$PROJECT_ROOT"/showcases/*; do
        [ -d "$app_dir" ] || continue
        [ -f "$app_dir/Dockerfile" ] || continue
        [ -f "$app_dir/app.yaml" ] || continue
        SHOWCASES+=("$(basename "$app_dir")")
    done
fi

if [ "${#SHOWCASES[@]}" -eq 0 ]; then
    echo "No showcase apps found." >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required." >&2
    exit 1
fi

if ! docker buildx version >/dev/null 2>&1; then
    echo "docker buildx is required." >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

extract_yaml_value() {
    local file="$1"
    local key="$2"

    awk -v key="$key" '
        $1 == key ":" {
            value = $2
            gsub(/["'\''\r]/, "", value)
            print value
            exit
        }
    ' "$file"
}

resolve_showcase_dir() {
    local item="$1"

    if [ -d "$item" ]; then
        (cd "$item" && pwd)
        return
    fi

    if [ -d "$PROJECT_ROOT/showcases/$item" ]; then
        (cd "$PROJECT_ROOT/showcases/$item" && pwd)
        return
    fi

    echo "Showcase not found: $item" >&2
    exit 1
}

for showcase in "${SHOWCASES[@]}"; do
    APP_DIR="$(resolve_showcase_dir "$showcase")"
    APP_NAME="$(basename "$APP_DIR")"
    APP_YAML="$APP_DIR/app.yaml"
    VERSION="$(extract_yaml_value "$APP_YAML" "version")"
    IMAGE_TAG="$(extract_yaml_value "$APP_YAML" "image")"
    VERSION="${VERSION:-0.0.0}"
    IMAGE_TAG="${IMAGE_TAG:-aipc/${APP_NAME}:${VERSION}}"

    BUNDLE_NAME="${APP_NAME}-${VERSION}-${ARCH}"
    BUNDLE_DIR="$OUTPUT_ROOT/$BUNDLE_NAME"
    IMAGE_TAR="$BUNDLE_DIR/${APP_NAME}-image.tar"
    BUNDLE_TGZ="$OUTPUT_ROOT/${BUNDLE_NAME}.tar.gz"
    STAGE_DIR="$APP_DIR/.aipc-build"

    echo "============================================"
    echo "  Building $APP_NAME $VERSION for linux/$ARCH"
    echo "  Image: $IMAGE_TAG"
    echo "============================================"

    rm -rf "$BUNDLE_DIR" "$BUNDLE_TGZ" "$STAGE_DIR"
    mkdir -p "$BUNDLE_DIR" "$STAGE_DIR"
    cp "$WHEEL_PATH" "$STAGE_DIR/"

    docker buildx build \
        --platform "linux/$ARCH" \
        --load \
        -t "$IMAGE_TAG" \
        "$APP_DIR"

    docker save "$IMAGE_TAG" -o "$IMAGE_TAR"

    shopt -s nullglob
    for yaml_file in "$APP_DIR"/*.yaml "$APP_DIR"/*.yml; do
        cp "$yaml_file" "$BUNDLE_DIR/"
    done
    shopt -u nullglob

    cat > "$BUNDLE_DIR/README.txt" <<EOF
$APP_NAME showcase bundle

Image:
  $IMAGE_TAG

Install:
  aipc-cli app install app.yaml ${APP_NAME}-image.tar

Manual image import:
  ctr -n aipc images import ${APP_NAME}-image.tar
EOF

    (
        cd "$BUNDLE_DIR"
        sha256sum * > SHA256SUMS
    )

    tar -C "$OUTPUT_ROOT" -czf "$BUNDLE_TGZ" "$BUNDLE_NAME"
    rm -rf "$STAGE_DIR"

    echo "Bundle: $BUNDLE_TGZ"
done
