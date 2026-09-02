#!/usr/bin/env bash
# Fetch model files for app images (Hailo model zoo, pinned compiles).
#
# Reads <app-dir>/models.manifest (TSV: sha256 <tab> filename <tab> url) and
# downloads each entry into <app-dir>/models/, verifying the SHA-256 before
# moving anything into place. Files already present with a matching hash are
# skipped, so re-running is cheap and safe. Vendored files that live in the
# repo without a manifest entry are left untouched.
#
# Usage:
#   scripts/fetch_models.sh <app-dir> [<app-dir>...]
#   scripts/fetch_models.sh --all          # every examples/*/showcases/* with a manifest
#
# Zoo compiles are pinned to ModelZoo/Compiled/v5.3.0/hailo15h to match the
# device firmware line (HailoRT 5.3.0 on HAILO15H). Do not bump versions
# without re-verifying with `hailortcli parse-hef` on target hardware.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

fetch_app_models() {
    local app_dir="$1"
    local manifest="$app_dir/models.manifest"
    [ -f "$manifest" ] || return 0

    local app_name
    app_name="$(basename "$app_dir")"
    local models_dir="$app_dir/models"
    mkdir -p "$models_dir"

    local count=0
    while IFS=$'\t' read -r sha filename url; do
        case "$sha" in ''|'#'*) continue ;; esac
        [ -n "$sha" ] && [ -n "$filename" ] && [ -n "$url" ] || {
            echo "[$app_name] skipping malformed manifest line" >&2
            continue
        }
        count=$((count + 1))

        local dest="$models_dir/$filename"
        if [ -f "$dest" ] && echo "$sha  $dest" | sha256sum -c --status >/dev/null 2>&1; then
            echo "[$app_name] $filename: up to date"
            continue
        fi

        echo "[$app_name] $filename: downloading..."
        local tmp="$dest.part"
        curl -fSL --retry 3 -o "$tmp" "$url"
        echo "$sha  $tmp" | sha256sum -c --status >/dev/null 2>&1 || {
            rm -f "$tmp"
            echo "[$app_name] ERROR: sha256 mismatch for $filename (expected $sha)" >&2
            return 1
        }
        mv "$tmp" "$dest"
        echo "[$app_name] $filename: verified OK"
    done < "$manifest"

    [ "$count" -gt 0 ] && echo "[$app_name] $count model file(s) ready under models/"
    return 0
}

if [ "${1:-}" = "--all" ]; then
    failed=0
    for dir in "$PROJECT_ROOT"/examples/*/ "$PROJECT_ROOT"/showcases/*/; do
        [ -f "$dir/models.manifest" ] || continue
        fetch_app_models "${dir%/}" || failed=1
    done
    exit "$failed"
fi

[ $# -ge 1 ] || { echo "Usage: $0 <app-dir>... | --all" >&2; exit 1; }

for app_dir in "$@"; do
    [ -d "$app_dir" ] || { echo "Error: app directory not found: $app_dir" >&2; exit 1; }
    fetch_app_models "$(cd "$app_dir" && pwd)"
done
