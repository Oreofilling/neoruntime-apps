# NeoRuntime Apps

[![Build showcase bundles](https://github.com/camthink-ai/neoruntime-apps/actions/workflows/showcase-artifacts.yml/badge.svg)](https://github.com/camthink-ai/neoruntime-apps/actions/workflows/showcase-artifacts.yml)

Official sample applications and templates for the NeoRuntime edge AI platform.

These applications are built against the NeoRuntime SDK and packaged as
containerized AIPC apps.

## Downloads

| Showcase | Latest ARM64 bundle |
| -------- | ------------------- |
| Model Showcase | [model-showcase-latest-arm64.tar.gz](https://github.com/camthink-ai/neoruntime-apps/releases/download/showcase-bundles-latest/model-showcase-latest-arm64.tar.gz) |
| Parking Lot | [parking-lot-latest-arm64.tar.gz](https://github.com/camthink-ai/neoruntime-apps/releases/download/showcase-bundles-latest/parking-lot-latest-arm64.tar.gz) |
| Gym Ops | [gym-ops-latest-arm64.tar.gz](https://github.com/camthink-ai/neoruntime-apps/releases/download/showcase-bundles-latest/gym-ops-latest-arm64.tar.gz) |
| Shelf Ops | [shelf-ops-latest-arm64.tar.gz](https://github.com/camthink-ai/neoruntime-apps/releases/download/showcase-bundles-latest/shelf-ops-latest-arm64.tar.gz) |

[All releases](https://github.com/camthink-ai/neoruntime-apps/releases)

## Repository Layout

- `templates/` - reusable app templates and manifest references
- `examples/` - small apps that demonstrate one platform capability at a time
- `tools/` - developer utilities, diagnostics, and plugin examples
- `showcases/` - richer end-to-end reference applications

## App Index

| Path | Type | Description |
| ---- | ---- | ----------- |
| `templates/basic/` | Template | Minimal app template and manifest reference |
| `examples/hello-world/` | Example | Basic lifecycle and logging example |
| `examples/person-detection/` | Example | Person detection with inference and events |
| `examples/people-counting/` | Example | People-counting starter application |
| `examples/object-detection/` | Example | Object detection and tracking starter |
| `examples/face-cascade/` | Example | Cascade inference example for face landmarks |
| `tools/clip-viewer/` | Tool | CLIP zero-shot image classification viewer |
| `tools/visualizer/` | Tool | Real-time inference result preview |
| `tools/parallel-benchmark/` | Tool | Parallel inference benchmark dashboard |
| `tools/plugin-rtsp/` | Tool | C++ RTSP plugin example |
| `showcases/model-showcase/` | Showcase | Multi-model web showcase with visualization |
| `showcases/parking-lot/` | Showcase | Parking-lot vehicle and plate extraction reference |
| `showcases/gym-ops/` | Showcase | Gym pose coaching and occupancy reference |
| `showcases/shelf-ops/` | Showcase | Shelf slot recognition: slot-anchored CLIP A–E classification, empty-slot & sales heatmap reference |

## Development

Install the SDK from PyPI, then work inside one app directory at a time:

```bash
cd examples/person-detection
python -m venv .venv
. .venv/bin/activate
pip install neoruntime-ipc-sdk
pip install -r requirements.txt
python app.py
```

Most apps include a `build.sh` script that packages a container image and
application manifest for deployment to an NeoRuntime device. Images install
`neoruntime-ipc-sdk==<version>` from PyPI inside the Dockerfile; the version
resolves to the latest release on PyPI at build time (`AIPC_SDK_VERSION`
overrides to pin a specific one).

Runtime credentials, device addresses, and generated `.aipc` packages are
intentionally not committed. Use environment variables and local deployment
configuration for device-specific values. Model HEFs fetched from the Hailo
model zoo are also uncommitted — see [Model Files](#model-files).

### Model Files

App images bundle their model dependencies so installs are plug-and-play on
fresh devices — no manual `/data/aipc/models` provisioning step.

- Each app lists its downloadable models in `models.manifest` (TSV:
  `sha256<TAB>filename<TAB>url`). `scripts/fetch_models.sh <app-dir>`
  downloads and verifies them into `<app-dir>/models/`; `build_app.sh` and
  the showcase bundle build run it automatically.
- HEFs with no public source (recompiled inputs, custom class sets) are
  vendored directly in `<app-dir>/models/` and committed via `.gitignore`
  negations.
- Dockerfiles copy `models/` into the image. Apps whose `app.yaml`
  bind-mounts the host model directory must copy to
  `/opt/aipc/bundled-models` instead of `/opt/aipc/models`, or the mount
  shadows the image content on device.
- Apps resolve model paths host-first (`MODEL_ROOT`), falling back to the
  bundled copy — devices that pre-provision models keep using their own.
- `spec.models` entries that auto-register a model at install time are only
  safe for models whose raw output needs no postprocess (embedding, depth)
  or whose app re-registers it with a variant. See `docs/app-permissions.md`.

## Showcase Bundles

Showcase bundles contain a Docker image tarball, `app.yaml`, companion YAML
files, and checksums. GitHub Actions builds these bundles when showcase files
or the build scripts change, and tag builds attach the bundles to the
GitHub Release.

To build locally, run the build script from any showcase directory (it wraps
`scripts/build_showcase_artifacts.sh`):

```bash
cd showcases/shelf-ops
./build.sh
```

The Python SDK (`neoruntime-ipc-sdk`) is installed from PyPI during the image
build, so a clean clone needs no sibling SDK repositories or tokens
(`AIPC_SDK_VERSION` pins a specific release). Models are fetched automatically at
build time via each showcase's `models.manifest` and baked into the image
(see [Model Files](#model-files)), so installed bundles run on fresh
devices; device-provisioned copies still take precedence where present.

To install a downloaded bundle on a device, extract it and run:

```bash
aipc-cli app install app.yaml <showcase>-image.tar
```

## Related Repositories

- `camthink-ai/neoruntime` - platform core
- `camthink-ai/neoruntime-sdks` - SDKs used by apps

## License

This repository is licensed under the MIT License. See [LICENSE](./LICENSE).
