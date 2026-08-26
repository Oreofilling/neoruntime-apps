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

Install the vendored SDK wheel (see `third_party/`), then work inside one app
directory at a time:

```bash
cd examples/person-detection
python -m venv .venv
. .venv/bin/activate
pip install ../../third_party/hailo_ipc_sdk-*.whl
pip install -r requirements.txt
python app.py
```

Most apps include a `build.sh` script that packages a container image and
application manifest for deployment to an NeoRuntime device.

Runtime credentials, device addresses, model files, and generated `.aipc`
packages are intentionally not committed. Use environment variables and local
deployment configuration for device-specific values.

## Showcase Bundles

Showcase bundles contain a Docker image tarball, `app.yaml`, companion YAML
files, and checksums. GitHub Actions builds these bundles when showcase files
or the vendored SDK wheel change, and tag builds attach the bundles to the
GitHub Release.

To build locally, run the build script from any showcase directory (it wraps
`scripts/build_showcase_artifacts.sh`):

```bash
cd showcases/shelf-ops
./build.sh
```

The Python SDK wheel is vendored in `third_party/`, so a clean clone builds
with no sibling SDK repositories or tokens (`--wheel` overrides, `dist/` and
sibling SDK repos remain fallbacks). Builds do not need model files: models
are mounted read-only from the device at runtime, never baked into images.

To install a downloaded bundle on a device, extract it and run:

```bash
aipc-cli app install app.yaml <showcase>-image.tar
```

## Related Repositories

- `camthink-ai/neoruntime` - platform core
- `camthink-ai/neoruntime-sdks` - SDKs used by apps

## License

This repository is licensed under the MIT License. See [LICENSE](./LICENSE).
