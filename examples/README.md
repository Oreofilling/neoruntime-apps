# Examples

Runnable, copyable apps built on the
[NeoRuntime IPC SDK](https://pypi.org/project/neoruntime-ipc-sdk/)
(`neoruntime-ipc-sdk`, version pinned by [`sdk.lock`](../sdk.lock)).

New to the SDK? The
[learning path](https://github.com/camthink-ai/neoruntime-sdks/blob/main/python/docs/learning-path.rst)
in the SDK docs walks from a first inference to shipping an app; the
ladder below is its stage 3.

## The ladder (start here)

Five rungs, in order. Each one is small enough to read in one sitting
(main files stay under ~150 lines), runs as a real installable app, and
introduces exactly one new idea. Rung 01 needs no device at all; rungs
02-05 need a NeoRuntime device with a camera.

| # | Example | New idea | Needs device | Model |
|---|---------|----------|--------------|-------|
| 1 | [01-hello-app](01-hello-app/) | identity, signals, exit codes — asking for nothing | no | — |
| 2 | [02-first-frame](02-first-frame/) | one frame off a stream, saved as JPEG | yes | — |
| 3 | [03-single-inference](03-single-inference/) | `spec.models` + results pushed to you (`subscribe`) | yes | yolov8n (bundled) |
| 4 | [04-register-model](04-register-model/) | runtime `register_model`/`unregister` + `InferencePipeline` | yes | yolov8n (bundled) |
| 5 | [05-live-detection](05-live-detection/) | the B-form live loop + your own web page (MJPEG) | yes | yolov8n (bundled) |

Two ways to get inference, in ladder terms:

- **A form** — the platform pushes results, you never touch pixels
  (rung 03's `InferenceClient.subscribe`).
- **B form** — pixels flow through your app: `FdMediaClient.subscribe`
  → `InferencePipeline` → `draw_detections` (rung 05).

## Full examples

Older, larger examples kept for reference; the ladder above supersedes
them as the entry point.

- [person-detection](person-detection/) — A-form subscription loop
- [object-detection](object-detection/) — B-form full pipeline
- [people-counting](people-counting/) — events + counting logic
- [hello-world](hello-world/) — minimal container, no SDK

## Building and installing

Every example directory carries the same four files:

```
app.py            the app (read this one)
app.yaml          identity + permissions + models for the platform
Dockerfile        image (SDK pinned via sdk.lock / --build-arg)
build.sh          wraps scripts/build_app.sh
```

```bash
./build.sh arm64                     # -> <id>-<version>-arm64.neoapp
tar xzf <id>-<version>-arm64.neoapp
aipc-cli app install <id> <id>-<version>-arm64/app.yaml \
    <id>-<version>-arm64/image.tar
aipc-cli app start <id>
aipc-cli app logs <id>
```

Model-bearing examples pin their `.hef` files in `models.manifest`
(sha256-verified at build time by `scripts/fetch_models.sh`).

## Offline tests

Each ladder rung has `tests/` that run without a device or even the
SDK installed (the SDK import is stubbed):

```bash
python3 -m pytest examples/01-hello-app/tests examples/02-first-frame/tests \
    examples/03-single-inference/tests examples/04-register-model/tests \
    examples/05-live-detection/tests -q
```

CI builds and smoke-tests every example on each push; the pytest suites
are for local development.

## Where to go after rung 05

- [sdk-teaching-demo](../showcases/sdk-teaching-demo/) — the five
  interactive stations (overlay, hardware routing, A-form events…);
  rung 05 is station 1 as a standalone app.
- [templates/basic](../templates/basic/) — starting point for your own.
- [showcases](../showcases/) — complete production-shaped apps
  (parking-lot, model-showcase, shelf-ops, gym-ops).
- The SDK's [learning path](https://github.com/camthink-ai/neoruntime-sdks/blob/main/python/docs/learning-path.rst)
  stages 4-5 — packaging a `.neoapp`, restart policies, diagnostics,
  and the error lessons.
