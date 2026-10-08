# SDK Teaching Demo — one installable interactive lesson

A single `.neoapp` that teaches the `neoruntime-ipc-sdk` by doing: install
it, open the page, and drive five hands-on stations against live camera
streams. No docs to read first — every station names the exact SDK calls
it exercises, and the mistakes it prevents are the ones we made on real
devices.

## Install

```bash
# device console: Apps → Import → sdk-teaching-demo-latest-arm64.neoapp
# or via aipc-cli:
aipc-cli app install sdk-teaching-demo <path-to-app.yaml> <path-to-image.tar>
aipc-cli app start sdk-teaching-demo
```

Open `http://<device>:8090/` directly, or **Visit App** in the console
(served under `/apps/sdk-teaching-demo/` — the page works behind the
reverse proxy).

Model note: the app runtime-registers its bundled
`hailo_yolov8n_384_640.hef` (4-class: person / vehicle / face /
license_plate). If the device already has `yolov8n_384_640` registered,
the existing registration is reused.

## The five stations

| # | Station | What you drive | SDK calls under the hood |
|---|---------|----------------|--------------------------|
| 1 | Live detection (B form) | engine on/off, draw on/off, score threshold, class filter | `FdMediaClient.subscribe("third", keep_fd=True)` → `InferencePipeline.run()` → `draw_detections()` → MJPEG |
| 2 | Render paths | local draw vs platform overlay, teaching-zone polygon | `OverlayClient.annotate_result("main", result, ttl_ms)` vs drawing pixels you own |
| 3 | Hardware routing | SOFTWARE_ONLY / PREFER_HARDWARE / HARDWARE_ONLY, refusal demo | `set_route_policy()`, `get_default_router().health()`, per-op hw/sw/fallback deltas |
| 4 | A form | start/stop the platform-side pipeline | `StreamPipeline("third", model, fps=10)` — platform subscribes, infers and draws |
| 5 | Events | watch gated republishes | `EventClient.publish("app/sdk-teaching-demo/detection", …)`, 5 s cooldown |

English is the default page language; toggle 中文 in the header.

## From this demo to your own app

Since v1.2 every station panel shows **the real code behind it** — a
condensed excerpt of `teaching_app.py` with a copy button — plus that
station's error lesson (DMA -2811, routing refusals, overlay `None` vs
`[]`, exit codes). The same payload is served at `GET /api/snippets`.

1. Copy [`templates/basic`](../../templates/basic) — the minimal app
   skeleton (app.yaml + Dockerfile + build.sh).
2. Write your engine against the same SDK surface the five stations
   drive.
3. `./build.sh arm64` → a `.neoapp` bundle.
4. Install like any app: `aipc-cli app install <id> <app.yaml> <image.tar>`.

SDK reference: [neoruntime-ipc-sdk on PyPI](https://pypi.org/project/neoruntime-ipc-sdk/) · [SDK source (python/)](https://github.com/camthink-ai/neoruntime-sdks/tree/main/python)

## Build

```bash
./build.sh arm64   # → ../../dist/showcases/sdk-teaching-demo-<ver>-arm64.neoapp
```

`python:3.11-slim` + `neoruntime-ipc-sdk` (pinned via `sdk.lock`) +
`opencv-python-headless` + `numpy`. Run the offline test suite with
`python3 -m pytest tests/ -q` (no SDK or device needed — the SDK module
is stubbed).

### Keeping the demo honest

- The demo's contracts target SDK **0.8** (`sdk.lock` pins the patch).
  `tests/test_sdk_compat.py` fails with instructions when the lock
  drifts outside `SUPPORTED_SDK_VERSIONS` — re-verify (offline suite +
  device e2e) before extending it.
- The in-page code excerpts carry anchors that must stay verbatim in
  `teaching_app.py`; `tests/test_snippets.py` enforces them, so a
  refactor fails the suite instead of silently rotting the lesson.

## Permissions (app.yaml) — why each one

- `video: [main.raw, third.raw, sub.raw]` — subscribe `third` for
  inference; annotate the console's `main` (normalized coords transfer
  across streams sharing the FOV).
- `inference.allow_register_model: true` — runtime `register_model()`
  like the SDK examples. No `models:` preload list: registering at
  runtime is itself one of the lessons.
- `events.publish: [inference/main, app/sdk-teaching-demo/*]` — the
  overlay path rides `inference/main`; the app's own detection events
  go to its topic.
- host `/data/aipc/models` mounted read-only at the same path —
  register paths must be host-real for ai-runtime.

## FAQ — contracts baked into this app

**Why does the engine exit 1 when it dies?** `restart_policy:
on-failure` treats exit 0 as a clean stop. An engine that dies without
a signal returns 1, so the supervisor revives the app instead of
leaving a zombie page.

**Why `third` and not `main` for inference?** Platform-side feeding has
no scaler: the stream geometry must equal the runtime-registered
model's input. `third` (640×384) matches; `main` 1080p rejects every
frame with `DMA input rejected: -2811` (Station 4 shows the numbers).
Feeding `main` requires a platform-managed model (`.bin` packaging),
not an SDK change.

**Why does every annotate carry `ttl_ms`?** Overlay layers expire; the
app re-publishes on its own cadence so boxes vanish within a second of
the app stopping. Cleanup clears with **empty lists** —
`detections=None` would keep the stale layer.

**Why is there no `models:` list in app.yaml?** A preload list turns
into a start-time requirement and hides the runtime `register_model()`
lesson this app teaches. The image bundles the HEF under
`/opt/aipc/bundled-models` as the fallback source; host
`/data/aipc/models/detection/` wins when present. The app never falls
back to the flat `/data/aipc/models/` root — on some devices that file
is a different compile of the same network.
