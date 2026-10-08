# 05-live-detection — the live loop, and a web page of your own

Rung 5, the capstone of the [examples ladder](../README.md) and the
code-level twin of sdk-teaching-demo station 1. Everything the earlier
rungs taught comes together, and the pixels flow through **your** app
(the "B form"): you pull frames, you run inference, you draw on the
pixels you own, and you serve the result from a web page the app hosts
itself.

```
FdMediaClient.subscribe     InferencePipeline.run      draw_detections
    (your frames)       (model + geometry handled)       (your pixels)
                                                                    |
                                                 Frame(...).to_jpeg_bytes
                                                                    |
                                latest-JPEG buffer --> /  (status JSON)
                                                        /snapshot.jpg
                                                        /stream  (MJPEG)
```

## What it teaches

- **The B-form loop** — `media.subscribe()` + `InferencePipeline`:
  frames arrive, the pipeline handles model geometry and pixel format,
  and the loop never exits (rung 01's SIGTERM shape is what stops it).
- **Drawing on your own pixels** — `draw_detections(image, objects)`
  with **normalized** bboxes draws correctly on the full-frame array,
  whatever geometry the model uses internally.
- **One-slot buffering** — viewers always get "now"; memory stays flat
  no matter how many tabs are open or how slow a viewer is.
- **The app is a web server** — `network: host` + an inbound port in
  `app.yaml`, `register_web_url("/")` so the console can reverse-proxy
  it, and only relative URLs in the page so it works both directly
  (`http://<device>:8090`) and behind the console
  (`/apps/live-detection/`).
- **Zero web-framework dependencies** — the stdlib `http.server` is
  enough for MJPEG; flask enters a codebase only when it earns it.

## Build and install

```bash
./build.sh arm64            # -> live-detection-1.0.0-arm64.neoapp
tar xzf live-detection-1.0.0-arm64.neoapp
aipc-cli app install live-detection live-detection-1.0.0-arm64/app.yaml \
    live-detection-1.0.0-arm64/image.tar
aipc-cli app start live-detection
```

Then open `http://<device>:8090/` (or the app's page in the console).
`aipc-cli app stop live-detection` shuts it down gracefully.

## Expected behavior

- The page shows the live stream with boxes and `label score` captions
  on every detection over `MIN_SCORE`.
- `/status` returns `{"sequence": N, "objects": M, "fps": F}` — refresh
  it while covering the camera: `objects` drops, `fps` does not.
- `aipc-cli app logs live-detection` stays quiet in steady state; the
  per-request noise is suppressed on purpose.

## Try changing

- Draw a counter badge on the frame (the teaching demo's `draw_badge`
  writes into the luma plane for NV12) — you own the pixels now.
- Publish a detection event through `EventClient` when `objects` goes
  0 → >0 (that is teaching-demo station 5's cooldown gate).

Where next: the [sdk-teaching-demo](../../showcases/sdk-teaching-demo/)
stations (overlay paths, hardware routing, A-form `StreamPipeline`,
events), the full apps in this directory, and
[templates/basic](../../templates/basic/) to start your own.
