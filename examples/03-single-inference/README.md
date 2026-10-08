# 03-single-inference — your first inference result

Rung 3 of the [examples ladder](../README.md). One subscription, one
result, exit. The heavy lifting is declarative: the model is declared in
`app.yaml` (`spec.models`), the platform registers and authorizes it at
install time, and this app only reads what the platform pushes back.

## What it teaches

- **`spec.models`** — declare a model dependency and the platform
  resolves it: registers the HEF, authorizes the id for this app, and
  injects the resolved id as `AIPC_MODEL_detector` (the spec key
  `detector` becomes the env suffix). A device that already has the id
  uses its own copy.
- **`inference.subscribe(stream=..., model=..., fps=...)`** — the
  platform runs inference on the stream and pushes
  `(frame_sequence, InferenceResult)` to your generator. Your app never
  touches pixels on this path.
- **Reading results** — `result.objects` is a list of
  `DetectedObject` with `.label`, `.score` and a **normalized** (0..1)
  `.bbox`, so boxes transfer across streams sharing the field of view.
- **Fail-fast diagnostics** — the model list is logged first; a missing
  model warns before subscribing, and a failed subscription exits 3
  with the three things to check (stream id, model id, permissions).

## Build and install

```bash
./build.sh arm64            # -> single-inference-1.0.0-arm64.neoapp
tar xzf single-inference-1.0.0-arm64.neoapp
aipc-cli app install single-inference single-inference-1.0.0-arm64/app.yaml \
    single-inference-1.0.0-arm64/image.tar
aipc-cli app start single-inference
aipc-cli app logs single-inference
```

## Expected output

```
[INFO] registered models: ['yolov8n', ...]
[INFO] subscribing stream=main model=yolov8n (fps<=2, first result only)
[INFO] frame 118: 2 object(s) over 0.30
[INFO]   person       0.87  x=0.31 y=0.42 w=0.18 h=0.41
[INFO]   person       0.64  x=0.62 y=0.45 w=0.14 h=0.35
```

Exit 3 with `inference failed` → check `STREAM_ID` (devices expose
`main`/`sub`), `MODEL_ID` (env override `AIPC_MODEL_detector`), and the
`video`/`inference` permissions in `app.yaml`.

## Try changing

- Raise `MIN_SCORE` to `0.8` and watch the kept count drop — the model
  still returns everything; filtering is yours.
- Set `fps: 10` in the `subscribe()` call and compare the result rate
  in the logs.

Next: [04-register-model](../04-register-model/) — owning the model
lifecycle at runtime.
