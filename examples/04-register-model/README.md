# 04-register-model — owning the model lifecycle at runtime

Rung 4 of the [examples ladder](../README.md). Rung 03 declared its
model in `app.yaml` and the platform resolved it. This rung does the
opposite: the app itself walks the full runtime lifecycle — register,
list, infer once, unregister — then exits.

## What it teaches

- **`register_model()` / `unregister_model()`** — and the permission
  that gates them: nothing works until `inference.allow_register_model:
  true` is in `app.yaml`. That line is the rung.
- **The already-registered guard** — re-registering an existing id
  breaks the postprocess link of anything using it. The app registers
  only when the id is missing, and unregisters only when *it* did the
  registering.
- **`model_variant=None` matters** — a bare string gets wrapped as a
  backend_function and inference fails `HAILO_INVALID_OPERATION`
  (parking-lot hit this in the field).
- **Model resolution order** — host model store first (a device may
  already carry this exact compile; some devices keep a *different*
  compile under the flat `models/` root), image-bundled copy last.
- **`Preprocessor.from_model()` + `InferencePipeline`** — the pipeline
  reads the model's input geometry/pixel format, so the app never
  hardcodes a resize. One `run(frame)`, one result.

## Build and install

```bash
./build.sh arm64            # -> register-model-1.0.0-arm64.neoapp
tar xzf register-model-1.0.0-arm64.neoapp
aipc-cli app install register-model register-model-1.0.0-arm64/app.yaml \
    register-model-1.0.0-arm64/image.tar
aipc-cli app start register-model
aipc-cli app logs register-model
```

## Expected output

```
[INFO] [1/4] registered 'yolov8n-demo' from /data/aipc/models/detection/yolov8n.hef
[INFO] [2/4] models now: ['yolov8n-demo', 'yolov8n', ...]
[INFO] [3/4] 2 object(s)
[INFO]     person       0.83
[INFO] [4/4] unregistered 'yolov8n-demo'
```

Run it twice: the second run logs `[1/4] 'yolov8n-demo' already
registered - reusing` only if something else registered the id in
between (this rung always unregisters on exit — check `list_models`
behavior yourself by removing that block).

Exit 3 with `model lifecycle failed` → almost always the missing
`allow_register_model: true` permission.

## Try changing

- Register under a second id (`MODEL_ID: "yolov8n-b"`) and leave both
  registered (delete the `[4/4]` block) — then compare `list_models`
  before/after in the logs.
- Break it on purpose: drop `allow_register_model` from `app.yaml`,
  reinstall, and read the refusal.

Next: [05-live-detection](../05-live-detection/) — the live loop, and
a web page of your own.
