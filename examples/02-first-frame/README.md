# 02-first-frame — one camera frame, as a file

Rung 2 of the [examples ladder](../README.md). The smallest camera app
possible: subscribe to nothing, infer nothing — grab **one** frame,
write `first-frame.jpg`, exit. It is the diagnostic you run first on a
new device: green output proves camera-daemon, the stream id, and the
SDK socket path all line up.

## What it teaches

- **`FdMediaClient` basics** — `list_streams()`, `get_stream_info()`
  (geometry/fps/format before you commit), `get_frame()` with a
  timeout.
- **Encoding without image stack** — `Frame.to_jpeg_bytes()`; no numpy,
  no cv2 anywhere in this rung.
- **One-shot exit codes** — 0 on success, 3 on environment problems
  (dead camera / wrong stream id). Paired with `restart_policy: never`
  in `app.yaml`: a dead camera surfaces as one failed run instead of an
  invisible retry loop.
- **Permissions** — the first rung that asks for something: `video:
  [main.raw, sub.raw]`. Install fails fast without it.

## Build and install

```bash
./build.sh arm64                       # -> first-frame-1.0.0-arm64.neoapp
tar xzf first-frame-1.0.0-arm64.neoapp
aipc-cli app install first-frame first-frame-1.0.0-arm64/app.yaml \
    first-frame-1.0.0-arm64/image.tar
aipc-cli app start first-frame         # runs once, exits
aipc-cli app logs first-frame
```

The JPEG lands on the device at `/data/aipc/data/first-frame/first-frame.jpg`
(the `volumes:` mapping in `app.yaml`).

## Expected output

```
[INFO] requesting one frame from 'main'
[INFO] stream 'main': 1920x1080 @ 25fps (NV12)
[INFO] wrote 240813 bytes -> /app/data/first-frame.jpg
```

Exit 3 with `no frame within 5000 ms` means the environment, not the
code: check `aipc-cli` diagnostics for camera-daemon, and try
`STREAM_ID: "sub"` in `app.yaml`.

## Try changing

- Point `OUT_PATH` at `/app/data/sub-frame.jpg` and `STREAM_ID` at
  `sub` — compare the two JPEGs (field of view vs resolution).
- Lower `TIMEOUT_MS` to 500 and watch the exit-3 path fire.

Next: [03-single-inference](../03-single-inference/) — your first
inference result.
