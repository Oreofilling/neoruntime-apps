# 07-platform-overlay

Rung 7 of the [examples ladder](../README.md): **boxes on the stream,
zero pixels in your app.** Same A-form subscription as rung 03, but
results are PUSHED to the platform's overlay — camera-daemon draws
them on every display stream. Contrast rung 05, where you drew boxes
yourself on frames you subscribed to.

## What it teaches

- `overlay.enable()` — the global overlay toggle; the console's
  AI-overlay switch is this same call
- `overlay.annotate_result(stream_id, result, ttl_ms=...)` — one call
  per result, rendered by the platform on every stream displaying
  `stream_id`
- ttl semantics: boxes fade after `ttl_ms`; set it to a few frame
  periods so a dropped frame does not leave frozen boxes
- Empty results still annotate — publishing "no detections" clears
  stale boxes EARLY instead of waiting out the ttl
- `permissions.events.publish: [inference/main]` — annotate rides the
  event bus on the `inference/<stream>` topic

## Files

```
app.py            the app (read this one)
app.yaml          identity + permissions (overlay event topic) + model
Dockerfile        image (SDK pinned via sdk.lock / --build-arg)
build.sh          wraps scripts/build_app.sh
```

## Build and install

```bash
./build.sh arm64                     # -> platform-overlay-1.0.0-arm64.neoapp
tar xzf platform-overlay-1.0.0-arm64.neoapp
aipc-cli app install platform-overlay platform-overlay-1.0.0-arm64/app.yaml \
    platform-overlay-1.0.0-arm64/image.tar
aipc-cli app start platform-overlay
aipc-cli app logs platform-overlay   # one "frame N: K box(es)" line per result
```

Then open the device's web console or RTSP stream: the boxes are on
the picture, drawn by the platform — this app never saw a pixel.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `STREAM_ID` | `main` | stream whose results annotate the display |
| `OVERLAY_TTL_MS` | `1500` | how long boxes stay valid before fading |

Where to go next: the [sdk-teaching-demo](../../showcases/sdk-teaching-demo/)
interactive stations, [templates/basic](../../templates/basic/) to start
your own, and the SDK docs' [learning path](https://github.com/camthink-ai/neoruntime-sdks/blob/main/python/docs/learning-path.rst)
stages 4-5 (packaging, restart policies, diagnostics).
