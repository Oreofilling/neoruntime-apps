# 01-hello-app — lifecycle only

Rung 1 of the [examples ladder](../README.md). The smallest useful
NeoRuntime app: it ticks once a second and stops gracefully when asked.
No camera, no model, no device daemons — but it is a complete,
installable app, and everything the later rungs add (frames, models,
loops, a web page) rides on the lifecycle this rung demonstrates.

## What it teaches

- **App identity** — `Config.get_app_id()` reads the id the platform
  injects; it works with no device connection at all.
- **Graceful shutdown** — SIGTERM (what `aipc-cli app stop` sends)
  flips a flag; the loop exits cleanly with code 0.
- **Exit codes vs `restart_policy`** — an unrecoverable error should
  exit non-zero and let `restart_policy: on-failure` restart you. Set
  `DEMO_FAIL=1` in `app.yaml` and the app deliberately crashes on its
  third tick; watch the restarts in `aipc-cli app logs hello-app`.
- **Asking for nothing** — this manifest requests no permissions at
  all. Compare it with rung 02 (video), 03/05 (inference) and 04
  (`allow_register_model`).

## Build and install

```bash
./build.sh arm64                       # -> hello-app-1.0.0-arm64.neoapp
tar xzf hello-app-1.0.0-arm64.neoapp
aipc-cli app install hello-app hello-app-1.0.0-arm64/app.yaml \
    hello-app-1.0.0-arm64/image.tar
aipc-cli app start hello-app
aipc-cli app logs hello-app            # watch the ticks
aipc-cli app stop hello-app            # SIGTERM -> "stopped after N ticks"
```

## Expected output

```
[hello-app] hello from aarch64
[hello-app] tick #1
[hello-app] tick #2
[hello-app] signal 15 -> graceful stop
[hello-app] stopped after 2 ticks
```

## Try changing

- Make the tick print `Config.is_debug()` too.
- Set `DEMO_FAIL: "1"` in `app.yaml`, reinstall, and count the restarts
  (the platform gives up after `restart_max_retries`).

Next: [02-first-frame](../02-first-frame/) — your first camera frame.
