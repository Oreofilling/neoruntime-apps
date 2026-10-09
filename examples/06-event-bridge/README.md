# 06-event-bridge

Rung 6 of the [examples ladder](../README.md): **detections in, events
out.** The inference side is rung 03's A form; the one new idea is the
event bus — `EventClient.publish()` turns results into events, and a
daemon tap `subscribe()`s the wildcard topic to watch them come back.

## What it teaches

- `events.publish(topic, payload)` — one dict becomes one bus event
- `events.subscribe(topic)` — a generator of `Event`
  (`.topic` / `.payload` / `.source`); wildcards work (`app/event-bridge/#`)
- Publish hygiene: a cooldown bounds events to one per window, and
  frames with no objects publish nothing
- `permissions.events.publish` in app.yaml — topics are allow-listed

## Files

```
app.py            the app (read this one)
app.yaml          identity + permissions (event topic allow-list) + model
Dockerfile        image (SDK pinned via sdk.lock / --build-arg)
build.sh          wraps scripts/build_app.sh
```

## Build and install

```bash
./build.sh arm64                     # -> event-bridge-1.0.0-arm64.neoapp
tar xzf event-bridge-1.0.0-arm64.neoapp
aipc-cli app install event-bridge event-bridge-1.0.0-arm64/app.yaml \
    event-bridge-1.0.0-arm64/image.tar
aipc-cli app start event-bridge
aipc-cli app logs event-bridge       # watch "published {...}" + "bus: ..."
```

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `STREAM_ID` | `main` | stream to subscribe results for |
| `EVENT_COOLDOWN_S` | `2` | min seconds between published events |
| `MIN_SCORE` | `0.3` | score floor for an object to count |

## Watching it from another app

Anything on the device (app or SSH session) that can reach the bus may
subscribe to `app/event-bridge/#` and will see this rung's events,
source-tagged `event-bridge`.

Next: [07-platform-overlay](../07-platform-overlay/) — push results to
the platform's overlay and let it draw the boxes for you.
