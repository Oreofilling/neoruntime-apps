#!/usr/bin/env python3
"""06-event-bridge — detections in, events out.

Rung 6 of the examples ladder (01-07). The inference side is rung 03's
A form (subscribe to platform-run results); the ONE new idea is the
event bus: turn results into events that other apps — or your own
tooling — can subscribe to, and tap a topic to watch them come back.

    inference.subscribe           events.publish                 events.subscribe
    (results pushed to you)  ->  (app/event-bridge/detections) ->  (daemon tap logs
                                                                      everything on # )

What it teaches
  * ``events.publish(topic, payload)`` — one dict becomes one bus event
  * ``events.subscribe(topic)`` — a generator of ``Event``
    (``.topic`` / ``.payload`` / ``.source``); wildcards work, so the
    tap listens on ``app/event-bridge/#``
  * publish hygiene: a 2 fps stream must not become a 2 Hz event fire
    hose — a cooldown publishes at most one event per window, and
    frames with no objects publish nothing at all
  * app.yaml ``permissions.events.publish`` — topics are allow-listed;
    publishing to an undeclared topic is rejected at runtime

Long-running (no exit on first result): SIGTERM/SIGINT stop cleanly.
Watch it from ANY other app: subscribe to "app/event-bridge/#".
"""

import logging
import os
import signal
import sys
import threading
import time

from neoruntime_ipc_sdk import EventClient, InferenceClient

STREAM_ID = os.environ.get("STREAM_ID", "main")
# Injected from app.yaml spec.models (key "detector"), like rung 03.
MODEL_ID = os.environ.get("AIPC_MODEL_detector", "yolov8n")
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.3"))
# At most one published event per window (seconds).
COOLDOWN_S = float(os.environ.get("EVENT_COOLDOWN_S", "2"))
PUBLISH_TOPIC = "app/event-bridge/detections"
TAP_TOPIC = "app/event-bridge/#"

EXIT_OK = 0

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="[%(levelname)s] %(message)s")
logger = logging.getLogger("event-bridge")


class Cooldown:
    """Allow at most one ``allow()`` per window; the first call passes."""

    def __init__(self, window_s):
        self.window_s = window_s
        self._last = None

    def allow(self, now):
        if self._last is None or now - self._last >= self.window_s:
            self._last = now
            return True
        return False


def event_payload(sequence, result):
    """Kept labels + count for one result — the event's payload dict."""
    labels = sorted(obj.label for obj in result.objects
                    if obj.score >= MIN_SCORE)
    return {"frame": sequence, "objects": len(labels), "labels": labels}


def maybe_publish(events, cooldown, sequence, result, now):
    """Publish one event if there are objects AND the cooldown allows.

    Frames with no objects return early — they publish nothing and do
    not consume the cooldown window.
    """
    payload = event_payload(sequence, result)
    if not payload["objects"]:
        return None
    if not cooldown.allow(now):
        return None
    events.publish(PUBLISH_TOPIC, payload)
    return payload


def tap_loop(events):
    """Daemon: log every event that comes back on TAP_TOPIC.

    Ends when the shared client is closed (that unblocks the iterator);
    a dead tap must never take the bridge down with it.
    """
    try:
        for event in events.subscribe(TAP_TOPIC):
            logger.info("bus: %s <- %s %s",
                        event.topic, event.source, event.payload)
    except Exception as exc:
        logger.warning("event tap ended: %r", exc)


class Bridge(threading.Thread):
    """subscribe -> maybe_publish, until stop() closes the clients."""

    def __init__(self):
        super().__init__(daemon=True, name="event-bridge")
        self._inference = None
        self._events = None

    def stop(self):
        # Closing the clients unblocks both iterators (results + tap).
        for client in (self._inference, self._events):
            if client is not None:
                try:
                    client.close()
                except Exception:
                    logger.exception("client close during stop")

    def run(self):
        self._events = EventClient()
        self._inference = InferenceClient()
        tap = threading.Thread(target=tap_loop, args=(self._events,),
                               daemon=True, name="event-tap")
        tap.start()
        cooldown = Cooldown(COOLDOWN_S)
        logger.info("bridging stream=%s model=%s -> %s (cooldown %.1fs)",
                    STREAM_ID, MODEL_ID, PUBLISH_TOPIC, COOLDOWN_S)
        try:
            # fps<=2: an event bridge does not need more than two looks/s.
            for sequence, result in self._inference.subscribe(
                    stream=STREAM_ID, model=MODEL_ID, fps=2):
                payload = maybe_publish(
                    self._events, cooldown, sequence, result,
                    time.monotonic())
                if payload:
                    logger.info("published %s", payload)
        except Exception:
            logger.exception("bridge loop died")
        finally:
            self._inference.close()
            self._events.close()


def main():
    bridge = Bridge()
    stopping = threading.Event()

    def _stop(signum, _frame):
        logger.info("signal %d -> stopping", signum)
        stopping.set()
        bridge.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    bridge.start()
    try:
        while not stopping.is_set():
            stopping.wait(1.0)
            if not bridge.is_alive():
                break  # loop died on its own; do not spin forever
    finally:
        bridge.stop()
        bridge.join(timeout=3.0)
    logger.info("bye")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
