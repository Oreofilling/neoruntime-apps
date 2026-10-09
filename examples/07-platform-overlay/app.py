#!/usr/bin/env python3
"""07-platform-overlay — boxes on the stream, zero pixels in your app.

Rung 7 of the examples ladder (01-07). Same A-form subscription as
rungs 03/06, but the ONE new idea runs the other way: you PUSH results
to the platform's overlay and camera-daemon draws the boxes onto the
RTSP/web streams for you. Your app never touches a pixel — contrast
rung 05, where you subscribed to frames and drew them yourself.

    inference.subscribe                overlay.annotate_result
    (results pushed to you)   ------>  (platform draws on every display
                                         stream; ttl fades stale boxes)

What it teaches
  * ``overlay.enable()`` — the global overlay toggle (label /
    confidence / thickness); the console's AI-overlay switch is this
    same call
  * ``overlay.annotate_result(stream_id, result, ttl_ms=...)`` — one
    call per result; boxes land on every stream that displays
    ``stream_id``, not just one
  * ttl semantics: a result stays valid for ``ttl_ms`` then fades —
    set it to a few frame periods so a dropped frame does not leave
    frozen boxes
  * empty results still annotate: an empty result publishes "no
    detections", clearing stale boxes EARLY instead of waiting for ttl
  * app.yaml ``permissions.events.publish: [inference/main]`` —
    annotate rides the event bus on the ``inference/<stream>`` topic

Long-running: SIGTERM/SIGINT stop cleanly (overlay disabled on exit).
"""

import logging
import os
import signal
import sys
import threading

from neoruntime_ipc_sdk import InferenceClient, OverlayClient

STREAM_ID = os.environ.get("STREAM_ID", "main")
# Injected from app.yaml spec.models (key "detector"), like rung 03.
MODEL_ID = os.environ.get("AIPC_MODEL_detector", "yolov8n")
# Boxes are valid this long; ~3x the frame period at fps<=2.
TTL_MS = int(os.environ.get("OVERLAY_TTL_MS", "1500"))

EXIT_OK = 0

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="[%(levelname)s] %(message)s")
logger = logging.getLogger("platform-overlay")


def annotate(overlay, sequence, result):
    """Push one result to the platform overlay; returns the box count.

    Called for EVERY result, empties included: an empty result
    publishes zero detections, which clears stale boxes immediately
    (otherwise they linger until ttl expires).
    """
    overlay.annotate_result(STREAM_ID, result, ttl_ms=TTL_MS)
    count = len(result.objects)
    logger.info("frame %d: %d box(es) -> %s (ttl %dms)",
                sequence, count, STREAM_ID, TTL_MS)
    return count


class Annotator(threading.Thread):
    """subscribe -> annotate, until stop() closes the clients."""

    def __init__(self):
        super().__init__(daemon=True, name="overlay-annotator")
        self._inference = None
        self._overlay = None

    def stop(self):
        # Closing the inference client unblocks the subscribe iterator.
        if self._inference is not None:
            try:
                self._inference.close()
            except Exception:
                logger.exception("inference close during stop")

    def run(self):
        self._overlay = OverlayClient()
        self._inference = InferenceClient()
        self._overlay.enable(show_label=True, show_confidence=True,
                             line_thickness=2)
        logger.info("annotating stream=%s model=%s onto display "
                    "streams (ttl %dms)", STREAM_ID, MODEL_ID, TTL_MS)
        try:
            # fps<=2: overlay ttl covers the gaps; more would not draw
            # better, only re-publish the same boxes.
            for sequence, result in self._inference.subscribe(
                    stream=STREAM_ID, model=MODEL_ID, fps=2):
                annotate(self._overlay, sequence, result)
        except Exception:
            logger.exception("annotator loop died")
        finally:
            try:
                self._overlay.disable()  # take our boxes off the streams
            except Exception:
                logger.exception("overlay disable during exit")
            self._inference.close()
            self._overlay.close()


def main():
    annotator = Annotator()
    stopping = threading.Event()

    def _stop(signum, _frame):
        logger.info("signal %d -> stopping", signum)
        stopping.set()
        annotator.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    annotator.start()
    try:
        while not stopping.is_set():
            stopping.wait(1.0)
            if not annotator.is_alive():
                break  # loop died on its own; do not spin forever
    finally:
        annotator.stop()
        annotator.join(timeout=3.0)
    logger.info("bye")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
