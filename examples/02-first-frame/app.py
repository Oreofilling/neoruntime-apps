#!/usr/bin/env python3
"""02-first-frame — your first camera frame, as a file.

Rung 2 of the examples ladder (01-07). No subscription, no model: grab
ONE frame from a video stream, write it as a JPEG, exit. It is a
one-shot diagnostic — run it to prove that camera-daemon, the stream
id, and the SDK socket all line up before you build anything heavier.

What it teaches
  * ``FdMediaClient``: ``list_streams`` / ``get_stream_info`` /
    ``get_frame`` — the three calls every camera app starts with
  * ``Frame.to_jpeg_bytes()`` — encode without numpy or cv2
  * exit codes as signals: 0 = frame written, 3 = environment problem
    (dead camera, wrong stream id). A one-shot app must not let
    ``restart_policy`` silently retry a dead camera — so this rung's
    app.yaml sets ``restart_policy: never``.
"""

import logging
import os
import sys

from neoruntime_ipc_sdk import FdMediaClient

STREAM_ID = os.environ.get("STREAM_ID", "main")
OUT_PATH = os.environ.get("OUT_PATH", "/app/data/first-frame.jpg")
TIMEOUT_MS = int(os.environ.get("TIMEOUT_MS", "5000"))

EXIT_OK = 0
EXIT_NO_FRAME = 3

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="[%(levelname)s] %(message)s")
logger = logging.getLogger("first-frame")


def grab_frame(media):
    """Describe the stream, then return one Frame (None on any failure)."""
    info = media.get_stream_info(STREAM_ID)
    if info is None:
        logger.error("stream '%s' not found; available: %s",
                     STREAM_ID, media.list_streams())
        return None
    logger.info("stream '%s': %dx%d @ %dfps (%s)",
                STREAM_ID, info.width, info.height, info.fps, info.format)
    return media.get_frame(STREAM_ID, timeout_ms=TIMEOUT_MS)


def main():
    logger.info("requesting one frame from '%s'", STREAM_ID)
    media = FdMediaClient()
    try:
        try:
            frame = grab_frame(media)
        except Exception:
            # A dead socket is an environment problem, not an app bug:
            # report it as such instead of crashing.
            logger.exception("frame grab failed")
            frame = None
        if frame is None:
            logger.error(
                "no frame within %d ms — check camera-daemon and the "
                "stream id; nothing will retry this for you "
                "(restart_policy: never)", TIMEOUT_MS)
            return EXIT_NO_FRAME
        with frame:
            data = frame.to_jpeg_bytes(quality=90)
        directory = os.path.dirname(OUT_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(OUT_PATH, "wb") as fh:
            fh.write(data)
        logger.info("wrote %d bytes -> %s", len(data), OUT_PATH)
        return EXIT_OK
    finally:
        media.close()


if __name__ == "__main__":
    sys.exit(main())
