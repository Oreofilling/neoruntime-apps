#!/usr/bin/env python3
"""03-single-inference — your first inference result.

Rung 3 of the examples ladder (01-07). The model is DECLARED in
app.yaml (spec.models): the platform registers it at install time,
authorizes the id for this app, and injects the resolved id as
AIPC_MODEL_detector. The code only subscribes to platform-run stream
inference and reads what comes back — on this path your app never
touches pixels (contrast rung 05, where pixels flow through you).

What it teaches
  * ``spec.models`` in app.yaml vs rung 04's runtime
    ``register_model()`` — declarative is the right default
  * ``inference.subscribe(stream=..., model=..., fps=...)`` — a
    generator of ``(frame_sequence, InferenceResult)``
  * ``DetectedObject.label / .score / .bbox`` — bbox is normalized
    0..1, so it transfers across streams that share the field of view

This rung exits after the FIRST result: proof that stream + model +
permissions all line up, before you commit to a loop (rung 05).
"""

import logging
import os
import sys

from neoruntime_ipc_sdk import InferenceClient

STREAM_ID = os.environ.get("STREAM_ID", "main")
# Injected by the platform from app.yaml spec.models (key "detector").
# The literal default only covers running outside app-manager.
MODEL_ID = os.environ.get("AIPC_MODEL_detector", "yolov8n")
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.3"))

EXIT_OK = 0
EXIT_NO_RESULT = 3

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="[%(levelname)s] %(message)s")
logger = logging.getLogger("single-inference")


def kept_objects(result):
    return [obj for obj in result.objects if obj.score >= MIN_SCORE]


def report(sequence, result):
    """Log one inference result; returns the kept object count."""
    kept = kept_objects(result)
    logger.info("frame %d: %d object(s) over %.2f",
                sequence, len(kept), MIN_SCORE)
    for obj in kept:
        bbox = obj.bbox
        logger.info("  %-12s %.2f  x=%.2f y=%.2f w=%.2f h=%.2f",
                    obj.label, obj.score,
                    bbox.x, bbox.y, bbox.width, bbox.height)
    return len(kept)


def main():
    inference = InferenceClient()
    try:
        models = [m.model_id for m in inference.list_models()]
        logger.info("registered models: %s", models)
        if MODEL_ID not in models:
            logger.warning("model '%s' not registered yet — install may "
                           "have skipped spec.models", MODEL_ID)
        logger.info("subscribing stream=%s model=%s (fps<=2, first "
                    "result only)", STREAM_ID, MODEL_ID)
        for sequence, result in inference.subscribe(
                stream=STREAM_ID, model=MODEL_ID, fps=2):
            report(sequence, result)
            return EXIT_OK
        logger.error("subscription ended without a result")
        return EXIT_NO_RESULT
    except Exception:
        logger.exception("inference failed — check stream id, model id, "
                         "and the video/inference permissions in app.yaml")
        return EXIT_NO_RESULT
    finally:
        inference.close()


if __name__ == "__main__":
    sys.exit(main())
