#!/usr/bin/env python3
"""04-register-model — own a model's lifecycle at runtime.

Rung 4 of the examples ladder (01-07). Rung 03's ``spec.models`` is the
right default, but apps that swap models per deployment, ship several
of them, or must not clobber an id that is already live use the runtime
API instead. This rung walks the full lifecycle once, then exits:

    [1/4] register_model    — skipped when the id already exists:
                              re-registering breaks the postprocess
                              link of anything using it
    [2/4] list_models       — your id among everything else on the box
    [3/4] one inference     — Preprocessor.from_model() reads the
                              model's input geometry and pixel format,
                              so the app never hardcodes a resize
    [4/4] unregister_model  — only when THIS run did the registering

Nothing works until app.yaml sets ``inference.allow_register_model:
true`` — that permission line IS the lesson of this rung.

Model resolution order (showcase convention): the host model store
first — a device may already carry this exact compile — then the copy
bundled in the image.
"""

import logging
import os
import sys

from neoruntime_ipc_sdk import (
    Config,
    FdMediaClient,
    InferenceClient,
    InferencePipeline,
    Preprocessor,
)

STREAM_ID = os.environ.get("STREAM_ID", "main")
MODEL_ID = os.environ.get("MODEL_ID", "yolov8n-demo")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "detection")
# Host store candidates first, image-bundled copy last. Some devices
# keep a DIFFERENT compile of the same filename under the flat models/
# root — the detection/ copy is the one matching our sha256.
CANDIDATE_PATHS = [
    "/data/aipc/models/detection/yolov8n.hef",
    "/data/aipc/models/yolov8n.hef",
    "/opt/aipc/models/yolov8n.hef",  # bundled in the image
]

EXIT_OK = 0
EXIT_ENV = 3

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="[%(levelname)s] %(message)s")
logger = logging.getLogger("register-model")


def resolve_model_path():
    for path in CANDIDATE_PATHS:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"yolov8n.hef not found in any of: {CANDIDATE_PATHS}")


def register(inference):
    """Register MODEL_ID; True when this call did the registering."""
    if any(m.model_id == MODEL_ID for m in inference.list_models()):
        logger.info("[1/4] '%s' already registered - reusing", MODEL_ID)
        return False
    path = resolve_model_path()
    # model_variant MUST be None here: a bare string gets wrapped as a
    # backend_function and inference fails HAILO_INVALID_OPERATION.
    inference.register_model(
        model_path=path,
        model_id=MODEL_ID,
        model_type=MODEL_TYPE,
        owner_id=Config.get_app_id(),
        model_variant=None,
    )
    logger.info("[1/4] registered '%s' from %s", MODEL_ID, path)
    return True


def infer_once(inference):
    """One frame -> one result through the pipeline; returns object count."""
    pipeline = InferencePipeline(
        client=inference,
        model_id=MODEL_ID,
        preprocessor=Preprocessor.from_model(inference, MODEL_ID),
    )
    media = FdMediaClient()
    try:
        frame = media.get_frame(STREAM_ID, timeout_ms=5000)
        if frame is None:
            raise RuntimeError(f"no frame from stream '{STREAM_ID}'")
        with frame:
            out = pipeline.run(frame)
            try:
                logger.info("[3/4] %d object(s)", len(out.objects))
                for obj in out.objects:
                    logger.info("    %-12s %.2f", obj.label, obj.score)
                return len(out.objects)
            finally:
                out.release()  # good habit even without retain_input
    finally:
        media.close()


def main():
    inference = InferenceClient()
    registered_here = False
    try:
        registered_here = register(inference)
        logger.info("[2/4] models now: %s",
                    [m.model_id for m in inference.list_models()])
        infer_once(inference)
        return EXIT_OK
    except Exception:
        logger.exception("model lifecycle failed — is "
                         "inference.allow_register_model true in app.yaml?")
        return EXIT_ENV
    finally:
        if registered_here:
            inference.unregister_model(MODEL_ID)
            logger.info("[4/4] unregistered '%s'", MODEL_ID)
        inference.close()


if __name__ == "__main__":
    sys.exit(main())
