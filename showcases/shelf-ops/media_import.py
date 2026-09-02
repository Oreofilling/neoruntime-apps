"""Media import helpers — upload validation + image decoding (image-first).

Pure helpers only (no hailo SDK imports) so they unit-test on the host. The
analysis itself lives in app.py::_analyze_import, which mirrors the live
_infer_once/_detect_frame chains on the decoded frame. Video import (server
sampling + per-frame analysis) extends this module later.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
from PIL import Image, ImageOps

# Long-edge cap for decoded imports: 4K-class is plenty for both the 224x224
# CLIP cell crops and the 640-letterboxed detector, and it bounds memory.
MAX_DECODE_EDGE = 3840

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}

# Magic-byte table; a file must match one of these, not just its name.
_IMAGE_MAGIC = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"BM", "bmp"),
)


def sniff_image(raw: bytes) -> str | None:
    """Format name from the leading bytes, or None if no known magic."""
    for magic, name in _IMAGE_MAGIC:
        if raw.startswith(magic):
            return name
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    return None


def validate_image_upload(raw: bytes, filename: str, max_mb: int) -> str:
    """Validate an uploaded image; return its extension or raise ValueError.

    Extension + size + magic bytes are all checked — the filename is client
    data, so the magic table is the authority on what PIL will be asked to
    decode. """
    if not raw:
        raise ValueError("empty upload")
    if len(raw) > max_mb * 1024 * 1024:
        raise ValueError(f"image exceeds {max_mb} MB limit")
    ext = "." + (filename.rsplit(".", 1)[-1].lower()
                 if "." in filename else "")
    if ext not in IMAGE_EXTENSIONS:
        raise ValueError(
            "unsupported image type — use JPEG, PNG, WebP or BMP")
    fmt = sniff_image(raw)
    if fmt is None:
        raise ValueError("not a valid JPEG/PNG/WebP/BMP image")
    return fmt  # sniffed format, not the (client-controlled) filename


def decode_image_bytes(raw: bytes) -> np.ndarray:
    """Decode validated bytes into an RGB ndarray (EXIF-transposed, capped).

    EXIF transpose keeps phone-portrait photos upright before cropping;
    the long-edge cap bounds the crops/NMS work that follows. """
    img = Image.open(io_bytes(raw))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    edge = max(w, h)
    if edge > MAX_DECODE_EDGE:
        scale = MAX_DECODE_EDGE / edge
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                         Image.LANCZOS)
    return np.asarray(img)


def io_bytes(raw: bytes):
    """BytesIO wrapper (kept tiny for test monkeypatching)."""
    import io
    return io.BytesIO(raw)


def jpeg_b64(frame_bgr: np.ndarray, quality: int = 88) -> str:
    """Encode a BGR frame as a base64 JPEG (annotated import result)."""
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def validate_video_upload(raw: bytes, filename: str, max_mb: int) -> str:
    """Validate an uploaded video; return its extension or raise ValueError.

    Same contract as validate_image_upload: extension + size + magic bytes
    all checked. MP4/MOV/M4V are ISO-BMFF containers — every one of them
    carries the 'ftyp' box header at bytes 4:8, which is the authority on
    what cv2.VideoCapture will be asked to open. """
    if not raw:
        raise ValueError("empty upload")
    if len(raw) > max_mb * 1024 * 1024:
        raise ValueError(f"video exceeds {max_mb} MB limit")
    ext = "." + (filename.rsplit(".", 1)[-1].lower()
                 if "." in filename else "")
    if ext not in VIDEO_EXTENSIONS:
        raise ValueError("unsupported video type — use MP4 or MOV")
    if raw[4:8] != b"ftyp":
        raise ValueError("not a valid MP4/MOV video")
    return ext


def sample_positions(total: int, limit: int) -> list[int]:
    """Evenly spaced indices into [0, total) — at most `limit` of them.

    Frame 0 and the last frame are always included; duplicates collapse
    when limit >= total. Used both for the analyzed keyframe indices of a
    video and (recursively) for picking which of those get annotated. """
    if total <= 0 or limit <= 0:
        return []
    n = min(total, limit)
    if n == 1:
        return [0]
    step = (total - 1) / (n - 1)
    return sorted(set(round(i * step) for i in range(n)))
