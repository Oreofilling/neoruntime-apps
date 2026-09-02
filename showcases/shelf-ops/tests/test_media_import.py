"""Unit tests for media_import.py (upload validation + decode helpers)."""
import base64
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from media_import import (
    MAX_DECODE_EDGE,
    decode_image_bytes,
    jpeg_b64,
    sample_positions,
    sniff_image,
    validate_image_upload,
    validate_video_upload,
)


def _png(w=32, h=24, color=(10, 120, 240)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


def _jpeg(w=40, h=30) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 200, 30)).save(buf, "JPEG")
    return buf.getvalue()


# ---- validation -------------------------------------------------------------


def test_validate_accepts_jpeg_and_png() -> None:
    assert validate_image_upload(_jpeg(), "shelf.jpg", 15) == "jpeg"
    assert validate_image_upload(_jpeg(), "shelf.JPEG", 15) == "jpeg"
    assert validate_image_upload(_png(), "shelf.png", 15) == "png"


def test_validate_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):   # empty body
        validate_image_upload(b"", "x.jpg", 15)
    with pytest.raises(ValueError):   # over the cap (tiny limit)
        validate_image_upload(_png() + b"\x00" * 1024, "x.png", 0)
    with pytest.raises(ValueError):   # unknown extension
        validate_image_upload(_png(), "x.gif", 15)
    # magic is the authority: a misnamed file still decodes (PIL ignores the
    # name), and the returned format is the sniffed one, not the filename's
    assert validate_image_upload(_png(), "x.jpg", 15) == "png"
    with pytest.raises(ValueError):   # no known magic at all
        validate_image_upload(b"definitely not an image", "x.jpg", 15)


def test_sniff_image_magic_table() -> None:
    assert sniff_image(_jpeg()) == "jpeg"
    assert sniff_image(_png()) == "png"
    assert sniff_image(b"BM\x00\x00rest") == "bmp"
    assert sniff_image(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "webp"
    assert sniff_image(b"RIFF\x00\x00\x00\x00WAVEfmt ") is None
    assert sniff_image(b"\x00" * 16) is None


# ---- decode -----------------------------------------------------------------


def test_decode_png_roundtrip() -> None:
    rgb = decode_image_bytes(_png(color=(10, 120, 240)))
    assert rgb.shape == (24, 32, 3)
    assert rgb.dtype == np.uint8
    assert tuple(rgb[12, 16]) == (10, 120, 240)  # center pixel survives


def test_decode_garbage_raises() -> None:
    with pytest.raises(Exception):
        decode_image_bytes(b"\xff\xd8\xff" + b"\x01" * 64)  # jpeg magic, junk


def test_decode_caps_long_edge() -> None:
    wide = Image.new("RGB", (MAX_DECODE_EDGE + 800, 100), (5, 5, 5))
    buf = io.BytesIO()
    wide.save(buf, "PNG")
    rgb = decode_image_bytes(buf.getvalue())
    assert max(rgb.shape[:2]) <= MAX_DECODE_EDGE


def test_decode_grayscale_becomes_rgb() -> None:
    buf = io.BytesIO()
    Image.new("L", (20, 10), 128).save(buf, "PNG")
    rgb = decode_image_bytes(buf.getvalue())
    assert rgb.shape == (10, 20, 3)


# ---- jpeg_b64 ---------------------------------------------------------------


def test_jpeg_b64_encodes_decodable_image() -> None:
    frame = np.zeros((60, 80, 3), np.uint8)
    frame[:, :40] = (0, 255, 0)
    payload = jpeg_b64(frame)
    raw = base64.b64decode(payload)
    assert raw[:3] == b"\xff\xd8\xff"      # JPEG magic
    back = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert back.shape == (60, 80, 3)
    assert back[30, 5, 1] > 200            # green half stays green-ish


# ---- video upload validation -------------------------------------------------

# every ISO-BMFF container (mp4/mov/m4v) starts with a length-prefenced
# 'ftyp' box at offset 4 — enough bytes to exercise the magic check
_MP4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 64


def test_validate_video_accepts_known_containers() -> None:
    assert validate_video_upload(_MP4, "clip.mp4", 200) == ".mp4"
    assert validate_video_upload(_MP4, "clip.MOV", 200) == ".mov"
    assert validate_video_upload(_MP4, "clip.m4v", 200) == ".m4v"


def test_validate_video_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):   # empty body
        validate_video_upload(b"", "x.mp4", 200)
    with pytest.raises(ValueError):   # over the cap (tiny limit)
        validate_video_upload(_MP4, "x.mp4", 0)
    with pytest.raises(ValueError):   # unsupported extension
        validate_video_upload(_MP4, "x.avi", 200)
    with pytest.raises(ValueError):   # no ftyp box where ISO-BMFF needs it
        validate_video_upload(b"\x00" * 64, "x.mp4", 200)


# ---- sample_positions --------------------------------------------------------


def test_sample_positions_even_and_bounded() -> None:
    # demo.mp4 shape: 367 frames capped at 48 -> endpoints kept, evenly spread
    pos = sample_positions(367, 48)
    assert len(pos) == 48 and pos[0] == 0 and pos[-1] == 366
    assert pos == sorted(set(pos))                    # strictly increasing
    assert max(np.diff(pos)) - min(np.diff(pos)) <= 1  # near-even spacing

    # fewer frames than the limit -> every frame, deduped
    assert sample_positions(10, 48) == list(range(10))
    assert sample_positions(1, 6) == [0]
    # keyframe selection is the same math over the sample list
    assert sample_positions(12, 6) == [0, 2, 4, 7, 9, 11]


def test_sample_positions_empty_cases() -> None:
    assert sample_positions(0, 48) == []
    assert sample_positions(367, 0) == []
    assert sample_positions(-1, 6) == []
