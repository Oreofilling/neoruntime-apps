"""Face region cropping from person bounding box.

Simplified approach: use the upper portion of the person bbox as the face
region. For frontal camera angles (gym entrance / equipment-facing), this
is sufficient. Can be upgraded to a dedicated face detection model later.
"""
from __future__ import annotations

import cv2
import numpy as np


def extract_face_crop(bgr: np.ndarray, person_bbox: dict,
                      top_ratio: float = 0.35,
                      min_size: int = 48) -> np.ndarray | None:
    """Crop face region from person bbox upper portion.

    Args:
        bgr: Full frame in BGR format.
        person_bbox: Dict with keys x, y, w, h (normalized 0-1).
        top_ratio: Fraction of bbox height to use as face region (default 0.35).
        min_size: Minimum crop dimension in pixels; return None if smaller.

    Returns:
        Face crop in BGR, or None if crop is too small or out of bounds.
    """
    h_img, w_img = bgr.shape[:2]
    x = int(person_bbox["x"] * w_img)
    y = int(person_bbox["y"] * h_img)
    w = int(person_bbox["w"] * w_img)
    h = int(person_bbox["h"] * h_img)

    # Face region: top portion of person bbox, centered horizontally
    face_h = int(h * top_ratio)
    # Shrink width slightly (face is narrower than shoulders)
    face_w = int(w * 0.7)
    face_x = x + (w - face_w) // 2
    face_y = y

    # Clamp to image bounds
    face_x = max(0, face_x)
    face_y = max(0, face_y)
    face_x2 = min(w_img, face_x + face_w)
    face_y2 = min(h_img, face_y + face_h)

    crop = bgr[face_y:face_y2, face_x:face_x2]
    if crop.shape[0] < min_size or crop.shape[1] < min_size:
        return None
    return crop


def prepare_face_input(crop: np.ndarray, size: tuple[int, int] = (112, 112)) -> np.ndarray:
    """Resize and normalize face crop for embedding model input.

    Args:
        crop: BGR face crop.
        size: Target size (height, width) for the embedding model.

    Returns:
        RGB float32 array of shape (size[0], size[1], 3), normalized to [0, 1].
    """
    resized = cv2.resize(crop, (size[1], size[0]))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0
