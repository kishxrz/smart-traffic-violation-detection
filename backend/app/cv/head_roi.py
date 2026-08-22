"""
backend/app/cv/head_roi.py
────────────────────────────
Head / Rider Region of Interest (ROI) extraction utilities.

Design:
  Extracts the upper portion (head/helmet region) of an associated rider's bounding box
  with safety padding to account for camera angles, posture, and helmet height.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from app.schemas.detection import BoundingBox

logger = logging.getLogger(__name__)


def extract_head_crop(
    frame: np.ndarray,
    rider_bbox: BoundingBox,
    top_fraction: float = 0.35,
    padding_fraction: float = 0.10,
    min_size_px: Tuple[int, int] = (24, 24),
) -> Optional[np.ndarray]:
    """
    Extract the head/helmet ROI crop from an image frame for a given rider bounding box.

    Args:
        frame:            BGR numpy array (H, W, 3).
        rider_bbox:       Rider BoundingBox (x1, y1, x2, y2).
        top_fraction:     Fraction of person height representing head (default 0.35).
        padding_fraction: Extra padding added to head ROI (default 0.10).
        min_size_px:      Minimum (width, height) in pixels to accept.

    Returns:
        Cropped BGR numpy array or None if invalid / out of bounds.
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = rider_bbox.to_xyxy()

    person_w = x2 - x1
    person_h = y2 - y1

    if person_w <= 0 or person_h <= 0:
        return None

    # Head height estimation
    head_h = int(person_h * top_fraction)
    pad_w = int(person_w * padding_fraction)
    pad_h = int(head_h * padding_fraction)

    # Compute crop coordinates with padding
    crop_x1 = max(0, x1 - pad_w)
    crop_y1 = max(0, y1 - pad_h)
    crop_x2 = min(w, x2 + pad_w)
    crop_y2 = min(h, y1 + head_h + pad_h)

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    if crop_w < min_size_px[0] or crop_h < min_size_px[1]:
        return None

    return frame[crop_y1:crop_y2, crop_x1:crop_x2]
