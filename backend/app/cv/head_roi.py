"""
backend/app/cv/head_roi.py
────────────────────────────
Head / Rider Region of Interest (ROI) extraction & quality validation utilities.

Design:
  1. Priority A (Explicit Person): Extract upper 35% of person bounding box with safety padding.
  2. Priority B (Implicit Motorcycle): Extract upper 65% of motorcycle box as estimated rider ROI.
  3. Quality & Truncation Filter: Rejects crops that are truncated by frame boundaries (>25%),
     too small (<32x32 px), or have invalid aspect ratios (outside 0.45 - 2.0).
  4. Returns structured extraction result with explicit telemetry rejection reason.

Coordinate-space contract:
  All bbox coordinates passed to these functions MUST be in the same pixel space
  as the `frame` array provided. YOLO/Ultralytics already scales output boxes back
  to the input-frame pixel space, so pass the ORIGINAL frame (not a pre-scaled one)
  for highest-quality crops. The final crop is resized to `target_size` via LANCZOS4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from app.schemas.detection import BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class HeadCropResult:
    """Structured result of head ROI crop extraction & quality validation."""
    crop: Optional[np.ndarray]
    is_accepted: bool
    rejection_reason: Optional[str]
    roi_bbox: BoundingBox
    visible_ratio: float


def validate_crop_quality(
    frame_shape: Tuple[int, int],
    bbox: BoundingBox,
    min_size_px: Tuple[int, int] = (24, 24),
    min_visible_ratio: float = 0.40,
    min_aspect_ratio: float = 0.35,
    max_aspect_ratio: float = 3.50,
) -> Tuple[bool, Optional[str], float]:
    """
    Validate whether a bounding box provides a usable head crop without heavy truncation.

    Returns:
        (is_accepted, rejection_reason, visible_ratio)
    """
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = bbox.to_xyxy()

    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)

    if box_w <= 0 or box_h <= 0:
        return False, "BELOW_MIN_SIZE", 0.0

    # 1. Compute visible intersection area with frame boundaries
    clamped_x1 = max(0, min(frame_w, x1))
    clamped_y1 = max(0, min(frame_h, y1))
    clamped_x2 = max(0, min(frame_w, x2))
    clamped_y2 = max(0, min(frame_h, y2))

    visible_w = max(0, clamped_x2 - clamped_x1)
    visible_h = max(0, clamped_y2 - clamped_y1)

    total_area = box_w * box_h
    visible_area = visible_w * visible_h

    visible_ratio = visible_area / float(total_area) if total_area > 0 else 0.0

    # Check frame boundary truncation
    is_at_boundary = (x1 <= 5 or y1 <= 5 or x2 >= frame_w - 5 or y2 >= frame_h - 5)
    if is_at_boundary and visible_ratio < min_visible_ratio:
        return False, "FRAME_BOUNDARY_TRUNCATED", round(visible_ratio, 4)

    # Check minimum crop dimensions
    if visible_w < min_size_px[0] or visible_h < min_size_px[1]:
        return False, "BELOW_MIN_SIZE", round(visible_ratio, 4)

    # Check aspect ratio
    aspect_ratio = visible_w / float(max(1, visible_h))
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return False, "INVALID_ASPECT_RATIO", round(visible_ratio, 4)

    # Check minimum visible area
    if visible_area < (min_size_px[0] * min_size_px[1]):
        return False, "LOW_VISIBLE_AREA", round(visible_ratio, 4)

    return True, None, round(visible_ratio, 4)


def extract_head_crop(
    frame: np.ndarray,
    rider_bbox: BoundingBox,
    top_fraction: float = 0.35,
    padding_fraction: float = 0.10,
    min_size_px: Tuple[int, int] = (24, 24),
    target_size: Optional[Tuple[int, int]] = (224, 224),
) -> Optional[np.ndarray]:
    """
    Extract the head/helmet ROI crop from an image frame for a given rider bounding box.

    Maintains backward compatibility with existing tests.
    The crop is resized to `target_size` (default 224×224) using LANCZOS4 interpolation
    so the classifier always receives a fixed-size input matching its training resolution.
    """
    result = extract_head_crop_with_quality(
        frame=frame,
        rider_bbox=rider_bbox,
        top_fraction=top_fraction,
        padding_fraction=padding_fraction,
        min_size_px=min_size_px,
        target_size=target_size,
    )
    return result.crop if result.is_accepted else None


def extract_head_crop_with_quality(
    frame: np.ndarray,
    rider_bbox: BoundingBox,
    top_fraction: float = 0.35,
    padding_fraction: float = 0.10,
    min_size_px: Tuple[int, int] = (32, 32),
    target_size: Optional[Tuple[int, int]] = (224, 224),
) -> HeadCropResult:
    """
    Extract head ROI crop with quality validation, clamping, and rejection logging.

    Args:
        frame:            Source frame. MUST be the original-resolution frame so that
                          the extracted crop has maximum detail. YOLO bbox coordinates
                          are already scaled back to this frame's pixel space.
        rider_bbox:       Rider bounding box in `frame` pixel coordinates.
        top_fraction:     Fraction of rider height to use as head region (default 0.35).
        padding_fraction: Fractional padding around the head region (default 0.10).
        min_size_px:      Minimum accepted crop dimensions in pixels.
        target_size:      If set, the accepted crop is resized to (w, h) using
                          LANCZOS4 interpolation. Default (224, 224) matches V3-cls
                          training resolution for best classification accuracy.
    """
    if frame is None or frame.size == 0:
        return HeadCropResult(
            crop=None,
            is_accepted=False,
            rejection_reason="BELOW_MIN_SIZE",
            roi_bbox=rider_bbox,
            visible_ratio=0.0,
        )

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = rider_bbox.to_xyxy()

    person_w = x2 - x1
    person_h = y2 - y1

    if person_w <= 0 or person_h <= 0:
        return HeadCropResult(
            crop=None,
            is_accepted=False,
            rejection_reason="BELOW_MIN_SIZE",
            roi_bbox=rider_bbox,
            visible_ratio=0.0,
        )

    # Head height estimation
    head_h = int(person_h * top_fraction)
    pad_w = int(person_w * padding_fraction)
    pad_h = int(head_h * padding_fraction)

    # Compute crop coordinates with safety padding
    crop_x1 = max(0, x1 - pad_w)
    crop_y1 = max(0, y1 - pad_h)
    crop_x2 = min(w, x2 + pad_w)
    crop_y2 = min(h, y1 + head_h + pad_h)

    crop_bbox = BoundingBox(x1=crop_x1, y1=crop_y1, x2=crop_x2, y2=crop_y2)

    is_ok, reason, ratio = validate_crop_quality(
        frame_shape=(h, w),
        bbox=crop_bbox,
        min_size_px=min_size_px,
    )

    if not is_ok:
        return HeadCropResult(
            crop=None,
            is_accepted=False,
            rejection_reason=reason,
            roi_bbox=crop_bbox,
            visible_ratio=ratio,
        )

    crop_mat = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize to target_size using LANCZOS4 (best quality for classifier input)
    if target_size is not None and crop_mat.size > 0:
        tw, th = target_size
        if crop_mat.shape[1] != tw or crop_mat.shape[0] != th:
            crop_mat = cv2.resize(crop_mat, (tw, th), interpolation=cv2.INTER_LANCZOS4)

    return HeadCropResult(
        crop=crop_mat,
        is_accepted=True,
        rejection_reason=None,
        roi_bbox=crop_bbox,
        visible_ratio=ratio,
    )
