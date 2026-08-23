"""
backend/tests/test_head_roi.py
───────────────────────────────
Unit test suite for head ROI extraction & quality validation.

Tests:
  1. Explicit person ROI extraction.
  2. Implicit motorcycle fallback ROI estimation.
  3. Frame-boundary coordinate clamping.
  4. Truncated ROI rejection (boundary overflow).
  5. Tiny ROI rejection (below min dimensions).
  6. Valid centered ROI acceptance.
  7. Invalid aspect ratio rejection.
  8. Multiple motorcycles with distinct ROIs.
  9. Stable ROI extraction across consecutive frames.
"""

import numpy as np
import pytest

from app.cv.head_roi import extract_head_crop_with_quality, validate_crop_quality
from app.schemas.detection import BoundingBox, TrackedObject
from app.violations.rider_association import RiderAssociator

FRAME_H, FRAME_W = 1080, 1920
SAMPLE_FRAME = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


class TestHeadROIExtractionUnitCases:
    def test_explicit_person_roi(self):
        """1. Explicit person ROI derives upper body/head region."""
        person_bbox = BoundingBox(x1=500, y1=200, x2=600, y2=500)
        res = extract_head_crop_with_quality(
            frame=SAMPLE_FRAME,
            rider_bbox=person_bbox,
            top_fraction=0.35,
        )
        assert res.is_accepted is True
        assert res.rejection_reason is None
        assert res.crop is not None
        assert res.crop.shape[0] > 0 and res.crop.shape[1] > 0

    def test_implicit_motorcycle_fallback_roi(self):
        """2. Implicit motorcycle fallback ROI extracts upper 65% of vehicle box."""
        associator = RiderAssociator(allow_motorcycle_crop_fallback=True)
        moto = TrackedObject(
            track_id=1, class_id=3, class_name="motorcycle", confidence=0.9,
            bbox=BoundingBox(x1=400, y1=300, x2=600, y2=700),
            frame_number=1, timestamp=0.1, trajectory=[], frames_tracked=5,
            first_seen_frame=1, last_seen_frame=1,
        )
        assocs = associator.associate([moto])
        assert len(assocs) == 1
        assert assocs[0].rider_bbox.y2 == 300 + int(400 * 0.65)

    def test_frame_boundary_clamping(self):
        """3. Coordinates extending beyond frame boundaries are clamped safely."""
        overflow_bbox = BoundingBox(x1=-20, y1=-10, x2=200, y2=300)
        res = extract_head_crop_with_quality(frame=SAMPLE_FRAME, rider_bbox=overflow_bbox)
        assert res.roi_bbox.x1 >= 0
        assert res.roi_bbox.y1 >= 0

    def test_truncated_roi_rejection(self):
        """4. Bounding boxes with heavy boundary truncation (<0.40 visible) are rejected."""
        # Bounding box mostly off-screen to the right
        boundary_bbox = BoundingBox(x1=1900, y1=500, x2=2300, y2=700)
        is_ok, reason, ratio = validate_crop_quality(
            frame_shape=(FRAME_H, FRAME_W),
            bbox=boundary_bbox,
            min_visible_ratio=0.40,
        )
        assert is_ok is False
        assert reason == "FRAME_BOUNDARY_TRUNCATED"

    def test_tiny_roi_rejection(self):
        """5. Tiny ROIs below (24, 24) are rejected."""
        tiny_bbox = BoundingBox(x1=100, y1=100, x2=110, y2=115)
        res = extract_head_crop_with_quality(
            frame=SAMPLE_FRAME,
            rider_bbox=tiny_bbox,
            min_size_px=(24, 24),
        )
        assert res.is_accepted is False
        assert res.rejection_reason == "BELOW_MIN_SIZE"

    def test_valid_centered_roi(self):
        """6. Centered, well-formed ROI is accepted with high visible ratio."""
        centered_bbox = BoundingBox(x1=500, y1=300, x2=700, y2=600)
        res = extract_head_crop_with_quality(frame=SAMPLE_FRAME, rider_bbox=centered_bbox)
        assert res.is_accepted is True
        assert res.visible_ratio >= 0.95

    def test_invalid_aspect_ratio(self):
        """7. ROIs with extreme aspect ratios (<0.35 or >3.5) are rejected."""
        narrow_bbox = BoundingBox(x1=100, y1=100, x2=125, y2=500)
        is_ok, reason, ratio = validate_crop_quality(
            frame_shape=(FRAME_H, FRAME_W),
            bbox=narrow_bbox,
            min_aspect_ratio=0.35,
            max_aspect_ratio=3.50,
        )
        assert is_ok is False
        assert reason == "INVALID_ASPECT_RATIO"

    def test_multiple_motorcycles(self):
        """8. Multiple motorcycles receive distinct ROI crops."""
        associator = RiderAssociator(allow_motorcycle_crop_fallback=True)
        m1 = TrackedObject(
            track_id=1, class_id=3, class_name="motorcycle", confidence=0.9,
            bbox=BoundingBox(x1=100, y1=200, x2=300, y2=500),
            frame_number=1, timestamp=0.1, trajectory=[], frames_tracked=5,
            first_seen_frame=1, last_seen_frame=1,
        )
        m2 = TrackedObject(
            track_id=2, class_id=3, class_name="motorcycle", confidence=0.9,
            bbox=BoundingBox(x1=600, y1=200, x2=800, y2=500),
            frame_number=1, timestamp=0.1, trajectory=[], frames_tracked=5,
            first_seen_frame=1, last_seen_frame=1,
        )
        assocs = associator.associate([m1, m2])
        assert len(assocs) == 2

        r1 = extract_head_crop_with_quality(SAMPLE_FRAME, assocs[0].rider_bbox)
        r2 = extract_head_crop_with_quality(SAMPLE_FRAME, assocs[1].rider_bbox)
        assert r1.is_accepted and r2.is_accepted

    def test_stable_roi_across_frames(self):
        """9. Stable ROI crop extraction across consecutive frames."""
        for step in range(5):
            bbox = BoundingBox(x1=400 + step * 2, y1=300, x2=600 + step * 2, y2=600)
            res = extract_head_crop_with_quality(SAMPLE_FRAME, bbox)
            assert res.is_accepted is True

    def test_target_size_default_224(self):
        """10. Accepted crops are resized to 224×224 by default (V3-cls training size)."""
        bbox = BoundingBox(x1=500, y1=200, x2=700, y2=600)
        res = extract_head_crop_with_quality(frame=SAMPLE_FRAME, rider_bbox=bbox)
        assert res.is_accepted is True
        assert res.crop is not None
        assert res.crop.shape[:2] == (224, 224), (
            f"Expected crop shape (224, 224) but got {res.crop.shape[:2]}. "
            "Crop must be resized to training resolution for V3-cls."
        )

    def test_target_size_custom(self):
        """11. target_size parameter is respected when set to a custom value."""
        bbox = BoundingBox(x1=500, y1=200, x2=700, y2=600)
        res = extract_head_crop_with_quality(
            frame=SAMPLE_FRAME, rider_bbox=bbox, target_size=(128, 128)
        )
        assert res.is_accepted is True
        assert res.crop.shape[:2] == (128, 128)

    def test_target_size_none_preserves_native(self):
        """12. target_size=None preserves the native crop dimensions (no resize)."""
        # 1920×1080 frame, bbox yielding a ~70×105px crop
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=500, y1=200, x2=600, y2=500)
        res = extract_head_crop_with_quality(
            frame=frame, rider_bbox=bbox, target_size=None
        )
        assert res.is_accepted is True
        # Native crop should NOT be 224×224
        assert res.crop.shape[:2] != (224, 224)

    def test_original_frame_higher_resolution(self):
        """13. Crop from a higher-res original frame is larger before resize than
        a crop from a downscaled frame at the same normalized bbox."""
        # Simulate: detection bbox in 640×640 space (proportional)
        # Original frame: 1920×1080. Scaled bbox maps to larger pixel region.
        original_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        small_frame    = np.random.randint(0, 255, (640,  640,  3), dtype=np.uint8)

        # Same rider at ~25% across, 30% down, size ~15% of frame
        # In 1920×1080: x1=480, y1=324, x2=768, y2=648
        # In 640×640:   x1=160, y1=192, x2=256, y2=384
        bbox_orig  = BoundingBox(x1=480, y1=324, x2=768, y2=648)
        bbox_small = BoundingBox(x1=160, y1=192, x2=256, y2=384)

        res_orig  = extract_head_crop_with_quality(original_frame, bbox_orig,  target_size=None)
        res_small = extract_head_crop_with_quality(small_frame,    bbox_small, target_size=None)

        assert res_orig.is_accepted and res_small.is_accepted
        # The original-res crop should be physically larger
        orig_px  = res_orig.crop.shape[0]  * res_orig.crop.shape[1]
        small_px = res_small.crop.shape[0] * res_small.crop.shape[1]
        assert orig_px > small_px, (
            "Crop from original-resolution frame must contain more pixels than "
            "crop from downscaled frame. Got "
            f"orig={orig_px}px vs small={small_px}px"
        )
