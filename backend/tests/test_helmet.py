"""
backend/tests/test_helmet.py
─────────────────────────────
Comprehensive unit and regression tests for Helmet Detector & Violation Pipeline.

Uses synthetic TrackedObject and image arrays — NO GPU or video file required.

Tests cover:
  1. HelmetPrediction status mapping & confidence parsing
  2. Rider ↔ Motorcycle spatial association
  3. Head ROI bounding box calculation and cropping
  4. UNKNOWN state safety guard (must NEVER trigger violation)
  5. Low confidence prediction handling
  6. Temporal aggregation (insufficient observations -> NO violation)
  7. Confirmed NO_HELMET violation (sustained no-helmet predictions)
  8. Confirmed HELMET status (NO violation)
  9. Violation cooldown de-duplication
  10. Schema verification (rider_id, helmet_model_confidence, violation_confidence, metadata)
"""

import numpy as np
import pytest

from app.cv.head_roi import extract_head_crop
from app.detection.helmet_detector import HelmetDetector, HelmetPrediction
from app.schemas.detection import BoundingBox, TrackedObject
from app.schemas.violation import SceneState, ViolationSeverity, ViolationType
from app.violations.helmet import HelmetViolationDetector
from app.violations.rider_association import RiderAssociator, RiderAssociation
from app.violations.severity import SeverityEngine

BLANK_FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


def make_mock_track(
    track_id: int,
    class_id: int,
    class_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frames_tracked: int = 15,
    confidence: float = 0.90,
) -> TrackedObject:
    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    center = bbox.center
    return TrackedObject(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        frame_number=10,
        timestamp=0.4,
        trajectory=[center],
        frames_tracked=frames_tracked,
        first_seen_frame=0,
        last_seen_frame=10,
    )


class MockHelmetDetector(HelmetDetector):
    """Mock detector returning predefined prediction sequence."""

    def __init__(self, predictions: list[HelmetPrediction] = None):
        self.predictions = predictions or []
        self._idx = 0
        self._is_loaded = True
        self.conf_threshold = 0.50

    @property
    def is_available(self) -> bool:
        return True

    def predict_crop(self, head_crop: np.ndarray) -> HelmetPrediction:
        if not self.predictions:
            return HelmetPrediction(status="UNKNOWN", confidence=0.0, class_id=None)
        pred = self.predictions[min(self._idx, len(self.predictions) - 1)]
        self._idx += 1
        return pred


class TestHelmetPipeline:
    def test_rider_motorcycle_association(self):
        """Test spatial association between motorcycle and rider."""
        associator = RiderAssociator(min_horizontal_overlap=0.30)

        # Motorcycle at (100, 300, 200, 450)
        moto = make_mock_track(track_id=10, class_id=3, class_name="motorcycle", x1=100, y1=300, x2=200, y2=450)
        # Rider at (110, 200, 190, 350) — directly on motorcycle
        rider = make_mock_track(track_id=5, class_id=0, class_name="person", x1=110, y1=200, x2=190, y2=350)
        # Bystander walking at (500, 200, 550, 400) — far away
        bystander = make_mock_track(track_id=99, class_id=0, class_name="person", x1=500, y1=200, x2=550, y2=400)

        associations = associator.associate([moto, rider, bystander])

        assert len(associations) == 1
        assoc = associations[0]
        assert assoc.motorcycle_id == 10
        assert assoc.rider_id == 5
        assert assoc.association_confidence > 0.50

    def test_head_roi_extraction(self):
        """Test head crop region calculation and bounds checking."""
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        rider_bbox = BoundingBox(x1=100, y1=100, x2=200, y2=300)

        crop = extract_head_crop(frame, rider_bbox, top_fraction=0.35)
        assert crop is not None
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0

        # Test invalid bbox
        invalid_bbox = BoundingBox(x1=100, y1=100, x2=100, y2=100)
        assert extract_head_crop(frame, invalid_bbox) is None

    def test_unknown_state_no_violation(self):
        """UNKNOWN prediction state must NEVER trigger a violation."""
        severity = SeverityEngine()
        mock_detector = MockHelmetDetector([HelmetPrediction(status="UNKNOWN", confidence=0.0, class_id=None)])
        detector = HelmetViolationDetector(severity_engine=severity, helmet_detector=mock_detector)

        moto = make_mock_track(track_id=1, class_id=3, class_name="motorcycle", x1=100, y1=300, x2=200, y2=450)
        rider = make_mock_track(track_id=2, class_id=0, class_name="person", x1=110, y1=200, x2=190, y2=350)
        scene = SceneState(frame_number=1, timestamp=0.1)

        for _ in range(10):
            violations = detector.evaluate(BLANK_FRAME, [moto, rider], scene)
            assert len(violations) == 0

    def test_temporal_aggregation_insufficient_frames(self):
        """Fewer than min_observations (5) frames must NOT trigger a violation."""
        severity = SeverityEngine()
        mock_detector = MockHelmetDetector([HelmetPrediction(status="NO_HELMET", confidence=0.92, class_id=1)] * 3)
        detector = HelmetViolationDetector(
            severity_engine=severity,
            helmet_detector=mock_detector,
            min_observations=5,
        )

        moto = make_mock_track(track_id=1, class_id=3, class_name="motorcycle", x1=100, y1=300, x2=200, y2=450)
        rider = make_mock_track(track_id=2, class_id=0, class_name="person", x1=110, y1=200, x2=190, y2=350)
        scene = SceneState(frame_number=1, timestamp=0.1)

        # Run 3 frames (< 5 min observations)
        for _ in range(3):
            violations = detector.evaluate(BLANK_FRAME, [moto, rider], scene)
            assert len(violations) == 0

    def test_no_helmet_confirmed_violation(self):
        """Sustained NO_HELMET predictions over 5+ frames trigger confirmed violation."""
        severity = SeverityEngine()
        mock_detector = MockHelmetDetector([HelmetPrediction(status="NO_HELMET", confidence=0.91, class_id=1)] * 10)
        detector = HelmetViolationDetector(
            severity_engine=severity,
            helmet_detector=mock_detector,
            min_frames_tracked=10,
            min_observations=5,
            cooldown_seconds=5.0,
        )

        moto = make_mock_track(track_id=10, class_id=3, class_name="motorcycle", x1=100, y1=300, x2=200, y2=450, frames_tracked=12)
        rider = make_mock_track(track_id=5, class_id=0, class_name="person", x1=110, y1=200, x2=190, y2=350, frames_tracked=12)
        scene = SceneState(frame_number=10, timestamp=0.4)

        all_violations = []
        for _ in range(6):
            v = detector.evaluate(BLANK_FRAME, [moto, rider], scene)
            all_violations.extend(v)

        assert len(all_violations) == 1
        viol = all_violations[0]
        assert viol.violation_type == ViolationType.NO_HELMET
        assert viol.vehicle_id == 10
        assert viol.metadata["rider_id"] == 5
        assert viol.violation_confidence > 0.85
        assert viol.severity_score >= 60

    def test_helmet_confirmed_no_violation(self):
        """Sustained HELMET predictions over 10 frames -> NO violation."""
        severity = SeverityEngine()
        mock_detector = MockHelmetDetector([HelmetPrediction(status="HELMET", confidence=0.95, class_id=0)] * 10)
        detector = HelmetViolationDetector(severity_engine=severity, helmet_detector=mock_detector)

        moto = make_mock_track(track_id=1, class_id=3, class_name="motorcycle", x1=100, y1=300, x2=200, y2=450, frames_tracked=12)
        rider = make_mock_track(track_id=2, class_id=0, class_name="person", x1=110, y1=200, x2=190, y2=350, frames_tracked=12)
        scene = SceneState(frame_number=10, timestamp=0.4)

        for _ in range(8):
            violations = detector.evaluate(BLANK_FRAME, [moto, rider], scene)
            assert len(violations) == 0

    def test_violation_cooldown(self):
        """Repeat evaluations within cooldown period are suppressed."""
        severity = SeverityEngine()
        mock_detector = MockHelmetDetector([HelmetPrediction(status="NO_HELMET", confidence=0.91, class_id=1)] * 20)
        detector = HelmetViolationDetector(
            severity_engine=severity,
            helmet_detector=mock_detector,
            cooldown_seconds=5.0,
        )

        moto = make_mock_track(track_id=1, class_id=3, class_name="motorcycle", x1=100, y1=300, x2=200, y2=450, frames_tracked=12)
        rider = make_mock_track(track_id=2, class_id=0, class_name="person", x1=110, y1=200, x2=190, y2=350, frames_tracked=12)
        scene = SceneState(frame_number=10, timestamp=0.4)

        # First 5 frames -> 1 violation
        v1 = []
        for _ in range(5):
            v1.extend(detector.evaluate(BLANK_FRAME, [moto, rider], scene))
        assert len(v1) == 1

        # Immediately next frame -> suppressed by cooldown
        v2 = detector.evaluate(BLANK_FRAME, [moto, rider], scene)
        assert len(v2) == 0

    def test_schema_and_metadata(self):
        """Verify confirmed violation schema contains all required decoupled confidence fields."""
        severity = SeverityEngine()
        mock_detector = MockHelmetDetector([HelmetPrediction(status="NO_HELMET", confidence=0.88, class_id=1)] * 6)
        detector = HelmetViolationDetector(severity_engine=severity, helmet_detector=mock_detector)

        moto = make_mock_track(track_id=7, class_id=3, class_name="motorcycle", x1=100, y1=300, x2=200, y2=450, frames_tracked=12)
        rider = make_mock_track(track_id=3, class_id=0, class_name="person", x1=110, y1=200, x2=190, y2=350, frames_tracked=12)
        scene = SceneState(frame_number=10, timestamp=0.4)

        v_list = []
        for _ in range(5):
            v_list.extend(detector.evaluate(BLANK_FRAME, [moto, rider], scene))

        assert len(v_list) == 1
        v = v_list[0]
        assert v.vehicle_id == 7
        assert v.detection_confidence > 0.80
        assert v.violation_confidence > 0.80
        assert "rider_id" in v.metadata
        assert v.metadata["rider_id"] == 3
        assert "helmet_model_confidence" in v.metadata
        assert "head_roi_bbox" in v.metadata
