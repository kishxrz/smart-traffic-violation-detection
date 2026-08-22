"""
backend/tests/test_helmet_pipeline.py
──────────────────────────────────────
Pipeline telemetry & coverage tests for Helmet Detection Pipeline.

Tests:
  1. Zero motorcycles scenario (produces non-meaningful diagnostic warning).
  2. Motorcycles present but no rider associated.
  3. Rider association with HELMET prediction.
  4. Rider association with NO_HELMET prediction.
  5. UNKNOWN predictions handling.
  6. Temporal confirmation threshold.
  7. Duplicate evidence cooldown.
"""

from pathlib import Path
import numpy as np
import pytest

from app.cv.head_roi import extract_head_crop
from app.detection.helmet_detector import HelmetDetector, HelmetPrediction
from app.schemas.detection import BoundingBox, TrackedObject
from app.schemas.violation import SceneState, ViolationType
from app.violations.helmet import HelmetViolationDetector
from app.violations.rider_association import RiderAssociator
from app.violations.severity import SeverityEngine

BLANK_FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


def make_track(track_id: int, class_id: int, class_name: str, x1: float, y1: float, x2: float, y2: float) -> TrackedObject:
    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    return TrackedObject(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=0.90,
        bbox=bbox,
        frame_number=10,
        timestamp=0.4,
        trajectory=[bbox.center],
        frames_tracked=15,
        first_seen_frame=0,
        last_seen_frame=10,
    )


class MockDetector(HelmetDetector):
    def __init__(self, pred: HelmetPrediction):
        self.pred = pred
        self._is_loaded = True

    @property
    def is_available(self) -> bool:
        return True

    def predict_crop(self, head_crop: np.ndarray, **kwargs) -> HelmetPrediction:
        return self.pred


class TestHelmetPipelineTelemetry:
    def test_zero_motorcycles_scenario(self):
        """Zero motorcycles tracked yields zero helmet crops & zero violations."""
        severity = SeverityEngine()
        mock = MockDetector(HelmetPrediction(status="NO_HELMET", confidence=0.9, class_id=1))
        detector = HelmetViolationDetector(severity_engine=severity, helmet_detector=mock)

        # Only cars in frame
        car = make_track(1, 2, "car", 100, 100, 300, 300)
        scene = SceneState(frame_number=1, timestamp=0.1)

        viols = detector.evaluate(BLANK_FRAME, [car], scene)
        assert len(viols) == 0

    def test_motorcycle_no_rider(self):
        """Motorcycle present without rider produces zero helmet crops & zero violations."""
        severity = SeverityEngine()
        mock = MockDetector(HelmetPrediction(status="NO_HELMET", confidence=0.9, class_id=1))
        detector = HelmetViolationDetector(severity_engine=severity, helmet_detector=mock)

        moto = make_track(10, 3, "motorcycle", 100, 300, 200, 450)
        scene = SceneState(frame_number=1, timestamp=0.1)

        viols = detector.evaluate(BLANK_FRAME, [moto], scene)
        assert len(viols) == 0

    def test_rider_association_helmet(self):
        """Rider with HELMET prediction produces zero violations."""
        severity = SeverityEngine()
        mock = MockDetector(HelmetPrediction(status="HELMET", confidence=0.95, class_id=0))
        detector = HelmetViolationDetector(severity_engine=severity, helmet_detector=mock)

        moto = make_track(10, 3, "motorcycle", 100, 300, 200, 450)
        rider = make_track(5, 0, "person", 110, 200, 190, 350)
        scene = SceneState(frame_number=1, timestamp=0.1)

        for _ in range(8):
            viols = detector.evaluate(BLANK_FRAME, [moto, rider], scene)
            assert len(viols) == 0

    def test_rider_association_no_helmet(self):
        """Rider with sustained NO_HELMET predictions triggers confirmed violation."""
        severity = SeverityEngine()
        mock = MockDetector(HelmetPrediction(status="NO_HELMET", confidence=0.90, class_id=1))
        detector = HelmetViolationDetector(severity_engine=severity, helmet_detector=mock)

        moto = make_track(10, 3, "motorcycle", 100, 300, 200, 450)
        rider = make_track(5, 0, "person", 110, 200, 190, 350)
        scene = SceneState(frame_number=1, timestamp=0.1)

        all_v = []
        for _ in range(6):
            v = detector.evaluate(BLANK_FRAME, [moto, rider], scene)
            all_v.extend(v)

        assert len(all_v) == 1
        assert all_v[0].violation_type == ViolationType.NO_HELMET
        assert all_v[0].vehicle_id == 10

    def test_unknown_predictions(self):
        """UNKNOWN predictions NEVER trigger violations."""
        severity = SeverityEngine()
        mock = MockDetector(HelmetPrediction(status="UNKNOWN", confidence=0.0, class_id=None))
        detector = HelmetViolationDetector(severity_engine=severity, helmet_detector=mock)

        moto = make_track(10, 3, "motorcycle", 100, 300, 200, 450)
        rider = make_track(5, 0, "person", 110, 200, 190, 350)
        scene = SceneState(frame_number=1, timestamp=0.1)

        for _ in range(10):
            viols = detector.evaluate(BLANK_FRAME, [moto, rider], scene)
            assert len(viols) == 0

    def test_duplicate_evidence_cooldown(self):
        """Violations for same motorcycle are suppressed during cooldown period."""
        severity = SeverityEngine()
        mock = MockDetector(HelmetPrediction(status="NO_HELMET", confidence=0.90, class_id=1))
        detector = HelmetViolationDetector(severity_engine=severity, helmet_detector=mock, cooldown_seconds=5.0)

        moto = make_track(10, 3, "motorcycle", 100, 300, 200, 450)
        rider = make_track(5, 0, "person", 110, 200, 190, 350)
        scene = SceneState(frame_number=1, timestamp=0.1)

        v1 = []
        for _ in range(5):
            v1.extend(detector.evaluate(BLANK_FRAME, [moto, rider], scene))
        assert len(v1) == 1

        v2 = detector.evaluate(BLANK_FRAME, [moto, rider], scene)
        assert len(v2) == 0


class TestRiderAssociatorUnitCases:
    def test_person_motorcycle_associated(self):
        """1. Person + motorcycle correctly associated."""
        associator = RiderAssociator(allow_motorcycle_crop_fallback=False)
        moto = make_track(1, 3, "motorcycle", 100, 300, 200, 450)
        rider = make_track(10, 0, "person", 110, 200, 190, 350)

        assocs = associator.associate([moto, rider])
        assert len(assocs) == 1
        assert assocs[0].motorcycle_id == 1
        assert assocs[0].rider_id == 10

    def test_person_outside_motorcycle_rejected(self):
        """2. Person outside motorcycle ROI rejected."""
        associator = RiderAssociator(allow_motorcycle_crop_fallback=False)
        moto = make_track(1, 3, "motorcycle", 100, 300, 200, 450)
        pedestrian = make_track(99, 0, "person", 800, 300, 880, 450)

        assocs = associator.associate([moto, pedestrian])
        assert len(assocs) == 0

    def test_multiple_motorcycles_multiple_riders(self):
        """3. Multiple motorcycles with multiple riders correctly paired."""
        associator = RiderAssociator(allow_motorcycle_crop_fallback=False)
        m1 = make_track(1, 3, "motorcycle", 100, 300, 200, 450)
        r1 = make_track(10, 0, "person", 110, 200, 190, 350)

        m2 = make_track(2, 3, "motorcycle", 500, 300, 600, 450)
        r2 = make_track(20, 0, "person", 510, 200, 590, 350)

        assocs = associator.associate([m1, r1, m2, r2])
        assert len(assocs) == 2
        assoc_map = {a.motorcycle_id: a.rider_id for a in assocs}
        assert assoc_map[1] == 10
        assert assoc_map[2] == 20

    def test_motorcycle_without_visible_rider_fallback(self):
        """4. Motorcycle without separate rider box uses fallback crop."""
        associator = RiderAssociator(allow_motorcycle_crop_fallback=True)
        moto = make_track(1, 3, "motorcycle", 100, 300, 200, 450)

        assocs = associator.associate([moto])
        assert len(assocs) == 1
        assert assocs[0].motorcycle_id == 1
        assert assocs[0].rider_id == 10001
        assert assocs[0].rider_bbox.y2 == 300 + int(150 * 0.65)

    def test_person_without_motorcycle(self):
        """5. Person without motorcycle yields 0 associations."""
        associator = RiderAssociator(allow_motorcycle_crop_fallback=True)
        pedestrian = make_track(5, 0, "person", 100, 200, 180, 350)

        assocs = associator.associate([pedestrian])
        assert len(assocs) == 0

    def test_stable_association_across_frames(self):
        """6. Stable association across consecutive frames."""
        associator = RiderAssociator(allow_motorcycle_crop_fallback=True)

        for frame in range(1, 10):
            moto = make_track(1, 3, "motorcycle", 100 + frame, 300, 200 + frame, 450)
            rider = make_track(10, 0, "person", 110 + frame, 200, 190 + frame, 350)
            assocs = associator.associate([moto, rider])

            assert len(assocs) == 1
            assert assocs[0].motorcycle_id == 1
            assert assocs[0].rider_id == 10

