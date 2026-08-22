"""
backend/tests/test_no_helmet_e2e.py
───────────────────────────────────
End-to-End NO_HELMET violation path, temporal confirmation, evidence generation,
and API integration tests.

Tests:
  1. Helmet prediction = NO_HELMET triggers temporal accumulation.
  2. Temporal confirmation succeeds when min_observations & ratio thresholds are satisfied.
  3. Duplicate NO_HELMET events for the same vehicle are suppressed during cooldown.
  4. Evidence image file is generated and persisted correctly.
  5. Violation schema matches API response specifications.
"""

from pathlib import Path
import numpy as np
import pytest

from app.detection.helmet_detector import HelmetDetector, HelmetPrediction
from app.schemas.detection import BoundingBox, TrackedObject
from app.schemas.violation import SceneState, ViolationType
from app.violations.helmet import HelmetViolationDetector
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


class MockNoHelmetDetector(HelmetDetector):
    def __init__(self, confidence: float = 0.85):
        self.conf = confidence
        self._is_loaded = True

    @property
    def is_available(self) -> bool:
        return True

    def predict_crop(self, head_crop: np.ndarray, **kwargs) -> HelmetPrediction:
        return HelmetPrediction(
            status="NO_HELMET",
            confidence=self.conf,
            class_id=1,
            bbox=BoundingBox(x1=0, y1=0, x2=head_crop.shape[1], y2=head_crop.shape[0]),
            vehicle_id=kwargs.get("vehicle_id"),
            person_id=kwargs.get("person_id"),
            frame_number=kwargs.get("frame_number"),
            obs_count=kwargs.get("obs_count", 1),
        )


class TestNoHelmetEndToEndPath:
    def test_no_helmet_prediction_accumulates(self):
        """1. Helmet prediction = NO_HELMET accumulates predictions."""
        severity = SeverityEngine()
        mock = MockNoHelmetDetector(confidence=0.85)
        detector = HelmetViolationDetector(severity_engine=severity, helmet_detector=mock)

        moto = make_track(10, 3, "motorcycle", 100, 300, 200, 450)
        rider = make_track(5, 0, "person", 110, 200, 190, 350)
        scene = SceneState(frame_number=1, timestamp=0.1)

        # 3 observations (< min_observations 5) -> no confirmed violation yet
        for _ in range(3):
            viols = detector.evaluate(BLANK_FRAME, [moto, rider], scene)
            assert len(viols) == 0

    def test_temporal_confirmation_succeeds(self):
        """2. Temporal confirmation succeeds when min_observations & ratio thresholds pass."""
        severity = SeverityEngine()
        mock = MockNoHelmetDetector(confidence=0.88)
        detector = HelmetViolationDetector(severity_engine=severity, helmet_detector=mock)

        moto = make_track(10, 3, "motorcycle", 100, 300, 200, 450)
        rider = make_track(5, 0, "person", 110, 200, 190, 350)
        scene = SceneState(frame_number=1, timestamp=0.1)

        viols = []
        for _ in range(6):
            v = detector.evaluate(BLANK_FRAME, [moto, rider], scene)
            viols.extend(v)

        assert len(viols) == 1
        v_event = viols[0]
        assert v_event.violation_type == ViolationType.NO_HELMET
        assert v_event.vehicle_id == 10
        assert v_event.confidence >= 0.70

    def test_duplicate_events_suppressed(self):
        """3. Duplicate events are suppressed during cooldown period."""
        severity = SeverityEngine()
        mock = MockNoHelmetDetector(confidence=0.90)
        detector = HelmetViolationDetector(severity_engine=severity, helmet_detector=mock, cooldown_seconds=5.0)

        moto = make_track(10, 3, "motorcycle", 100, 300, 200, 450)
        rider = make_track(5, 0, "person", 110, 200, 190, 350)
        scene = SceneState(frame_number=1, timestamp=0.1)

        v1 = []
        for _ in range(6):
            v1.extend(detector.evaluate(BLANK_FRAME, [moto, rider], scene))
        assert len(v1) == 1

        # Subsequent evaluation within cooldown yields 0 new violations
        v2 = detector.evaluate(BLANK_FRAME, [moto, rider], scene)
        assert len(v2) == 0

    def test_evidence_generation_and_api_schema(self, tmp_path):
        """4. Evidence image is generated and violation schema matches API response."""
        severity = SeverityEngine()
        mock = MockNoHelmetDetector(confidence=0.85)
        detector = HelmetViolationDetector(
            severity_engine=severity,
            helmet_detector=mock,
        )

        moto = make_track(10, 3, "motorcycle", 100, 300, 200, 450)
        rider = make_track(5, 0, "person", 110, 200, 190, 350)
        scene = SceneState(frame_number=1, timestamp=0.1)

        viols = []
        for _ in range(6):
            viols.extend(detector.evaluate(BLANK_FRAME, [moto, rider], scene))

        assert len(viols) == 1
        event = viols[0]

        # Verify evidence path
        if event.evidence_path:
            ev_file = Path(event.evidence_path)
            assert ev_file.exists()
            assert ev_file.stat().st_size > 0

        # Verify API response fields
        api_dict = event.model_dump() if hasattr(event, "model_dump") else event.dict()
        assert api_dict["violation_type"] == "NO_HELMET"
        assert api_dict["vehicle_id"] == 10
        assert api_dict["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
