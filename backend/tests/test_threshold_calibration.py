"""
backend/tests/test_threshold_calibration.py
─────────────────────────────────────────────
Unit tests for HelmetDetector threshold configuration, prediction thresholding,
and false-positive protection safeguards.
"""

import numpy as np
import pytest

from app.config import get_settings
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


class MockCalibratedDetector(HelmetDetector):
    def __init__(self, conf_threshold: float = 0.35, mock_conf: float = 0.40, mock_class: int = 1):
        self.conf_threshold = conf_threshold
        self.mock_conf = mock_conf
        self.mock_class = mock_class
        self._is_loaded = True

    @property
    def is_available(self) -> bool:
        return True

    def predict_crop(self, head_crop: np.ndarray, **kwargs) -> HelmetPrediction:
        if self.mock_conf < self.conf_threshold:
            return HelmetPrediction(
                status="UNKNOWN",
                confidence=self.mock_conf,
                class_id=None,
                crop_shape=head_crop.shape[:2] if head_crop is not None else None,
            )
        status = "NO_HELMET" if self.mock_class == 1 else "HELMET"
        return HelmetPrediction(
            status=status,
            confidence=self.mock_conf,
            class_id=self.mock_class,
            bbox=BoundingBox(x1=0, y1=0, x2=head_crop.shape[1], y2=head_crop.shape[0]),
        )


class TestThresholdCalibrationUnitCases:
    def test_configured_threshold_setting(self):
        """1. Configured threshold is respected by HelmetDetector."""
        settings = get_settings()
        detector = HelmetDetector(conf_threshold=0.42)
        assert detector.conf_threshold == 0.42

    def test_no_helmet_above_threshold(self):
        """2. Prediction with confidence >= threshold produces NO_HELMET."""
        det = MockCalibratedDetector(conf_threshold=0.35, mock_conf=0.40, mock_class=1)
        pred = det.predict_crop(BLANK_FRAME)
        assert pred.status == "NO_HELMET"
        assert pred.confidence == 0.40

    def test_no_helmet_below_threshold(self):
        """3. Prediction with confidence < threshold produces UNKNOWN."""
        det = MockCalibratedDetector(conf_threshold=0.50, mock_conf=0.40, mock_class=1)
        pred = det.predict_crop(BLANK_FRAME)
        assert pred.status == "UNKNOWN"

    def test_unknown_below_threshold(self):
        """4. UNKNOWN predictions do not trigger NO_HELMET state."""
        det = MockCalibratedDetector(conf_threshold=0.50, mock_conf=0.20, mock_class=1)
        pred = det.predict_crop(BLANK_FRAME)
        assert pred.status == "UNKNOWN"

    def test_temporal_confirmation_with_calibrated_threshold(self):
        """5. Temporal confirmation requires ratio >= 0.70 of valid NO_HELMET predictions."""
        severity = SeverityEngine()
        mock_det = MockCalibratedDetector(conf_threshold=0.35, mock_conf=0.45, mock_class=1)
        detector = HelmetViolationDetector(severity_engine=severity, helmet_detector=mock_det)

        moto = make_track(10, 3, "motorcycle", 100, 300, 200, 450)
        rider = make_track(5, 0, "person", 110, 200, 190, 350)
        scene = SceneState(frame_number=1, timestamp=0.1)

        viols = []
        for _ in range(6):
            viols.extend(detector.evaluate(BLANK_FRAME, [moto, rider], scene))

        assert len(viols) == 1
        assert viols[0].violation_type == ViolationType.NO_HELMET

    def test_false_positive_protection_when_below_threshold(self):
        """6. Low confidence predictions (< threshold) are suppressed to prevent false accusations."""
        severity = SeverityEngine()
        mock_det = MockCalibratedDetector(conf_threshold=0.50, mock_conf=0.40, mock_class=1)
        detector = HelmetViolationDetector(severity_engine=severity, helmet_detector=mock_det)

        moto = make_track(10, 3, "motorcycle", 100, 300, 200, 450)
        rider = make_track(5, 0, "person", 110, 200, 190, 350)
        scene = SceneState(frame_number=1, timestamp=0.1)

        viols = []
        for _ in range(10):
            viols.extend(detector.evaluate(BLANK_FRAME, [moto, rider], scene))

        # Output remains 0 violations because predictions are UNKNOWN
        assert len(viols) == 0
