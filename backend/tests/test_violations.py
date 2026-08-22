"""
backend/tests/test_violations.py
──────────────────────────────────
Unit tests for violation detectors.

Uses mock frames (numpy zeros) and mock TrackedObjects.
No YOLO model or GPU required.
"""

import numpy as np
import pytest

from app.schemas.detection import BoundingBox, TrackedObject
from app.schemas.violation import (
    SceneState, TrafficLightState, Violation, ViolationType,
)
from app.violations.geometry import crosses_line
from app.violations.severity import SeverityEngine
from app.violations.red_light import RedLightViolationDetector
from app.violations.wrong_way import WrongWayViolationDetector
from app.violations.lane import LaneViolationDetector
from app.cv.roi import PolygonROI, ROIType


def make_tracked(
    track_id: int,
    x1: float, y1: float, x2: float, y2: float,
    class_id: int = 2,  # car
    class_name: str = "car",
    frames_tracked: int = 10,
    trajectory: list = None,
    confidence: float = 0.88,
    frame_number: int = 1,
    timestamp: float = 1.0,
) -> TrackedObject:
    if trajectory is None:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        trajectory = [(cx, cy - 10), (cx, cy)]  # Simple downward movement

    return TrackedObject(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        frame_number=frame_number,
        timestamp=timestamp,
        trajectory=trajectory,
        frames_tracked=frames_tracked,
        first_seen_frame=0,
        last_seen_frame=frame_number,
    )


BLANK_FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


class TestRedLightViolationDetector:
    def setup_method(self):
        self.severity = SeverityEngine()
        self.detector = RedLightViolationDetector(
            severity_engine=self.severity,
            stop_line_y=400.0,
            cooldown_seconds=0.0,  # No cooldown in tests
        )

    def _make_scene(self, light: TrafficLightState, frame: int = 1) -> SceneState:
        return SceneState(
            frame_number=frame,
            timestamp=float(frame),
            traffic_light_state=light,
            stop_line_y=400.0,
        )

    def test_no_violation_on_green(self):
        """No violation when light is green, even if vehicle crosses line."""
        obj = make_tracked(
            1, 300, 395, 400, 440,
            trajectory=[(350, 395), (350, 410)],  # Crosses stop line
        )
        scene = self._make_scene(TrafficLightState.GREEN)
        violations = self.detector.evaluate(BLANK_FRAME, [obj], scene)
        assert violations == []

    def test_violation_on_red_crossing(self):
        """Violation when red light and vehicle crosses stop line going down."""
        obj = make_tracked(
            1, 300, 395, 400, 440,
            trajectory=[(350, 395), (350, 410)],  # Crosses 400 downward
        )
        scene = self._make_scene(TrafficLightState.RED)
        violations = self.detector.evaluate(BLANK_FRAME, [obj], scene)
        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.RED_LIGHT
        assert violations[0].vehicle_id == 1

    def test_no_violation_if_not_crossing(self):
        """No violation if vehicle is below stop line but didn't cross it."""
        obj = make_tracked(
            1, 300, 410, 400, 460,
            trajectory=[(350, 415), (350, 420)],  # Both below line, no crossing
        )
        scene = self._make_scene(TrafficLightState.RED)
        violations = self.detector.evaluate(BLANK_FRAME, [obj], scene)
        assert violations == []

    def test_only_vehicles_trigger(self):
        """Persons (class 0) should not trigger red-light violations."""
        obj = make_tracked(
            1, 300, 395, 400, 440,
            class_id=0, class_name="person",
            trajectory=[(350, 395), (350, 410)],
        )
        scene = self._make_scene(TrafficLightState.RED)
        violations = self.detector.evaluate(BLANK_FRAME, [obj], scene)
        assert violations == []


class TestWrongWayViolationDetector:
    def setup_method(self):
        self.severity = SeverityEngine()
        self.detector = WrongWayViolationDetector(
            severity_engine=self.severity,
            expected_direction="right",
            min_displacement_px=3.0,
            min_frames_tracked=3,
            cooldown_seconds=0.0,
        )

    def _scene(self, frame: int = 10) -> SceneState:
        return SceneState(
            frame_number=frame,
            timestamp=float(frame),
            expected_direction="right",
        )

    def test_wrong_way_left(self):
        """Vehicle moving left when expected direction is right → violation."""
        # bbox center = (200+250)/2, (300+400)/2 = (225, 350)
        # trajectory[-2] = (275, 350), trajectory[-1] = (225, 350)
        # movement: dx = 225-275 = -50 → LEFT
        obj = make_tracked(
            1, 200, 300, 250, 400,
            trajectory=[(275, 350), (225, 350)],  # Moving left
            frames_tracked=10,
        )
        violations = self.detector.evaluate(BLANK_FRAME, [obj], self._scene())
        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.WRONG_WAY

    def test_correct_direction_no_violation(self):
        """Vehicle moving right → no violation."""
        obj = make_tracked(
            1, 200, 300, 300, 400,
            trajectory=[(200, 350), (250, 350)],  # Moving right
            frames_tracked=10,
        )
        violations = self.detector.evaluate(BLANK_FRAME, [obj], self._scene())
        assert violations == []

    def test_new_track_ignored(self):
        """Vehicles with few tracked frames should be ignored."""
        obj = make_tracked(
            1, 200, 300, 300, 400,
            trajectory=[(250, 350), (200, 350)],  # Moving left
            frames_tracked=2,  # Below min_frames_tracked=3
        )
        violations = self.detector.evaluate(BLANK_FRAME, [obj], self._scene())
        assert violations == []


class TestSeverityEngine:
    def test_base_severity_wrong_way(self):
        engine = SeverityEngine()
        severity = engine.calculate(ViolationType.WRONG_WAY, vehicle_id=1, confidence=0.9)
        assert severity == "CRITICAL"

    def test_low_confidence_downgrades(self):
        engine = SeverityEngine()
        # NO_HELMET base = HIGH, low confidence → MEDIUM
        severity = engine.calculate(ViolationType.NO_HELMET, vehicle_id=1, confidence=0.4)
        assert severity == "MEDIUM"

    def test_repeat_offender_escalation(self):
        engine = SeverityEngine()
        # Commit RED_LIGHT 4 times — 4th should escalate
        for _ in range(3):
            engine.calculate(ViolationType.RED_LIGHT, vehicle_id=1, confidence=0.9)
        severity = engine.calculate(ViolationType.RED_LIGHT, vehicle_id=1, confidence=0.9)
        # HIGH → CRITICAL after 3 repeats
        assert severity == "CRITICAL"
