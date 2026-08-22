"""
backend/tests/test_wrong_way.py
─────────────────────────────────
Regression tests for WrongWayViolationDetector.

Uses synthetic TrackedObject instances — no video, GPU, or camera required.

Tests cover:
  1. Insufficient trajectory history (< min_frames_tracked or min_trajectory_points)
  2. Tiny displacement (under min_displacement_px)
  3. Stationary vehicle with bounding box jitter
  4. Vehicle moving in correct legal direction
  5. Vehicle moving in wrong direction (sustained)
  6. Temporary single-step direction jitter (should NOT trigger)
  7. Explicit direction vector configurations (e.g. (1.0, 0.0) vs (-1.0, 0.0))
  8. Violation cooldown de-duplication
  9. Explainable severity level, numeric score, and reasons calculation
"""

import numpy as np
import pytest

from app.schemas.detection import BoundingBox, TrackedObject
from app.schemas.violation import SceneState, ViolationType, ViolationSeverity
from app.violations.wrong_way import WrongWayViolationDetector
from app.violations.severity import SeverityEngine

BLANK_FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


def make_mock_track(
    track_id: int,
    trajectory: list,
    frames_tracked: int = None,
    confidence: float = 0.88,
    class_id: int = 2,  # car
    class_name: str = "car",
    frame_number: int = 20,
) -> TrackedObject:
    if frames_tracked is None:
        frames_tracked = len(trajectory)

    curr_x, curr_y = trajectory[-1]
    bbox = BoundingBox(
        x1=curr_x - 25,
        y1=curr_y - 20,
        x2=curr_x + 25,
        y2=curr_y + 20,
    )

    return TrackedObject(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        frame_number=frame_number,
        timestamp=frame_number / 25.0,
        trajectory=trajectory,
        frames_tracked=frames_tracked,
        first_seen_frame=0,
        last_seen_frame=frame_number,
    )


class TestWrongWayRegression:
    def setup_method(self):
        self.severity = SeverityEngine()
        self.detector = WrongWayViolationDetector(
            severity_engine=self.severity,
            expected_direction="RIGHT",  # Moving right (increasing X) is legal
            min_displacement_px=30.0,
            min_frames_tracked=15,
            min_trajectory_points=10,
            sustained_ratio_threshold=0.70,
            tolerance_angle_degrees=120.0,
            cooldown_seconds=5.0,
        )

    def _scene(self, frame_num: int = 20) -> SceneState:
        return SceneState(
            frame_number=frame_num,
            timestamp=frame_num / 25.0,
            expected_direction="RIGHT",
        )

    def test_insufficient_trajectory_history(self):
        """Vehicle tracked for only 5 frames (< 15) must NOT trigger a violation."""
        trajectory = [(500 - i * 5, 300) for i in range(5)]  # Moving left (wrong way)
        obj = make_mock_track(track_id=1, trajectory=trajectory, frames_tracked=5)

        violations = self.detector.evaluate(BLANK_FRAME, [obj], self._scene())
        assert len(violations) == 0

    def test_tiny_displacement(self):
        """Vehicle with net displacement 10px (< 30px) must NOT trigger a violation."""
        # 15 frames, but moving only 0.7px per frame -> total 10.5px displacement
        trajectory = [(500 - i * 0.7, 300) for i in range(15)]
        obj = make_mock_track(track_id=1, trajectory=trajectory, frames_tracked=15)

        violations = self.detector.evaluate(BLANK_FRAME, [obj], self._scene())
        assert len(violations) == 0

    def test_stationary_vehicle_jitter(self):
        """Stationary vehicle jittering +/- 2px around a center point must NOT trigger."""
        base_x, base_y = 400.0, 300.0
        jitter_pattern = [(-1, 1), (2, -1), (-2, 2), (1, -2), (-1, -1)]
        trajectory = []
        for i in range(20):
            jx, jy = jitter_pattern[i % len(jitter_pattern)]
            trajectory.append((base_x + jx, base_y + jy))

        obj = make_mock_track(track_id=1, trajectory=trajectory, frames_tracked=20)
        violations = self.detector.evaluate(BLANK_FRAME, [obj], self._scene())
        assert len(violations) == 0

    def test_correct_direction(self):
        """Vehicle moving RIGHT (increasing X) when expected is RIGHT -> NO violation."""
        trajectory = [(100 + i * 5, 300) for i in range(15)]  # Net disp = 70px RIGHT
        obj = make_mock_track(track_id=1, trajectory=trajectory, frames_tracked=15)

        violations = self.detector.evaluate(BLANK_FRAME, [obj], self._scene())
        assert len(violations) == 0

    def test_sustained_wrong_direction(self):
        """Vehicle moving LEFT (decreasing X) consistently over 15 frames -> Violation!"""
        trajectory = [(500 - i * 5, 300) for i in range(15)]  # Net disp = 70px LEFT
        obj = make_mock_track(track_id=1, trajectory=trajectory, frames_tracked=15)

        violations = self.detector.evaluate(BLANK_FRAME, [obj], self._scene())
        assert len(violations) == 1

        v = violations[0]
        assert v.violation_type == ViolationType.WRONG_WAY
        assert v.vehicle_id == 1
        assert v.violation_confidence > 0.70
        assert v.severity_score >= 65
        assert len(v.severity_reasons) >= 2
        assert "Base WRONG_WAY severity" in v.severity_reasons[0]

    def test_direction_jitter_single_step(self):
        """Vehicle mostly moving RIGHT (legal), but has 2 noisy steps left -> NO violation."""
        # 12 steps right, 2 steps left
        trajectory = [(100 + i * 5, 300) for i in range(10)]
        trajectory += [(trajectory[-1][0] - 3, 300), (trajectory[-1][0] - 2, 300)]
        trajectory += [(trajectory[-1][0] + 5, 300) for _ in range(5)]

        obj = make_mock_track(track_id=1, trajectory=trajectory, frames_tracked=len(trajectory))
        violations = self.detector.evaluate(BLANK_FRAME, [obj], self._scene())
        assert len(violations) == 0

    def test_explicit_direction_vector(self):
        """Test configured expected vector (-1.0, 0.0) -> legal is LEFT."""
        detector = WrongWayViolationDetector(
            severity_engine=self.severity,
            expected_direction=(-1.0, 0.0),  # Legal travel is LEFT
            min_displacement_px=25.0,
            min_frames_tracked=10,
            min_trajectory_points=8,
            cooldown_seconds=0.0,
        )

        # Vehicle moving RIGHT (increasing X) -> WRONG WAY when legal is LEFT
        trajectory = [(100 + i * 5, 300) for i in range(15)]
        obj = make_mock_track(track_id=1, trajectory=trajectory, frames_tracked=15)

        violations = detector.evaluate(
            BLANK_FRAME,
            [obj],
            SceneState(frame_number=20, timestamp=0.8, expected_direction=(-1.0, 0.0)),
        )
        assert len(violations) == 1
        assert violations[0].violation_type == ViolationType.WRONG_WAY

    def test_violation_cooldown(self):
        """Second call within cooldown period must be suppressed."""
        trajectory = [(500 - i * 5, 300) for i in range(15)]
        obj = make_mock_track(track_id=1, trajectory=trajectory, frames_tracked=15)

        # First evaluation -> violation
        v1 = self.detector.evaluate(BLANK_FRAME, [obj], self._scene(frame_num=20))
        assert len(v1) == 1

        # Immediate second evaluation -> cooldown suppressed
        v2 = self.detector.evaluate(BLANK_FRAME, [obj], self._scene(frame_num=21))
        assert len(v2) == 0

    def test_severity_explanation(self):
        """Verify severity calculation returns score, level, and reasons."""
        level, score, reasons = self.severity.calculate_detailed(
            violation_type=ViolationType.WRONG_WAY,
            vehicle_id=10,
            detection_confidence=0.92,
            violation_confidence=0.88,
            displacement_px=65.0,
            sustained_frames=18,
        )

        assert isinstance(level, ViolationSeverity)
        assert 0 <= score <= 100
        assert len(reasons) >= 3
        assert any("High violation confidence" in r for r in reasons)
        assert any("Sustained wrong-way movement" in r for r in reasons)
