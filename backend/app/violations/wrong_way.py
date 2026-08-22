"""
backend/app/violations/wrong_way.py
─────────────────────────────────────
Wrong-way driving violation detector.

Algorithm:
  1. Compute movement vector from prev_center → curr_center.
  2. Determine dominant direction (left/right/up/down).
  3. Compare against configured expected direction.
  4. Apply minimum displacement threshold to filter tracker jitter.

Configurable per camera setup:
  - expected_direction: the legal direction of travel for this road segment.
  - min_displacement: minimum pixel movement to consider significant.
  - min_frames_tracked: vehicle must be tracked for N frames before checking
    (new detections may have unreliable initial trajectories).

Why not use angle directly?
  For most straight roads, dominant direction analysis is more robust
  than angle thresholds. Angle-based detection is available via
  geometry.movement_angle_degrees() for diagonal roads.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Set

import numpy as np

from app.schemas.detection import TrackedObject
from app.schemas.violation import SceneState, Violation, ViolationType
from app.violations.base import ViolationDetector
from app.violations.geometry import movement_direction
from app.violations.severity import SeverityEngine

logger = logging.getLogger(__name__)

VEHICLE_CLASS_IDS: Set[int] = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Legal directions and their opposites
OPPOSITE_DIRECTIONS = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}


class WrongWayViolationDetector(ViolationDetector):
    """
    Detects vehicles moving in the wrong direction.
    """

    def __init__(
        self,
        severity_engine: SeverityEngine,
        expected_direction: str = "right",
        min_displacement_px: float = 5.0,
        min_frames_tracked: int = 5,
        cooldown_seconds: float = 4.0,
    ) -> None:
        """
        Args:
            severity_engine:     Shared severity calculator.
            expected_direction:  Legal direction of travel: "left"|"right"|"up"|"down".
            min_displacement_px: Minimum movement to trigger direction check.
                                 Prevents jitter in stopped vehicles from triggering.
            min_frames_tracked:  Skip recently-appeared vehicles whose trajectory
                                 is not yet reliable.
            cooldown_seconds:    Minimum seconds between violation records for
                                 the same vehicle.
        """
        if expected_direction not in OPPOSITE_DIRECTIONS:
            raise ValueError(
                f"expected_direction must be one of {list(OPPOSITE_DIRECTIONS.keys())}"
            )
        self._severity = severity_engine
        self._expected_direction = expected_direction
        self._min_displacement = min_displacement_px
        self._min_frames_tracked = min_frames_tracked
        self._cooldown = cooldown_seconds
        self._last_violation_time: Dict[int, float] = {}

    @property
    def violation_type(self) -> ViolationType:
        return ViolationType.WRONG_WAY

    def evaluate(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        scene_state: SceneState,
    ) -> List[Violation]:
        violations: List[Violation] = []

        expected = scene_state.expected_direction or self._expected_direction

        for obj in tracked_objects:
            if obj.class_id not in VEHICLE_CLASS_IDS:
                continue

            if obj.frames_tracked < self._min_frames_tracked:
                continue

            prev = obj.prev_center
            if prev is None:
                continue

            direction = movement_direction(
                prev, obj.center, min_displacement=self._min_displacement
            )
            if direction is None:
                continue  # Not enough movement

            wrong_way = direction == OPPOSITE_DIRECTIONS.get(expected)
            if not wrong_way:
                continue

            now = time.time()
            if self._is_on_cooldown(obj.track_id, now):
                continue

            self._last_violation_time[obj.track_id] = now
            severity = self._severity.calculate(
                ViolationType.WRONG_WAY, obj.track_id, obj.confidence
            )

            logger.info(
                "WRONG WAY violation: vehicle #%d (%s) moving %s (expected: %s)",
                obj.track_id, obj.class_name, direction, expected,
            )

            violations.append(
                Violation(
                    violation_type=ViolationType.WRONG_WAY,
                    vehicle_id=obj.track_id,
                    vehicle_class=obj.class_name,
                    confidence=obj.confidence,
                    severity=severity,
                    frame_number=scene_state.frame_number,
                    timestamp=scene_state.timestamp,
                    metadata={
                        "detected_direction": direction,
                        "expected_direction": expected,
                        "prev_center": list(prev),
                        "curr_center": list(obj.center),
                        "displacement_px": (
                            (obj.center[0] - prev[0]) ** 2
                            + (obj.center[1] - prev[1]) ** 2
                        )
                        ** 0.5,
                    },
                )
            )

        return violations

    def _is_on_cooldown(self, vehicle_id: int, now: float) -> bool:
        last = self._last_violation_time.get(vehicle_id, 0.0)
        return (now - last) < self._cooldown

    def reset(self) -> None:
        self._last_violation_time.clear()
