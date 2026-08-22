"""
backend/app/violations/lane.py
────────────────────────────────
Lane violation detector.

Lane boundaries are represented as polygons.
A lane violation occurs when a vehicle's center crosses from
one lane polygon into a restricted zone (not a valid adjacent lane).

Design:
  - Lanes are defined as PolygonROI objects.
  - Each frame, we determine which lane each vehicle is in.
  - If a vehicle was in Lane 1 and is now in Lane 2 AND that crossing
    is marked as restricted, a violation is recorded.

For v1, any lane boundary crossing triggers a violation if the
`restricted_crossings` set is non-empty. Full lane-change legality
logic (indicator detection, gradual vs abrupt crossing) requires
additional infrastructure.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Set

import numpy as np

from app.cv.roi import PolygonROI, point_in_polygon
from app.schemas.detection import TrackedObject
from app.schemas.violation import SceneState, Violation, ViolationType
from app.violations.base import ViolationDetector
from app.violations.severity import SeverityEngine

logger = logging.getLogger(__name__)

VEHICLE_CLASS_IDS: Set[int] = {2, 3, 5, 7}


class LaneViolationDetector(ViolationDetector):
    """
    Detects vehicles crossing restricted lane boundaries.
    """

    def __init__(
        self,
        severity_engine: SeverityEngine,
        lane_rois: List[PolygonROI],
        restricted_crossings: Optional[Set[tuple]] = None,
        cooldown_seconds: float = 3.0,
    ) -> None:
        """
        Args:
            severity_engine:      Shared severity calculator.
            lane_rois:            Ordered list of lane polygons
                                  (left to right or as appropriate).
            restricted_crossings: Set of (from_lane_idx, to_lane_idx) tuples
                                  that constitute a violation.
                                  None = all boundary crossings are violations.
            cooldown_seconds:     Cooldown per vehicle.
        """
        self._severity = severity_engine
        self._lanes = lane_rois
        self._restricted = restricted_crossings  # None = all crossings
        self._cooldown = cooldown_seconds
        self._vehicle_lane: Dict[int, Optional[int]] = {}   # track_id → lane_idx
        self._last_violation_time: Dict[int, float] = {}

    @property
    def violation_type(self) -> ViolationType:
        return ViolationType.LANE_VIOLATION

    def evaluate(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        scene_state: SceneState,
    ) -> List[Violation]:
        violations: List[Violation] = []

        for obj in tracked_objects:
            if obj.class_id not in VEHICLE_CLASS_IDS:
                continue

            cx, cy = obj.center
            current_lane = self._detect_lane(cx, cy)

            prev_lane = self._vehicle_lane.get(obj.track_id)

            # Update lane memory
            self._vehicle_lane[obj.track_id] = current_lane

            if prev_lane is None or current_lane is None:
                continue

            if prev_lane == current_lane:
                continue  # No lane change

            # Lane change detected
            crossing = (prev_lane, current_lane)
            is_restricted = (
                self._restricted is None or crossing in self._restricted
            )

            if not is_restricted:
                continue

            now = time.time()
            if self._is_on_cooldown(obj.track_id, now):
                continue

            self._last_violation_time[obj.track_id] = now
            severity = self._severity.calculate(
                ViolationType.LANE_VIOLATION, obj.track_id, obj.confidence
            )

            from_label = (
                self._lanes[prev_lane].label if prev_lane < len(self._lanes) else str(prev_lane)
            )
            to_label = (
                self._lanes[current_lane].label if current_lane < len(self._lanes) else str(current_lane)
            )

            logger.info(
                "LANE VIOLATION: vehicle #%d (%s) moved from %s to %s",
                obj.track_id, obj.class_name, from_label, to_label,
            )

            violations.append(
                Violation(
                    violation_type=ViolationType.LANE_VIOLATION,
                    vehicle_id=obj.track_id,
                    vehicle_class=obj.class_name,
                    confidence=obj.confidence,
                    severity=severity,
                    frame_number=scene_state.frame_number,
                    timestamp=scene_state.timestamp,
                    metadata={
                        "from_lane": from_label,
                        "to_lane": to_label,
                        "crossing": list(crossing),
                    },
                )
            )

        return violations

    def _detect_lane(self, cx: float, cy: float) -> Optional[int]:
        """Return the index of the lane polygon containing this point, or None."""
        for i, lane in enumerate(self._lanes):
            if lane.contains_point(cx, cy):
                return i
        return None

    def _is_on_cooldown(self, vehicle_id: int, now: float) -> bool:
        last = self._last_violation_time.get(vehicle_id, 0.0)
        return (now - last) < self._cooldown

    def reset(self) -> None:
        self._vehicle_lane.clear()
        self._last_violation_time.clear()
