"""
backend/app/violations/wrong_way.py
─────────────────────────────────────
Wrong-way driving violation detector.

Screen Coordinate System Documentation:
  - Origin (0,0) is top-left of the image.
  - X increases to the RIGHT.
  - Y increases DOWNWARDS.

Expected Direction Representation:
  - String names:
      "RIGHT": vector (1.0, 0.0)
      "LEFT" : vector (-1.0, 0.0)
      "DOWN" : vector (0.0, 1.0)
      "UP"   : vector (0.0, -1.0)
  - Custom vector tuple: expected_direction_vector = (ex, ey)

Algorithm & Safeguards:
  1. Filter out non-vehicle classes and recently created tracks (< min_frames_tracked).
  2. Ensure sufficient trajectory history points (>= min_trajectory_points).
  3. Verify net displacement exceeds min_displacement_px to ignore stationary jitter.
  4. Evaluate direction consistency across recent trajectory segments.
  5. Compute cosine of angle between movement vectors and expected direction.
  6. Require sustained wrong-way movement ratio >= sustained_ratio_threshold.
  7. Calculate transparent rule-based violation_confidence and explainable severity.
  8. Apply cooldown de-duplication per vehicle track ID.
"""

from __future__ import annotations

import math
import logging
import time
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from app.schemas.detection import TrackedObject
from app.schemas.violation import SceneState, Violation, ViolationSeverity, ViolationType
from app.violations.base import ViolationDetector
from app.violations.severity import SeverityEngine

logger = logging.getLogger(__name__)

VEHICLE_CLASS_IDS: Set[int] = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Named directions mapped to unit vectors (Screen Coordinates: +X right, +Y down)
DIRECTION_VECTORS: Dict[str, Tuple[float, float]] = {
    "RIGHT": (1.0, 0.0),
    "LEFT": (-1.0, 0.0),
    "DOWN": (0.0, 1.0),
    "UP": (0.0, -1.0),
}


def parse_direction_vector(
    direction: Union[str, Tuple[float, float]]
) -> Tuple[float, float]:
    """Convert string name or tuple to normalized direction vector."""
    if isinstance(direction, str):
        name = direction.upper()
        if name in DIRECTION_VECTORS:
            return DIRECTION_VECTORS[name]
        raise ValueError(f"Unknown direction name '{direction}'. Valid: {list(DIRECTION_VECTORS.keys())}")
    
    vx, vy = float(direction[0]), float(direction[1])
    mag = math.hypot(vx, vy)
    if mag == 0:
        raise ValueError("Expected direction vector cannot be zero length.")
    return (vx / mag, vy / mag)


class WrongWayViolationDetector(ViolationDetector):
    """
    Detects vehicles moving in the wrong direction with trajectory safeguards.
    """

    def __init__(
        self,
        severity_engine: SeverityEngine,
        expected_direction: Union[str, Tuple[float, float]] = "RIGHT",
        min_displacement_px: float = 30.0,
        min_frames_tracked: int = 15,
        min_trajectory_points: int = 10,
        sustained_ratio_threshold: float = 0.70,
        tolerance_angle_degrees: float = 120.0,
        cooldown_seconds: float = 5.0,
    ) -> None:
        """
        Args:
            severity_engine:           Shared severity engine instance.
            expected_direction:        Legal travel direction ("RIGHT", "LEFT", "UP", "DOWN" or (ex, ey)).
            min_displacement_px:       Minimum net pixel displacement over tracking history.
            min_frames_tracked:        Minimum frames tracked before checking (ignores newly created tracks).
            min_trajectory_points:     Minimum trajectory points required.
            sustained_ratio_threshold: Fraction of trajectory steps that must be wrong-way (e.g. 0.70).
            tolerance_angle_degrees:   Angle threshold (e.g. >120° deviation from legal direction = wrong way).
            cooldown_seconds:          Cooldown seconds per vehicle ID to avoid duplicate alerts.
        """
        self._severity = severity_engine
        self._expected_direction_raw = expected_direction
        self._expected_vector = parse_direction_vector(expected_direction)
        self._min_displacement = min_displacement_px
        self._min_frames_tracked = min_frames_tracked
        self._min_trajectory_points = min_trajectory_points
        self._sustained_ratio_threshold = sustained_ratio_threshold
        self._cos_threshold = math.cos(math.radians(tolerance_angle_degrees))
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

        # Allow scene state to override expected direction if configured
        expected_vector = self._expected_vector
        if scene_state.expected_direction:
            try:
                expected_vector = parse_direction_vector(scene_state.expected_direction)
            except Exception as err:
                logger.warning("Invalid scene_state expected direction '%s': %s", scene_state.expected_direction, err)

        ex, ey = expected_vector

        for obj in tracked_objects:
            if obj.class_id not in VEHICLE_CLASS_IDS:
                continue

            # Safeguard 1: Ignore newly created tracks (< min_frames_tracked)
            if obj.frames_tracked < self._min_frames_tracked:
                continue

            # Safeguard 2: Ignore tracks with insufficient trajectory history
            if len(obj.trajectory) < self._min_trajectory_points:
                continue

            # Trajectory window to analyze
            window = obj.trajectory[-self._min_trajectory_points:]
            start_pt = window[0]
            curr_pt = window[-1]

            # Safeguard 3: Minimum net displacement check
            dx = curr_pt[0] - start_pt[0]
            dy = curr_pt[1] - start_pt[1]
            net_displacement = math.hypot(dx, dy)

            if net_displacement < self._min_displacement:
                continue  # Stationary or jittering vehicle

            # Safeguard 4: Evaluate trajectory segment vectors
            wrong_way_steps = 0
            total_steps = 0

            for i in range(1, len(window)):
                p_prev = window[i - 1]
                p_curr = window[i]
                step_dx = p_curr[0] - p_prev[0]
                step_dy = p_curr[1] - p_prev[1]
                step_len = math.hypot(step_dx, step_dy)

                if step_len < 1.5:
                    continue  # Ignore tiny frame-to-frame noise steps

                total_steps += 1
                # Cosine of angle between step vector and legal direction vector
                cos_theta = (step_dx * ex + step_dy * ey) / (step_len * 1.0)
                if cos_theta <= self._cos_threshold or cos_theta < 0:
                    wrong_way_steps += 1

            if total_steps == 0:
                continue

            sustained_ratio = wrong_way_steps / total_steps
            if sustained_ratio < self._sustained_ratio_threshold:
                continue  # Not sustained wrong-way movement

            # Safeguard 5: Cooldown check
            now = time.time()
            if self._is_on_cooldown(obj.track_id, now):
                continue

            # Calculate rule-based violation confidence
            # Formula: 0.3 * detection_conf + 0.3 * min(1.0, net_displacement / (2 * min_disp)) + 0.4 * sustained_ratio
            disp_ratio = min(1.0, net_displacement / (self._min_displacement * 2.0))
            violation_confidence = round(
                0.3 * obj.confidence + 0.3 * disp_ratio + 0.4 * sustained_ratio,
                4,
            )

            # Calculate severity score & explanation
            severity, severity_score, severity_reasons = self._severity.calculate_detailed(
                violation_type=ViolationType.WRONG_WAY,
                vehicle_id=obj.track_id,
                detection_confidence=obj.confidence,
                violation_confidence=violation_confidence,
                displacement_px=net_displacement,
                sustained_frames=len(window),
            )

            self._last_violation_time[obj.track_id] = now

            logger.info(
                "WRONG WAY violation detected: vehicle #%d (%s), net disp=%.1fpx, sustained=%.0f%%, severity=%s (%d pts)",
                obj.track_id, obj.class_name, net_displacement, sustained_ratio * 100, severity.value, severity_score,
            )

            violations.append(
                Violation(
                    violation_type=ViolationType.WRONG_WAY,
                    vehicle_id=obj.track_id,
                    vehicle_class=obj.class_name,
                    confidence=violation_confidence,  # For backward compatibility
                    detection_confidence=obj.confidence,
                    violation_confidence=violation_confidence,
                    severity=severity,
                    severity_score=severity_score,
                    severity_reasons=severity_reasons,
                    frame_number=scene_state.frame_number,
                    timestamp=scene_state.timestamp,
                    metadata={
                        "expected_direction_vector": [ex, ey],
                        "net_displacement_px": round(net_displacement, 2),
                        "sustained_wrong_ratio": round(sustained_ratio, 2),
                        "total_trajectory_points": len(obj.trajectory),
                        "analyzed_window_points": len(window),
                        "start_center": list(start_pt),
                        "curr_center": list(curr_pt),
                    },
                )
            )

        return violations

    def _is_on_cooldown(self, vehicle_id: int, now: float) -> bool:
        last = self._last_violation_time.get(vehicle_id, 0.0)
        return (now - last) < self._cooldown

    def reset(self) -> None:
        self._last_violation_time.clear()
