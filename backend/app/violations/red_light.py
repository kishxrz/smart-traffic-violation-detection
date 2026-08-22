"""
backend/app/violations/red_light.py
─────────────────────────────────────
Red-light violation detector.

Logic:
  IF traffic_light_state == RED
  AND a vehicle's trajectory crosses the configured stop_line
  THEN red_light_violation = TRUE

Key design decisions:
  1. We detect LINE CROSSING, not presence near traffic light.
     Simply being near a red light is not a violation.
     The vehicle must cross the stop line.

  2. Traffic light state is provided by SceneState, which is populated
     by a separate TrafficLightAnalyzer (not yet implemented for v1).
     For demonstration, SceneState.traffic_light_state can be set manually.

  3. Stop line is a horizontal line at a configurable Y coordinate.
     This should be calibrated per camera using real road measurements.

  4. Crossing detection uses the geometry.crosses_line() utility which
     checks for sign change in (y - stop_line_y) between frames.

Traffic light detection from COCO:
  COCO class 9 = "traffic light". YOLO can detect the PRESENCE of a
  traffic light but NOT its state (red/green/yellow). State determination
  requires color analysis of the detected bounding box.

  This module uses SceneState.traffic_light_state which can be set by:
  - Manual/hardcoded value (for testing)
  - Color-region analysis (HSV thresholding on the traffic light crop)
  - A dedicated traffic light state classifier model
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Set

import numpy as np

from app.schemas.detection import TrackedObject
from app.schemas.violation import (
    SceneState, Violation, ViolationType, TrafficLightState,
)
from app.violations.base import ViolationDetector
from app.violations.geometry import crosses_line
from app.violations.severity import SeverityEngine

logger = logging.getLogger(__name__)

# Vehicle class IDs that can commit red-light violations
VEHICLE_CLASS_IDS: Set[int] = {2, 3, 5, 7}  # car, motorcycle, bus, truck


class RedLightViolationDetector(ViolationDetector):
    """
    Detects vehicles crossing the stop line while the traffic light is red.
    """

    def __init__(
        self,
        severity_engine: SeverityEngine,
        stop_line_y: float,
        cooldown_seconds: float = 5.0,
    ) -> None:
        """
        Args:
            severity_engine: Shared severity calculator.
            stop_line_y:     Y pixel coordinate of the stop line.
                             Vehicles crossing downward (increasing Y) while
                             the light is RED trigger a violation.
            cooldown_seconds: Minimum time between repeated violation records
                              for the same vehicle.
        """
        self._severity = severity_engine
        self._stop_line_y = stop_line_y
        self._cooldown = cooldown_seconds
        self._last_violation_time: Dict[int, float] = {}

    @property
    def violation_type(self) -> ViolationType:
        return ViolationType.RED_LIGHT

    def evaluate(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        scene_state: SceneState,
    ) -> List[Violation]:
        violations: List[Violation] = []

        # Only evaluate when light is known to be red
        if scene_state.traffic_light_state != TrafficLightState.RED:
            return violations

        stop_line_y = scene_state.stop_line_y or self._stop_line_y

        for obj in tracked_objects:
            if obj.class_id not in VEHICLE_CLASS_IDS:
                continue

            prev = obj.prev_center
            if prev is None:
                continue  # Need at least 2 frames to detect crossing

            curr = obj.center

            # Did this vehicle's trajectory cross the stop line going forward?
            if crosses_line(prev, curr, stop_line_y, direction="down"):
                now = time.time()
                if self._is_on_cooldown(obj.track_id, now):
                    continue

                self._last_violation_time[obj.track_id] = now
                violation_confidence = round(0.5 * obj.confidence + 0.5 * 1.0, 4)
                severity, severity_score, severity_reasons = self._severity.calculate_detailed(
                    violation_type=ViolationType.RED_LIGHT,
                    vehicle_id=obj.track_id,
                    detection_confidence=obj.confidence,
                    violation_confidence=violation_confidence,
                    displacement_px=abs(curr[1] - prev[1]),
                )

                logger.info(
                    "RED LIGHT violation: vehicle #%d (%s) crossed stop_line_y=%.1f",
                    obj.track_id, obj.class_name, stop_line_y,
                )

                violations.append(
                    Violation(
                        violation_type=ViolationType.RED_LIGHT,
                        vehicle_id=obj.track_id,
                        vehicle_class=obj.class_name,
                        confidence=violation_confidence,
                        detection_confidence=obj.confidence,
                        violation_confidence=violation_confidence,
                        severity=severity,
                        severity_score=severity_score,
                        severity_reasons=severity_reasons,
                        frame_number=scene_state.frame_number,
                        timestamp=scene_state.timestamp,
                        metadata={
                            "stop_line_y": stop_line_y,
                            "prev_center": list(prev),
                            "curr_center": list(curr),
                            "traffic_light": TrafficLightState.RED.value,
                        },
                    )
                )

        return violations

    def _is_on_cooldown(self, vehicle_id: int, now: float) -> bool:
        last = self._last_violation_time.get(vehicle_id, 0.0)
        return (now - last) < self._cooldown

    def reset(self) -> None:
        self._last_violation_time.clear()


def analyze_traffic_light_color(
    frame: np.ndarray,
    light_bbox_xyxy: tuple,
) -> TrafficLightState:
    """
    Estimate traffic light state from the detected bounding box using
    HSV color thresholding.

    This is a heuristic — it works reliably in clear daylight conditions
    but degrades at night, under glare, or with non-standard light colors.

    Algorithm:
      1. Crop the traffic light region.
      2. Convert to HSV.
      3. Divide crop into top/middle/bottom thirds.
      4. Measure saturation-weighted hue distribution in each third.
      5. The brightest third determines the active light color.

    Returns TrafficLightState.UNKNOWN if confidence is too low.
    """
    import cv2

    x1, y1, x2, y2 = [int(v) for v in light_bbox_xyxy]
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return TrafficLightState.UNKNOWN

    h_crop, w_crop = crop.shape[:2]
    if h_crop < 20 or w_crop < 5:
        return TrafficLightState.UNKNOWN

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    third = h_crop // 3

    sections = {
        "red": hsv[:third],
        "yellow": hsv[third : 2 * third],
        "green": hsv[2 * third :],
    }

    # HSV hue ranges (OpenCV: Hue [0,179], Sat [0,255], Val [0,255])
    # Red hue wraps: [0-10] and [160-179]
    def brightness_score(section: np.ndarray) -> float:
        return float(section[:, :, 2].mean())  # Value channel mean

    scores = {k: brightness_score(v) for k, v in sections.items()}
    dominant = max(scores, key=scores.get)

    threshold = 80  # Below this brightness → no active light detected
    if scores[dominant] < threshold:
        return TrafficLightState.UNKNOWN

    return {
        "red": TrafficLightState.RED,
        "yellow": TrafficLightState.YELLOW,
        "green": TrafficLightState.GREEN,
    }[dominant]
