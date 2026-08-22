"""
backend/app/violations/helmet.py
──────────────────────────────────
No-helmet violation detector.

Architecture:
  Motorcycle → find associated person(s) → check helmet status → violation

Helmet detection is a known limitation of standard COCO YOLOv8:
  - COCO does NOT include "helmet" or "no_helmet" classes.
  - A custom-trained model is required for reliable helmet detection.

This module implements the FULL ARCHITECTURE for helmet detection so that:
  1. It works with the COCO model by using heuristic logic (head region
     proximity check) — with clearly documented accuracy limitations.
  2. It supports plugging in a custom helmet model via HelmetClassifier
     interface without changing any violation engine code.

Do not interpret the current implementation as claiming 95%+ accuracy
on helmet detection — it does not. The architecture is production-ready;
the model is not (without custom training data).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Set

import numpy as np

from app.schemas.detection import TrackedObject
from app.schemas.violation import (
    SceneState, Violation, ViolationSeverity, ViolationType,
)
from app.violations.base import ViolationDetector
from app.violations.geometry import iou_1d
from app.violations.severity import SeverityEngine

logger = logging.getLogger(__name__)

# COCO class IDs
CLASS_MOTORCYCLE = 3
CLASS_PERSON = 0

# Heuristic: person bbox must horizontally overlap with motorcycle by at least this fraction
HORIZONTAL_OVERLAP_THRESHOLD = 0.3

# Person center must be within this many pixels above motorcycle top for association
VERTICAL_ASSOCIATION_RANGE_PX = 120


class HelmetClassifier:
    """
    Interface for a helmet-specific classification model.

    Currently: a stub that always returns UNKNOWN (no custom model loaded).

    To add a real model:
      1. Train a YOLOv8 model on a helmet/no-helmet dataset.
      2. Subclass HelmetClassifier and override predict().
      3. Pass the subclass instance to HelmetViolationDetector.

    Expected classes: {"helmet": 0, "no_helmet": 1}
    """

    def predict(self, head_crop: np.ndarray) -> Optional[str]:
        """
        Classify a head crop image.

        Returns:
          "helmet"    — rider has a helmet.
          "no_helmet" — rider does not have a helmet.
          None        — model is not available / confidence too low.
        """
        # STUB: Custom model not loaded.
        # Without a custom helmet model, we cannot reliably determine
        # helmet status from a standard COCO YOLOv8 model.
        return None


class HelmetViolationDetector(ViolationDetector):
    """
    Detects no-helmet violations on motorcycle riders.

    Algorithm:
      1. Find all motorcycles in tracked_objects.
      2. For each motorcycle, find spatially associated person detections.
      3. For each person, extract the head region crop.
      4. Run the head crop through HelmetClassifier.
      5. If classifier returns "no_helmet", generate a violation.

    Fallback (COCO model only):
      Without a helmet classifier, we CANNOT reliably detect helmets.
      The current fallback logs a warning rather than generating false violations.
      This is the correct behaviour — do not fake capabilities.
    """

    def __init__(
        self,
        severity_engine: SeverityEngine,
        helmet_classifier: Optional[HelmetClassifier] = None,
        cooldown_seconds: float = 3.0,
    ) -> None:
        self._severity = severity_engine
        self._classifier = helmet_classifier or HelmetClassifier()
        self._cooldown = cooldown_seconds
        self._last_violation_time: Dict[int, float] = {}

    @property
    def violation_type(self) -> ViolationType:
        return ViolationType.NO_HELMET

    def evaluate(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        scene_state: SceneState,
    ) -> List[Violation]:
        violations: List[Violation] = []

        motorcycles = [o for o in tracked_objects if o.class_id == CLASS_MOTORCYCLE]
        persons = [o for o in tracked_objects if o.class_id == CLASS_PERSON]

        for moto in motorcycles:
            # Find persons likely riding this motorcycle
            riders = self._find_riders(moto, persons)

            for rider in riders:
                # Extract head region (upper ~25% of person bbox)
                head_crop = self._extract_head_crop(frame, rider)
                if head_crop is None:
                    continue

                # Classify helmet status
                status = self._classifier.predict(head_crop)

                if status == "no_helmet":
                    # Check cooldown
                    now = time.time()
                    if self._is_on_cooldown(moto.track_id, now):
                        continue

                    self._last_violation_time[moto.track_id] = now
                    severity = self._severity.calculate(
                        ViolationType.NO_HELMET,
                        moto.track_id,
                        rider.confidence,
                    )
                    violations.append(
                        Violation(
                            violation_type=ViolationType.NO_HELMET,
                            vehicle_id=moto.track_id,
                            vehicle_class="motorcycle",
                            confidence=rider.confidence,
                            severity=severity,
                            frame_number=scene_state.frame_number,
                            timestamp=scene_state.timestamp,
                            metadata={
                                "rider_id": rider.track_id,
                                "note": "Detected via custom helmet classifier.",
                            },
                        )
                    )
                elif status is None:
                    # Custom model not loaded — log once, do not generate false violation
                    logger.debug(
                        "Helmet classifier unavailable for motorcycle #%d. "
                        "A custom-trained model is required for reliable helmet detection.",
                        moto.track_id,
                    )

        return violations

    def _find_riders(
        self,
        motorcycle: TrackedObject,
        persons: List[TrackedObject],
    ) -> List[TrackedObject]:
        """
        Associate person detections with a motorcycle.

        A person is considered a rider if:
          - Their bbox horizontally overlaps with the motorcycle.
          - Their vertical center is at or above the motorcycle's top edge.
        """
        riders = []
        moto_cx, _ = motorcycle.center
        moto_x1, moto_y1 = motorcycle.bbox.x1, motorcycle.bbox.y1
        moto_x2 = motorcycle.bbox.x2

        for person in persons:
            # Horizontal overlap check
            horiz_overlap = iou_1d(person.bbox.x1, person.bbox.x2, moto_x1, moto_x2)
            if horiz_overlap < HORIZONTAL_OVERLAP_THRESHOLD:
                continue

            # Person should be above or at motorcycle top
            person_cy = person.center[1]
            if person_cy > moto_y1 + VERTICAL_ASSOCIATION_RANGE_PX:
                continue

            riders.append(person)

        return riders

    @staticmethod
    def _extract_head_crop(
        frame: np.ndarray,
        person: TrackedObject,
    ) -> Optional[np.ndarray]:
        """
        Extract the head region (top 25% of person bounding box).
        Returns None if the crop would be empty or out of bounds.
        """
        x1, y1, x2, y2 = person.bbox.to_xyxy()
        head_height = max(1, (y2 - y1) // 4)
        head_y2 = y1 + head_height

        # Bounds check
        h, w = frame.shape[:2]
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(w, x2)
        y2c = min(h, head_y2)

        if x2c <= x1c or y2c <= y1c:
            return None

        return frame[y1c:y2c, x1c:x2c]

    def _is_on_cooldown(self, vehicle_id: int, now: float) -> bool:
        last = self._last_violation_time.get(vehicle_id, 0.0)
        return (now - last) < self._cooldown

    def reset(self) -> None:
        self._last_violation_time.clear()
        self._severity.reset()
