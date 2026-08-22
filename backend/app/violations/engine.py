"""
backend/app/violations/engine.py
──────────────────────────────────
The Violation Engine — orchestrates all violation detectors.

This is the main entry point for violation detection.
It receives tracked objects + scene state and runs each registered
detector, collecting and deduplicating results.

Architecture:
  ViolationEngine
    ├── HelmetViolationDetector
    ├── RedLightViolationDetector
    ├── WrongWayViolationDetector
    └── LaneViolationDetector

Adding a new violation type:
  1. Create a new class implementing ViolationDetector.
  2. Register it with engine.register(detector).
  That's all — no changes to existing code required.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from app.schemas.detection import TrackedObject
from app.schemas.violation import SceneState, Violation
from app.violations.base import ViolationDetector

logger = logging.getLogger(__name__)


class ViolationEngine:
    """
    Runs all registered violation detectors against each frame.

    Usage:
        engine = ViolationEngine()
        engine.register(HelmetViolationDetector(...))
        engine.register(RedLightViolationDetector(...))

        violations = engine.evaluate(frame, tracked_objects, scene_state)
    """

    def __init__(self) -> None:
        self._detectors: List[ViolationDetector] = []
        self._total_violations = 0

    def register(self, detector: ViolationDetector) -> None:
        """Register a violation detector with the engine."""
        self._detectors.append(detector)
        logger.info("Registered violation detector: %s", detector.__class__.__name__)

    def evaluate(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        scene_state: SceneState,
    ) -> List[Violation]:
        """
        Run all detectors and return the combined violation list.

        Each detector manages its own cooldown state — the engine
        does not deduplicate across detectors.
        """
        all_violations: List[Violation] = []

        for detector in self._detectors:
            try:
                violations = detector.evaluate(frame, tracked_objects, scene_state)
                all_violations.extend(violations)
            except Exception as exc:
                logger.error(
                    "Error in %s.evaluate(): %s",
                    detector.__class__.__name__, exc,
                    exc_info=True,
                )

        if all_violations:
            self._total_violations += len(all_violations)
            logger.info(
                "Frame %d: %d new violations detected (total: %d)",
                scene_state.frame_number,
                len(all_violations),
                self._total_violations,
            )

        return all_violations

    def reset(self) -> None:
        """Reset all detectors and counters for a new session."""
        for detector in self._detectors:
            detector.reset()
        self._total_violations = 0
        logger.debug("ViolationEngine reset.")

    @property
    def registered_detectors(self) -> List[str]:
        return [d.__class__.__name__ for d in self._detectors]
