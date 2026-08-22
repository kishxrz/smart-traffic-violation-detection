"""
backend/app/violations/base.py
────────────────────────────────
Abstract base class for all violation detectors.

Key architectural point for interviews:
  This is where the distinction between "object detection" and
  "traffic-rule reasoning" becomes explicit.

  YOLO detects objects. The ViolationDetector interprets them.

  A ViolationDetector never calls YOLO directly. It only receives:
    - The current frame (for evidence drawing)
    - Tracked objects (who is where, moving how fast/in which direction)
    - Scene state (traffic light color, stop line position, etc.)

  This separation means:
    - The violation logic can be unit-tested with mock detections.
    - Different detectors can be enabled/disabled at runtime.
    - New violations can be added without touching existing code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from app.schemas.detection import TrackedObject
from app.schemas.violation import SceneState, Violation, ViolationType


class ViolationDetector(ABC):
    """
    Abstract interface for a single type of traffic violation.

    Each violation detector:
      - Is stateful (may remember which vehicles have been flagged).
      - Implements a cooldown to avoid flooding the violation log.
      - Returns Violation objects — the canonical violation schema.
    """

    @property
    @abstractmethod
    def violation_type(self) -> ViolationType:
        """The specific violation this detector handles."""
        ...

    @abstractmethod
    def evaluate(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        scene_state: SceneState,
    ) -> List[Violation]:
        """
        Evaluate the current frame for this violation type.

        Args:
            frame:           Current BGR frame (used for evidence images).
            tracked_objects: All objects tracked in this frame.
            scene_state:     Interpreted traffic scene (light color, etc.).

        Returns:
            List of confirmed Violation objects (usually 0 or 1 per frame
            per vehicle, due to cooldown logic).
        """
        ...

    def reset(self) -> None:
        """Reset per-session state (called between analysis sessions)."""
