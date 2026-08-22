"""
backend/app/tracking/tracker.py
─────────────────────────────────
Abstract base class for object trackers.

Why tracking is necessary:
  Object detection identifies WHAT and WHERE in each frame independently.
  It cannot answer "is this Car in frame 100 the same car as in frame 95?"
  Tracking assigns persistent IDs so we can:
    - Measure trajectories (direction of travel)
    - Count unique vehicles (not re-count the same car 1000 times)
    - Detect violations that unfold over time (e.g., crossing a stop line)
    - Associate evidence images with the correct vehicle

The tracker is a separate abstraction so its implementation
can be swapped (e.g., replace IoU tracker with DeepSORT or ByteTrack)
without touching the violation engine or analytics code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.schemas.detection import Detection, TrackedObject


class BaseTracker(ABC):
    """
    Abstract interface for object trackers.

    Input:  List[Detection] from a single frame.
    Output: List[TrackedObject] — same objects with persistent track_ids.
    """

    @abstractmethod
    def update(
        self,
        detections: List[Detection],
        frame_number: int,
        timestamp: float,
    ) -> List[TrackedObject]:
        """
        Update tracker state with new detections.

        Args:
            detections:   Raw detections from the current frame.
            frame_number: Current frame index.
            timestamp:    Seconds since video start.

        Returns:
            List of TrackedObject with assigned track_ids.
            May include previously tracked objects that were not
            detected in this frame (depending on implementation).
        """
        ...

    def reset(self) -> None:
        """Reset all track state. Call between unrelated video sessions."""

    def get_active_tracks(self) -> List[TrackedObject]:
        """Return all currently active (visible) tracked objects."""
        return []
