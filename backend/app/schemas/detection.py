"""
backend/app/schemas/detection.py
─────────────────────────────────
Pydantic schemas for detection-related data flowing through the system.
These are the canonical data structures — used by the API, violation engine,
and analytics. Never pass raw YOLO Result objects between modules.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
import uuid


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def to_xywh(self) -> Tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.width), int(self.height)


class Detection(BaseModel):
    """
    A single raw detection from the object detector.

    One Detection = one detected object in one frame.
    This is the output of the YOLO inference step before tracking.
    """
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    frame_number: int
    timestamp: float  # seconds since video start

    @property
    def center(self) -> Tuple[float, float]:
        return self.bbox.center


class TrackedObject(BaseModel):
    """
    A detection that has been assigned a persistent tracking ID.

    The tracker upgrades raw Detections into TrackedObjects by
    maintaining identity across frames.
    """
    track_id: int                        # Persistent ID across frames
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    frame_number: int
    timestamp: float

    # Trajectory (last N center points for direction analysis)
    trajectory: List[Tuple[float, float]] = Field(default_factory=list)

    # Lifecycle
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    frames_tracked: int = 0

    @property
    def center(self) -> Tuple[float, float]:
        return self.bbox.center

    @property
    def prev_center(self) -> Optional[Tuple[float, float]]:
        """Center point from the previous frame."""
        if len(self.trajectory) >= 2:
            return self.trajectory[-2]
        return None

    @property
    def movement_vector(self) -> Optional[Tuple[float, float]]:
        """
        (dx, dy) movement since last frame.
        Positive dx = moving right; positive dy = moving down.
        """
        if self.prev_center is None:
            return None
        cx, cy = self.center
        px, py = self.prev_center
        return (cx - px, cy - py)


class FrameResult(BaseModel):
    """All detections and tracked objects produced for a single frame."""
    frame_number: int
    timestamp: float
    detections: List[Detection] = Field(default_factory=list)
    tracked_objects: List[TrackedObject] = Field(default_factory=list)
    processing_time_ms: float = 0.0

    @property
    def detection_count(self) -> int:
        return len(self.detections)
