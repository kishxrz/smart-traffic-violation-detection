"""
backend/tests/test_tracking.py
────────────────────────────────
Unit tests for the IoU-based object tracker.

Uses mock Detection objects — no GPU or video required.
"""

import pytest

from app.schemas.detection import BoundingBox, Detection
from app.tracking.object_tracker import ObjectTracker, compute_iou


def make_detection(
    x1: float, y1: float, x2: float, y2: float,
    class_id: int = 2,
    class_name: str = "car",
    confidence: float = 0.85,
    frame_number: int = 0,
    timestamp: float = 0.0,
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        frame_number=frame_number,
        timestamp=timestamp,
    )


class TestComputeIoU:
    def test_perfect_overlap(self):
        box = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        b = BoundingBox(x1=60, y1=60, x2=110, y2=110)
        assert compute_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        # Intersection: 50x50 = 2500; Union: 10000+10000-2500 = 17500
        assert compute_iou(a, b) == pytest.approx(2500 / 17500)


class TestObjectTracker:
    def setup_method(self):
        self.tracker = ObjectTracker(iou_threshold=0.3, max_age=5)

    def test_new_detection_creates_track(self):
        det = make_detection(10, 10, 50, 50)
        tracks = self.tracker.update([det], frame_number=0, timestamp=0.0)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1

    def test_same_vehicle_keeps_id(self):
        """A vehicle that barely moves should keep its track ID."""
        det1 = make_detection(10, 10, 50, 50, frame_number=0)
        det2 = make_detection(12, 12, 52, 52, frame_number=1)  # Slight movement

        self.tracker.update([det1], frame_number=0, timestamp=0.0)
        tracks = self.tracker.update([det2], frame_number=1, timestamp=0.1)

        assert len(tracks) == 1
        assert tracks[0].track_id == 1  # Same ID

    def test_new_vehicle_gets_new_id(self):
        """A vehicle appearing far from existing tracks gets a new ID."""
        det1 = make_detection(10, 10, 50, 50, frame_number=0)
        det2 = make_detection(500, 500, 600, 600, frame_number=1)  # Far away

        self.tracker.update([det1], frame_number=0, timestamp=0.0)
        # Frame 1: both vehicles present
        tracks = self.tracker.update([det1, det2], frame_number=1, timestamp=0.1)

        track_ids = {t.track_id for t in tracks}
        assert len(track_ids) == 2
        assert 1 in track_ids
        assert 2 in track_ids

    def test_stale_track_removed(self):
        """Tracks not matched for max_age frames should be removed."""
        tracker = ObjectTracker(iou_threshold=0.3, max_age=2)
        det = make_detection(10, 10, 50, 50, frame_number=0)
        tracker.update([det], frame_number=0, timestamp=0.0)

        # 3 frames with no detections → track should be removed
        for i in range(1, 4):
            tracks = tracker.update([], frame_number=i, timestamp=float(i) * 0.1)

        # After max_age+1 frames, the track should be gone
        assert len(tracks) == 0

    def test_trajectory_accumulates(self):
        """Track trajectory should grow with each update."""
        det1 = make_detection(10, 10, 50, 50, frame_number=0)
        det2 = make_detection(15, 15, 55, 55, frame_number=1)
        det3 = make_detection(20, 20, 60, 60, frame_number=2)

        self.tracker.update([det1], frame_number=0, timestamp=0.0)
        self.tracker.update([det2], frame_number=1, timestamp=0.1)
        tracks = self.tracker.update([det3], frame_number=2, timestamp=0.2)

        assert len(tracks[0].trajectory) == 3

    def test_reset_clears_state(self):
        det = make_detection(10, 10, 50, 50, frame_number=0)
        self.tracker.update([det], frame_number=0, timestamp=0.0)
        self.tracker.reset()

        tracks = self.tracker.update([det], frame_number=0, timestamp=0.0)
        assert tracks[0].track_id == 1  # IDs restart from 1
