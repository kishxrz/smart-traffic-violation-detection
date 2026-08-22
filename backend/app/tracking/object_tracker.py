"""
backend/app/tracking/object_tracker.py
────────────────────────────────────────
IoU-based centroid tracker with trajectory recording.

Algorithm:
  1. For each new detection, compute IoU overlap with all existing tracks.
  2. Use the Hungarian algorithm (scipy.optimize.linear_sum_assignment)
     to find the globally optimal detection-to-track assignment.
  3. Matched tracks are updated with the new detection bbox.
  4. Unmatched detections become new tracks with fresh IDs.
  5. Tracks with no match for more than `max_age` frames are removed.

Why IoU-based over Euclidean distance?
  IoU is scale-invariant — a large truck and a small motorcycle are
  both handled correctly because IoU measures proportional overlap,
  not absolute pixel distance.

Limitations:
  - Fails when two identical vehicles pass directly in front of each
    other (ID switch). DeepSORT or ByteTrack solve this with re-ID
    features from a CNN embedding network.
  - This implementation is intentionally simple and CPU-efficient
    so the system runs without a GPU.

Interface note:
  ObjectTracker is a concrete implementation of BaseTracker.
  It can be replaced with a DeepSORT wrapper without changing
  any violation or analytics code.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.schemas.detection import BoundingBox, Detection, TrackedObject
from app.tracking.tracker import BaseTracker

logger = logging.getLogger(__name__)

MAX_TRAJECTORY_LENGTH = 50  # Keep last N positions per track


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """
    Compute Intersection over Union between two bounding boxes.

    IoU = Area(A ∩ B) / Area(A ∪ B)

    Range: [0.0, 1.0]
      0.0 = no overlap
      1.0 = perfect overlap
    """
    ix1 = max(box_a.x1, box_b.x1)
    iy1 = max(box_a.y1, box_b.y1)
    ix2 = min(box_a.x2, box_b.x2)
    iy2 = min(box_a.y2, box_b.y2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection == 0.0:
        return 0.0

    area_a = box_a.area
    area_b = box_b.area
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class _Track:
    """Internal state for a single tracked object."""

    def __init__(self, track_id: int, detection: Detection, frame_number: int) -> None:
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.first_seen_frame = frame_number
        self.last_seen_frame = frame_number
        self.frames_tracked = 1
        self.frames_missing = 0
        self.trajectory: List[Tuple[float, float]] = [detection.bbox.center]

    def update(self, detection: Detection, frame_number: int) -> None:
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.last_seen_frame = frame_number
        self.frames_tracked += 1
        self.frames_missing = 0

        center = detection.bbox.center
        self.trajectory.append(center)
        if len(self.trajectory) > MAX_TRAJECTORY_LENGTH:
            self.trajectory.pop(0)

    def to_tracked_object(self, timestamp: float) -> TrackedObject:
        return TrackedObject(
            track_id=self.track_id,
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            bbox=self.bbox,
            frame_number=self.last_seen_frame,
            timestamp=timestamp,
            trajectory=list(self.trajectory),
            first_seen_frame=self.first_seen_frame,
            last_seen_frame=self.last_seen_frame,
            frames_tracked=self.frames_tracked,
        )


class ObjectTracker(BaseTracker):
    """
    IoU-based multi-object tracker using the Hungarian algorithm
    for optimal detection-to-track assignment.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
    ) -> None:
        """
        Args:
            iou_threshold: Minimum IoU to consider a detection and track
                           as the same object.
            max_age:       Frames a track can go unmatched before removal.
                           At 10 FPS, max_age=30 means 3 seconds of memory.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age

        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1

    def update(
        self,
        detections: List[Detection],
        frame_number: int,
        timestamp: float,
    ) -> List[TrackedObject]:
        """
        Assign tracking IDs to new detections via IoU matching.

        Returns only currently visible (matched) tracks.
        """
        if not detections:
            # Age all tracks and remove stale ones
            self._age_and_prune(frame_number)
            return []

        if not self._tracks:
            # No existing tracks — create a new track for each detection
            for det in detections:
                self._new_track(det, frame_number)
            return [t.to_tracked_object(timestamp) for t in self._tracks.values()]

        # ── Hungarian assignment ───────────────────────────────────────────
        track_ids = list(self._tracks.keys())
        tracks = [self._tracks[tid] for tid in track_ids]

        # Build IoU cost matrix (rows = tracks, cols = detections)
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = compute_iou(track.bbox, det.bbox)

        # Use linear_sum_assignment to maximise IoU (minimise negative IoU)
        try:
            from scipy.optimize import linear_sum_assignment
            row_inds, col_inds = linear_sum_assignment(-cost_matrix)
        except ImportError:
            # Fallback: greedy matching (still correct, less optimal)
            row_inds, col_inds = self._greedy_match(cost_matrix)

        matched_track_ids = set()
        matched_det_ids = set()

        for r, c in zip(row_inds, col_inds):
            if cost_matrix[r, c] >= self.iou_threshold:
                tracks[r].update(detections[c], frame_number)
                matched_track_ids.add(track_ids[r])
                matched_det_ids.add(c)

        # New tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_det_ids:
                self._new_track(det, frame_number)

        # Age unmatched tracks
        for i, tid in enumerate(track_ids):
            if tid not in matched_track_ids:
                self._tracks[tid].frames_missing += 1

        # Prune stale tracks
        self._age_and_prune(frame_number)

        return [
            t.to_tracked_object(timestamp)
            for t in self._tracks.values()
            if t.frames_missing == 0
        ]

    def _new_track(self, detection: Detection, frame_number: int) -> None:
        track = _Track(self._next_id, detection, frame_number)
        self._tracks[self._next_id] = track
        self._next_id += 1

    def _age_and_prune(self, frame_number: int) -> None:
        stale = [
            tid for tid, t in self._tracks.items()
            if t.frames_missing > self.max_age
        ]
        for tid in stale:
            logger.debug("Track #%d removed after %d missing frames.", tid, self.max_age)
            del self._tracks[tid]

    @staticmethod
    def _greedy_match(
        cost_matrix: np.ndarray,
    ) -> Tuple[List[int], List[int]]:
        """Greedy fallback if scipy is not available."""
        rows, cols = [], []
        used_cols: set = set()
        for r in range(cost_matrix.shape[0]):
            best_c = -1
            best_v = -1.0
            for c in range(cost_matrix.shape[1]):
                if c not in used_cols and cost_matrix[r, c] > best_v:
                    best_v = cost_matrix[r, c]
                    best_c = c
            if best_c >= 0:
                rows.append(r)
                cols.append(best_c)
                used_cols.add(best_c)
        return rows, cols

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def get_active_tracks(self) -> List[TrackedObject]:
        return [
            t.to_tracked_object(0.0)
            for t in self._tracks.values()
            if t.frames_missing == 0
        ]
