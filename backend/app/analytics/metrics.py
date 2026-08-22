"""
backend/app/analytics/metrics.py
──────────────────────────────────
Analytics computation functions.

All functions are pure (stateless) — they take data and return results.
State is managed by the AnalyticsAccumulator which maintains running
totals and is updated per frame.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from app.schemas.analytics import AnalyticsSummary, TrafficVolumePoint
from app.schemas.detection import FrameResult, TrackedObject
from app.schemas.violation import Violation


class AnalyticsAccumulator:
    """
    Accumulates per-frame metrics across an entire analysis session.

    Updated once per processed frame; queried to produce AnalyticsSummary.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.total_frames = 0
        self.total_detections = 0
        self.total_violations = 0
        self.total_inference_ms = 0.0

        self._unique_track_ids: set = set()
        self._class_counts: Dict[str, int] = defaultdict(int)
        self._violation_type_counts: Dict[str, int] = defaultdict(int)
        self._violation_severity_counts: Dict[str, int] = defaultdict(int)
        self._confidence_sum = 0.0
        self._confidence_count = 0
        self._volume_over_time: List[TrafficVolumePoint] = []

    def update(
        self,
        frame_result: FrameResult,
        violations: List[Violation],
    ) -> None:
        """Update accumulators with results from one frame."""
        self.total_frames += 1
        self.total_detections += frame_result.detection_count
        self.total_violations += len(violations)
        self.total_inference_ms += frame_result.processing_time_ms

        # Track unique vehicle IDs
        for obj in frame_result.tracked_objects:
            self._unique_track_ids.add(obj.track_id)
            self._class_counts[obj.class_name] += 1
            self._confidence_sum += obj.confidence
            self._confidence_count += 1

        # Violation breakdown
        for v in violations:
            self._violation_type_counts[v.violation_type] += 1
            self._violation_severity_counts[v.severity] += 1

        # Traffic volume snapshot (one per frame, pruned later)
        if len(frame_result.tracked_objects) > 0:
            self._volume_over_time.append(
                TrafficVolumePoint(
                    time_seconds=frame_result.timestamp,
                    count=len(frame_result.tracked_objects),
                )
            )

    def build_summary(self, fps: float = 25.0) -> AnalyticsSummary:
        """Produce the final AnalyticsSummary from accumulated data."""
        avg_conf = (
            self._confidence_sum / self._confidence_count
            if self._confidence_count > 0
            else 0.0
        )
        avg_ms = (
            self.total_inference_ms / self.total_frames
            if self.total_frames > 0
            else 0.0
        )
        processing_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

        return AnalyticsSummary(
            session_id=self.session_id,
            total_frames_processed=self.total_frames,
            total_vehicles_detected=self.total_detections,
            unique_vehicles_tracked=len(self._unique_track_ids),
            total_violations=self.total_violations,
            violations_by_type=dict(self._violation_type_counts),
            violations_by_severity=dict(self._violation_severity_counts),
            avg_confidence=round(avg_conf, 4),
            avg_inference_time_ms=round(avg_ms, 2),
            processing_fps=round(processing_fps, 2),
            vehicles_by_class=dict(self._class_counts),
            traffic_volume_over_time=self._volume_over_time,
        )

    def reset(self) -> None:
        self.__init__(self.session_id)
