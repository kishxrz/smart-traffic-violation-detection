"""
backend/app/schemas/analytics.py
─────────────────────────────────
Pydantic schemas for analytics and reporting endpoints.
"""

from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, Field


class VehicleClassCount(BaseModel):
    class_name: str
    count: int


class ViolationTypeCount(BaseModel):
    violation_type: str
    count: int


class TrafficVolumePoint(BaseModel):
    """Vehicle count at a specific time bucket."""
    time_seconds: float
    count: int


class AnalyticsSummary(BaseModel):
    """
    High-level summary returned by GET /api/analytics/summary.
    """
    session_id: str
    total_frames_processed: int = 0
    total_vehicles_detected: int = 0
    unique_vehicles_tracked: int = 0

    # Violation counts
    total_violations: int = 0
    violations_by_type: Dict[str, int] = Field(default_factory=dict)
    violations_by_severity: Dict[str, int] = Field(default_factory=dict)

    # Performance
    avg_confidence: float = 0.0
    avg_inference_time_ms: float = 0.0
    processing_fps: float = 0.0

    # Traffic composition
    vehicles_by_class: Dict[str, int] = Field(default_factory=dict)

    # Time series (one bucket per second)
    traffic_volume_over_time: List[TrafficVolumePoint] = Field(default_factory=list)

    @property
    def violations_per_minute(self) -> float:
        if self.total_frames_processed == 0:
            return 0.0
        # Rough estimate; proper calculation needs FPS info
        return self.total_violations / max(1, self.total_frames_processed / 600)
