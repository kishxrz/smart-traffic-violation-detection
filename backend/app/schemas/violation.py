"""
backend/app/schemas/violation.py
─────────────────────────────────
Pydantic schemas for violations, scene state, and traffic light signals.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ViolationType(str, Enum):
    NO_HELMET = "NO_HELMET"
    RED_LIGHT = "RED_LIGHT"
    WRONG_WAY = "WRONG_WAY"
    LANE_VIOLATION = "LANE_VIOLATION"


class ViolationSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TrafficLightState(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    UNKNOWN = "UNKNOWN"


class Violation(BaseModel):
    """
    A confirmed traffic violation record.

    This is the canonical output of the violation engine —
    every violation module produces objects of this type.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    violation_type: ViolationType
    vehicle_id: int
    vehicle_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    severity: ViolationSeverity
    frame_number: int
    timestamp: float              # seconds since video start
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evidence_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


class SceneState(BaseModel):
    """
    The interpreted state of the traffic scene at a given frame.

    This is the bridge between raw detections and violation logic.
    The violation engine uses SceneState rather than raw YOLO outputs.
    """
    frame_number: int
    timestamp: float
    traffic_light_state: TrafficLightState = TrafficLightState.UNKNOWN
    active_violations: List[Violation] = Field(default_factory=list)

    # Configurable scene parameters (set per analysis session)
    stop_line_y: Optional[float] = None        # pixel Y coordinate
    expected_direction: Optional[str] = None   # "left" | "right" | "up" | "down"
    lane_count: int = 1


class ViolationSummary(BaseModel):
    """Aggregated violation statistics for the API response."""
    total: int = 0
    by_type: Dict[str, int] = Field(default_factory=dict)
    by_severity: Dict[str, int] = Field(default_factory=dict)
    violations: List[Violation] = Field(default_factory=list)
