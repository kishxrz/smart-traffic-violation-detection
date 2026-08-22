"""
backend/app/api/dependencies.py
─────────────────────────────────
FastAPI dependency injection.

These dependency functions are used with FastAPI's Depends() system
to inject shared resources (detector, tracker, engine) into route handlers
without creating them on every request.

The get_* functions use Python module-level state as a simple
singleton pattern. For production, use lifespan context or Redis.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import HTTPException

from app.config import Settings, get_settings
from app.detection.detector import BaseDetector
from app.detection.models import create_detector
from app.tracking.object_tracker import ObjectTracker
from app.violations.engine import ViolationEngine
from app.violations.severity import SeverityEngine
from app.violations.helmet import HelmetViolationDetector
from app.violations.red_light import RedLightViolationDetector
from app.violations.wrong_way import WrongWayViolationDetector
from app.violations.lane import LaneViolationDetector
from app.evidence.evidence_generator import EvidenceGenerator
from app.analytics.metrics import AnalyticsAccumulator

logger = logging.getLogger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────────
_detector: Optional[BaseDetector] = None
_tracker: Optional[ObjectTracker] = None
_violation_engine: Optional[ViolationEngine] = None
_evidence_generator: Optional[EvidenceGenerator] = None
_analytics: Optional[AnalyticsAccumulator] = None


def get_detector() -> BaseDetector:
    global _detector
    if _detector is None:
        settings = get_settings()
        _detector = create_detector(settings)
    return _detector


def get_tracker() -> ObjectTracker:
    global _tracker
    if _tracker is None:
        settings = get_settings()
        _tracker = ObjectTracker(
            iou_threshold=settings.iou_threshold,
            max_age=60,
        )
    return _tracker


def get_violation_engine() -> ViolationEngine:
    global _violation_engine
    if _violation_engine is None:
        settings = get_settings()
        severity = SeverityEngine()

        engine = ViolationEngine()
        engine.register(HelmetViolationDetector(severity_engine=severity))
        engine.register(
            RedLightViolationDetector(
                severity_engine=severity,
                stop_line_y=400.0,  # Default; overridden per session
            )
        )
        engine.register(
            WrongWayViolationDetector(
                severity_engine=severity,
                expected_direction="AUTO",
            )
        )
        # LaneViolationDetector requires lane ROIs — added on session start
        _violation_engine = engine

    return _violation_engine


def get_evidence_generator() -> EvidenceGenerator:
    global _evidence_generator
    if _evidence_generator is None:
        settings = get_settings()
        _evidence_generator = EvidenceGenerator(
            output_dir=settings.evidence_abs_dir,
            cooldown_seconds=settings.evidence_cooldown_seconds,
        )
    return _evidence_generator


def get_analytics() -> AnalyticsAccumulator:
    global _analytics
    if _analytics is None:
        _analytics = AnalyticsAccumulator(session_id="default")
    return _analytics


def reset_session() -> None:
    """Reset all stateful components between analysis sessions."""
    global _tracker, _violation_engine, _analytics
    if _tracker:
        _tracker.reset()
    if _violation_engine:
        _violation_engine.reset()
    _analytics = AnalyticsAccumulator(session_id="default")
    logger.info("Session reset complete.")
