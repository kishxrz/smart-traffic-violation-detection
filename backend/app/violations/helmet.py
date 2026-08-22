"""
backend/app/violations/helmet.py
──────────────────────────────────
No-helmet violation detector with rider association, custom YOLO helmet classifier,
and temporal evidence aggregation.

Pipeline Architecture:
  Frame → Object Tracker → Rider ↔ Motorcycle Spatial Associator → Head ROI Extractor
        → Custom Helmet YOLO Detector → Temporal Evidence Accumulator → Violation Engine

Temporal Evidence Safeguards:
  1. A single frame prediction NEVER triggers a violation.
  2. Rider history collects helmet predictions across consecutive frames.
  3. Requires min_frames_tracked >= 10, min_observations >= 5, and no_helmet_ratio >= 0.70.
  4. UNKNOWN state NEVER triggers a violation.
  5. Decouples detection_confidence, helmet_model_confidence, and violation_confidence.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from app.cv.head_roi import extract_head_crop
from app.detection.helmet_detector import HelmetDetector, HelmetPrediction
from app.schemas.detection import TrackedObject
from app.schemas.violation import (
    SceneState, Violation, ViolationSeverity, ViolationType,
)
from app.violations.base import ViolationDetector
from app.violations.rider_association import RiderAssociator, RiderAssociation
from app.violations.severity import SeverityEngine

logger = logging.getLogger(__name__)

CLASS_MOTORCYCLE = 3
CLASS_PERSON = 0


class HelmetViolationDetector(ViolationDetector):
    """
    Production No-Helmet Violation Detector featuring rider association,
    custom helmet model classification, and temporal aggregation.
    """

    def __init__(
        self,
        severity_engine: SeverityEngine,
        helmet_detector: Optional[HelmetDetector] = None,
        min_frames_tracked: int = 10,
        min_observations: int = 5,
        no_helmet_ratio_threshold: float = 0.70,
        cooldown_seconds: float = 5.0,
    ) -> None:
        self._severity = severity_engine
        self._helmet_detector = helmet_detector or HelmetDetector()
        self._associator = RiderAssociator()

        self._min_frames_tracked = min_frames_tracked
        self._min_observations = min_observations
        self._no_helmet_ratio_threshold = no_helmet_ratio_threshold
        self._cooldown = cooldown_seconds

        # motorcycle_id → List[HelmetPrediction]
        self._prediction_history: Dict[int, List[HelmetPrediction]] = defaultdict(list)
        self._last_violation_time: Dict[int, float] = {}

    @property
    def violation_type(self) -> ViolationType:
        return ViolationType.NO_HELMET

    def evaluate(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        scene_state: SceneState,
    ) -> List[Violation]:
        violations: List[Violation] = []

        # 1. Associate riders with motorcycles
        associations: List[RiderAssociation] = self._associator.associate(tracked_objects)

        for assoc in associations:
            moto_id = assoc.motorcycle_id
            rider_id = assoc.rider_id

            # Find tracked objects for metadata & confidence
            moto_obj = next((o for o in tracked_objects if o.track_id == moto_id), None)
            rider_obj = next((o for o in tracked_objects if o.track_id == rider_id), None)

            if moto_obj is None or rider_obj is None:
                continue

            # Temporal Safeguard 1: Skip newly tracked motorcycles (< min_frames_tracked)
            if moto_obj.frames_tracked < self._min_frames_tracked:
                continue

            # 2. Extract head ROI crop
            head_crop = extract_head_crop(frame, rider_obj.bbox)
            if head_crop is None or head_crop.shape[0] < 24 or head_crop.shape[1] < 24:
                continue

            # Safeguard: Rider association confidence threshold
            if assoc.association_confidence < 0.30:
                continue

            # 3. Predict helmet status via custom HelmetDetector (Model v2)
            history = self._prediction_history[moto_id]
            prediction = self._helmet_detector.predict_crop(
                head_crop,
                vehicle_id=moto_id,
                person_id=rider_id,
                frame_number=scene_state.frame_number,
                obs_count=len(history) + 1,
            )

            # Store prediction in history window (keep max 15 recent observations)
            history.append(prediction)
            if len(history) > 15:
                history.pop(0)

            # Temporal Safeguard 2: Require minimum observation history
            if len(history) < self._min_observations:
                continue

            # Temporal Safeguard 3: Calculate no_helmet prediction ratio
            no_helmet_count = sum(1 for p in history if p.status == "NO_HELMET")
            valid_obs = [p for p in history if p.status in ("HELMET", "NO_HELMET")]

            if not valid_obs:
                continue  # All UNKNOWN -> NO violation!

            no_helmet_ratio = no_helmet_count / len(valid_obs)

            # Safeguard 4: UNKNOWN state or low ratio NEVER triggers a violation
            if no_helmet_ratio < self._no_helmet_ratio_threshold:
                continue

            # Cooldown check
            now = time.time()
            if self._is_on_cooldown(moto_id, now):
                continue

            # Calculate confidences
            det_conf = round(min(moto_obj.confidence, rider_obj.confidence), 4)
            valid_confs = [p.confidence for p in valid_obs if p.status == "NO_HELMET"]
            helmet_model_conf = round(sum(valid_confs) / len(valid_confs), 4) if valid_confs else 0.80

            violation_conf = round(
                0.3 * det_conf + 0.4 * helmet_model_conf + 0.3 * no_helmet_ratio,
                4,
            )

            # Calculate explainable severity
            severity, severity_score, severity_reasons = self._severity.calculate_detailed(
                violation_type=ViolationType.NO_HELMET,
                vehicle_id=moto_id,
                detection_confidence=det_conf,
                violation_confidence=violation_conf,
                sustained_frames=len(history),
            )

            severity_reasons.append(
                f"No-helmet confirmed in {no_helmet_count}/{len(valid_obs)} observations ({no_helmet_ratio*100:.0f}%)"
            )
            severity_reasons.append(
                f"Associated rider #{rider_id} on motorcycle #{moto_id} (assoc conf: {assoc.association_confidence:.2f})"
            )

            self._last_violation_time[moto_id] = now

            logger.info(
                "NO HELMET violation confirmed: motorcycle #%d, rider #%d, ratio=%.0f%%, severity=%s (%d pts)",
                moto_id, rider_id, no_helmet_ratio * 100, severity.value, severity_score,
            )

            violations.append(
                Violation(
                    violation_type=ViolationType.NO_HELMET,
                    vehicle_id=moto_id,
                    vehicle_class="motorcycle",
                    confidence=violation_conf,
                    detection_confidence=det_conf,
                    violation_confidence=violation_conf,
                    severity=severity,
                    severity_score=severity_score,
                    severity_reasons=severity_reasons,
                    frame_number=scene_state.frame_number,
                    timestamp=scene_state.timestamp,
                    metadata={
                        "rider_id": rider_id,
                        "rider_bbox": list(rider_obj.bbox.to_xyxy()),
                        "motorcycle_bbox": list(moto_obj.bbox.to_xyxy()),
                        "head_roi_bbox": list(assoc.head_roi_bbox.to_xyxy()),
                        "association_confidence": assoc.association_confidence,
                        "helmet_model_confidence": helmet_model_conf,
                        "no_helmet_observation_ratio": round(no_helmet_ratio, 2),
                        "total_observations": len(history),
                    },
                )
            )

        return violations

    def _is_on_cooldown(self, vehicle_id: int, now: float) -> bool:
        last = self._last_violation_time.get(vehicle_id, 0.0)
        return (now - last) < self._cooldown

    def reset(self) -> None:
        self._prediction_history.clear()
        self._last_violation_time.clear()
