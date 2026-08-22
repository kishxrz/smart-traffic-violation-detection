"""
backend/app/violations/severity.py
────────────────────────────────────
Transparent, rule-based severity scoring engine.

Design philosophy:
  Severity is NOT pretended to be an ML model.
  It is an explicit, auditable rule table that outputs:
    1. Numeric severity score (0 - 100)
    2. Severity level (LOW, MEDIUM, HIGH, CRITICAL)
    3. Human-readable, transparent reasons for the rating

Scoring factors:
  1. Base severity per violation type (legal risk weight)
  2. Sustained duration / displacement factor
  3. Rule-based violation confidence
  4. Repeat offender escalation
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from app.schemas.violation import ViolationSeverity, ViolationType

# Base score (0-100) per violation type
BASE_SEVERITY_SCORES: Dict[ViolationType, int] = {
    ViolationType.NO_HELMET: 55,
    ViolationType.RED_LIGHT: 70,
    ViolationType.WRONG_WAY: 75,
    ViolationType.LANE_VIOLATION: 40,
}

BASE_SEVERITY_LEVELS: Dict[ViolationType, ViolationSeverity] = {
    ViolationType.NO_HELMET: ViolationSeverity.HIGH,
    ViolationType.RED_LIGHT: ViolationSeverity.HIGH,
    ViolationType.WRONG_WAY: ViolationSeverity.CRITICAL,
    ViolationType.LANE_VIOLATION: ViolationSeverity.MEDIUM,
}


def score_to_level(score: int) -> ViolationSeverity:
    if score >= 85:
        return ViolationSeverity.CRITICAL
    if score >= 65:
        return ViolationSeverity.HIGH
    if score >= 45:
        return ViolationSeverity.MEDIUM
    return ViolationSeverity.LOW


class SeverityEngine:
    """
    Stateful severity calculator providing explainable severity scores.
    """

    def __init__(self) -> None:
        # vehicle_id → {violation_type → count}
        self._violation_counts: Dict[int, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def calculate_detailed(
        self,
        violation_type: ViolationType,
        vehicle_id: int,
        detection_confidence: float,
        violation_confidence: float = 0.8,
        displacement_px: float = 0.0,
        sustained_frames: int = 1,
    ) -> Tuple[ViolationSeverity, int, List[str]]:
        """
        Compute transparent severity score, level, and explanation reasons.

        Returns:
            Tuple of (ViolationSeverity, severity_score_0_100, List[reasons])
        """
        reasons: List[str] = []
        base_score = BASE_SEVERITY_SCORES.get(violation_type, 50)
        reasons.append(f"Base {violation_type.value} severity score ({base_score} pts)")

        score = base_score

        # Factor 1: Violation confidence
        if violation_confidence >= 0.85:
            score += 10
            reasons.append(f"High violation confidence {violation_confidence:.2f} (+10 pts)")
        elif violation_confidence < 0.50:
            score -= 15
            reasons.append(f"Low violation confidence {violation_confidence:.2f} (-15 pts)")

        # Factor 2: Sustained movement / displacement
        if sustained_frames >= 15:
            score += 10
            reasons.append(f"Sustained wrong-way movement over {sustained_frames} frames (+10 pts)")
        elif sustained_frames >= 8:
            score += 5
            reasons.append(f"Sustained movement over {sustained_frames} frames (+5 pts)")

        if displacement_px >= 50.0:
            score += 5
            reasons.append(f"Significant displacement {displacement_px:.1f}px (+5 pts)")

        # Factor 3: Repeat offender escalation
        count = self._violation_counts[vehicle_id][violation_type.value]
        if count >= 3:
            score += 20
            reasons.append(f"Repeat offender: vehicle #{vehicle_id} has {count} previous violations (+20 pts)")
        elif count >= 1:
            score += 10
            reasons.append(f"Repeat offender: vehicle #{vehicle_id} has {count} previous violation (+10 pts)")

        # Record this occurrence
        self._violation_counts[vehicle_id][violation_type.value] += 1

        final_score = max(0, min(100, score))
        level = score_to_level(final_score)

        return level, final_score, reasons

    def calculate(
        self,
        violation_type: ViolationType,
        vehicle_id: int,
        confidence: float,
    ) -> ViolationSeverity:
        """Legacy compatibility wrapper."""
        level, _, _ = self.calculate_detailed(
            violation_type=violation_type,
            vehicle_id=vehicle_id,
            detection_confidence=confidence,
            violation_confidence=confidence,
        )
        return level

    def reset(self) -> None:
        self._violation_counts.clear()
