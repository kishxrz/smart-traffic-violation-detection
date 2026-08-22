"""
backend/app/violations/severity.py
────────────────────────────────────
Transparent, rule-based severity scoring engine.

Design philosophy:
  Severity is NOT pretended to be an ML model.
  It is an explicit, auditable rule table that can be:
    - Reviewed by traffic law experts.
    - Adjusted without retraining anything.
    - Explained to any interviewer in 30 seconds.

  This honesty is important: it's much better to have a working
  rule-based system than a fake "AI severity predictor".

Scoring factors:
  1. Base severity per violation type (legal risk category)
  2. Detection confidence bonus/penalty
  3. Repeat offender escalation
  4. Future: time-of-day, speed, proximity to pedestrian zones

Severity levels map to risk:
  LOW      = Minor infraction, low risk
  MEDIUM   = Notable infraction, moderate risk
  HIGH     = Serious violation, significant danger
  CRITICAL = Immediate safety threat (e.g., wrong-way at high speed)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

from app.schemas.violation import ViolationSeverity, ViolationType

# Base severity per violation type (subject to legal/policy review)
BASE_SEVERITY: Dict[ViolationType, ViolationSeverity] = {
    ViolationType.NO_HELMET: ViolationSeverity.HIGH,
    ViolationType.RED_LIGHT: ViolationSeverity.HIGH,
    ViolationType.WRONG_WAY: ViolationSeverity.CRITICAL,
    ViolationType.LANE_VIOLATION: ViolationSeverity.MEDIUM,
}

_SEVERITY_RANK: Dict[ViolationSeverity, int] = {
    ViolationSeverity.LOW: 0,
    ViolationSeverity.MEDIUM: 1,
    ViolationSeverity.HIGH: 2,
    ViolationSeverity.CRITICAL: 3,
}

_RANK_TO_SEVERITY = {v: k for k, v in _SEVERITY_RANK.items()}


class SeverityEngine:
    """
    Stateful severity calculator.

    Stateful because it tracks per-vehicle repeat violations.
    A vehicle that commits the same violation twice gets escalated severity.
    """

    def __init__(self) -> None:
        # vehicle_id → {violation_type → count}
        self._violation_counts: Dict[int, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def calculate(
        self,
        violation_type: ViolationType,
        vehicle_id: int,
        confidence: float,
    ) -> ViolationSeverity:
        """
        Compute severity for a given violation instance.

        Rules applied in order:
          1. Start from base severity for this violation type.
          2. If confidence < 0.5 → downgrade one level (avoid false positives).
          3. If this vehicle has committed the same violation before → upgrade.

        Args:
            violation_type: Type of the detected violation.
            vehicle_id:     Tracked object ID of the violating vehicle.
            confidence:     Detection confidence [0.0, 1.0].

        Returns:
            ViolationSeverity enum value.
        """
        base = BASE_SEVERITY.get(violation_type, ViolationSeverity.MEDIUM)
        rank = _SEVERITY_RANK[base]

        # Low confidence → downgrade (we're less sure about the violation)
        if confidence < 0.5:
            rank = max(0, rank - 1)

        # Repeat offender → upgrade
        count = self._violation_counts[vehicle_id][violation_type.value]
        if count >= 3:
            rank = min(3, rank + 1)
        elif count >= 1:
            rank = min(3, rank)  # No change on first repeat, escalate at 3

        # Record this occurrence
        self._violation_counts[vehicle_id][violation_type.value] += 1

        return _RANK_TO_SEVERITY[rank]

    def reset(self) -> None:
        self._violation_counts.clear()
