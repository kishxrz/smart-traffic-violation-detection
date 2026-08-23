"""
backend/app/evidence/evidence_generator.py
────────────────────────────────────────────
Generates annotated evidence images when violations are confirmed.

Evidence image format:
  violation_{YYYYMMDD}_{HHMMSS}_{type}_vehicle_{id}.jpg

Cooldown/deduplication:
  Per (vehicle_id, violation_type) pair, evidence is only saved once
  per cooldown window. This prevents flooding disk with 30 near-identical
  frames for the same violation event.

Each evidence image contains:
  - Original frame
  - Vehicle bounding box (colored by severity)
  - Violation type label
  - Vehicle ID
  - Confidence score
  - Timestamp
  - Frame number
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.schemas.detection import BoundingBox, TrackedObject
from app.schemas.violation import Violation, ViolationSeverity
from app.cv.visualization import draw_violation_overlay, SEVERITY_COLORS

logger = logging.getLogger(__name__)


class EvidenceGenerator:
    """
    Saves annotated evidence frames for confirmed violations.
    """

    def __init__(
        self,
        output_dir: Path,
        cooldown_seconds: float = 3.0,
        jpeg_quality: int = 90,
    ) -> None:
        """
        Args:
            output_dir:       Directory to save evidence images.
            cooldown_seconds: Minimum time between evidence saves for the
                              same (vehicle_id, violation_type) pair.
            jpeg_quality:     JPEG compression quality [1, 100].
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cooldown = cooldown_seconds
        self._jpeg_quality = jpeg_quality
        # (vehicle_id, violation_type) → last save timestamp
        self._last_saved: Dict[Tuple[int, str], float] = {}

    def save(
        self,
        frame: np.ndarray,
        violation: Violation,
        bbox: Optional[BoundingBox] = None,
    ) -> Optional[str]:
        """
        Save an evidence image for a violation.

        Args:
            frame:     The current BGR frame.
            violation: The violation record.
            bbox:      Bounding box of the violating vehicle (for annotation).

        Returns:
            Absolute path to the saved image, or None if skipped (cooldown).
        """
        key = (violation.vehicle_id, violation.violation_type)
        now = time.time()

        if self._is_on_cooldown(key, now):
            return None

        self._last_saved[key] = now

        # Generate filename
        ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        v_type = violation.violation_type.replace(" ", "_").lower()
        filename = (
            f"violation_{ts_str}_{v_type}_vehicle_{violation.vehicle_id}.jpg"
        )
        filepath = self.output_dir / filename

        # Annotate frame
        annotated = draw_violation_overlay(frame, violation, bbox)

        # Save as JPEG
        success = cv2.imwrite(
            str(filepath),
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )

        if success:
            logger.info("Evidence saved: %s", filepath)
            return str(filepath)
        else:
            logger.error("Failed to save evidence image: %s", filepath)
            return None

    def _is_on_cooldown(self, key: Tuple[int, str], now: float) -> bool:
        last = self._last_saved.get(key, 0.0)
        return (now - last) < self._cooldown

    def save_batch(
        self,
        frame: np.ndarray,
        violations: List[Violation],
        tracked_objects_by_id: Dict[int, TrackedObject],
    ) -> List[str]:
        """
        Save evidence for a list of violations from the same frame.

        Returns list of saved file paths (excluding skipped cooldowns).
        """
        saved_paths: List[str] = []
        for violation in violations:
            tracked = tracked_objects_by_id.get(violation.vehicle_id)
            bbox = tracked.bbox if tracked else None
            path = self.save(frame, violation, bbox)
            if path:
                saved_paths.append(path)
                violation.evidence_path = path
        return saved_paths
