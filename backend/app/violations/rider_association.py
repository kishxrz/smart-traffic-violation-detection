"""
backend/app/violations/rider_association.py
─────────────────────────────────────────────
Spatial association module connecting motorcycle riders with their vehicles.

Problem solved:
  Running helmet classification indiscriminately across an entire video frame leads to
  high false-positive rates (e.g., pedestrians walking near a street, bystanders).
  A helmet violation must strictly evaluate a PERSON riding a MOTORCYCLE.

Algorithm:
  For each detected motorcycle:
    1. Filter person detections with horizontal IoU >= min_horizontal_overlap.
    2. Check vertical alignment: person's center Y must be above motorcycle top edge + padding.
    3. Calculate composite association confidence score based on horizontal overlap
       and spatial proximity.
    4. Select top associated rider(s) for the motorcycle.

Returns structured RiderAssociation objects containing:
  - motorcycle_track_id
  - rider_track_id
  - association_confidence
  - motorcycle_bbox
  - rider_bbox
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.schemas.detection import BoundingBox, TrackedObject
from app.violations.geometry import iou_1d

logger = logging.getLogger(__name__)

CLASS_PERSON = 0
CLASS_MOTORCYCLE = 3


@dataclass
class RiderAssociation:
    """Represents a spatial link between a motorcycle and its rider."""
    motorcycle_id: int
    rider_id: int
    association_confidence: float
    motorcycle_bbox: BoundingBox
    rider_bbox: BoundingBox

    @property
    def head_roi_bbox(self) -> BoundingBox:
        """Estimate head ROI (upper ~30% of rider bounding box)."""
        x1, y1, x2, y2 = self.rider_bbox.to_xyxy()
        height = max(1, y2 - y1)
        head_y2 = y1 + int(height * 0.35)
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=head_y2)


class RiderAssociator:
    """
    State-free spatial association engine connecting riders and motorcycles.
    """

    def __init__(
        self,
        min_horizontal_overlap: float = 0.30,
        vertical_range_px: float = 140.0,
        min_association_confidence: float = 0.40,
    ) -> None:
        self.min_horizontal_overlap = min_horizontal_overlap
        self.vertical_range_px = vertical_range_px
        self.min_association_confidence = min_association_confidence

    def associate(
        self,
        tracked_objects: List[TrackedObject],
    ) -> List[RiderAssociation]:
        """
        Associate person detections with motorcycle detections in a single frame.

        Args:
            tracked_objects: List of TrackedObject instances in current frame.

        Returns:
            List of RiderAssociation objects.
        """
        motorcycles = [o for o in tracked_objects if o.class_id == CLASS_MOTORCYCLE]
        persons = [o for o in tracked_objects if o.class_id == CLASS_PERSON]

        associations: List[RiderAssociation] = []

        for moto in motorcycles:
            best_rider: Optional[TrackedObject] = None
            best_score: float = 0.0

            moto_x1, moto_y1 = moto.bbox.x1, moto.bbox.y1
            moto_x2, moto_y2 = moto.bbox.x2, moto.bbox.y2

            for person in persons:
                p_x1, p_y1 = person.bbox.x1, person.bbox.y1
                p_x2, p_y2 = person.bbox.x2, person.bbox.y2

                # 1. Horizontal IoU overlap
                horiz_overlap = iou_1d(p_x1, p_x2, moto_x1, moto_x2)
                if horiz_overlap < self.min_horizontal_overlap:
                    continue

                # 2. Vertical alignment check: Rider center Y should be above or near motorcycle top
                person_cy = person.center[1]
                if person_cy > moto_y1 + self.vertical_range_px:
                    continue

                # Rider's feet (y2) should be near or inside motorcycle
                if p_y2 < moto_y1 - 50:
                    continue  # Person is floating far above motorcycle

                # 3. Composite score: weighted combination of horizontal overlap & vertical proximity
                vert_dist = abs(person_cy - moto_y1)
                vert_proximity_score = max(0.0, 1.0 - (vert_dist / self.vertical_range_px))
                score = round(0.6 * horiz_overlap + 0.4 * vert_proximity_score, 4)

                if score > best_score and score >= self.min_association_confidence:
                    best_score = score
                    best_rider = person

            if best_rider is not None:
                associations.append(
                    RiderAssociation(
                        motorcycle_id=moto.track_id,
                        rider_id=best_rider.track_id,
                        association_confidence=best_score,
                        motorcycle_bbox=moto.bbox,
                        rider_bbox=best_rider.bbox,
                    )
                )

        return associations
