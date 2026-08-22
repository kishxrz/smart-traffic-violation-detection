"""
backend/app/cv/roi.py
─────────────────────
Region of Interest (ROI) utilities.

An ROI defines which part of the frame is relevant for a given
analysis task. This avoids false positives from vehicles outside
the monitored zone and reduces compute by masking irrelevant areas.

Supported ROI shapes:
  - Rectangle  (axis-aligned)
  - Polygon    (arbitrary convex/concave polygon)

Interview talking points:
  - ROIs are defined once per camera setup (calibration step).
  - cv2.pointPolygonTest() uses the winding-number algorithm.
  - cv2.fillPoly() builds a binary mask — bitwise AND with frame
    zeroes out pixels outside the ROI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Type alias for a 2D point
Point = Tuple[float, float]


class ROIType(str, Enum):
    TRAFFIC = "traffic"       # General monitored traffic zone
    LANE = "lane"             # Individual lane boundary
    STOP_LINE = "stop_line"   # Line vehicles must not cross on red
    VEHICLE = "vehicle"       # Detection zone for vehicles
    HELMET = "helmet"         # Head region above motorcycle detection


@dataclass
class RectROI:
    """Axis-aligned rectangular ROI."""
    x: int
    y: int
    width: int
    height: int
    roi_type: ROIType = ROIType.TRAFFIC
    label: str = ""

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def corners(self) -> List[Point]:
        return [
            (self.x, self.y),
            (self.x2, self.y),
            (self.x2, self.y2),
            (self.x, self.y2),
        ]

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x2 and self.y <= py <= self.y2

    def apply_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return frame with everything outside this ROI zeroed out."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (self.x, self.y), (self.x2, self.y2), 255, -1)
        return cv2.bitwise_and(frame, frame, mask=mask)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Return cropped sub-image of this ROI."""
        return frame[self.y:self.y2, self.x:self.x2]


@dataclass
class PolygonROI:
    """
    Arbitrary polygon ROI.

    Polygons are more flexible than rectangles and can accurately
    represent angled camera perspectives, curved road segments,
    or intersection zones.
    """
    vertices: List[Point]          # [(x0,y0), (x1,y1), ...]
    roi_type: ROIType = ROIType.TRAFFIC
    label: str = ""

    def _np_vertices(self) -> np.ndarray:
        return np.array(self.vertices, dtype=np.int32)

    def contains_point(self, px: float, py: float) -> bool:
        """
        Test whether a point lies inside this polygon.

        cv2.pointPolygonTest() returns:
          +d  — inside  (d = distance to nearest edge)
           0  — on edge
          -d  — outside
        """
        result = cv2.pointPolygonTest(
            self._np_vertices(), (float(px), float(py)), measureDist=False
        )
        return result >= 0

    def apply_mask(self, frame: np.ndarray) -> np.ndarray:
        """Zero out pixels outside the polygon."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self._np_vertices()], 255)
        return cv2.bitwise_and(frame, frame, mask=mask)

    def draw(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        fill_alpha: float = 0.0,
    ) -> np.ndarray:
        """
        Draw the polygon on a copy of the frame.

        Args:
            fill_alpha: 0.0 = outline only; >0.0 = semi-transparent fill.
        """
        out = frame.copy()
        pts = self._np_vertices()

        if fill_alpha > 0.0:
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0, out)

        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)

        if self.label:
            cx = int(np.mean([v[0] for v in self.vertices]))
            cy = int(np.mean([v[1] for v in self.vertices]))
            cv2.putText(
                out, self.label, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
        return out


def point_in_polygon(point: Point, polygon: List[Point]) -> bool:
    """
    Standalone polygon containment test (no class instantiation needed).

    Uses OpenCV's winding-number implementation internally.
    """
    pts = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0


def build_lane_rois(
    frame_width: int,
    frame_height: int,
    num_lanes: int,
    top_y: int,
    bottom_y: int,
    left_x: int,
    right_x: int,
) -> List[PolygonROI]:
    """
    Divide a trapezoidal road region into equal-width lane polygons.

    This is a convenience factory used when exact lane coordinates
    are unknown — it approximates lanes by equal division.
    Proper calibration should replace this with surveyed coordinates.
    """
    lane_width = (right_x - left_x) / num_lanes
    rois: List[PolygonROI] = []

    for i in range(num_lanes):
        x_left_bottom = int(left_x + i * lane_width)
        x_right_bottom = int(left_x + (i + 1) * lane_width)
        # Top of lane (perspective narrows toward vanishing point)
        scale = (top_y - frame_height * 0.3) / (bottom_y - frame_height * 0.3)
        center_bottom = (x_left_bottom + x_right_bottom) / 2
        half_top = lane_width * scale / 2
        x_left_top = int(center_bottom - half_top)
        x_right_top = int(center_bottom + half_top)

        rois.append(
            PolygonROI(
                vertices=[
                    (x_left_top, top_y),
                    (x_right_top, top_y),
                    (x_right_bottom, bottom_y),
                    (x_left_bottom, bottom_y),
                ],
                roi_type=ROIType.LANE,
                label=f"Lane {i + 1}",
            )
        )

    return rois
