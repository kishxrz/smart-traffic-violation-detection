"""
backend/app/cv/perspective.py
──────────────────────────────
Perspective transformation utilities.

Purpose:
  Traffic cameras are mounted at an angle, creating a projective
  (perspective) distortion. Real-world distance and direction
  analysis requires working in a top-down (bird's-eye) view.

Mathematics:
  A perspective transformation is represented by a 3×3 homography
  matrix H such that:

      [x', y', w'] = H × [x, y, 1]ᵀ
      (x_corrected, y_corrected) = (x'/w', y'/w')

  Four point correspondences fully determine H.
  OpenCV uses Direct Linear Transform (DLT) with SVD internally.

Applications in this system:
  1. Lane width measurement (parallel in bird's-eye = consistent lane width)
  2. Vehicle trajectory analysis (direction in bird's-eye = actual road direction)
  3. Stop-line crossing detection (more accurate than pixel-based)

See docs/computer-vision.md for the full mathematical derivation.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Four 2D points defining a quadrilateral
Quad = List[Tuple[float, float]]  # [top-left, top-right, bottom-right, bottom-left]


class PerspectiveTransformer:
    """
    Wraps OpenCV's perspective transform for traffic camera calibration.

    Usage:
        transformer = PerspectiveTransformer()
        transformer.calibrate(src_points, dst_points, frame_size)
        bird_eye = transformer.warp(frame)
        original_pt = transformer.inverse_point(bird_eye_pt)
    """

    def __init__(self) -> None:
        self._M: Optional[np.ndarray] = None        # 3×3 homography matrix
        self._M_inv: Optional[np.ndarray] = None    # Inverse homography
        self._dst_size: Optional[Tuple[int, int]] = None

    def calibrate(
        self,
        src_quad: Quad,
        dst_size: Tuple[int, int],
    ) -> None:
        """
        Compute the homography matrix from four source points.

        Args:
            src_quad:  Four points in the original camera frame that
                       correspond to a real-world rectangle on the road
                       (e.g., a known road marking or intersection corners).
                       Order: [top-left, top-right, bottom-right, bottom-left].
            dst_size:  (width, height) of the output bird's-eye image.

        The destination quad is always the full dst_size rectangle:
            [0,0] → [W,0] → [W,H] → [0,H]
        """
        w, h = dst_size
        self._dst_size = dst_size

        dst_quad: Quad = [(0, 0), (w, 0), (w, h), (0, h)]

        src_pts = np.float32(src_quad)
        dst_pts = np.float32(dst_quad)

        self._M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self._M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

        logger.debug("Perspective calibration complete. M:\n%s", self._M)

    @property
    def is_calibrated(self) -> bool:
        return self._M is not None

    def warp(self, frame: np.ndarray) -> np.ndarray:
        """
        Transform a camera frame to bird's-eye view.

        Args:
            frame: BGR frame from the traffic camera.

        Returns:
            Warped bird's-eye frame with shape (dst_height, dst_width, 3).
        """
        self._require_calibration()
        w, h = self._dst_size  # type: ignore[misc]
        return cv2.warpPerspective(frame, self._M, (w, h))

    def warp_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a single point from camera coords to bird's-eye coords.

        Uses matrix multiplication rather than warping a whole frame —
        efficient for transforming bounding box centers.
        """
        self._require_calibration()
        pt = np.float32([[list(point)]])
        warped = cv2.perspectiveTransform(pt, self._M)
        x, y = warped[0][0]
        return float(x), float(y)

    def inverse_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a bird's-eye point back to original camera coordinates.
        Used for drawing violation overlays on the original frame.
        """
        self._require_calibration()
        pt = np.float32([[list(point)]])
        original = cv2.perspectiveTransform(pt, self._M_inv)
        x, y = original[0][0]
        return float(x), float(y)

    def _require_calibration(self) -> None:
        if not self.is_calibrated:
            raise RuntimeError(
                "PerspectiveTransformer has not been calibrated. "
                "Call calibrate() with source quad points first."
            )


def estimate_default_quad(
    frame_width: int,
    frame_height: int,
    top_fraction: float = 0.55,
    bottom_fraction: float = 0.95,
    top_inset_fraction: float = 0.35,
    bottom_inset_fraction: float = 0.10,
) -> Quad:
    """
    Estimate a reasonable perspective quad for a straight road camera.

    This is a heuristic starting point — proper calibration should
    use known road markings measured in real-world coordinates.

    The quad approximates the road surface as a trapezoid:
      - Narrow at the top (far end of road, perspective compression)
      - Wide at the bottom (near end, close to camera)
    """
    w, h = frame_width, frame_height
    top_y = int(h * top_fraction)
    bot_y = int(h * bottom_fraction)
    top_left_x = int(w * top_inset_fraction)
    top_right_x = int(w * (1 - top_inset_fraction))
    bot_left_x = int(w * bottom_inset_fraction)
    bot_right_x = int(w * (1 - bottom_inset_fraction))

    return [
        (top_left_x, top_y),
        (top_right_x, top_y),
        (bot_right_x, bot_y),
        (bot_left_x, bot_y),
    ]
