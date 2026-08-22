"""
backend/app/violations/geometry.py
────────────────────────────────────
Geometric utility functions used by multiple violation detectors.

These functions are pure (no side effects, no state) so they can be
unit-tested without any CV or ML dependencies.

Functions:
  - crosses_line()       : Has a trajectory crossed a horizontal/vertical line?
  - movement_direction() : What direction is a vehicle moving?
  - angle_between()      : Angle between two vectors in degrees.
  - point_side_of_line() : Which side of a line is a point on?
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

# 2D point type alias
Point = Tuple[float, float]


def crosses_line(
    prev: Point,
    curr: Point,
    line_y: float,
    direction: str = "any",
) -> bool:
    """
    Determine if a trajectory segment crosses a horizontal line.

    Args:
        prev:      Previous center point (x, y).
        curr:      Current center point (x, y).
        line_y:    Y coordinate of the horizontal line.
        direction: "up"  = only detect crossing from below to above (decreasing Y)
                   "down"= only detect crossing from above to below (increasing Y)
                   "any" = detect any crossing.

    Returns:
        True if the segment prev→curr crosses the line.

    Interview note:
      A crossing is detected by checking if line_y lies strictly between
      prev_y and curr_y. This is a simple sign-change test:
        (prev_y - line_y) and (curr_y - line_y) have opposite signs.
    """
    prev_y, curr_y = prev[1], curr[1]

    # Sign of (y - line_y) changes on crossing
    crossed = (prev_y - line_y) * (curr_y - line_y) < 0

    if not crossed:
        return False

    if direction == "down":
        return curr_y > prev_y   # Moving downward (increasing Y)
    elif direction == "up":
        return curr_y < prev_y   # Moving upward (decreasing Y)
    return True                  # "any" direction


def crosses_vertical_line(
    prev: Point,
    curr: Point,
    line_x: float,
    direction: str = "any",
) -> bool:
    """Detect crossing of a vertical line at x=line_x."""
    prev_x, curr_x = prev[0], curr[0]
    crossed = (prev_x - line_x) * (curr_x - line_x) < 0

    if not crossed:
        return False

    if direction == "right":
        return curr_x > prev_x
    elif direction == "left":
        return curr_x < prev_x
    return True


def movement_direction(
    prev: Point,
    curr: Point,
    min_displacement: float = 3.0,
) -> Optional[str]:
    """
    Compute the dominant movement direction.

    Args:
        prev:             Previous position.
        curr:             Current position.
        min_displacement: Minimum pixel displacement to avoid noise.
                          Tiny tracker jitter should not trigger violations.

    Returns:
        "left" | "right" | "up" | "down" | None (not enough movement).

    Implementation:
      We decompose the motion vector into dx, dy components and
      return the direction of the dominant component.
    """
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    magnitude = math.sqrt(dx * dx + dy * dy)

    if magnitude < min_displacement:
        return None

    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"


def vector_angle_degrees(vector: Point) -> float:
    """
    Compute angle of a 2D vector in screen coordinates in degrees [0, 360).
    Screen coords: +X is 0° (right), +Y is 90° (down).
    """
    vx, vy = vector
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(vy, vx)) % 360.0


def angular_difference_degrees(v1: Point, v2: Point) -> float:
    """
    Compute the minimal unsigned angular difference between two 2D vectors in degrees [0, 180].
    """
    len1 = math.hypot(v1[0], v1[1])
    len2 = math.hypot(v2[0], v2[1])
    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0
    dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def movement_angle_degrees(prev: Point, curr: Point) -> Optional[float]:
    """
    Angle of movement in degrees (0° = right, 90° = up, 180° = left, 270° = down).
    Returns None if the movement is negligible.
    """
    dx = curr[0] - prev[0]
    dy = -(curr[1] - prev[1])  # Invert Y (screen coords increase downward)

    if abs(dx) < 0.001 and abs(dy) < 0.001:
        return None

    return math.degrees(math.atan2(dy, dx)) % 360


def point_side_of_line(
    point: Point,
    line_start: Point,
    line_end: Point,
) -> float:
    """
    Return the signed area of the triangle formed by line_start, line_end, point.

    Positive = point is to the left of the line (counter-clockwise).
    Negative = point is to the right.
    Zero     = point is on the line.

    Used for lane-boundary determination.
    """
    return (
        (line_end[0] - line_start[0]) * (point[1] - line_start[1])
        - (line_end[1] - line_start[1]) * (point[0] - line_start[0])
    )


def iou_1d(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """
    1D IoU (overlap fraction) between two line segments.
    Used for associating motorcycle and person detections vertically.
    """
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - intersection
    return intersection / union if union > 0 else 0.0
