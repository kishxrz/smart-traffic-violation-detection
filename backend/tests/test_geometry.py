"""
backend/tests/test_geometry.py
────────────────────────────────
Unit tests for geometric utility functions.

These tests run WITHOUT any GPU, model weights, or camera input.
They only test the mathematical logic used by violation detectors.

Design rationale:
  Traffic-rule logic (line crossing, direction calculation) must be
  independently testable. Tests that require a GPU slow down CI
  and make TDD impractical.
"""

import pytest

from app.violations.geometry import (
    crosses_line,
    crosses_vertical_line,
    iou_1d,
    movement_direction,
    movement_angle_degrees,
    point_side_of_line,
)


# ── crosses_line ──────────────────────────────────────────────────────────────

class TestCrossesLine:
    def test_crossing_downward(self):
        """Vehicle moving from above line to below line."""
        assert crosses_line(prev=(100, 390), curr=(105, 410), line_y=400.0)

    def test_crossing_upward(self):
        """Vehicle moving from below line to above line."""
        assert crosses_line(prev=(100, 410), curr=(105, 390), line_y=400.0)

    def test_no_crossing_both_above(self):
        assert not crosses_line(prev=(100, 380), curr=(110, 385), line_y=400.0)

    def test_no_crossing_both_below(self):
        assert not crosses_line(prev=(100, 410), curr=(110, 420), line_y=400.0)

    def test_direction_filter_down_only(self):
        """Should only detect downward crossing."""
        # Downward crossing — should detect
        assert crosses_line(
            prev=(100, 390), curr=(105, 410), line_y=400.0, direction="down"
        )
        # Upward crossing — should NOT detect with direction="down"
        assert not crosses_line(
            prev=(100, 410), curr=(105, 390), line_y=400.0, direction="down"
        )

    def test_direction_filter_up_only(self):
        assert crosses_line(
            prev=(100, 410), curr=(105, 390), line_y=400.0, direction="up"
        )
        assert not crosses_line(
            prev=(100, 390), curr=(105, 410), line_y=400.0, direction="up"
        )

    def test_exactly_on_line_no_cross(self):
        """A point sitting exactly on the line does not constitute a crossing."""
        assert not crosses_line(prev=(100, 400), curr=(110, 400), line_y=400.0)


# ── crosses_vertical_line ────────────────────────────────────────────────────

class TestCrossesVerticalLine:
    def test_crossing_rightward(self):
        assert crosses_vertical_line(
            prev=(390, 100), curr=(410, 105), line_x=400.0, direction="right"
        )

    def test_no_crossing_leftward_with_right_filter(self):
        assert not crosses_vertical_line(
            prev=(410, 100), curr=(390, 105), line_x=400.0, direction="right"
        )


# ── movement_direction ───────────────────────────────────────────────────────

class TestMovementDirection:
    def test_moving_right(self):
        assert movement_direction((100, 200), (150, 205)) == "right"

    def test_moving_left(self):
        assert movement_direction((150, 200), (100, 205)) == "left"

    def test_moving_down(self):
        assert movement_direction((200, 100), (205, 150)) == "down"

    def test_moving_up(self):
        assert movement_direction((200, 150), (205, 100)) == "up"

    def test_no_movement_returns_none(self):
        """Jitter below threshold should return None."""
        assert movement_direction((200, 200), (201, 201), min_displacement=5.0) is None

    def test_zero_movement_returns_none(self):
        assert movement_direction((200, 200), (200, 200)) is None


# ── iou_1d ───────────────────────────────────────────────────────────────────

class TestIoU1D:
    def test_perfect_overlap(self):
        assert iou_1d(0, 10, 0, 10) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert iou_1d(0, 5, 10, 15) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # [0,10] ∩ [5,15] = [5,10] = 5; ∪ = 10+10-5 = 15
        assert iou_1d(0, 10, 5, 15) == pytest.approx(5 / 15)

    def test_contained(self):
        # [2,8] inside [0,10]: intersection=6, union=10
        assert iou_1d(2, 8, 0, 10) == pytest.approx(6 / 10)


# ── point_side_of_line ───────────────────────────────────────────────────────

class TestPointSideOfLine:
    def test_left_of_line(self):
        """
        In screen coordinates (Y increases downward), a point ABOVE a
        rightward line has a negative Y offset → signed area is negative.
        We test for the correct sign given screen-coord convention.
        """
        # Line from (0,0) to (10,0) going right.
        # Point at (5, -1) is above the line in screen coords.
        # Cross product: (10-0)*(−1−0) − (0−0)*(5−0) = 10*(−1) = −10 < 0
        result = point_side_of_line(
            point=(5, -1),
            line_start=(0, 0),
            line_end=(10, 0),
        )
        assert result < 0  # Above the line in screen coords

    def test_right_of_line(self):
        # Point at (5, 1) is below the line.
        # Cross product: 10*(1) − 0*(5) = 10 > 0
        result = point_side_of_line(
            point=(5, 1),
            line_start=(0, 0),
            line_end=(10, 0),
        )
        assert result > 0  # Below the line in screen coords

    def test_on_line(self):
        result = point_side_of_line(
            point=(5, 0),
            line_start=(0, 0),
            line_end=(10, 0),
        )
        assert result == pytest.approx(0.0)
