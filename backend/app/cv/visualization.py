"""
backend/app/cv/visualization.py
────────────────────────────────
Drawing utilities for annotating frames with detection results.

All drawing happens on a copy of the frame — original data is never mutated.
Color scheme is consistent throughout the system:
  - Green  (#00FF00) : Normal / tracked vehicle
  - Red    (#FF0000) : Violation
  - Orange (#FF8000) : Warning / yellow light
  - Blue   (#0080FF) : ROI / informational
  - White  (#FFFFFF) : Text / labels
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.schemas.detection import BoundingBox, TrackedObject
from app.schemas.violation import Violation, ViolationType, ViolationSeverity

logger = logging.getLogger(__name__)

# ── Color constants (BGR) ─────────────────────────────────────────────────────
COLOR_NORMAL = (0, 220, 0)        # Green
COLOR_VIOLATION = (0, 0, 220)     # Red
COLOR_WARNING = (0, 130, 255)     # Orange
COLOR_INFO = (220, 130, 0)        # Blue
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_STOP_LINE = (0, 0, 255)     # Bright red

# Severity → color mapping
SEVERITY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "LOW": (0, 200, 200),
    "MEDIUM": (0, 130, 255),
    "HIGH": (0, 0, 220),
    "CRITICAL": (150, 0, 220),
}


def draw_tracked_object(
    frame: np.ndarray,
    obj: TrackedObject,
    violations: Optional[List[str]] = None,
    show_trajectory: bool = True,
    show_id: bool = True,
) -> np.ndarray:
    """
    Draw bounding box, track ID, class, and optionally trajectory.

    Color is green for normal vehicles, red if any violation is active.
    """
    out = frame.copy() if frame is not None else frame
    has_violation = bool(violations)
    color = COLOR_VIOLATION if has_violation else COLOR_NORMAL

    x1, y1, x2, y2 = obj.bbox.to_xyxy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    if show_id:
        label = f"#{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
        if violations:
            label += f" [{', '.join(violations)}]"
        _draw_label(out, label, (x1, y1), color)

    if show_trajectory and len(obj.trajectory) >= 2:
        pts = np.array(
            [(int(p[0]), int(p[1])) for p in obj.trajectory[-20:]],
            dtype=np.int32,
        )
        cv2.polylines(out, [pts], isClosed=False, color=color, thickness=1)

    return out


def draw_stop_line(
    frame: np.ndarray,
    y: int,
    label: str = "STOP LINE",
) -> np.ndarray:
    """Draw a horizontal stop line across the frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.line(out, (0, y), (w, y), COLOR_STOP_LINE, 2, cv2.LINE_AA)
    cv2.putText(
        out, label, (10, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_STOP_LINE, 1, cv2.LINE_AA,
    )
    return out


def draw_roi_polygon(
    frame: np.ndarray,
    vertices: List[Tuple[float, float]],
    color: Tuple[int, int, int] = COLOR_INFO,
    label: str = "",
    fill_alpha: float = 0.08,
) -> np.ndarray:
    """Draw a filled polygon ROI on the frame."""
    out = frame.copy()
    pts = np.array([(int(x), int(y)) for x, y in vertices], dtype=np.int32)

    if fill_alpha > 0:
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, fill_alpha, out, 1 - fill_alpha, 0, out)

    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=1)

    if label:
        cx = int(np.mean([v[0] for v in vertices]))
        cy = int(np.mean([v[1] for v in vertices]))
        cv2.putText(out, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return out


def draw_violation_overlay(
    frame: np.ndarray,
    violation: Violation,
    bbox: Optional[BoundingBox] = None,
) -> np.ndarray:
    """
    Draw a prominent violation overlay on the frame.

    Used for evidence generation — makes the violation immediately visible.
    """
    out = frame.copy()
    sev_color = SEVERITY_COLORS.get(violation.severity, COLOR_VIOLATION)

    if bbox:
        x1, y1, x2, y2 = bbox.to_xyxy()
        # Thick colored border
        cv2.rectangle(out, (x1, y1), (x2, y2), sev_color, 3)

        lines = [
            f"VIOLATION: {violation.violation_type}",
            f"Vehicle #{violation.vehicle_id}",
            f"Conf: {violation.confidence:.2f}  Sev: {violation.severity}",
        ]

        if "rider_id" in violation.metadata:
            lines.append(f"Rider #{violation.metadata['rider_id']}")

        y_text = max(y1 - 10, 80)
        for i, line in enumerate(lines):
            _draw_label(out, line, (x1, y_text - i * 18), sev_color, font_scale=0.45)

        # Draw head ROI if present in metadata
        if "head_roi_bbox" in violation.metadata:
            hx1, hy1, hx2, hy2 = violation.metadata["head_roi_bbox"]
            cv2.rectangle(out, (int(hx1), int(hy1)), (int(hx2), int(hy2)), COLOR_WARNING, 2)
            _draw_label(out, "HEAD ROI [NO HELMET]", (int(hx1), int(hy1)), COLOR_WARNING, font_scale=0.40)

    # Timestamp banner at top
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    cv2.putText(
        out, f"[{ts}] Frame {violation.frame_number}",
        (10, out.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA,
    )
    return out


def draw_hud(
    frame: np.ndarray,
    fps: float,
    frame_number: int,
    vehicle_count: int,
    violation_count: int,
    traffic_light: str = "UNKNOWN",
) -> np.ndarray:
    """
    Draw a heads-up display (HUD) panel on the top-left of the frame.
    Shows real-time statistics without cluttering the detection view.
    """
    out = frame.copy()
    panel_h, panel_w = 100, 260
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    lines = [
        f"FPS: {fps:.1f}   Frame: {frame_number}",
        f"Vehicles: {vehicle_count}",
        f"Violations: {violation_count}",
        f"Traffic Light: {traffic_light}",
    ]
    tl_color_map = {
        "RED": (0, 0, 255),
        "YELLOW": (0, 200, 255),
        "GREEN": (0, 220, 0),
        "UNKNOWN": COLOR_WHITE,
    }
    for i, line in enumerate(lines):
        color = tl_color_map.get(traffic_light, COLOR_WHITE) if i == 3 else COLOR_WHITE
        cv2.putText(
            out, line, (8, 20 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
        )
    return out


# ── Internal helpers ──────────────────────────────────────────────────────────

def _draw_label(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.45,
) -> None:
    """Draw text with a dark background for readability."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    x, y = origin
    # Background rectangle
    cv2.rectangle(
        frame,
        (x, y - th - baseline - 2),
        (x + tw + 4, y + baseline),
        COLOR_BLACK, -1,
    )
    cv2.putText(
        frame, text, (x + 2, y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA,
    )
