"""
backend/app/violations/wrong_way.py
─────────────────────────────────────
Wrong-way driving violation detector with perspective & congestion safeguards.

Screen Coordinate System:
  - Origin (0,0) is at top-left of the image.
  - X increases RIGHTWARDS.
  - Y increases DOWNWARDS.

Supported Directions:
  - Named strings:
      "RIGHT"      : vector (1.0, 0.0)    (0°)
      "LEFT"       : vector (-1.0, 0.0)   (180°)
      "DOWN"       : vector (0.0, 1.0)    (90°) — typical towards-camera traffic
      "UP"         : vector (0.0, -1.0)   (270°) — typical away-from-camera traffic
      "DOWN_LEFT"  : vector (-0.707, 0.707) (135°)
      "DOWN_RIGHT" : vector (0.707, 0.707)  (45°)
      "UP_LEFT"    : vector (-0.707, -0.707) (225°)
      "UP_RIGHT"   : vector (0.707, -0.707)  (315°)
      "AUTO"       : Auto-calibrate direction from dominant traffic movement
  - Custom tuple: (dx, dy) 2D vector

Safeguards & Multi-Stage Pipeline:
  1. Class Filtering: Only evaluate vehicles (cars, motorcycles, buses, trucks).
  2. History Guard: Ignore new tracks (< min_frames_tracked) or short trajectories (< min_trajectory_points).
  3. Congestion / Stationary Guard: Skip vehicles with displacement < min_displacement_px or speed < min_speed_px_per_frame.
  4. Trajectory Smoothing: Compute smoothed movement vector over trajectory window using moving average.
  5. Angular Difference: Compute exact unsigned angle difference Δθ = |θ_vehicle - θ_expected| in degrees [0, 180].
  6. Sustained Disagreement: Require Δθ > min_wrong_way_angle_deg across at least sustained_ratio_threshold of trajectory steps.
  7. Confidence Decoupling: Compute separate detection_confidence, movement_confidence, and violation_confidence.
  8. Explainable Severity: Severity calculated from sustained duration, displacement, angular deviation, and repeat status.
  9. Detailed Debug Metadata: Includes direction_angle_deg, expected_angle_deg, angular_difference_deg, trajectory_distance_px, etc.
"""

from __future__ import annotations

import math
import logging
import time
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from app.schemas.detection import TrackedObject
from app.schemas.violation import SceneState, Violation, ViolationSeverity, ViolationType
from app.violations.base import ViolationDetector
from app.violations.geometry import angular_difference_degrees, vector_angle_degrees
from app.violations.severity import SeverityEngine

logger = logging.getLogger(__name__)

VEHICLE_CLASS_IDS: Set[int] = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Named directions mapped to normalized unit vectors in screen coordinates
DIRECTION_VECTORS: Dict[str, Tuple[float, float]] = {
    "RIGHT": (1.0, 0.0),
    "LEFT": (-1.0, 0.0),
    "DOWN": (0.0, 1.0),
    "UP": (0.0, -1.0),
    "DOWN_LEFT": (-0.7071, 0.7071),
    "DOWN_RIGHT": (0.7071, 0.7071),
    "UP_LEFT": (-0.7071, -0.7071),
    "UP_RIGHT": (0.7071, -0.7071),
}


def parse_direction_vector(
    direction: Union[str, Tuple[float, float]]
) -> Optional[Tuple[float, float]]:
    """Convert string name or tuple to normalized direction vector. Returns None if 'AUTO'."""
    if isinstance(direction, str):
        name = direction.upper().strip()
        if name == "AUTO":
            return None
        if name in DIRECTION_VECTORS:
            return DIRECTION_VECTORS[name]
        raise ValueError(f"Unknown direction name '{direction}'. Valid: {list(DIRECTION_VECTORS.keys())} or 'AUTO'")
    
    vx, vy = float(direction[0]), float(direction[1])
    mag = math.hypot(vx, vy)
    if mag < 1e-6:
        raise ValueError("Expected direction vector cannot be zero length.")
    return (vx / mag, vy / mag)


def smooth_trajectory(trajectory: List[Tuple[float, float]], window_size: int = 3) -> List[Tuple[float, float]]:
    """Apply moving average smoothing to a 2D trajectory to filter single-frame box jitter."""
    if len(trajectory) < window_size:
        return trajectory

    smoothed: List[Tuple[float, float]] = []
    half = window_size // 2
    n = len(trajectory)

    for i in range(n):
        start_idx = max(0, i - half)
        end_idx = min(n, i + half + 1)
        pts = trajectory[start_idx:end_idx]
        avg_x = sum(p[0] for p in pts) / len(pts)
        avg_y = sum(p[1] for p in pts) / len(pts)
        smoothed.append((avg_x, avg_y))

    return smoothed


class WrongWayViolationDetector(ViolationDetector):
    """
    Detects wrong-way vehicles with trajectory smoothing, angular tolerance, and congestion guards.
    """

    def __init__(
        self,
        severity_engine: SeverityEngine,
        expected_direction: Union[str, Tuple[float, float]] = "AUTO",
        min_displacement_px: float = 35.0,
        min_speed_px_per_frame: float = 0.8,
        min_frames_tracked: int = 15,
        min_trajectory_points: int = 10,
        sustained_ratio_threshold: float = 0.75,
        min_wrong_way_angle_deg: float = 120.0,
        cooldown_seconds: float = 5.0,
    ) -> None:
        """
        Args:
            severity_engine:           Shared severity engine.
            expected_direction:        Legal travel direction ("RIGHT", "LEFT", "DOWN", "UP", vector tuple, or "AUTO").
            min_displacement_px:       Minimum net displacement required to evaluate motion.
            min_speed_px_per_frame:    Minimum average speed per frame (ignores stationary congestion).
            min_frames_tracked:        Minimum frames tracked before checking (ignores newly created tracks).
            min_trajectory_points:     Minimum trajectory points required.
            sustained_ratio_threshold: Fraction of trajectory steps that must exceed min_wrong_way_angle_deg.
            min_wrong_way_angle_deg:   Minimum angular difference Δθ required (e.g. > 120° deviation).
            cooldown_seconds:          Cooldown seconds per vehicle ID to prevent duplicate alerts.
        """
        self._severity = severity_engine
        self._expected_direction_raw = expected_direction
        self._configured_vector = parse_direction_vector(expected_direction)
        self._min_displacement = min_displacement_px
        self._min_speed = min_speed_px_per_frame
        self._min_frames_tracked = min_frames_tracked
        self._min_trajectory_points = min_trajectory_points
        self._sustained_ratio_threshold = sustained_ratio_threshold
        self._min_wrong_way_angle_deg = min_wrong_way_angle_deg
        self._cooldown = cooldown_seconds
        self._last_violation_time: Dict[int, float] = {}

    @property
    def violation_type(self) -> ViolationType:
        return ViolationType.WRONG_WAY

    def evaluate(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        scene_state: SceneState,
    ) -> List[Violation]:
        violations: List[Violation] = []

        # Determine effective expected direction vector
        expected_vector = self._configured_vector
        if scene_state.expected_direction:
            try:
                parsed = parse_direction_vector(scene_state.expected_direction)
                if parsed is not None:
                    expected_vector = parsed
            except Exception as err:
                logger.warning("Invalid scene_state expected direction '%s': %s", scene_state.expected_direction, err)

        # Auto-calibration fallback: If vector is None ("AUTO"), estimate dominant motion from current active tracks
        if expected_vector is None:
            expected_vector = self._estimate_auto_direction(tracked_objects)

        if expected_vector is None:
            return violations  # Cannot determine expected direction yet

        ex, ey = expected_vector
        expected_angle_deg = vector_angle_degrees((ex, ey))

        for obj in tracked_objects:
            if obj.class_id not in VEHICLE_CLASS_IDS:
                continue

            # Safeguard 1: Ignore newly created tracks
            if obj.frames_tracked < self._min_frames_tracked:
                continue

            # Safeguard 2: Ignore tracks with insufficient trajectory history
            if len(obj.trajectory) < self._min_trajectory_points:
                continue

            # Apply trajectory smoothing to filter 1-frame box jitter
            smoothed = smooth_trajectory(obj.trajectory[-self._min_trajectory_points:])
            start_pt = smoothed[0]
            curr_pt = smoothed[-1]

            # Net displacement & speed calculation
            dx = curr_pt[0] - start_pt[0]
            dy = curr_pt[1] - start_pt[1]
            net_displacement = math.hypot(dx, dy)
            obs_frames = len(smoothed)
            avg_speed = net_displacement / max(1, obs_frames - 1)

            # Safeguard 3: Ignore stationary or crawling congestion traffic
            if net_displacement < self._min_displacement or avg_speed < self._min_speed:
                continue

            # Overall movement vector & vehicle angle
            vehicle_vector = (dx / net_displacement, dy / net_displacement)
            vehicle_angle_deg = vector_angle_degrees(vehicle_vector)
            net_angular_diff_deg = angular_difference_degrees(vehicle_vector, expected_vector)

            # Safeguard 4: Evaluate trajectory step vectors for sustained wrong-way movement
            wrong_way_steps = 0
            total_steps = 0
            step_angles = []

            for i in range(1, len(smoothed)):
                p_prev = smoothed[i - 1]
                p_curr = smoothed[i]
                step_dx = p_curr[0] - p_prev[0]
                step_dy = p_curr[1] - p_prev[1]
                step_len = math.hypot(step_dx, step_dy)

                if step_len < 1.0:
                    continue  # Skip negligible step movement

                total_steps += 1
                step_vec = (step_dx / step_len, step_dy / step_len)
                step_diff_deg = angular_difference_degrees(step_vec, expected_vector)
                step_angles.append(step_diff_deg)

                if step_diff_deg >= self._min_wrong_way_angle_deg:
                    wrong_way_steps += 1

            if total_steps == 0:
                continue

            sustained_ratio = wrong_way_steps / total_steps

            # Safeguard 5: Require sustained angular disagreement
            if net_angular_diff_deg < self._min_wrong_way_angle_deg or sustained_ratio < self._sustained_ratio_threshold:
                continue

            # Safeguard 6: Cooldown check
            now = time.time()
            if self._is_on_cooldown(obj.track_id, now):
                continue

            # Decoupled Confidence Calculations:
            # 1. detection_confidence: YOLO object detection score
            det_conf = obj.confidence

            # 2. movement_confidence: Measure of trajectory stability and displacement
            disp_factor = min(1.0, net_displacement / (self._min_displacement * 2.0))
            speed_factor = min(1.0, avg_speed / (self._min_speed * 2.0))
            movement_confidence = round(0.5 * disp_factor + 0.5 * speed_factor, 4)

            # 3. violation_confidence: Rule-based score combining det_conf, movement_conf, sustained_ratio & angular deviation
            angle_severity_factor = min(1.0, (net_angular_diff_deg - 90.0) / 90.0)  # 180° = 1.0
            violation_confidence = round(
                0.25 * det_conf + 0.35 * movement_confidence + 0.25 * sustained_ratio + 0.15 * angle_severity_factor,
                4,
            )

            # Calculate explainable severity
            severity, severity_score, severity_reasons = self._severity.calculate_detailed(
                violation_type=ViolationType.WRONG_WAY,
                vehicle_id=obj.track_id,
                detection_confidence=det_conf,
                violation_confidence=violation_confidence,
                displacement_px=net_displacement,
                sustained_frames=obs_frames,
            )

            # Add specific wrong-way factors to severity_reasons
            severity_reasons.append(
                f"Angular deviation {net_angular_diff_deg:.1f}° vs legal direction (thresh: >{self._min_wrong_way_angle_deg:.0f}°)"
            )
            severity_reasons.append(
                f"Sustained wrong-way movement in {wrong_way_steps}/{total_steps} trajectory segments ({sustained_ratio*100:.0f}%)"
            )

            self._last_violation_time[obj.track_id] = now

            logger.info(
                "WRONG WAY violation: vehicle #%d (%s), Δθ=%.1f°, disp=%.1fpx, sustained=%.0f%%, severity=%s (%d pts)",
                obj.track_id, obj.class_name, net_angular_diff_deg, net_displacement, sustained_ratio * 100, severity.value, severity_score,
            )

            violations.append(
                Violation(
                    violation_type=ViolationType.WRONG_WAY,
                    vehicle_id=obj.track_id,
                    vehicle_class=obj.class_name,
                    confidence=violation_confidence,
                    detection_confidence=det_conf,
                    violation_confidence=violation_confidence,
                    severity=severity,
                    severity_score=severity_score,
                    severity_reasons=severity_reasons,
                    frame_number=scene_state.frame_number,
                    timestamp=scene_state.timestamp,
                    metadata={
                        "direction_angle_deg": round(vehicle_angle_deg, 1),
                        "expected_angle_deg": round(expected_angle_deg, 1),
                        "angular_difference_deg": round(net_angular_diff_deg, 1),
                        "trajectory_distance_px": round(net_displacement, 1),
                        "avg_speed_px_per_frame": round(avg_speed, 2),
                        "frames_observed": obs_frames,
                        "movement_vector": [round(vehicle_vector[0], 3), round(vehicle_vector[1], 3)],
                        "expected_vector": [round(ex, 3), round(ey, 3)],
                        "sustained_wrong_ratio": round(sustained_ratio, 2),
                        "movement_confidence": movement_confidence,
                    },
                )
            )

        return violations

    def _estimate_auto_direction(
        self, tracked_objects: List[TrackedObject]
    ) -> Optional[Tuple[float, float]]:
        """Estimate dominant traffic direction from active vehicle trajectories."""
        valid_vectors = []
        for obj in tracked_objects:
            if obj.class_id in VEHICLE_CLASS_IDS and len(obj.trajectory) >= 5:
                pts = obj.trajectory
                dx = pts[-1][0] - pts[0][0]
                dy = pts[-1][1] - pts[0][1]
                mag = math.hypot(dx, dy)
                if mag >= 15.0:
                    valid_vectors.append((dx / mag, dy / mag))

        if len(valid_vectors) < 2:
            return None  # Need at least 2 moving vehicles to estimate

        mean_dx = sum(v[0] for v in valid_vectors) / len(valid_vectors)
        mean_dy = sum(v[1] for v in valid_vectors) / len(valid_vectors)
        mag = math.hypot(mean_dx, mean_dy)
        if mag < 1e-4:
            return None
        return (mean_dx / mag, mean_dy / mag)

    def _is_on_cooldown(self, vehicle_id: int, now: float) -> bool:
        last = self._last_violation_time.get(vehicle_id, 0.0)
        return (now - last) < self._cooldown

    def reset(self) -> None:
        self._last_violation_time.clear()
