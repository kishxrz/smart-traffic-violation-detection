# Violation Detection Logic

## Overview

This document describes how each traffic violation is detected.
The key architectural principle is:

> **YOLO detects objects. Our code applies traffic rules.**

No violation decision is made inside the detection or tracking modules.
All violation logic is in `backend/app/violations/`.

---

## Common Infrastructure

### ViolationDetector Interface

All violation detectors implement:

```python
class ViolationDetector(ABC):
    def evaluate(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        scene_state: SceneState,
    ) -> List[Violation]:
        ...
```

This enforces:
- Testability without video/GPU
- Pluggability (swap implementations without touching the engine)
- Separation of detection logic from orchestration

### SceneState

`SceneState` is the interpreted traffic scene state that violation detectors
receive. It is populated by the API route or inference pipeline before calling
the violation engine.

Key fields:
- `traffic_light_state`: RED | GREEN | YELLOW | UNKNOWN
- `stop_line_y`: pixel Y coordinate of the stop line
- `expected_direction`: configured legal direction of travel

---

## No-Helmet Violation

### Architecture

```
Tracked motorcycles
    ↓
Find associated person detections (spatial overlap)
    ↓
Extract head region (upper 25% of person bbox)
    ↓
Run HelmetClassifier.predict(head_crop)
    ↓
If "no_helmet" → generate Violation
```

### Association Logic

A person is associated with a motorcycle if:
1. Their bounding boxes have horizontal IoU > 0.30.
2. The person's center Y is within 120px above the motorcycle's top edge.

### HelmetClassifier

Currently: a stub returning `None` (no custom model loaded).

**This is intentional.** The standard COCO YOLOv8 model does not include
helmet/no-helmet classes. Claiming helmet detection from COCO YOLO would be
dishonest.

To add real detection:
1. Train YOLOv8 on a helmet dataset (classes: `helmet`, `no_helmet`).
2. Subclass `HelmetClassifier`, override `predict()`.
3. Pass the instance to `HelmetViolationDetector`.

---

## Red-Light Violation

### Logic

```
IF traffic_light_state == RED
AND tracked_vehicle.trajectory crosses stop_line_y (going forward)
THEN violation = TRUE
```

### Line Crossing Detection

```python
def crosses_line(prev, curr, line_y):
    return (prev.y - line_y) * (curr.y - line_y) < 0
```

A crossing is detected by the sign change of (y - line_y) between frames.
This requires exactly 2 tracked positions — no crossing is reported on the
first frame a vehicle appears.

### Traffic Light State

Currently populated from:
- `SceneState.traffic_light_state` (manually set or from color analysis)
- `analyze_traffic_light_color()` — HSV brightness per third of the detected bbox

Limitation: COCO YOLO detects the presence of a traffic light (class 9)
but not its color. Color analysis is a heuristic.

---

## Wrong-Way Violation

### Logic

```
dx = curr_center.x - prev_center.x
dy = curr_center.y - prev_center.y
dominant_direction = "right" if |dx| > |dy| and dx > 0 else ...

IF dominant_direction != expected_direction
AND magnitude > min_displacement
AND frames_tracked > min_frames_tracked
THEN violation = TRUE
```

### Design decisions

- `min_displacement_px = 5.0`: Prevents stopped vehicles from triggering
  violations due to tracker jitter (sub-pixel movements).
- `min_frames_tracked = 5`: New detections may have unreliable trajectories.
  We wait until a vehicle has been tracked for 5 frames before evaluating direction.
- Per-violation cooldown: prevents 30 consecutive violation records for
  the same event.

---

## Lane Violation

### Logic

```
For each frame:
  current_lane = first polygon containing vehicle_center
  prev_lane = recorded from previous frame

IF current_lane != prev_lane
AND (prev_lane, current_lane) in restricted_crossings
THEN violation = TRUE
```

### Lane polygons

Lanes are defined as `PolygonROI` objects. Containment test uses:
```python
cv2.pointPolygonTest(polygon_pts, (cx, cy), measureDist=False) >= 0
```

`restricted_crossings = None` means ALL boundary crossings are violations.
This can be refined to allow legal lane changes (e.g., only left→right or
only at specific road segments).

---

## Severity Engine

The severity engine is **not ML-based**. It uses an explicit rule table:

| Violation | Base Severity |
|-----------|--------------|
| NO_HELMET | HIGH |
| RED_LIGHT | HIGH |
| WRONG_WAY | CRITICAL |
| LANE_VIOLATION | MEDIUM |

Modifiers:
1. `confidence < 0.5` → downgrade one level
2. Same vehicle commits 3+ violations of the same type → upgrade one level

This is transparent and auditable — important for any system used in
legal/enforcement contexts.

---

## Evidence Generation

For each confirmed violation:
1. Draw violation overlay on current frame.
2. Add vehicle bbox (colored by severity), violation type label, timestamp.
3. Save as JPEG with filename: `violation_YYYYMMDD_HHMMSS_{type}_vehicle_{id}.jpg`

Cooldown: same (vehicle_id, violation_type) pair is not saved more than once
per `EVIDENCE_COOLDOWN_SECONDS` (default: 3 seconds).
