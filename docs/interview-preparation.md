# Interview Preparation Guide

This document prepares you to explain every component of the Smart Traffic AI
system in a technical interview. Answers are written at the level expected
from a senior ML/CV engineer.

---

## OpenCV

### Why OpenCV?

OpenCV (Open Source Computer Vision Library) is the industry standard for
real-time image processing. It provides:

1. Hardware-accelerated pixel operations via SIMD intrinsics.
2. Optimized implementations of standard algorithms (Canny, SIFT, etc.).
3. Direct integration with NumPy arrays (zero-copy via shared memory).
4. `cv2.VideoCapture` — a unified interface to files, cameras, and streams.

**Interview answer:** "OpenCV handles the low-level image manipulation that
YOLO doesn't need to know about — resizing, color space conversions, ROI
masking, perspective warping, and video I/O."

---

### Preprocessing

**Gaussian blur:** Convolves the image with a Gaussian kernel to attenuate
high-frequency noise before edge detection. We use kernel_size=5 (σ≈1).

**CLAHE:** Contrast Limited Adaptive Histogram Equalization — improves local
contrast without amplifying noise. We apply it to the L channel in L*a*b*
space to avoid hue shifts (important for traffic light detection).

**Canny:** 4-step edge detector: smoothing → Sobel gradient → non-max
suppression → hysteresis thresholding. Used for lane boundary localization.

**Morphological operations:** Erosion/dilation applied to binary masks.
Opening removes small noise blobs; closing fills holes.

---

### ROI (Region of Interest)

An ROI is a sub-region of the frame that bounds the monitored traffic zone.

**Why:** False positives from vehicles on side roads, pedestrians on
pavements, or ad-boards can inflate violation counts. ROIs prevent this.

**Implementation:**
```python
mask = np.zeros(frame.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [polygon_vertices], 255)
roi_frame = cv2.bitwise_and(frame, frame, mask=mask)
```

**Point containment:**
```python
cv2.pointPolygonTest(polygon, point, measureDist=False) >= 0
```

---

### Perspective Transformation

**Problem:** A road camera at an angle makes vehicles appear smaller at
distance, making direction and distance measurements inaccurate.

**Solution:** Homography — a 3×3 matrix H that maps camera coordinates to
a top-down bird's-eye view. 4 point correspondences uniquely determine H.

**OpenCV:**
```python
M = cv2.getPerspectiveTransform(src_quad, dst_quad)
bird_eye = cv2.warpPerspective(frame, M, (w, h))
```

---

## Deep Learning

### CNN Fundamentals

A Convolutional Neural Network extracts hierarchical features:
- Early layers: edges, textures.
- Mid layers: shapes, object parts.
- Late layers: semantic concepts (car, person).

Convolution: `(I * K)(x,y) = Σ I(x+i, y+j) × K(i, j)`

Pooling reduces spatial dimensions, increasing receptive field.
Batch Normalization stabilizes training.
ReLU introduces non-linearity.

---

### YOLO (You Only Look Once)

**Architecture:**
- **Backbone:** CSP-DarkNet (YOLOv8) — feature extraction.
- **Neck:** PANet — multi-scale feature fusion (detects small and large objects).
- **Head:** Anchor-free detection head (YOLOv8) — outputs [cx, cy, w, h, conf, class_probs].

**Single-pass:** Unlike R-CNN which uses a separate region proposal step,
YOLO processes the entire image once and predicts all boxes simultaneously.

**Interview:** "YOLO trades some recall for speed. For traffic cameras at
10-30 FPS, it's the right choice. We use YOLOv8n for CPU deployment and
YOLOv8m for better accuracy on a GPU."

---

### Bounding Boxes and Confidence

A bounding box is (x1, y1, x2, y2) in pixel coordinates.
The confidence score = P(object) × IoU(predicted, ground truth).
We filter at conf=0.45 — a tunable threshold balancing precision/recall.

---

### IoU (Intersection over Union)

```
IoU = Area(A ∩ B) / Area(A ∪ B)
```

Range [0.0, 1.0]. Used for:
1. NMS — suppress duplicate boxes (IoU > threshold → keep higher-conf box).
2. Tracking assignment — match detections to existing tracks.
3. Model evaluation — mAP calculation uses IoU thresholds.

---

### NMS (Non-Maximum Suppression)

Multiple anchors may detect the same object. NMS:
1. Sort boxes by confidence (highest first).
2. Keep the top box.
3. Remove all other boxes with IoU > threshold against the kept box.
4. Repeat.

```python
keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.45)
```

---

### Precision, Recall, mAP

```
Precision = TP / (TP + FP)  — of what we detected, how much was correct?
Recall    = TP / (TP + FN)  — of all real objects, how many did we find?
```

**mAP@50:** Mean Average Precision at IoU threshold 0.5.
**mAP@50:95:** Averaged over IoU thresholds 0.50, 0.55, …, 0.95.
This is the primary YOLO benchmark metric — harder to game with low-IoU detections.

---

## Object Tracking

### Why tracking is necessary

Detection tells us "there is a car in this frame at (x1, y1, x2, y2)".
It cannot tell us "this is the same car as in the previous frame".

Without tracking:
- Cannot measure direction of travel.
- Cannot count unique vehicles (a car detected in 300 frames would count 300 times).
- Cannot accumulate trajectory for violation detection.

### Our Tracker: IoU + Hungarian Algorithm

**Assignment problem:** Given N detections and M existing tracks, find the
optimal pairing that maximises total IoU overlap.

**Hungarian algorithm (scipy.linear_sum_assignment):**
Solves the linear assignment problem in O(N³) — optimal globally, not greedy.

**Trajectory:** Each track stores its last 50 center positions.
Movement vector = curr_center - prev_center.

### Limitations

- **ID switches** when two identical vehicles cross in front of each other.
- **Solution:** DeepSORT or ByteTrack add re-identification embeddings from
  a separate CNN to distinguish vehicles even when they overlap.

---

## Traffic Violation Logic

### Separation of concerns

```
YOLO detects: class, bbox, confidence
Our logic concludes: violation type, vehicle identity, severity
```

This is the key architectural insight — YOLO does not "know" about traffic
laws. Our violation engine applies configurable rules.

### Line Crossing

The fundamental operation for red-light and stop-line detection:

```python
# A trajectory segment crosses a line if the y values straddle line_y
(prev_y - line_y) * (curr_y - line_y) < 0
```

### Direction Vectors

```python
dx = curr_center[0] - prev_center[0]
dy = curr_center[1] - prev_center[1]
direction = "right" if dx > |dy| and dx > 0 else ...
```

Minimum displacement threshold prevents stopped-vehicle jitter from
triggering false wrong-way violations.

### Lane Polygons

Each lane is a polygon (PolygonROI). Per-frame, we classify each vehicle
into a lane using `cv2.pointPolygonTest`. A lane change is detected when
the current-frame lane differs from the previous-frame lane.

---

## FastAPI Backend

### Architecture

```
HTTP request
  → CORS middleware
  → Router (routes/detection.py, routes/violations.py, etc.)
  → Dependency injection (get_detector(), get_tracker(), etc.)
  → Business logic (frame_processor, violation_engine)
  → Pydantic schema serialization
  → JSON response
```

### Pydantic schemas

All request and response data is validated through Pydantic models. This
ensures type safety and generates automatic OpenAPI documentation.

### Dependency injection

FastAPI's `Depends()` injects shared resources (detector, tracker) without
singletons scattered throughout business logic. Resources are created once
at startup and reused across requests.

---

## Docker

### Multi-stage build

```dockerfile
# Stage 1: install dependencies
FROM python:3.13-slim AS deps
RUN pip install --prefix=/opt/venv -r requirements.txt

# Stage 2: runtime (no build tools, smaller image)
FROM python:3.13-slim AS runtime
COPY --from=deps /opt/venv /usr/local
COPY app/ ./app/
```

**Why multi-stage?** The final image doesn't include pip, compilers, or
build artifacts — only the installed packages. Smaller attack surface,
faster deployment.

### CPU vs GPU inference

- CPU: Always available, ~5-10 FPS on YOLOv8n, sufficient for recorded video.
- GPU (CUDA): ~100+ FPS on a modern GPU, required for live streams.
- Apple Silicon (MPS): Native acceleration via PyTorch MPS backend.

Switch via: `DEVICE=cuda` in `.env`.

---

## Frontend Dashboard

### API communication

```typescript
const response = await fetch(`${API_URL}/api/analytics/summary`);
const data = await response.json();
```

### State management

React `useEffect` hooks with `useState` for local component state.
SWR or React Query can be added for caching and automatic revalidation.

### Real-time updates

For live stream processing, WebSockets would replace REST polling.
The backend would push frame results via a WebSocket endpoint.
