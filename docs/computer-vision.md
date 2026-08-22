# Computer Vision Technical Reference

## Overview

This document explains the OpenCV operations used in the Smart Traffic AI
pipeline and their mathematical basis.

---

## 1. Image Preprocessing

### Gaussian Blur

**Purpose:** Reduces high-frequency sensor noise before edge detection or
object detection inference.

**Mathematics:**
```
G(x, y) = (1 / 2πσ²) × exp(-(x² + y²) / 2σ²)
```
Convolution of input I with Gaussian kernel G:
```
I_blurred(x, y) = Σ G(x-i, y-j) × I(i, j)
```

**When to use:** Before Canny edge detection or when the camera produces
noisy low-light footage.

**OpenCV call:**
```python
cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
```

---

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Purpose:** Improves local contrast in overexposed or underexposed frames
without washing out already-bright regions.

**Why CLAHE over global histogram equalization?**
Global HE amplifies noise in uniform regions. CLAHE divides the image into
tiles, equalizes each tile's histogram, and clips the enhancement to a
configurable limit before bilinear interpolation between tiles.

**Implementation detail:**
We convert BGR → L*a*b*, apply CLAHE only to the L (luminance) channel,
then convert back to BGR. This avoids shifting hue values — critical for
accurate traffic light color detection.

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l_eq = clahe.apply(l)
result = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
```

---

### Canny Edge Detection

**The Canny algorithm:**
1. **Gaussian smoothing** — reduce noise.
2. **Sobel gradient** — compute image gradient magnitude and direction.
3. **Non-maximum suppression** — thin edges to 1-pixel width.
4. **Hysteresis thresholding** — keep strong edges (> high_threshold),
   discard weak ones (< low_threshold), keep medium ones if connected to strong.

**Uses in this system:** Lane boundary localization, stop-line detection.

```python
edges = cv2.Canny(gray, low_threshold=50, high_threshold=150)
```

---

### Morphological Operations

**Erosion:** Shrinks foreground regions (removes small noise blobs).
**Dilation:** Expands foreground regions.
**Opening (erosion then dilation):** Removes small noise.
**Closing (dilation then erosion):** Fills small holes.

Uses in this system: Cleaning binary masks for ROI analysis.

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
```

---

## 2. Region of Interest (ROI)

### Purpose

ROIs define the monitored zone within a camera frame. Processing only
the ROI:
- Eliminates false positives from off-road objects.
- Reduces compute by ignoring irrelevant pixels.

### Rectangle ROI

Simple axis-aligned box:
```python
cv2.rectangle(mask, (x, y), (x2, y2), 255, -1)
```

### Polygon ROI

Arbitrary shape using `cv2.fillPoly()`:
```python
pts = np.array(vertices, dtype=np.int32)
cv2.fillPoly(mask, [pts], 255)
result = cv2.bitwise_and(frame, frame, mask=mask)
```

### Point-in-Polygon Test

Uses OpenCV's `pointPolygonTest()`, which implements the winding-number
or ray-casting algorithm:
```python
result = cv2.pointPolygonTest(pts, (px, py), measureDist=False)
# result >= 0 means inside or on boundary
```

---

## 3. Perspective Transformation

### Problem

Traffic cameras are mounted at an angle. This creates perspective
distortion — objects farther away appear smaller and their positions
are compressed. This makes direct distance and direction measurements
unreliable.

### Solution: Homography

A perspective transformation is described by a 3×3 homography matrix H:

```
[x', y', w']ᵀ = H × [x, y, 1]ᵀ

(x_corrected, y_corrected) = (x'/w', y'/w')
```

**Computing H:** Given 4 point correspondences between source and
destination quadrilaterals, OpenCV uses the Direct Linear Transform
(DLT) algorithm with SVD to find the least-squares H.

```python
# src_quad: 4 points on the road in camera view
# dst_size: output bird's-eye image dimensions
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
bird_eye = cv2.warpPerspective(frame, M, dst_size)
```

### Inverse Transform

To draw annotations on the original frame after analyzing in bird's-eye:
```python
M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
original_pt = cv2.perspectiveTransform(bird_eye_pt, M_inv)
```

### Applications in This System

1. **Lane width measurement** — parallel lanes in bird's-eye view have
   consistent pixel width, allowing width estimation.
2. **Trajectory direction** — a vehicle moving "right" in bird's-eye is
   genuinely moving in the road direction.
3. **Stop-line crossing** — more accurate in top-down coordinates.

---

## 4. Video Processing

### Frame Extraction

```python
cap = cv2.VideoCapture(source)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    process(frame)
```

### Frame Skip

At 30 FPS, processing every 3rd frame gives 10 FPS of inference.
This reduces GPU load by 3× with negligible impact on violation detection
(vehicles don't disappear in 100ms).

```python
if frame_number % frame_skip == 0:
    yield frame_number, timestamp, frame
```

### Output Video Writing

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
writer.write(annotated_frame)
```

---

## 5. Traffic Light Color Detection

### Approach

1. Detect traffic light bounding box via YOLO (class 9 in COCO).
2. Crop the bounding box region.
3. Convert to HSV (Hue-Saturation-Value).
4. Divide into thirds (top=red zone, mid=yellow zone, bottom=green zone).
5. Measure average brightness (Value channel) in each third.
6. The brightest third indicates the active light.

### Limitations

- Fails at night (all regions are dark → UNKNOWN).
- Fails under glare or non-standard fixtures.
- Does not distinguish between a real light and a red advertisement board.
- Production quality requires a dedicated classifier model.

```python
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
brightness_top = hsv[:third, :, 2].mean()    # Red zone
brightness_mid = hsv[third:2*third, :, 2].mean()  # Yellow zone
brightness_bot = hsv[2*third:, :, 2].mean()  # Green zone
```
