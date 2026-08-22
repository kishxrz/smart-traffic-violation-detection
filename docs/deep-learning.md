# Deep Learning Reference

## YOLOv8 Architecture

```
Input (640×640×3)
    ↓
Backbone: CSP-DarkNet
  → Multi-scale feature maps: P3, P4, P5
    ↓
Neck: PANet (Path Aggregation Network)
  → Fuses features from multiple scales
  → Top-down + bottom-up connections
    ↓
Head: Anchor-free detection
  → 3 output scales (80×80, 40×40, 20×20)
  → Per cell: [x, y, w, h] + cls_probs
    ↓
Post-processing: NMS (Non-Maximum Suppression)
    ↓
Detections: [(class_id, confidence, x1, y1, x2, y2), ...]
```

### Why anchor-free (YOLOv8) vs anchor-based (YOLOv5)?

YOLOv5 uses predefined anchor boxes (aspect ratios computed from dataset statistics).
YOLOv8 directly predicts box coordinates without anchors:
- Simpler training (no anchor tuning per dataset)
- Better generalization to new object shapes
- Slightly faster inference

---

## Training Pipeline

### Transfer Learning Strategy

```
1. Load pretrained COCO weights (80 classes, 3M-100M parameters)
2. Replace the detection head for the new number of classes
3. Freeze backbone for first N epochs (optional, for small datasets)
4. Train the full network
5. Save best.pt (highest val/mAP50)
```

### Loss Functions

YOLOv8 uses three loss components:

1. **Box regression loss (CIoU):**
   Complete IoU = IoU + distance penalty + aspect ratio penalty
   Better than MSE for bounding box regression.

2. **Classification loss (BCE):**
   Binary Cross-Entropy per class.

3. **DFL (Distribution Focal Loss):**
   Improves localization accuracy by predicting a distribution over box edge coordinates.

Total: `L = λ_box × L_box + λ_cls × L_cls + λ_dfl × L_dfl`

---

## Evaluation Metrics

### Precision, Recall, F1

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × (P × R) / (P + R)
```

A detection is True Positive if:
- The predicted class matches ground truth
- IoU(predicted_box, ground_truth_box) ≥ threshold (typically 0.5)

### Average Precision (AP)

AP is the area under the Precision-Recall curve.
A perfect detector has AP = 1.0.

### mAP@50

Mean AP across all classes, at IoU threshold 0.50.

### mAP@50:95

Mean AP averaged over IoU thresholds 0.50, 0.55, 0.60, …, 0.95.
This is the primary COCO benchmark metric — harder to game.

---

## Inference Performance (Reference — COCO pretrained)

| Model | mAP@50 | mAP@50:95 | Params | CPU FPS* |
|-------|--------|-----------|--------|---------|
| YOLOv8n | 52.9% | 37.3% | 3.2M | ~12 |
| YOLOv8s | 61.8% | 44.9% | 11.2M | ~7 |
| YOLOv8m | 67.2% | 50.2% | 25.9M | ~3 |

*Approximate on modern CPU at 640×640 input.

> **These are COCO benchmark numbers, not traffic-specific numbers.**
> Traffic-specific evaluation requires a labeled traffic dataset.

---

## Object Tracker (IoU + Hungarian)

### Why not just re-detect every frame?

Re-detecting every frame:
- Loses vehicle identity between frames
- Cannot compute direction of travel
- Cannot count unique vehicles
- Cannot accumulate trajectory for violation detection

### Hungarian Algorithm

Given N detections and M existing tracks, the optimal assignment
that maximises total IoU is computed in O(N³) time using scipy:

```python
from scipy.optimize import linear_sum_assignment
row_inds, col_inds = linear_sum_assignment(-cost_matrix)
```

For each matched (detection, track) pair with IoU ≥ threshold:
- Track is updated with the new detection bbox
- Trajectory is extended

Unmatched detections → new tracks.
Tracks unmatched for > max_age frames → removed.

### Limitations and Alternatives

| Method | Strength | Limitation |
|--------|----------|-----------|
| IoU tracker (this system) | No extra model, fast | ID switches on overlap |
| DeepSORT | Re-ID embeddings prevent switches | Requires re-ID model |
| ByteTrack | High recall, low-confidence detections | Needs tuning |
| StrongSORT | State-of-the-art accuracy | Complex setup |

---

## Custom Helmet Model Training Guide

### 1. Data Collection

Collect images from traffic cameras showing motorcycle riders.
Aim for variety: day, night, different motorcycle types, different helmet colors.

Minimum: 2,500 images per class (helmet, no_helmet).

### 2. Annotation

Using Roboflow (recommended):
1. Create project → Object Detection
2. Upload images
3. Annotate: draw bounding box around head region
4. Label: `helmet` or `no_helmet`
5. Export as YOLOv8 format

### 3. Training

```bash
python scripts/train.py \
    --task helmet \
    --data datasets/helmet/data.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --device cpu
```

Expected training time: 4-8 hours on CPU for 100 epochs.
With GPU (RTX 3080): ~30 minutes.

### 4. Deployment

```bash
cp runs/detect/helmet/weights/best.pt models/helmet_v1.pt
# Set in .env:
HELMET_MODEL_PATH=models/helmet_v1.pt
```

The `HelmetClassifier` in `app/violations/helmet.py` needs to be
subclassed to load and run this model.
