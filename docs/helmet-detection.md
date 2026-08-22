# Custom YOLO Helmet & Rider Safety Detection

## Architecture Overview

```
Traffic Frame
    ↓
General YOLO Detector (YOLOv8)
    ↓
Detections: Person (COCO 0) & Motorcycle (COCO 3)
    ↓
Rider ↔ Motorcycle Spatial Associator (rider_association.py)
    ↓
Head Region of Interest (ROI) Extractor (head_roi.py)
    ↓
Custom Helmet YOLO Detector (helmet_detector.py)
    ↓
Helmet Classification: HELMET | NO_HELMET | UNKNOWN
    ↓
Temporal Evidence Accumulator (helmet.py)
  (min_frames_tracked >= 10, min_observations >= 5, no_helmet_ratio >= 0.70)
    ↓
Confirmed NO_HELMET Violation + Evidence Snapshot
    ↓
FastAPI Backend & Next.js Dashboard
```

---

## Dataset Infrastructure & Setup Guide

### Directory Hierarchy

```
datasets/
└── helmet/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

### data.yaml Format

```yaml
path: datasets/helmet
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: helmet
  1: no_helmet
```

### Annotation Format (YOLO Normalized Coordinates)

Each label `.txt` file corresponds 1-to-1 with an image `.jpg` file:
```
<class_id> <x_center> <y_center> <width> <height>
```
Where coordinates are normalized floats in $[0.0, 1.0]$.
- `0`: helmet
- `1`: no_helmet

### Recommended Public Datasets

1. **Roboflow "Motorcycle Helmet Detection" Dataset**:
   - Source: Roboflow Universe
   - License: CC BY 4.0 (Open Source with Attribution)
   - Size: ~3,500 annotated images
   - Classes: `helmet`, `no_helmet`
2. **Kaggle Helmet Detection Dataset**:
   - Source: Kaggle Datasets
   - License: Open Data Commons Attribution License (ODC-By)

---

## Dataset Validation Script

Run `scripts/validate_helmet_dataset.py` to inspect label files, check coordinate bounds $[0, 1]$, and print class distributions:

```bash
python scripts/validate_helmet_dataset.py --data datasets/helmet/data.yaml
```

---

## Training Pipeline & Augmentations

### Transfer Learning Strategy

Start from a pretrained COCO backbone (`yolov8s.pt` or `yolov8n.pt`):
- The lower layers (backbone/neck) already extract edge, texture, and shape features.
- Fine-tuning the detection head on custom helmet crops achieves high precision with far fewer training samples.

### Augmentations for Traffic Camera Conditions

Configured in `scripts/train_helmet.py`:
- `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4`: Jitter hue, saturation, and value to simulate varying weather, shadows, and night lighting.
- `scale=0.5`: Scale jitter to handle varying rider distances from camera.
- `fliplr=0.5`: Left-right flip for orientation invariance.
- `mosaic=1.0`: Composite 4 images into 1 to improve small-object detection in crowded scenes.

### Training Command

```bash
python scripts/train_helmet.py \
    --data datasets/helmet/data.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --batch 16 \
    --device cpu
```

Outputs are saved to `runs/detect/helmet/` with `weights/best.pt`.
To deploy:
```bash
cp runs/detect/helmet/weights/best.pt models/helmet_v1.pt
```

---

## Model Evaluation & Metrics

Run `scripts/evaluate_helmet.py`:

```bash
python scripts/evaluate_helmet.py --model models/helmet_v1.pt --data datasets/helmet/data.yaml
```

Reports REAL metrics only:
- **Precision**: Fraction of predicted no-helmet violations that are correct.
- **Recall**: Fraction of actual un-helmeted riders detected.
- **mAP@50**: Mean Average Precision at IoU threshold 0.50.
- **mAP@50:95**: Mean AP across IoU thresholds from 0.50 to 0.95.
- **Inference Latency**: Benchmarked in milliseconds per head crop.

---

## Rider ↔ Motorcycle Spatial Association Math

A `person` detection is associated with a `motorcycle` if:
1. Horizontal 1D IoU:
   $$\text{IoU}_{1D}(x1_{\text{person}}, x2_{\text{person}}, x1_{\text{moto}}, x2_{\text{moto}}) \ge 0.30$$
2. Vertical position:
   $$y_{\text{center, person}} \le y1_{\text{moto}} + 140\,\text{px}$$
3. Composite association score:
   $$S_{\text{assoc}} = 0.6 \cdot \text{IoU}_{1D} + 0.4 \cdot \left(1 - \frac{|y_{\text{center, person}} - y1_{\text{moto}}|}{140}\right)$$

---

## Temporal Evidence Validation & False Positive Protection

To prevent false violations caused by single-frame motion blur or temporary occlusion:
1. **Rider History Window**: Accumulates predictions across consecutive frames for each tracked motorcycle (`_prediction_history`).
2. **Observation Thresholds**:
   - `min_frames_tracked >= 10` (ignores newly created tracks)
   - `min_observations >= 5`
   - `no_helmet_ratio >= 0.70` (at least 70% of valid observations must predict `NO_HELMET`)
3. **UNKNOWN State Guard**: If predictions are `UNKNOWN` (confidence below 0.50 or missing weights), the system **NEVER** triggers a violation.

---

## Technical Interview Q&A (Resume Preparation)

**Q: How did you train the helmet detector?**
> "We used transfer learning on YOLOv8 Nano/Small initialized with COCO weights. We fine-tuned the model on head crop images annotated with `helmet` (class 0) and `no_helmet` (class 1), using HSV color jitter, scale augmentation, and mosaic compositing."

**Q: How did you associate riders with motorcycles?**
> "We implemented a two-pass spatial association engine (`RiderAssociator`). First, general YOLO detects `person` (class 0) and `motorcycle` (class 3). We calculate 1D horizontal IoU overlap between person and motorcycle bounding boxes alongside vertical center alignment. Riders are paired with motorcycles based on a composite spatial proximity score."

**Q: Why can't you confirm a violation from a single frame?**
> "Single-frame vision predictions suffer from motion blur, head rotation, and transient occlusion. We implemented a temporal accumulator that collects predictions for each tracked motorcycle over time. A `NO_HELMET` violation is confirmed only when at least 70% of valid observations across 5+ frames predict `NO_HELMET`."

**Q: How do you differentiate detection confidence from violation confidence?**
> "YOLO detection confidence measures bounding box presence. Violation confidence is a rule-based composite score:
> $0.3 \cdot \text{det\_conf} + 0.4 \cdot \text{helmet\_model\_conf} + 0.3 \cdot \text{no\_helmet\_ratio}$. This prevents a false violation from claiming 90% confidence merely because YOLO was sure a motorcycle was present."
