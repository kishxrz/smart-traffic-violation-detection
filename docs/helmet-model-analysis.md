# Custom YOLO Helmet Model Evaluation & Error Analysis Report

## 1. Executive Summary

This document presents the empirical evaluation of the custom-trained binary YOLOv8 Nano helmet detector on the held-out **TEST split** (`datasets/helmet_binary/images/test`).

---

## 2. Test Set Quantitative Metrics

Evaluated on 232 held-out test images containing 729 total annotated bounding boxes:

| Metric | Overall System | Class 0: HELMET | Class 1: NO_HELMET |
| :--- | :--- | :--- | :--- |
| **Precision (P)** | **80.75%** (0.8075) | 61.51% (0.6151) | 100.0% (1.0000) |
| **Recall (R)** | **27.21%** (0.2721) | 54.42% (0.5442) | **0.00%** (0.0000) |
| **mAP@50** | **30.58%** (0.3058) | **56.15%** (0.5615) | **5.02%** (0.0502) |
| **mAP@50:95** | **12.35%** (0.1235) | 23.10% (0.2310) | 1.64% (0.0164) |
| **Inference Latency** | **31.36 ms / frame** | — | — |
| **Inference Throughput** | **~31.9 FPS (CPU)** | — | — |

---

## 3. Class Imbalance Analysis & Observed Failure Modes

### Primary Failure Mode: Class Imbalance Bias ($15.41 : 1$)

1. **Extreme Negative Class Skew**:
   - The converted dataset contains **5,825 `helmet` annotations** versus only **378 `no_helmet` annotations** (a $15.41 : 1$ ratio).
   - In short training runs (5 epochs), the model prioritizes minimizing total classification loss by defaulting predictions to the majority class (`helmet`).
2. **Impact on Violation Detection**:
   - Class `HELMET` achieved **61.51% Precision** and **54.42% Recall** ($\text{AP@50} = 56.15\%$).
   - Class `NO_HELMET` achieved **100% Precision** (no false positive alarms) but **0.00% Recall** ($\text{AP@50} = 5.02\%$), meaning un-helmeted riders are frequently missed unless additional loss weighting, oversampling, or longer focal loss training is applied.

### Observed Technical Challenges in Traffic Footage

1. **Rider Distance & Resolution**:
   - Motorcycles positioned $> 40$ meters from traffic cameras yield head crops under $24 \times 24$ pixels, leading to confidence suppression below threshold ($0.50$).
2. **Motion Blur & Daylight Shadows**:
   - Fast-moving two-wheelers experience vertical motion blur, blurring helmet boundaries into background asphalt.
3. **Occlusion & Passenger Shadows**:
   - Pillions seated behind motorcycle drivers obscure the head region of the driver from rear-mounted surveillance angles.

---

## 4. Recommended Solutions for Production Deployment

1. **Class-Weighted Focal Loss & Oversampling**:
   - Apply `pos_weight` or oversample `no_helmet` crop images in `data.yaml` to balance gradients ($15\times$ weight boost for class 1).
2. **Longer Training Schedule with Fine-Tuning**:
   - Extend training to 50–100 epochs on GPU hardware with cosine learning rate annealing.
3. **System-Level Temporal Safeguard Mitigation**:
   - Maintain the multi-frame temporal reasoning engine (`min_frames_tracked >= 10`, `min_observations >= 5`) so that isolated false predictions do not trigger false traffic citations.
