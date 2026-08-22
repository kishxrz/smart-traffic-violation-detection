"""
scripts/calibrate_helmet_threshold.py
───────────────────────────────────────
Empirical Threshold Calibration Script for Helmet Classifier (models/helmet_v2.pt).

Evaluates models/helmet_v2.pt across confidence thresholds [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]
on the held-out TEST set (datasets/helmet_crops/images/test/).
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.detection.helmet_detector import HelmetDetector


def calibrate_thresholds(
    model_path: str = "models/helmet_v2.pt",
    test_crops_dir: str = "datasets/helmet_crops/images/test",
    test_labels_dir: str = "datasets/helmet_crops/labels/test",
    thresholds: List[float] = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20],
) -> List[Dict[str, any]]:
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Model file not found: {model_file}")
        sys.exit(1)

    label_files = glob.glob(os.path.join(test_labels_dir, "*.txt"))
    test_dataset: List[Tuple[str, int]] = []

    for lpath in label_files:
        with open(lpath) as f:
            lines = f.readlines()
        if not lines:
            continue
        gt_cls = int(lines[0].split()[0])  # 0 = HELMET, 1 = NO_HELMET
        stem = os.path.splitext(os.path.basename(lpath))[0]
        ipath = os.path.join(test_crops_dir, stem + ".jpg")
        if os.path.exists(ipath):
            test_dataset.append((ipath, gt_cls))

    total_test = len(test_dataset)
    helmet_gt_count = sum(1 for _, gt in test_dataset if gt == 0)
    no_helmet_gt_count = sum(1 for _, gt in test_dataset if gt == 1)

    print(f"Calibration Dataset Loaded: {total_test} total test crops ({helmet_gt_count} HELMET, {no_helmet_gt_count} NO_HELMET)")

    # Pre-load images to memory for fast calibration
    cached_images = [(cv2.imread(ipath), gt) for ipath, gt in test_dataset]

    results_table = []

    for thresh in thresholds:
        detector = HelmetDetector(model_path=model_file, conf_threshold=thresh)

        tp_helmet = 0
        fp_helmet = 0
        fn_helmet = 0

        tp_no_helmet = 0
        fp_no_helmet = 0
        fn_no_helmet = 0

        unknown_count = 0

        for img_mat, gt_cls in cached_images:
            if img_mat is None:
                continue

            pred = detector.predict_crop(img_mat)

            if pred.status == "UNKNOWN":
                unknown_count += 1
                if gt_cls == 0:
                    fn_helmet += 1
                elif gt_cls == 1:
                    fn_no_helmet += 1
            elif pred.status == "HELMET":
                if gt_cls == 0:
                    tp_helmet += 1
                else:
                    fp_helmet += 1
                    fn_no_helmet += 1
            elif pred.status == "NO_HELMET":
                if gt_cls == 1:
                    tp_no_helmet += 1
                else:
                    fp_no_helmet += 1
                    fn_helmet += 1

        helmet_prec = round(tp_helmet / max(1, tp_helmet + fp_helmet), 4)
        helmet_rec = round(tp_helmet / max(1, helmet_gt_count), 4)

        no_helmet_prec = round(tp_no_helmet / max(1, tp_no_helmet + fp_no_helmet), 4)
        no_helmet_rec = round(tp_no_helmet / max(1, no_helmet_gt_count), 4)

        unknown_pct = round((unknown_count / max(1, total_test)) * 100, 2)

        res_row = {
            "threshold": thresh,
            "helmet_precision": helmet_prec,
            "helmet_recall": helmet_rec,
            "no_helmet_precision": no_helmet_prec,
            "no_helmet_recall": no_helmet_rec,
            "no_helmet_fp": fp_no_helmet,
            "no_helmet_fn": fn_no_helmet,
            "unknown_count": unknown_count,
            "unknown_pct": unknown_pct,
        }
        results_table.append(res_row)

    print("\n" + "=" * 80)
    print(" THRESHOLD CALIBRATION MATRIX (HELD-OUT TEST SET)")
    print("=" * 80)
    print(f" {'Thresh':<7} | {'NO_HELMET Prec':<15} | {'NO_HELMET Rec':<15} | {'FP (False Accusations)':<22} | {'UNKNOWN %':<10}")
    print("-" * 80)
    for r in results_table:
        print(f" {r['threshold']:<7.2f} | {r['no_helmet_precision']:<15.2%} | {r['no_helmet_recall']:<15.2%} | {r['no_helmet_fp']:<22} | {r['unknown_pct']:<10.1f}%")
    print("=" * 80)

    return results_table


if __name__ == "__main__":
    calibrate_thresholds()
