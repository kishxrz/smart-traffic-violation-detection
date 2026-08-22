"""
scripts/prepare_helmet_crop_dataset.py
───────────────────────────────────────
Crop-based Helmet Dataset Generator with Class Balancing.

Problem Solved:
  1. Distribution Mismatch: The inference pipeline feeds cropped head/rider ROIs to the
     HelmetDetector, but v1 was trained on full 640x640 traffic scenes with tiny targets.
  2. Class Imbalance: 'no_helmet' annotations represent only ~6% of the dataset.

Algorithm:
  For each image in datasets/helmet_binary/ (train, val, test):
    - For each annotated bounding box (class 0: helmet, class 1: no_helmet):
      - Extract a context-padded crop (2.5x bbox dimensions) centered on the head region.
      - Clamp crop boundaries to image dimensions [0, W] x [0, H].
      - Transform original bounding box to local crop coordinates [0, 1].
      - Save crop image and transformed label.
  In TRAIN split only:
    - Oversample 'no_helmet' crops ~5x to achieve ~1:3 class balance without data leakage.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Context padding factor (2.5x original bbox dimensions)
PADDING_FACTOR = 2.5
MIN_CROP_DIM_PX = 96
OVERSAMPLE_FACTOR_TRAIN_NO_HELMET = 5  # Oversample no_helmet in train to reach ~1:3 ratio


def extract_and_save_crop(
    img: np.ndarray,
    cls_id: int,
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    out_img_path: Path,
    out_lbl_path: Path,
) -> bool:
    """
    Extract context-padded crop around target bbox and write transformed YOLO label.
    """
    h, w = img.shape[:2]

    # Target bbox in absolute pixels
    x1_target = (cx - bw / 2) * w
    y1_target = (cy - bh / 2) * h
    x2_target = (cx + bw / 2) * w
    y2_target = (cy + bh / 2) * h

    target_w = max(1.0, x2_target - x1_target)
    target_h = max(1.0, y2_target - y1_target)
    center_x = (x1_target + x2_target) / 2.0
    center_y = (y1_target + y2_target) / 2.0

    # Crop dimension with context padding
    crop_w = max(float(MIN_CROP_DIM_PX), target_w * PADDING_FACTOR)
    crop_h = max(float(MIN_CROP_DIM_PX), target_h * PADDING_FACTOR)

    # Crop bounds clamped to image
    crop_x1 = max(0, int(round(center_x - crop_w / 2.0)))
    crop_y1 = max(0, int(round(center_y - crop_h / 2.0)))
    crop_x2 = min(w, int(round(center_x + crop_w / 2.0)))
    crop_y2 = min(h, int(round(center_y + crop_h / 2.0)))

    actual_crop_w = crop_x2 - crop_x1
    actual_crop_h = crop_y2 - crop_y1

    if actual_crop_w < 16 or actual_crop_h < 16:
        return False

    crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # Transform target bbox into local crop coordinates
    local_x1 = max(0.0, x1_target - crop_x1)
    local_y1 = max(0.0, y1_target - crop_y1)
    local_x2 = min(float(actual_crop_w), x2_target - crop_x1)
    local_y2 = min(float(actual_crop_h), y2_target - crop_y1)

    local_bw = local_x2 - local_x1
    local_bh = local_y2 - local_y1
    local_cx = local_x1 + local_bw / 2.0
    local_cy = local_y1 + local_bh / 2.0

    # Normalize to crop dimensions [0, 1]
    norm_cx = round(local_cx / actual_crop_w, 6)
    norm_cy = round(local_cy / actual_crop_h, 6)
    norm_bw = round(local_bw / actual_crop_w, 6)
    norm_bh = round(local_bh / actual_crop_h, 6)

    # Save cropped image & label
    cv2.imwrite(str(out_img_path), crop_img)
    label_text = f"{cls_id} {norm_cx:.6f} {norm_cy:.6f} {norm_bw:.6f} {norm_bh:.6f}\n"
    out_lbl_path.write_text(label_text, encoding="utf-8")

    return True


def prepare_crop_dataset(
    src_dir: Path = Path("datasets/helmet_binary"),
    dst_dir: Path = Path("datasets/helmet_crops"),
) -> None:
    logger.info("=" * 60)
    logger.info(" GENERATING CROP-BASED HELMET DATASET")
    logger.info(" Source Path     : %s", src_dir)
    logger.info(" Destination Path: %s", dst_dir)
    logger.info(" Context Padding : %.1fx bbox", PADDING_FACTOR)
    logger.info("=" * 60)

    if dst_dir.exists():
        shutil.rmtree(dst_dir)

    splits = [
        ("train", src_dir / "images" / "train", src_dir / "labels" / "train"),
        ("val", src_dir / "images" / "val", src_dir / "labels" / "val"),
        ("test", src_dir / "images" / "test", src_dir / "labels" / "test"),
    ]

    crop_dims: List[Tuple[int, int]] = []

    for split_name, img_dir, lbl_dir in splits:
        out_img_dir = dst_dir / "images" / split_name
        out_lbl_dir = dst_dir / "labels" / split_name
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        is_train = (split_name == "train")
        class_counts: Counter = Counter()
        crops_created = 0

        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

        for img_path in img_files:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                # Determine oversample repeats (5x for no_helmet in TRAIN only)
                repeats = OVERSAMPLE_FACTOR_TRAIN_NO_HELMET if (is_train and cls_id == 1) else 1

                for rep in range(repeats):
                    rep_suffix = f"_rep{rep}" if rep > 0 else ""
                    out_name = f"{img_path.stem}_box{idx}{rep_suffix}"
                    out_img_path = out_img_dir / f"{out_name}.jpg"
                    out_lbl_path = out_lbl_dir / f"{out_name}.txt"

                    success = extract_and_save_crop(img, cls_id, cx, cy, bw, bh, out_img_path, out_lbl_path)
                    if success:
                        crops_created += 1
                        class_counts[cls_id] += 1
                        c_img = cv2.imread(str(out_img_path))
                        if c_img is not None:
                            crop_dims.append((c_img.shape[1], c_img.shape[0]))

        print(f"\nSplit [{split_name.upper()}]:")
        print(f"  Total Crops Created : {crops_created}")
        print(f"    Class 0 (helmet)   : {class_counts[0]}")
        print(f"    Class 1 (no_helmet): {class_counts[1]}")
        ratio = round(class_counts[0] / max(1, class_counts[1]), 2)
        print(f"    Ratio (helmet : no_helmet): {ratio}:1")

    # Generate data.yaml
    data_yaml = dst_dir / "data.yaml"
    yaml_data = {
        "path": "datasets/helmet_crops",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,
        "names": {
            0: "helmet",
            1: "no_helmet",
        },
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    widths = [d[0] for d in crop_dims]
    heights = [d[1] for d in crop_dims]

    print("\n" + "=" * 60)
    print(" CROP DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f" Data Config Saved : {data_yaml}")
    print(f" Total Crops        : {len(crop_dims)}")
    print(f" Crop Widths (px)   : min={min(widths)}, max={max(widths)}, avg={int(np.mean(widths))}")
    print(f" Crop Heights (px)  : min={min(heights)}, max={max(heights)}, avg={int(np.mean(heights))}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_crop_dataset()
