"""
scripts/validate_helmet_crops.py
────────────────────────────────
Validation script for Crop-based Helmet Dataset.

Verifies:
  - Image and label file pairing across train/val/test splits.
  - Class IDs (0: helmet, 1: no_helmet).
  - Normalized bbox coordinates within [0, 1].
  - Bbox width and height > 0.
  - Zero image corruption.
  - Zero data leakage between splits (train, val, test).
  - Detailed statistics report.

Usage:
    python scripts/validate_helmet_crops.py --data datasets/helmet_crops/data.yaml
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_crop_split(
    split_name: str,
    img_dir: Path,
    lbl_dir: Path,
) -> Tuple[int, int, Counter, List[str], List[Tuple[int, int]]]:
    errors: List[str] = []
    class_counts: Counter = Counter()
    dimensions: List[Tuple[int, int]] = []

    if not img_dir.exists():
        errors.append(f"Images directory missing: {img_dir}")
        return 0, 0, class_counts, errors, dimensions

    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    label_count = 0

    for img_path in img_files:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            errors.append(f"Missing label file for: {img_path.name}")
            continue

        # Check image decodability
        img = cv2.imread(str(img_path))
        if img is None or img.size == 0:
            errors.append(f"Corrupted image file: {img_path.name}")
            continue

        dimensions.append((img.shape[1], img.shape[0]))
        label_count += 1

        content = lbl_path.read_text(encoding="utf-8").strip()
        if not content:
            errors.append(f"Empty label file: {lbl_path.name}")
            continue

        lines = content.splitlines()
        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                errors.append(f"{lbl_path.name}:{line_num} Invalid line item count")
                continue

            try:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                errors.append(f"{lbl_path.name}:{line_num} Non-numeric values")
                continue

            if cls_id not in (0, 1):
                errors.append(f"{lbl_path.name}:{line_num} Invalid class ID {cls_id}")

            if not (-0.01 <= cx <= 1.01 and -0.01 <= cy <= 1.01 and 0.0 < w <= 1.01 and 0.0 < h <= 1.01):
                errors.append(f"{lbl_path.name}:{line_num} Bbox out of range [0, 1]")

            class_counts[cls_id] += 1

    return len(img_files), label_count, class_counts, errors, dimensions


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Crop-based Helmet Dataset.")
    parser.add_argument("--data", default="datasets/helmet_crops/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    yaml_path = Path(args.data)
    if not yaml_path.exists():
        logger.error("Data config not found: %s", yaml_path)
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root_dir = yaml_path.parent
    names = cfg.get("names", {0: "helmet", 1: "no_helmet"})

    print("\n" + "=" * 60)
    print(" HELMET CROP DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f" Data Config: {yaml_path}")
    print(f" Classes    : {names}")
    print("=" * 60)

    splits = ["train", "val", "test"]
    all_errors: List[str] = []
    total_images = 0
    total_labels = 0
    total_class_counts: Counter = Counter()

    stem_sets: Dict[str, set] = {}

    for split in splits:
        img_dir = root_dir / "images" / split
        lbl_dir = root_dir / "labels" / split

        stems = {f.stem.split("_rep")[0].split("_box")[0] for f in img_dir.glob("*")}
        stem_sets[split] = stems

        n_img, n_lbl, counts, errs, dims = validate_crop_split(split, img_dir, lbl_dir)
        total_images += n_img
        total_labels += n_lbl
        total_class_counts.update(counts)
        all_errors.extend(errs)

        print(f"\nSplit [{split.upper()}]:")
        print(f"  Crops / Images: {n_img}")
        print(f"  Valid Labels  : {n_lbl}")
        print(f"    Class 0 (helmet)   : {counts[0]}")
        print(f"    Class 1 (no_helmet): {counts[1]}")
        ratio = round(counts[0] / max(1, counts[1]), 2)
        print(f"    Class Ratio (helmet : no_helmet): {ratio}:1")

    # Check data leakage between train/val/test source images
    leak_train_val = stem_sets["train"].intersection(stem_sets["val"])
    leak_train_test = stem_sets["train"].intersection(stem_sets["test"])
    leak_val_test = stem_sets["val"].intersection(stem_sets["test"])

    print("\n" + "=" * 60)
    print(" DATA LEAKAGE VERIFICATION")
    print("=" * 60)
    print(f" Train vs Val Overlap  : {len(leak_train_val)} images")
    print(f" Train vs Test Overlap : {len(leak_train_test)} images")
    print(f" Val vs Test Overlap   : {len(leak_val_test)} images")

    if leak_train_val or leak_train_test or leak_val_test:
        all_errors.append("DATA LEAKAGE DETECTED across splits!")

    print("\n" + "=" * 60)
    print(" FINAL SUMMARY")
    print("=" * 60)
    print(f" Total Crops        : {total_images}")
    print(f" Total Helmet (0)   : {total_class_counts[0]}")
    print(f" Total No_Helmet (1): {total_class_counts[1]}")
    print(f" Total Errors       : {len(all_errors)}")
    print("=" * 60)

    if all_errors:
        print(f"\nValidation Errors Found ({len(all_errors)} total):")
        for err in all_errors[:10]:
            print(f"  - {err}")
    else:
        print("\nCrop dataset validation passed cleanly! 0 errors found.")


if __name__ == "__main__":
    main()
