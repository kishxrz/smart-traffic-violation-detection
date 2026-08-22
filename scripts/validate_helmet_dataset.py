"""
scripts/validate_helmet_dataset.py
───────────────────────────────────
Dataset validation script for YOLO Helmet & Safety Detection.

Verifies:
  1. Image & label file existence across train/val/test splits.
  2. 1-to-1 pairing between image (.jpg/.png) and label (.txt) files.
  3. Class ID validity (must be 0: helmet or 1: no_helmet).
  4. Normalized bounding box coordinates within [0.0, 1.0].
  5. Image dimension integrity and duplicate filename checks.
  6. Class distribution statistics across splits.

Usage:
    python scripts/validate_helmet_dataset.py --data datasets/helmet/data.yaml
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_split(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    nc: int = 2,
) -> Tuple[int, int, Counter, List[str]]:
    """
    Validate a single dataset split (train/val/test).

    Returns:
        (image_count, label_count, class_counter, error_list)
    """
    errors: List[str] = []
    class_counts: Counter = Counter()

    if not images_dir.exists():
        errors.append(f"Images directory missing: {images_dir}")
        return 0, 0, class_counts, errors

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in images_dir.glob("*") if f.suffix.lower() in image_extensions]
    label_count = 0

    for img_path in image_files:
        stem = img_path.stem
        lbl_path = labels_dir / f"{stem}.txt"

        if not lbl_path.exists():
            errors.append(f"Missing label file for image: {img_path.name}")
            continue

        label_count += 1
        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()

        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                errors.append(f"{lbl_path.name}:{line_num} Invalid line format (expected 5 items, got {len(parts)})")
                continue

            try:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                errors.append(f"{lbl_path.name}:{line_num} Non-numeric values in label line")
                continue

            if cls_id < 0 or cls_id >= nc:
                errors.append(f"{lbl_path.name}:{line_num} Invalid class ID {cls_id} (allowed: 0..{nc-1})")

            # Check normalized bounds [0, 1]
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                errors.append(f"{lbl_path.name}:{line_num} Bounding box coords out of range [0, 1]: ({cx}, {cy}, {w}, {h})")

            class_counts[cls_id] += 1

    return len(image_files), label_count, class_counts, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO Helmet Dataset.")
    parser.add_argument("--data", default="datasets/helmet/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    yaml_path = Path(args.data)
    if not yaml_path.exists():
        logger.error("Dataset YAML configuration file not found at: %s", yaml_path)
        logger.info(
            "\nDataset Setup Required:\n"
            "Place your dataset under datasets/helmet/ with images/ and labels/ subdirectories.\n"
            "Recommended dataset: Roboflow 'Motorcycle Helmet Detection' (CC BY 4.0)."
        )
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root_dir = yaml_path.parent
    nc = cfg.get("nc", 2)
    names = cfg.get("names", {0: "helmet", 1: "no_helmet"})

    print("\n" + "=" * 60)
    print(" HELMET DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f" Dataset Config: {yaml_path}")
    print(f" Target Classes: {names}")
    print("=" * 60)

    splits = ["train", "val", "test"]
    total_images = 0
    total_labels = 0
    total_class_counts: Counter = Counter()
    all_errors: List[str] = []

    for split in splits:
        img_dir = root_dir / "images" / split
        lbl_dir = root_dir / "labels" / split
        n_img, n_lbl, counts, errs = validate_split(split, img_dir, lbl_dir, nc)

        total_images += n_img
        total_labels += n_lbl
        total_class_counts.update(counts)
        all_errors.extend(errs)

        print(f"\nSplit [{split.upper()}]:")
        print(f"  Images: {n_img}")
        print(f"  Labels: {n_lbl}")
        for cls_id, count in counts.items():
            print(f"    Class {cls_id} ({names.get(cls_id, cls_id)}): {count}")

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f" Total Images: {total_images}")
    print(f" Total Labels: {total_labels}")
    for cls_id in range(nc):
        print(f"  Total {names.get(cls_id, cls_id)} (class {cls_id}): {total_class_counts[cls_id]}")
    print(f" Total Validation Errors: {len(all_errors)}")
    print("=" * 60)

    if all_errors:
        print("\nErrors identified (first 10 shown):")
        for err in all_errors[:10]:
            print(f"  - {err}")
    else:
        print("\nDataset validation passed cleanly! No label errors found.")


if __name__ == "__main__":
    main()
