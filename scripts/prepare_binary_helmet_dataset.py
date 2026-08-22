"""
scripts/prepare_binary_helmet_dataset.py
──────────────────────────────────────────
Converts the 4-class Roboflow helmet dataset into a clean binary dataset.

Source Mapping (datasets/helmet/):
  Class 0 (full-faced)  → Binary Class 0 (helmet)
  Class 1 (half-faced)  → Binary Class 0 (helmet)
  Class 2 (invalid)     → DISCARD (neither helmet nor no_helmet)
  Class 3 (no helmet)   → Binary Class 1 (no_helmet)

Target Destination (datasets/helmet_binary/):
  Binary Class 0: helmet
  Binary Class 1: no_helmet

Policy for empty annotations:
  Images where all annotations were 'invalid' are retained as background/negative images
  (containing 0 annotations) in the training set to help reduce false positive rate in YOLO object detectors.
"""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Class Mapping Definition
CLASS_MAP = {
    0: 0,  # full-faced -> helmet
    1: 0,  # half-faced -> helmet
    # 2 is DISCARDED
    3: 1,  # no helmet  -> no_helmet
}

CLASS_NAMES = {
    0: "full-faced",
    1: "half-faced",
    2: "invalid",
    3: "no helmet",
}


def process_split(
    split_name: str,
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
) -> Tuple[int, int, Counter, Counter, int]:
    """
    Process a single split (train/valid/test), copying images and mapping labels.

    Returns:
        (images_processed, labels_written, binary_class_counter, conversion_source_counter, discarded_invalid_count)
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in src_img_dir.glob("*") if f.suffix.lower() in image_extensions]

    images_processed = 0
    labels_written = 0
    binary_counts: Counter = Counter()
    source_counts: Counter = Counter()
    discarded_invalid = 0

    for img_path in image_files:
        stem = img_path.stem
        lbl_path = src_lbl_dir / f"{stem}.txt"

        # Copy image to target
        dst_img_path = dst_img_dir / img_path.name
        shutil.copy2(img_path, dst_img_path)
        images_processed += 1

        binary_lines: List[str] = []

        if lbl_path.exists():
            content = lbl_path.read_text(encoding="utf-8").strip()
            if content:
                for line in content.splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        src_cls = int(parts[0])
                        cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]
                    except ValueError:
                        continue

                    source_counts[src_cls] += 1

                    if src_cls in CLASS_MAP:
                        target_cls = CLASS_MAP[src_cls]
                        binary_lines.append(f"{target_cls} {cx} {cy} {w} {h}")
                        binary_counts[target_cls] += 1
                        labels_written += 1
                    elif src_cls == 2:
                        discarded_invalid += 1

        # Write converted label file
        dst_lbl_path = dst_lbl_dir / f"{stem}.txt"
        dst_lbl_path.write_text("\n".join(binary_lines) + ("\n" if binary_lines else ""), encoding="utf-8")

    return images_processed, labels_written, binary_counts, source_counts, discarded_invalid


def create_binary_dataset(
    src_root: Path = Path("datasets/helmet"),
    dst_root: Path = Path("datasets/helmet_binary"),
) -> None:
    logger.info("=" * 60)
    logger.info(" PREPARING BINARY HELMET DATASET")
    logger.info(" Source Path     : %s", src_root)
    logger.info(" Destination Path: %s", dst_root)
    logger.info("=" * 60)

    if dst_root.exists():
        logger.info("Cleaning existing binary dataset directory: %s", dst_root)
        shutil.rmtree(dst_root)

    splits = [
        ("train", src_root / "train" / "images", src_root / "train" / "labels", dst_root / "images" / "train", dst_root / "labels" / "train"),
        ("val", src_root / "valid" / "images", src_root / "valid" / "labels", dst_root / "images" / "val", dst_root / "labels" / "val"),
        ("test", src_root / "test" / "images", src_root / "test" / "labels", dst_root / "images" / "test", dst_root / "labels" / "test"),
    ]

    total_images = 0
    total_labels = 0
    total_binary_counts: Counter = Counter()
    total_source_counts: Counter = Counter()
    total_discarded_invalid = 0

    for split_name, s_img, s_lbl, d_img, d_lbl in splits:
        n_img, n_lbl, b_counts, s_counts, n_disc = process_split(
            split_name, s_img, s_lbl, d_img, d_lbl
        )
        total_images += n_img
        total_labels += n_lbl
        total_binary_counts.update(b_counts)
        total_source_counts.update(s_counts)
        total_discarded_invalid += n_disc

        print(f"\nSplit [{split_name.upper()}]:")
        print(f"  Images processed: {n_img}")
        print(f"  Labels written: {n_lbl}")
        print(f"    Binary Class 0 (helmet): {b_counts[0]}")
        print(f"    Binary Class 1 (no_helmet): {b_counts[1]}")
        print(f"    Discarded 'invalid' annotations: {n_disc}")

    # Create binary data.yaml
    binary_data_yaml = dst_root / "data.yaml"
    yaml_content = {
        "path": "datasets/helmet_binary",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,
        "names": {
            0: "helmet",
            1: "no_helmet",
        },
    }
    with open(binary_data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print("\n" + "=" * 60)
    print(" BINARY CONVERSION CONVERSION CONVERSION SUMMARY")
    print("=" * 60)
    print(f" Target data.yaml created at: {binary_data_yaml}")
    print(f" Total Images Processed: {total_images}")
    print(f" Total Converted Labels: {total_labels}")
    print("\n Source Class Breakdown:")
    for src_cls, count in total_source_counts.items():
        name = CLASS_NAMES.get(src_cls, str(src_cls))
        action = "-> helmet (class 0)" if src_cls in (0, 1) else ("-> DISCARDED" if src_cls == 2 else "-> no_helmet (class 1)")
        print(f"  Class {src_cls} ({name:12s}): {count:5d}  {action}")

    total_class_helmet = total_binary_counts[0]
    total_class_no_helmet = total_binary_counts[1]

    print("\n Target Binary Class Breakdown:")
    print(f"  Class 0 (helmet)   : {total_class_helmet:5d}")
    print(f"  Class 1 (no_helmet): {total_class_no_helmet:5d}")
    print(f"  Discarded Invalid  : {total_discarded_invalid:5d}")

    ratio = round(total_class_helmet / max(1, total_class_no_helmet), 2)
    print(f"\n Imbalance Ratio (helmet : no_helmet): {ratio}:1")
    print("=" * 60)


if __name__ == "__main__":
    create_binary_dataset()
