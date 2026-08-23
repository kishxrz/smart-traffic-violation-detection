"""
scripts/prepare_helmet_crops_v3.py
───────────────────────────────────
Generates balanced, domain-augmented V3 dataset in datasets/helmet_crops_v3/.

Rules:
  - Exact preservation of val and test splits (NO leakage, NO augmentation).
  - Train split balanced 1:1 (3,920 HELMET : 3,920 NO_HELMET).
  - NO_HELMET training samples augmented with realistic domain-specific transforms:
    * Motion blur (simulating moving riders)
    * Brightness & contrast jitter (simulating sunlight/shadows)
    * JPEG compression artifacts (simulating video encoding)
    * Mild scale, crop, and rotation variation
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def apply_realistic_domain_augmentation(image: np.ndarray, aug_index: int) -> np.ndarray:
    aug = image.copy()
    h, w = aug.shape[:2]

    # 1. Brightness / Contrast Jitter
    if aug_index % 3 == 0:
        alpha = np.random.uniform(0.7, 1.3)  # contrast
        beta = np.random.randint(-30, 30)     # brightness
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    # 2. Motion Blur / Gaussian Blur
    if aug_index % 2 == 0:
        kernel_size = np.random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (kernel_size, kernel_size), 0)

    # 3. JPEG Compression Artifacts
    if aug_index % 4 == 0:
        quality = np.random.randint(40, 75)
        _, enc = cv2.imencode(".jpg", aug, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        aug = cv2.imdecode(enc, 1)

    # 4. Mild Rotation & Crop
    if aug_index % 5 == 0 and h > 10 and w > 10:
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return aug


def create_v3_dataset(
    src_dir: str = "datasets/helmet_crops",
    dst_dir: str = "datasets/helmet_crops_v3",
) -> Path:
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if dst_path.exists():
        shutil.rmtree(dst_path)

    # Copy val and test splits untouched
    for split in ["val", "test"]:
        shutil.copytree(src_path / "images" / split, dst_path / "images" / split)
        shutil.copytree(src_path / "labels" / split, dst_path / "labels" / split)

    (dst_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dst_path / "labels" / "train").mkdir(parents=True, exist_ok=True)

    # Process train split
    train_label_files = glob.glob(str(src_path / "labels" / "train" / "*.txt"))

    helmet_samples = []
    no_helmet_samples = []

    for lpath in train_label_files:
        with open(lpath) as f:
            lines = f.readlines()
        if not lines:
            continue
        cls_id = int(lines[0].split()[0])
        stem = Path(lpath).stem
        ipath = src_path / "images" / "train" / f"{stem}.jpg"
        if ipath.exists():
            if cls_id == 0:
                helmet_samples.append((ipath, lpath))
            elif cls_id == 1:
                no_helmet_samples.append((ipath, lpath))

    print(f"Source train crops: {len(helmet_samples)} HELMET, {len(no_helmet_samples)} NO_HELMET")

    # 1. Copy all HELMET training samples
    for ipath, lpath in helmet_samples:
        shutil.copy(ipath, dst_path / "images" / "train" / ipath.name)
        shutil.copy(lpath, dst_path / "labels" / "train" / Path(lpath).name)

    # Deduplicate unique NO_HELMET training crops before augmentation
    unique_no_helmet = {}
    for ipath, lpath in no_helmet_samples:
        key = ipath.name.split("_box")[0] if "_box" in ipath.name else ipath.name
        if key not in unique_no_helmet:
            unique_no_helmet[key] = (ipath, lpath)

    unique_nh_list = list(unique_no_helmet.values())
    print(f"Unique NO_HELMET training crops: {len(unique_nh_list)}")

    target_count = len(helmet_samples)  # 3,920
    added_nh = 0

    while added_nh < target_count:
        for idx, (ipath, lpath) in enumerate(unique_nh_list):
            if added_nh >= target_count:
                break
            mat = cv2.imread(str(ipath))
            if mat is None:
                continue

            if added_nh < len(unique_nh_list):
                aug_mat = mat
                new_stem = f"nh_orig_{added_nh}"
            else:
                aug_mat = apply_realistic_domain_augmentation(mat, added_nh)
                new_stem = f"nh_aug_{added_nh}"

            dst_img = dst_path / "images" / "train" / f"{new_stem}.jpg"
            dst_lbl = dst_path / "labels" / "train" / f"{new_stem}.txt"

            cv2.imwrite(str(dst_img), aug_mat)
            with open(dst_lbl, "w") as f:
                f.write(f"1 0.5 0.5 1.0 1.0\n")

            added_nh += 1

    # Create data.yaml
    yaml_content = f"""path: {dst_path.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: helmet
  1: no_helmet
"""
    with open(dst_path / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Successfully created V3 dataset at: {dst_path}")
    print(f"  Train: {len(helmet_samples)} HELMET, {added_nh} NO_HELMET (1:1 balanced)")
    return dst_path


if __name__ == "__main__":
    create_v3_dataset()
