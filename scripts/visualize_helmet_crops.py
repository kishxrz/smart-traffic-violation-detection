"""
scripts/visualize_helmet_crops.py
──────────────────────────────────
Visual verification script for Crop-based Helmet Dataset.

Generates a composite sample grid showing:
  - GREEN bounding box + 'HELMET' for class 0
  - RED bounding box + 'NO_HELMET' for class 1
  - Context padding around original head target
  - Saves to datasets/helmet_crops/samples_visualization.jpg
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2
import numpy as np


def visualize_helmet_crops(output_grid_path: str = "datasets/helmet_crops/samples_visualization.jpg") -> None:
    splits = ["train", "val", "test"]
    sample_items = []

    for split in splits:
        img_paths = glob.glob(f"datasets/helmet_crops/images/{split}/*.jpg")
        helmet_samples = []
        no_helmet_samples = []

        for p in img_paths:
            lbl_p = p.replace("images", "labels").replace(".jpg", ".txt")
            if not os.path.exists(lbl_p):
                continue
            content = open(lbl_p, "r", encoding="utf-8").read().strip()
            if not content:
                continue
            lines = content.splitlines()
            cls_id = int(lines[0].split()[0])
            if cls_id == 0 and len(helmet_samples) < 1:
                helmet_samples.append((split, p, lbl_p))
            elif cls_id == 1 and len(no_helmet_samples) < 1:
                no_helmet_samples.append((split, p, lbl_p))

            if helmet_samples and no_helmet_samples:
                break

        sample_items.extend(helmet_samples + no_helmet_samples)

    annotated_crops = []

    for split, img_p, lbl_p in sample_items[:6]:
        img = cv2.imread(img_p)
        if img is None:
            continue
        h, w = img.shape[:2]

        lines = open(lbl_p, "r", encoding="utf-8").read().strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            if cls_id == 0:
                color = (0, 220, 0)  # Green HELMET
                text = "HELMET"
            else:
                color = (0, 0, 220)  # Red NO_HELMET
                text = "NO_HELMET"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, text, (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

        # Header banner
        cv2.putText(
            img, f"[{split.upper()}]",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

        annotated_crops.append(cv2.resize(img, (320, 320)))

    if annotated_crops:
        while len(annotated_crops) < 6:
            annotated_crops.append(np.zeros((320, 320, 3), dtype=np.uint8))

        row1 = np.hstack(annotated_crops[:3])
        row2 = np.hstack(annotated_crops[3:6])
        grid = np.vstack([row1, row2])

        cv2.imwrite(output_grid_path, grid)
        print(f"Sample crop visualization saved to: {output_grid_path}")


if __name__ == "__main__":
    visualize_helmet_crops()
