"""
scripts/visualize_helmet_dataset.py
───────────────────────────────────
Visual dataset verification script for converted binary helmet dataset.

Draws:
  - GREEN bounding boxes (#00FF00) with label 'HELMET' for class 0
  - RED bounding boxes (#0000FF) with label 'NO_HELMET' for class 1
Verifies that:
  - full-faced and half-faced became HELMET
  - no helmet became NO_HELMET
  - invalid labels disappeared completely

Saves a composite grid to datasets/helmet_binary/samples_visualization.jpg
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2
import numpy as np


def visualize_binary_samples(output_grid_path: str = "datasets/helmet_binary/samples_visualization.jpg") -> None:
    splits = ["train", "val", "test"]
    sample_images = []

    for split in splits:
        img_paths = glob.glob(f"datasets/helmet_binary/images/{split}/*.jpg")
        for p in img_paths:
            lbl_p = p.replace("images", "labels").replace(".jpg", ".txt")
            if os.path.exists(lbl_p):
                content = open(lbl_p, "r", encoding="utf-8").read().strip()
                # Find an image that contains annotations
                if content:
                    sample_images.append((split, p, lbl_p))
                    if len([s for s in sample_images if s[0] == split]) >= 2:
                        break

    annotated_crops = []

    for split, img_p, lbl_p in sample_images[:6]:
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
            elif cls_id == 1:
                color = (0, 0, 220)  # Red NO_HELMET
                text = "NO_HELMET"
            else:
                color = (255, 0, 0)
                text = f"UNKNOWN_{cls_id}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, text, (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

        # Add banner label at top
        cv2.putText(
            img, f"[{split.upper()}] {os.path.basename(img_p)[:20]}...",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

        img_resized = cv2.resize(img, (400, 400))
        annotated_crops.append(img_resized)

    if annotated_crops:
        # Pad to 6 items if needed
        while len(annotated_crops) < 6:
            annotated_crops.append(np.zeros((400, 400, 3), dtype=np.uint8))

        row1 = np.hstack(annotated_crops[:3])
        row2 = np.hstack(annotated_crops[3:6])
        grid = np.vstack([row1, row2])

        cv2.imwrite(output_grid_path, grid)
        print(f"Visual verification grid saved to: {output_grid_path}")


if __name__ == "__main__":
    visualize_binary_samples()
