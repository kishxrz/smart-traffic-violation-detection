"""
scripts/train_helmet_v3.py
───────────────────────────
Trains Model V3 using transfer learning on datasets/helmet_crops_v3/.

Hyperparameters:
  - Architecture: YOLOv8n (yolov8n.pt)
  - Image size: 320x320
  - Epochs: 30
  - Batch size: 16
  - Patience: 8 (early stopping)
  - Optimizer: auto (SGD/AdamW)
  - Output directory: runs/helmet/v3
  - Final model location: models/helmet_v3.pt
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def train_v3(
    data_yaml: str = "datasets/helmet_crops_v3/data.yaml",
    epochs: int = 3,
    imgsz: int = 320,
    batch_size: int = 128,
    device_name: str = "cpu",
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Helmet V3 Training on device: {device}")

    data_path = Path(data_yaml).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {data_path}")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project="runs/helmet",
        name="v3",
        exist_ok=True,
        patience=8,
        verbose=True,
    )

    # Locate best.pt
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = save_dir / "best.pt"

    print(f"Training completed. Best weights saved at: {best_weights}")

    target_model_path = Path("models/helmet_v3.pt")
    target_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, target_model_path)
    print(f"Successfully copied final Model V3 to: {target_model_path.resolve()}")

    return target_model_path


if __name__ == "__main__":
    train_v3()
