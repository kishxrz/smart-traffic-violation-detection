"""
scripts/train_helmet_crop.py
─────────────────────────────
Training script for Custom Crop-based YOLO Helmet Detector (Model v2).

Optimized for:
  - Input: datasets/helmet_crops/data.yaml (cropped head/rider ROIs, matching inference pipeline)
  - Class balancing: Train split oversampled 5x for no_helmet class (ratio ~2.99:1)
  - Saved Model Target: models/helmet_v2.pt (does NOT overwrite v1)

Usage:
    python scripts/train_helmet_crop.py --epochs 30 --batch 32
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def train_crop_model(
    data_yaml: str = "datasets/helmet_crops/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 320,
    batch: int = 32,
    lr0: float = 0.01,
    device: str = "auto",
    workers: int = 4,
    patience: int = 10,
    output_dir: str = "runs",
) -> Path:
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        logger.error("Dataset config data.yaml not found at: %s", yaml_path)
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics library not found. Install via: pip install ultralytics")
        sys.exit(1)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print(" HELMET CROP DETECTOR (MODEL V2) TRAINING SUMMARY")
    print("=" * 60)
    print(f" Data Config : {data_yaml}")
    print(f" Base Model  : {base_model}")
    print(f" Image Size  : {imgsz}x{imgsz}")
    print(f" Epochs      : {epochs} | Batch: {batch}")
    print(f" Device      : {device} (CUDA: {torch.cuda.is_available()})")
    print("=" * 60 + "\n")

    model = YOLO(base_model)

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        device=device,
        workers=workers,
        patience=patience,
        project=output_dir,
        name="helmet_crop",
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        save=True,
        plots=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    if best_weights.exists():
        target_model_path = Path("models/helmet_v2.pt")
        target_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, target_model_path)
        logger.info("Model v2 best weights successfully deployed to: %s", target_model_path)

    return best_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Custom Crop-based YOLO Helmet Detector (Model v2).")
    parser.add_argument("--data", default="datasets/helmet_crops/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained base model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output", default="runs", help="Output directory")
    args = parser.parse_args()

    train_crop_model(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
