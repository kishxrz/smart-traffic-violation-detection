"""
scripts/train_helmet.py
────────────────────────
Training script for custom YOLO Helmet & Rider Safety Detector using PyTorch/Ultralytics.

Features:
  - Transfer learning starting from pretrained YOLO weights (default: yolov8n.pt).
  - Target dataset: datasets/helmet_binary/data.yaml (binary 0: helmet, 1: no_helmet).
  - Automatically detects CUDA/GPU vs CPU.
  - Sensible augmentations tailored for traffic camera conditions:
      - scale, translation, rotation, horizontal flip, HSV jitter, mosaic
  - Early stopping (patience=15) to prevent overfitting.
  - Automatically copies trained best weights to models/helmet_v1.pt for deployment.

Usage:
    python scripts/train_helmet.py --data datasets/helmet_binary/data.yaml --epochs 30 --batch 16
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


def train_helmet_model(
    data_yaml: str,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    lr0: float,
    device: str,
    workers: int,
    patience: int,
    output_dir: str,
) -> Path:
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        logger.error("Dataset config data.yaml not found at: %s", yaml_path)
        logger.info(
            "Please run python scripts/prepare_binary_helmet_dataset.py to prepare datasets/helmet_binary/."
        )
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics library not found. Install via: pip install ultralytics")
        sys.exit(1)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info(" STARTING HELMET DETECTOR TRAINING")
    logger.info(" Base Model : %s", base_model)
    logger.info(" Data Config: %s", data_yaml)
    logger.info(" Epochs     : %d | Batch: %d | Image Size: %d", epochs, batch, imgsz)
    logger.info(" Device     : %s (CUDA Available: %s)", device, torch.cuda.is_available())
    logger.info("=" * 60)

    model = YOLO(base_model)

    # Train with traffic-oriented augmentations & early stopping
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
        name="helmet",
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
        target_model_path = Path("models/helmet_v1.pt")
        target_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, target_model_path)
        logger.info("Best weights successfully deployed to: %s", target_model_path)

    return best_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Custom YOLO Helmet Detector.")
    parser.add_argument("--data", default="datasets/helmet_binary/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained base model (yolov8n.pt / yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output", default="runs", help="Output directory")
    args = parser.parse_args()

    train_helmet_model(
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
