"""
scripts/train_helmet.py
────────────────────────
Training script for custom YOLO Helmet & Rider Safety Detector using PyTorch/Ultralytics.

Features:
  - Transfer learning starting from pretrained YOLO weights (yolov8n.pt or yolov8s.pt).
  - Sensible augmentations tailored for traffic camera conditions:
      - scale, translation, moderate rotation
      - horizontal flip (where orientation invariant)
      - HSV color jitter (for varying daylight/night conditions)
      - exposure and motion blur
  - Early stopping (patience=20) to prevent overfitting.
  - Automatically saves training metrics, confusion matrix, plots, and best weights to runs/detect/helmet/weights/best.pt.

Usage:
    python scripts/train_helmet.py \\
        --data datasets/helmet/data.yaml \\
        --model yolov8s.pt \\
        --epochs 100 \\
        --imgsz 640 \\
        --batch 16 \\
        --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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
) -> None:
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        logger.error("Dataset config data.yaml not found at: %s", yaml_path)
        logger.info(
            "Please populate datasets/helmet/ with images and labels before training.\n"
            "Run python scripts/validate_helmet_dataset.py to verify dataset readiness."
        )
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics library not found. Install via: pip install ultralytics")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(" STARTING HELMET DETECTOR TRAINING")
    logger.info(" Model Base : %s", base_model)
    logger.info(" Data Config: %s", data_yaml)
    logger.info(" Epochs     : %d | Batch: %d | Image Size: %d", epochs, batch, imgsz)
    logger.info(" Device     : %s", device)
    logger.info("=" * 60)

    model = YOLO(base_model)

    # Train with traffic-oriented augmentations
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
        # Sensible traffic camera augmentations
        hsv_h=0.015,     # HSV hue jitter
        hsv_s=0.7,       # HSV saturation jitter
        hsv_v=0.4,       # HSV value/exposure jitter
        degrees=10.0,    # Small rotation jitter
        translate=0.1,   # Small translation jitter
        scale=0.5,       # Scale jitter (handles varying distance to camera)
        fliplr=0.5,      # Horizontal flip
        mosaic=1.0,      # Mosaic augmentation for small objects
        save=True,
        plots=True,
        verbose=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("Training complete!")
    logger.info("Best weights saved to: %s", best_weights)
    logger.info("To deploy, copy weights to models/: cp %s models/helmet_v1.pt", best_weights)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Custom YOLO Helmet Detector.")
    parser.add_argument("--data", default="datasets/helmet/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Pretrained base model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
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
