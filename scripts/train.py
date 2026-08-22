"""
scripts/train.py
─────────────────
Training script for custom YOLO models.

Usage:
    python scripts/train.py \\
        --task helmet \\
        --data datasets/helmet/data.yaml \\
        --model yolov8s.pt \\
        --epochs 100 \\
        --imgsz 640 \\
        --batch 16 \\
        --device cpu

Tasks supported:
    helmet    — Helmet/no-helmet detection for motorcycle rider safety.
    (future)  — Traffic sign detection, license plate, speed estimation.

Training Notes:
    1. Start from a pretrained COCO model (yolov8n.pt or yolov8s.pt).
       This is called transfer learning — the backbone already understands
       shapes and textures; only the head needs to learn new classes.

    2. Freeze the backbone for the first 10 epochs if your dataset is small (<2000 images).
       This prevents catastrophic forgetting.

    3. Monitor val/mAP50 — stop early if it plateaus for 20 epochs.

    4. The best model weights are saved to:
           runs/detect/{task}/weights/best.pt

    5. Copy to models/ for deployment:
           cp runs/detect/helmet/weights/best.pt models/helmet_v1.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_TASKS = {"helmet"}


def train(
    task: str,
    data_yaml: str,
    base_model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> None:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"task must be one of {SUPPORTED_TASKS}")

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    logger.info("Starting training: task=%s epochs=%d device=%s", task, epochs, device)
    model = YOLO(base_model)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/detect",
        name=task,
        patience=20,      # Early stopping
        save=True,
        plots=True,       # Save training plots
        verbose=True,
    )

    logger.info("Training complete. Best weights: %s", results.save_dir / "weights/best.pt")
    logger.info("Copy to models/: cp %s models/%s_v1.pt", results.save_dir / "weights/best.pt", task)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a custom YOLO model.")
    parser.add_argument("--task", choices=list(SUPPORTED_TASKS), required=True)
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Base pretrained model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    args = parser.parse_args()

    train(
        task=args.task,
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
