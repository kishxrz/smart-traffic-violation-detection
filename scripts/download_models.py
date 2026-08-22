"""
scripts/download_models.py
───────────────────────────
Download pretrained YOLO model weights into the models/ directory.

Usage:
    python scripts/download_models.py [--size n|s|m|l|x]

This script:
  1. Creates the models/ directory if needed.
  2. Uses Ultralytics to download the specified YOLOv8 model.
  3. Moves the weights to models/yolov8{size}.pt.

You only need to run this once. After that, set YOLO_MODEL_PATH in .env.

Note on custom models:
  Custom helmet, traffic-light-state, or vehicle-type models
  must be trained separately and placed in models/ manually.
  See docs/deep-learning.md for training instructions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def download_yolo(size: str = "n") -> Path:
    """Download a YOLOv8 model of the given size."""
    valid_sizes = ("n", "s", "m", "l", "x")
    if size not in valid_sizes:
        raise ValueError(f"size must be one of {valid_sizes}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = f"yolov8{size}.pt"
    dest = MODELS_DIR / model_name

    if dest.exists():
        logger.info("Model already exists: %s", dest)
        return dest

    logger.info("Downloading YOLOv8-%s model...", size.upper())

    try:
        from ultralytics import YOLO
        # Ultralytics downloads to its cache then we copy to models/
        model = YOLO(model_name)
        # The weights file gets cached by ultralytics
        cached = Path(model.ckpt_path) if hasattr(model, "ckpt_path") else None
        if cached and cached.exists() and cached != dest:
            import shutil
            shutil.copy2(cached, dest)
            logger.info("Model saved to: %s", dest)
        else:
            logger.info("Model available via Ultralytics cache.")
    except ImportError:
        logger.error(
            "ultralytics is not installed. Run: pip install ultralytics"
        )
        sys.exit(1)

    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YOLO model weights.")
    parser.add_argument(
        "--size",
        choices=("n", "s", "m", "l", "x"),
        default="n",
        help="YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)",
    )
    args = parser.parse_args()

    path = download_yolo(args.size)
    print(f"\nModel ready: {path}")
    print(f"\nSet in .env:\n  YOLO_MODEL_PATH={path}")


if __name__ == "__main__":
    main()
