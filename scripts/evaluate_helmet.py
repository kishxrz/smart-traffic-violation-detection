"""
scripts/evaluate_helmet.py
───────────────────────────
Model evaluation script for Custom YOLO Helmet Detector.

Reports REAL metrics only:
  - Precision
  - Recall
  - mAP@50
  - mAP@50:95
  - Confusion Matrix
  - Inference Latency (ms) & FPS

Usage:
    python scripts/evaluate_helmet.py --model models/helmet_v1.pt --data datasets/helmet/data.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def measure_latency(model_path: str, n_runs: int = 30) -> dict:
    """Measure inference latency of the helmet model on a dummy head crop."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        sys.exit(1)

    if not Path(model_path).exists():
        logger.warning("Model file not found at %s. Latency benchmark skipped.", model_path)
        return {}

    logger.info("Benchmarking model latency: %s", model_path)
    model = YOLO(model_path)
    dummy_crop = np.zeros((128, 128, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        model.predict(dummy_crop, verbose=False)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy_crop, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "n_runs": n_runs,
        "mean_ms": round(float(np.mean(times)), 2),
        "std_ms": round(float(np.std(times)), 2),
        "min_ms": round(float(np.min(times)), 2),
        "max_ms": round(float(np.max(times)), 2),
        "fps_estimate": round(1000 / float(np.mean(times)), 1),
    }


def evaluate_model(model_path: str, data_yaml: str, device: str = "cpu") -> None:
    """Run Ultralytics validation on test/val set."""
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        logger.error("Dataset YAML not found at: %s", yaml_path)
        return

    model_file = Path(model_path)
    if not model_file.exists():
        logger.error("Model file not found at: %s", model_file)
        logger.info("To train a custom model: python scripts/train_helmet.py")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        sys.exit(1)

    logger.info("Evaluating model %s on %s", model_path, data_yaml)
    model = YOLO(model_path)
    results = model.val(data=data_yaml, device=device, verbose=True)

    print("\n" + "=" * 60)
    print(" MODEL METRICS (CUSTOM HELMET DETECTOR)")
    print("=" * 60)
    print(f" mAP@50    : {results.box.map50:.4f}")
    print(f" mAP@50:95 : {results.box.map:.4f}")
    print(f" Precision : {results.box.mp:.4f}")
    print(f" Recall    : {results.box.mr:.4f}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Custom YOLO Helmet Detector.")
    parser.add_argument("--model", default="models/helmet_v1.pt", help="Path to trained model weights")
    parser.add_argument("--data", default="datasets/helmet/data.yaml", help="Path to data.yaml")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    args = parser.parse_args()

    latency = measure_latency(args.model)
    if latency:
        print("\nSYSTEM METRICS (Inference Latency):")
        for k, v in latency.items():
            print(f"  {k}: {v}")

    evaluate_model(args.model, args.data, args.device)


if __name__ == "__main__":
    main()
