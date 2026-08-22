"""
scripts/evaluate.py
────────────────────
Model evaluation script.

Reports:
  - mAP@50 and mAP@50:95 using Ultralytics validation
  - Per-class precision and recall
  - Inference latency statistics
  - Confusion matrix (if dataset labels available)

Usage:
    python scripts/evaluate.py --model models/yolov8n.pt --data path/to/data.yaml

IMPORTANT:
  This script reports genuine metrics — no fabricated numbers.
  If no labeled dataset is available, only inference latency is reported.
  mAP metrics require ground-truth annotations.

Dataset format: YOLO format (images/ + labels/ directories).
See datasets/README.md for the expected structure.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


def measure_latency(model_path: str, n_runs: int = 50) -> dict:
    """Measure inference latency on a blank frame."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        sys.exit(1)

    logger.info("Loading model: %s", model_path)
    model = YOLO(model_path)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        model.predict(dummy, verbose=False)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "n_runs": n_runs,
        "mean_ms": round(np.mean(times), 2),
        "std_ms": round(np.std(times), 2),
        "min_ms": round(np.min(times), 2),
        "max_ms": round(np.max(times), 2),
        "fps_estimate": round(1000 / np.mean(times), 1),
    }


def evaluate_on_dataset(model_path: str, data_yaml: str, device: str = "cpu") -> None:
    """Run Ultralytics validation on a labeled dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        sys.exit(1)

    logger.info("Evaluating model: %s on dataset: %s", model_path, data_yaml)
    model = YOLO(model_path)
    results = model.val(data=data_yaml, device=device, verbose=True)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"mAP@50:     {results.box.map50:.4f}")
    print(f"mAP@50:95:  {results.box.map:.4f}")
    print(f"Precision:  {results.box.mp:.4f}")
    print(f"Recall:     {results.box.mr:.4f}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO model.")
    parser.add_argument("--model", default="yolov8n.pt", help="Model path")
    parser.add_argument("--data", default=None, help="data.yaml path for mAP evaluation")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--latency-runs", type=int, default=50)
    args = parser.parse_args()

    # Always measure latency
    latency = measure_latency(args.model, args.latency_runs)
    print("\nLatency Report:")
    for k, v in latency.items():
        print(f"  {k}: {v}")

    # mAP evaluation requires a dataset
    if args.data:
        evaluate_on_dataset(args.model, args.data, args.device)
    else:
        print(
            "\nNote: mAP evaluation skipped (no --data provided). "
            "Provide a data.yaml to get precision/recall/mAP metrics."
        )


if __name__ == "__main__":
    main()
