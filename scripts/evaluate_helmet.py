"""
scripts/evaluate_helmet.py
───────────────────────────
Model evaluation script for Custom YOLO Helmet Detector.

Evaluates against the held-out TEST split (datasets/helmet_binary/images/test).

Reports REAL metrics only:
  - Precision & Recall
  - mAP@50 & mAP@50:95
  - Per-class metrics (helmet vs no_helmet)
  - Inference Latency (ms) & FPS benchmark
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
    dummy_crop = np.zeros((320, 320, 3), dtype=np.uint8)

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
    """Run Ultralytics validation on held-out TEST set."""
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        logger.error("Dataset YAML not found at: %s", yaml_path)
        return

    model_file = Path(model_path)
    if not model_file.exists():
        logger.error("Model file not found at: %s", model_file)
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        sys.exit(1)

    logger.info("Evaluating model %s on TEST split of %s", model_path, data_yaml)
    model = YOLO(str(model_file))
    results = model.val(data=str(yaml_path), split="test", device=device, verbose=True)

    print("\n" + "=" * 60)
    print(" MODEL TEST SET EVALUATION METRICS")
    print("=" * 60)
    print(f" Overall mAP@50    : {results.box.map50:.4f}")
    print(f" Overall mAP@50:95 : {results.box.map:.4f}")
    print(f" Overall Precision : {results.box.mp:.4f}")
    print(f" Overall Recall    : {results.box.mr:.4f}")
    print("-" * 60)

    # Class-wise metrics
    names = results.names
    for i, cls_name in names.items():
        if i < len(results.box.p):
            p = results.box.p[i]
            r = results.box.r[i]
            ap50 = results.box.ap50[i]
            print(f" Class [{cls_name.upper()}]: Precision={p:.4f}, Recall={r:.4f}, AP@50={ap50:.4f}")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Custom YOLO Helmet Detector.")
    parser.add_argument("--model", default="models/helmet_v1.pt", help="Path to trained model weights")
    parser.add_argument("--data", default="datasets/helmet_binary/data.yaml", help="Path to data.yaml")
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
