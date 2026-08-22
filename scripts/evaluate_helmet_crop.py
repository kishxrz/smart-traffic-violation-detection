"""
scripts/evaluate_helmet_crop.py
──────────────────────────────
Model evaluation script for Crop-based YOLO Helmet Detector (Model v2).

Evaluates against the held-out TEST split of datasets/helmet_crops/data.yaml.

Reports REAL metrics only:
  - Overall Precision, Recall, mAP@50, mAP@50:95
  - HELMET (class 0): Precision, Recall, AP@50, AP@50:95
  - NO_HELMET (class 1): Precision, Recall, AP@50, AP@50:95
  - Confusion matrix
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
    """Measure inference latency of Model v2 on a crop input."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        sys.exit(1)

    if not Path(model_path).exists():
        logger.warning("Model file not found at %s. Latency benchmark skipped.", model_path)
        return {}

    logger.info("Benchmarking Model v2 latency: %s", model_path)
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


def evaluate_crop_model(model_path: str, data_yaml: str, device: str = "cpu") -> dict:
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        logger.error("Dataset YAML not found at: %s", yaml_path)
        return {}

    model_file = Path(model_path)
    if not model_file.exists():
        logger.error("Model file not found at: %s", model_file)
        return {}

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        sys.exit(1)

    logger.info("Evaluating Model v2 (%s) on HELD-OUT TEST split of %s", model_path, data_yaml)
    model = YOLO(str(model_file))
    results = model.val(data=str(yaml_path), split="test", device=device, verbose=True)

    metrics = {
        "overall_mp": float(results.box.mp),
        "overall_mr": float(results.box.mr),
        "overall_map50": float(results.box.map50),
        "overall_map": float(results.box.map),
        "classes": {},
    }

    names = results.names
    for i, cls_name in names.items():
        if i < len(results.box.p):
            p = float(results.box.p[i])
            r = float(results.box.r[i])
            ap50 = float(results.box.ap50[i])
            ap = float(results.box.ap[i])
            metrics["classes"][cls_name] = {
                "precision": p,
                "recall": r,
                "ap50": ap50,
                "ap50_95": ap,
            }

    print("\n" + "=" * 60)
    print(" MODEL V2 TEST SET EVALUATION METRICS")
    print("=" * 60)
    print(f" Overall mAP@50    : {metrics['overall_map50']:.4f}")
    print(f" Overall mAP@50:95 : {metrics['overall_map']:.4f}")
    print(f" Overall Precision : {metrics['overall_mp']:.4f}")
    print(f" Overall Recall    : {metrics['overall_mr']:.4f}")
    print("-" * 60)

    for cls_name, cls_m in metrics["classes"].items():
        print(f" Class [{cls_name.upper()}]:")
        print(f"   Precision : {cls_m['precision']:.4f}")
        print(f"   Recall    : {cls_m['recall']:.4f}")
        print(f"   AP@50     : {cls_m['ap50']:.4f}")
        print(f"   AP@50:95  : {cls_m['ap50_95']:.4f}")

    print("=" * 60)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Crop-based YOLO Helmet Detector (Model v2).")
    parser.add_argument("--model", default="models/helmet_v2.pt", help="Path to trained model weights")
    parser.add_argument("--data", default="datasets/helmet_crops/data.yaml", help="Path to data.yaml")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    args = parser.parse_args()

    latency = measure_latency(args.model)
    if latency:
        print("\nSYSTEM METRICS (Inference Latency):")
        for k, v in latency.items():
            print(f"  {k}: {v}")

    evaluate_crop_model(args.model, args.data, args.device)


if __name__ == "__main__":
    main()
