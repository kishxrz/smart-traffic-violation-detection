"""
scripts/inspect_real_helmet_crops.py
──────────────────────────────────────
Real-World Helmet Model Validation & Contact Sheet Generator.

Extracts all real-world head crops from input traffic video, runs standalone
HelmetDetector (models/helmet_v2.pt) inference, generates annotated contact sheet grid,
and computes quantitative crop quality metrics.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.cv.head_roi import extract_head_crop
from app.cv.video_processor import VideoProcessor
from app.detection.helmet_detector import HelmetDetector
from app.detection.yolo_detector import YOLODetector
from app.tracking.object_tracker import ObjectTracker
from app.violations.rider_association import RiderAssociator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def inspect_real_helmet_crops(
    video_path: str = r"C:\Users\Kishore\Desktop\INT 650\VID_20250930_161454.mp4",
    model_path: str = "models/helmet_v2.pt",
    output_dir: str = "evidence/helmet_validation",
) -> Dict[str, any]:
    video_file = Path(video_path)
    model_file = Path(model_path)
    out_path = Path(output_dir)
    crops_dir = out_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLODetector()
    tracker = ObjectTracker()
    associator = RiderAssociator(allow_motorcycle_crop_fallback=True)
    helmet_detector = HelmetDetector(model_path=model_file)

    extracted_records: List[Dict[str, any]] = []
    latencies: List[float] = []

    with VideoProcessor(str(video_file), frame_skip=1) as vp:
        for frame_num, ts, frame in vp.frames():
            dets = yolo.detect(frame, frame_number=frame_num, timestamp=ts)
            tracked_objects = tracker.update(dets, frame_number=frame_num, timestamp=ts)
            associations = associator.associate(tracked_objects)

            for assoc in associations:
                rider_obj = next((o for o in tracked_objects if o.track_id == assoc.rider_id), None)
                rider_bbox = rider_obj.bbox if rider_obj is not None else assoc.rider_bbox
                if rider_bbox is None:
                    continue

                head_crop = extract_head_crop(frame, rider_bbox)
                if head_crop is None or head_crop.shape[0] < 24 or head_crop.shape[1] < 24:
                    continue

                # Run standalone model v2 inference
                t0 = time.perf_counter()
                pred = helmet_detector.predict_crop(head_crop)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

                crop_filename = f"crop_f{frame_num}_m{assoc.motorcycle_id}_r{assoc.rider_id}.jpg"
                crop_file_path = crops_dir / crop_filename
                cv2.imwrite(str(crop_file_path), head_crop)

                # Assess crop content quality visually/algorithmically based on dimensions and aspect ratio
                h, w = head_crop.shape[:2]
                aspect_ratio = w / max(1, h)

                # Usable head crop criteria: height >= 32px, width >= 32px, reasonable aspect ratio
                is_usable = (h >= 32 and w >= 32 and 0.4 <= aspect_ratio <= 2.2)

                record = {
                    "frame_number": frame_num,
                    "timestamp": round(ts, 2),
                    "motorcycle_id": assoc.motorcycle_id,
                    "rider_id": assoc.rider_id,
                    "rider_roi": [rider_bbox.x1, rider_bbox.y1, rider_bbox.x2, rider_bbox.y2],
                    "crop_path": str(crop_file_path),
                    "crop_shape": [h, w, 3],
                    "predicted_status": pred.status,
                    "confidence": round(pred.confidence, 4),
                    "class_id": pred.class_id,
                    "is_usable": is_usable,
                }
                extracted_records.append(record)

    # Save records JSON
    with open(out_path / "crop_records.json", "w") as f:
        json.dump(extracted_records, f, indent=2)

    # Separate records by status
    helmet_recs = [r for r in extracted_records if r["predicted_status"] == "HELMET"]
    no_helmet_recs = [r for r in extracted_records if r["predicted_status"] == "NO_HELMET"]
    unknown_recs = [r for r in extracted_records if r["predicted_status"] == "UNKNOWN"]

    # Select representative samples for contact sheet
    selected_helmet = helmet_recs[:10]
    selected_no_helmet = no_helmet_recs[:10]
    selected_unknown = unknown_recs[:10]

    contact_samples = selected_helmet + selected_no_helmet + selected_unknown

    # Build Contact Sheet Grid (target tile: 200x200)
    tile_size = 200
    cols = 5
    num_samples = len(contact_samples)
    rows = max(1, (num_samples + cols - 1) // cols)

    grid_img = np.zeros((rows * (tile_size + 40), cols * tile_size, 3), dtype=np.uint8)

    for idx, rec in enumerate(contact_samples):
        r_idx = idx // cols
        c_idx = idx % cols

        crop_mat = cv2.imread(rec["crop_path"])
        if crop_mat is None:
            continue

        resized_crop = cv2.resize(crop_mat, (tile_size, tile_size))

        # Color banner based on status
        if rec["predicted_status"] == "HELMET":
            banner_color = (0, 200, 0)
        elif rec["predicted_status"] == "NO_HELMET":
            banner_color = (0, 0, 255)
        else:
            banner_color = (128, 128, 128)

        # Draw banner at top of tile
        cv2.rectangle(resized_crop, (0, 0), (tile_size, 28), (0, 0, 0), -1)
        lbl = f"{rec['predicted_status']} {rec['confidence']:.2f} (F{rec['frame_number']})"
        cv2.putText(resized_crop, lbl, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, banner_color, 1, cv2.LINE_AA)

        x_start = c_idx * tile_size
        y_start = r_idx * (tile_size + 40)
        grid_img[y_start : y_start + tile_size, x_start : x_start + tile_size] = resized_crop

    grid_out_path = out_path / "helmet_validation_grid.jpg"
    cv2.imwrite(str(grid_out_path), grid_img)
    logger.info("Saved contact sheet grid to: %s", grid_out_path)

    # Compute Statistics
    confidences = [r["confidence"] for r in extracted_records]
    helmet_confs = [r["confidence"] for r in helmet_recs]
    unknown_confs = [r["confidence"] for r in unknown_recs]

    total_crops = len(extracted_records)
    usable_crops = sum(1 for r in extracted_records if r["is_usable"])
    poor_crops = total_crops - usable_crops

    avg_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0
    avg_helmet_conf = round(float(np.mean(helmet_confs)), 4) if helmet_confs else 0.0
    avg_unknown_conf = round(float(np.mean(unknown_confs)), 4) if unknown_confs else 0.0
    min_conf = round(float(np.min(confidences)), 4) if confidences else 0.0
    max_conf = round(float(np.max(confidences)), 4) if confidences else 0.0
    avg_latency = round(float(np.mean(latencies)), 2) if latencies else 0.0

    print("\n" + "=" * 65)
    print(" REAL-WORLD HELMET VALIDATION REPORT")
    print("=" * 65)
    print(" Dataset                : Roboflow Motorcycle/Rider (Binary Schema)")
    print(" Training/Test Input    : 320x320 context-padded head crops (balanced)")
    print(f" Real Traffic Video     : {video_file.name}")
    print("-" * 65)
    print(f" Total Crops Extracted  : {total_crops}")
    print(f" Usable Head Crops      : {usable_crops}")
    print(f" Poor/Invalid Crops     : {poor_crops}")
    print("-" * 65)
    print(f" HELMET Status Count    : {len(helmet_recs)} (Avg Conf: {avg_helmet_conf})")
    print(f" NO_HELMET Status Count : {len(no_helmet_recs)}")
    print(f" UNKNOWN Status Count   : {len(unknown_recs)} (Avg Conf: {avg_unknown_conf})")
    print("-" * 65)
    print(f" Overall Avg Confidence : {avg_conf} (Min: {min_conf}, Max: {max_conf})")
    print(f" Standalone Model Latency: {avg_latency} ms / crop")
    print("=" * 65)

    return {
        "total_crops": total_crops,
        "usable_crops": usable_crops,
        "poor_crops": poor_crops,
        "helmet_count": len(helmet_recs),
        "no_helmet_count": len(no_helmet_recs),
        "unknown_count": len(unknown_recs),
        "avg_confidence": avg_conf,
        "avg_helmet_conf": avg_helmet_conf,
        "avg_unknown_conf": avg_unknown_conf,
        "min_confidence": min_conf,
        "max_confidence": max_conf,
        "avg_latency_ms": avg_latency,
        "grid_path": str(grid_out_path),
    }


if __name__ == "__main__":
    inspect_real_helmet_crops()
