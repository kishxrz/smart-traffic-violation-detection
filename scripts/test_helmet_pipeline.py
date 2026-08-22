"""
scripts/test_helmet_pipeline.py
────────────────────────────────
End-to-End Helmet Pipeline Validation & Telemetry Analysis Script.

Usage:
    python scripts/test_helmet_pipeline.py --video evidence/videos/test_h264.mp4 --debug
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config import get_settings
from app.cv.head_roi import extract_head_crop
from app.cv.video_encoder import encode_to_browser_mp4
from app.cv.video_processor import VideoProcessor
from app.cv.visualization import draw_helmet_debug, draw_tracked_object, draw_violation_overlay
from app.detection.helmet_detector import HelmetDetector, HelmetPrediction
from app.detection.yolo_detector import YOLODetector
from app.schemas.violation import SceneState
from app.tracking.object_tracker import ObjectTracker
from app.violations.helmet import HelmetViolationDetector
from app.violations.rider_association import RiderAssociation, RiderAssociator
from app.violations.severity import SeverityEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_helmet_pipeline_test(
    video_path: str,
    model_path: str = "models/helmet_v2.pt",
    debug: bool = True,
    out_video_path: Optional[str] = None,
    evidence_dir: str = "evidence/helmet_validation",
    sample_limit: int = 5,
) -> Dict[str, any]:
    video_file = Path(video_path)
    if not video_file.exists():
        logger.error("Input video file not found at: %s", video_file)
        sys.exit(1)

    model_file = Path(model_path)
    if not model_file.exists():
        logger.error("Helmet model file not found at: %s", model_file)
        sys.exit(1)

    ev_path = Path(evidence_dir)
    ev_path.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing components for Helmet Pipeline Test...")
    yolo = YOLODetector()
    tracker = ObjectTracker()
    associator = RiderAssociator()
    severity = SeverityEngine()
    helmet_detector = HelmetDetector(model_path=model_file)
    helmet_violation_detector = HelmetViolationDetector(
        severity_engine=severity,
        helmet_detector=helmet_detector,
    )

    # Telemetry Counters
    frames_processed = 0
    total_detections = 0
    unique_persons: Set[int] = set()
    unique_motorcycles: Set[int] = set()
    associations_count = 0
    head_crops_count = 0

    helmet_pred_count = 0
    no_helmet_pred_count = 0
    unknown_pred_count = 0
    confirmed_violations = 0

    helmet_latencies: List[float] = []
    saved_samples: Dict[str, int] = defaultdict(int)

    temp_raw_out = ev_path / f"temp_{video_file.stem}_debug.mp4"
    video_writer = None

    t_start = time.perf_counter()

    with VideoProcessor(str(video_file), frame_skip=1) as vp:
        fps = vp.metadata.fps
        width, height = vp.metadata.resolution

        if debug:
            out_path_target = out_video_path or str(ev_path / f"{video_file.stem}_debug_annotated.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(temp_raw_out), fourcc, fps, (width, height))

        for frame_num, ts, frame in vp.frames():
            frames_processed += 1
            annotated_frame = frame.copy() if debug else None

            # 1. Run general YOLO object detection
            dets = yolo.detect(frame, frame_number=frame_num, timestamp=ts)
            total_detections += len(dets)

            # 2. Run object tracking
            tracked_objects = tracker.update(dets, frame_number=frame_num, timestamp=ts)

            # Track unique entities
            for obj in tracked_objects:
                if obj.class_name == "person":
                    unique_persons.add(obj.track_id)
                elif obj.class_name == "motorcycle":
                    unique_motorcycles.add(obj.track_id)

                if debug and annotated_frame is not None:
                    annotated_frame = draw_tracked_object(annotated_frame, obj, show_id=True)

            # 3. Perform Rider ↔ Motorcycle Spatial Association
            associations: List[RiderAssociation] = associator.associate(tracked_objects)
            associations_count += len(associations)

            for assoc in associations:
                moto_id = assoc.motorcycle_id
                rider_id = assoc.rider_id

                moto_obj = next((o for o in tracked_objects if o.track_id == moto_id), None)
                rider_obj = next((o for o in tracked_objects if o.track_id == rider_id), None)

                rider_bbox = rider_obj.bbox if rider_obj is not None else assoc.rider_bbox
                if rider_bbox is None:
                    continue

                # 4. Extract Head ROI Crop
                head_crop = extract_head_crop(frame, rider_bbox)
                if head_crop is None or head_crop.shape[0] < 24 or head_crop.shape[1] < 24:
                    continue

                head_crops_count += 1

                # 5. Predict Helmet Status via Model v2
                t0_inf = time.perf_counter()
                pred = helmet_detector.predict_crop(
                    head_crop,
                    vehicle_id=moto_id,
                    person_id=rider_id,
                    frame_number=frame_num,
                    obs_count=len(helmet_violation_detector._prediction_history[moto_id]) + 1,
                )
                t1_inf = time.perf_counter()
                helmet_latencies.append((t1_inf - t0_inf) * 1000)

                # Categorize prediction
                if pred.status == "HELMET":
                    helmet_pred_count += 1
                elif pred.status == "NO_HELMET":
                    no_helmet_pred_count += 1
                else:
                    unknown_pred_count += 1

                # Save representative sample crop
                if saved_samples[pred.status] < sample_limit:
                    sample_name = f"{pred.status}_m{moto_id}_r{rider_id}_f{frame_num}.jpg"
                    sample_path = ev_path / sample_name
                    cv2.imwrite(str(sample_path), head_crop)
                    saved_samples[pred.status] += 1

                # Debug annotation overlay
                if debug and annotated_frame is not None:
                    obs_count = len(helmet_violation_detector._prediction_history[moto_id]) + 1
                    annotated_frame = draw_helmet_debug(
                        annotated_frame,
                        vehicle_id=moto_id,
                        rider_id=rider_id,
                        status=pred.status,
                        confidence=pred.confidence,
                        obs_count=obs_count,
                        total_obs=10,
                        rider_bbox=rider_bbox,
                    )

            # 6. Evaluate temporal violations
            scene_state = SceneState(frame_number=frame_num, timestamp=ts)
            viols = helmet_violation_detector.evaluate(frame, tracked_objects, scene_state)

            if viols:
                confirmed_violations += len(viols)
                for v in viols:
                    if debug and annotated_frame is not None:
                        annotated_frame = draw_violation_overlay(annotated_frame, v)

            if debug and video_writer is not None and annotated_frame is not None:
                video_writer.write(annotated_frame)

    t_end = time.perf_counter()
    total_pipeline_time = t_end - t_start

    if video_writer is not None:
        video_writer.release()
        out_path_target = out_video_path or str(ev_path / f"{video_file.stem}_debug_annotated.mp4")
        encode_to_browser_mp4(str(temp_raw_out), out_path_target)
        if temp_raw_out.exists():
            temp_raw_out.unlink()

    avg_latency = round(float(np.mean(helmet_latencies)), 2) if helmet_latencies else 0.0
    end_to_end_fps = round(frames_processed / max(total_pipeline_time, 0.001), 2)

    report = {
        "video_path": str(video_file),
        "model_path": str(model_file),
        "frames_processed": frames_processed,
        "total_detections": total_detections,
        "unique_persons": len(unique_persons),
        "unique_motorcycles": len(unique_motorcycles),
        "associations_count": associations_count,
        "head_crops_count": head_crops_count,
        "helmet_predictions": helmet_pred_count + no_helmet_pred_count + unknown_pred_count,
        "helmet_status_count": helmet_pred_count,
        "no_helmet_status_count": no_helmet_pred_count,
        "unknown_status_count": unknown_pred_count,
        "confirmed_violations": confirmed_violations,
        "avg_helmet_latency_ms": avg_latency,
        "end_to_end_fps": end_to_end_fps,
        "sample_crops_saved": dict(saved_samples),
    }

    # Print Telemetry Coverage Summary
    print("\n" + "=" * 65)
    print(" REAL VIDEO HELMET PIPELINE TELEMETRY COVERAGE REPORT")
    print("=" * 65)
    print(f" Input Video               : {report['video_path']}")
    print(f" Helmet Model              : {report['model_path']}")
    print(f" Frames Processed          : {report['frames_processed']}")
    print(f" Total Object Detections   : {report['total_detections']}")
    print(f" Unique Persons Tracked    : {report['unique_persons']}")
    print(f" Unique Motorcycles Tracked: {report['unique_motorcycles']}")
    print(f" Rider-Motorcycle Pairs    : {report['associations_count']}")
    print(f" Head Crops Extracted      : {report['head_crops_count']}")
    print("-" * 65)
    print(f" Helmet Model Predictions  : {report['helmet_predictions']}")
    print(f"   - HELMET status         : {report['helmet_status_count']}")
    print(f"   - NO_HELMET status      : {report['no_helmet_status_count']}")
    print(f"   - UNKNOWN status        : {report['unknown_status_count']}")
    print(f" Confirmed Violations      : {report['confirmed_violations']}")
    print("-" * 65)
    print(f" Standalone Model Latency  : {report['avg_helmet_latency_ms']} ms / crop")
    print(f" End-to-End Pipeline FPS   : {report['end_to_end_fps']} FPS")
    print("=" * 65)

    if report["unique_motorcycles"] == 0:
        print("\nWARNING / DIAGNOSTIC NOTICE:")
        print("  'Helmet detector was not meaningfully exercised because no motorcycles were tracked.'")
        print("  Zero violations in this video cannot be used as proof of zero false positives.\n")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Test End-to-End Helmet Pipeline on Video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="models/helmet_v2.pt", help="Path to helmet model weights")
    parser.add_argument("--debug", action="store_true", help="Generate annotated debug video overlay")
    parser.add_argument("--out", default=None, help="Output annotated video path")
    parser.add_argument("--evidence_dir", default="evidence/helmet_validation", help="Sample evidence output dir")
    parser.add_argument("--sample_limit", type=int, default=5, help="Max sample crops to save per prediction status")
    args = parser.parse_args()

    run_helmet_pipeline_test(
        video_path=args.video,
        model_path=args.model,
        debug=args.debug,
        out_video_path=args.out,
        evidence_dir=args.evidence_dir,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
