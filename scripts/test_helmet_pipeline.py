"""
scripts/test_helmet_pipeline.py
────────────────────────────────
End-to-End Helmet Pipeline Validation & ROI Quality Analysis Script.

Usage:
    python scripts/test_helmet_pipeline.py --video evidence/videos/test_h264.mp4 --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config import get_settings
from app.cv.head_roi import extract_head_crop_with_quality
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


def generate_roi_comparison_image(
    samples: List[Dict[str, any]],
    out_file_path: Path,
) -> None:
    """
    Generate 12-sample comparison visualization grid:
    LEFT: Frame with motorcycle box + rider ROI
    RIGHT: Actual crop sent to helmet_v2.pt
    """
    if not samples:
        return

    num_samples = min(12, len(samples))
    sample_items = samples[:num_samples]

    tile_w, tile_h = 320, 240
    row_height = tile_h + 30
    col_width = tile_w * 2 + 20

    rows = num_samples
    canvas_w = col_width
    canvas_h = rows * row_height

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, s in enumerate(sample_items):
        frame_mat = cv2.imread(s["frame_path"]) if "frame_path" in s and s["frame_path"] else None
        crop_mat = cv2.imread(s["crop_path"]) if "crop_path" in s and s["crop_path"] else None

        y_off = i * row_height

        # Left tile: Original frame region
        if frame_mat is not None:
            f_resized = cv2.resize(frame_mat, (tile_w, tile_h))
        else:
            f_resized = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

        # Right tile: Crop sent to model
        if crop_mat is not None and crop_mat.size > 0:
            c_resized = cv2.resize(crop_mat, (tile_w, tile_h))
        else:
            c_resized = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            cv2.putText(c_resized, s.get("rejection_reason", "REJECTED"), (20, tile_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw left and right tiles
        canvas[y_off:y_off + tile_h, 0:tile_w] = f_resized
        canvas[y_off:y_off + tile_h, tile_w + 10:tile_w * 2 + 10] = c_resized

        # Draw header text overlay
        st = s.get("status", "REJECTED")
        cf = s.get("confidence", 0.0)
        fid = s.get("frame_number", 0)
        mid = s.get("motorcycle_id", 0)
        q = s.get("quality_status", "REJECTED")

        banner_text = f"F#{fid} M#{mid} | ROI: {q} | {st} ({cf:.2f})"
        color = (0, 255, 0) if q == "ACCEPTED" else (0, 0, 255)

        cv2.putText(canvas, banner_text, (10, y_off + tile_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite(str(out_file_path), canvas)
    logger.info("Saved ROI comparison visualization to: %s", out_file_path)


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
    crop_samples_dir = ev_path / "improved_crops"
    crop_samples_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing components for Improved Helmet Pipeline Test...")
    yolo = YOLODetector()
    tracker = ObjectTracker()
    associator = RiderAssociator(allow_motorcycle_crop_fallback=True)
    severity = SeverityEngine()
    helmet_detector = HelmetDetector(model_path=model_file)
    helmet_violation_detector = HelmetViolationDetector(
        severity_engine=severity,
        helmet_detector=helmet_detector,
    )

    # Detailed ROI Telemetry Counters
    frames_processed = 0
    total_detections = 0
    unique_persons: Set[int] = set()
    unique_motorcycles: Set[int] = set()

    total_associations = 0
    candidate_rois = 0
    accepted_rois = 0
    rejected_rois = 0

    rejections_by_reason = {
        "FRAME_BOUNDARY_TRUNCATED": 0,
        "BELOW_MIN_SIZE": 0,
        "INVALID_ASPECT_RATIO": 0,
        "LOW_VISIBLE_AREA": 0,
    }

    helmet_model_calls = 0
    helmet_pred_count = 0
    no_helmet_pred_count = 0
    unknown_pred_count = 0
    confirmed_violations = 0

    helmet_latencies: List[float] = []
    comparison_samples: List[Dict[str, any]] = []

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

            # 1. Run YOLO object detection
            dets = yolo.detect(frame, frame_number=frame_num, timestamp=ts)
            total_detections += len(dets)

            # 2. Run object tracking
            tracked_objects = tracker.update(dets, frame_number=frame_num, timestamp=ts)

            for obj in tracked_objects:
                if obj.class_name == "person":
                    unique_persons.add(obj.track_id)
                elif obj.class_name == "motorcycle":
                    unique_motorcycles.add(obj.track_id)

                if debug and annotated_frame is not None:
                    annotated_frame = draw_tracked_object(annotated_frame, obj, show_id=True)

            # 3. Rider-Motorcycle Spatial Association
            associations: List[RiderAssociation] = associator.associate(tracked_objects)
            total_associations += len(associations)

            for assoc in associations:
                candidate_rois += 1
                moto_id = assoc.motorcycle_id
                rider_id = assoc.rider_id

                rider_obj = next((o for o in tracked_objects if o.track_id == rider_id), None)
                rider_bbox = rider_obj.bbox if rider_obj is not None else assoc.rider_bbox
                if rider_bbox is None:
                    continue

                # 4. Extract Head Crop with Quality Validation & Clamping
                crop_res = extract_head_crop_with_quality(
                    frame=frame,
                    rider_bbox=rider_bbox,
                    top_fraction=0.35,
                    padding_fraction=0.10,
                    min_size_px=(32, 32),
                )

                sample_record = {
                    "frame_number": frame_num,
                    "motorcycle_id": moto_id,
                    "rider_id": rider_id,
                    "quality_status": "ACCEPTED" if crop_res.is_accepted else "REJECTED",
                    "rejection_reason": crop_res.rejection_reason,
                    "visible_ratio": crop_res.visible_ratio,
                    "frame_path": None,
                    "crop_path": None,
                    "status": "REJECTED",
                    "confidence": 0.0,
                }

                if not crop_res.is_accepted:
                    rejected_rois += 1
                    reason = crop_res.rejection_reason or "BELOW_MIN_SIZE"
                    rejections_by_reason[reason] = rejections_by_reason.get(reason, 0) + 1

                    if len(comparison_samples) < 12 and reason == "FRAME_BOUNDARY_TRUNCATED":
                        frame_img_path = crop_samples_dir / f"frame_f{frame_num}_m{moto_id}.jpg"
                        cv2.imwrite(str(frame_img_path), frame)
                        sample_record["frame_path"] = str(frame_img_path)
                        comparison_samples.append(sample_record)

                    continue

                # ROI Accepted
                accepted_rois += 1
                head_crop = crop_res.crop

                # Save sample crop image
                crop_img_path = crop_samples_dir / f"crop_f{frame_num}_m{moto_id}.jpg"
                cv2.imwrite(str(crop_img_path), head_crop)
                sample_record["crop_path"] = str(crop_img_path)

                if len(comparison_samples) < 12:
                    frame_img_path = crop_samples_dir / f"frame_f{frame_num}_m{moto_id}.jpg"
                    cv2.imwrite(str(frame_img_path), frame)
                    sample_record["frame_path"] = str(frame_img_path)

                # 5. Model v2 Inference
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
                helmet_model_calls += 1

                sample_record["status"] = pred.status
                sample_record["confidence"] = round(pred.confidence, 4)

                if len(comparison_samples) < 12:
                    comparison_samples.append(sample_record)

                if pred.status == "HELMET":
                    helmet_pred_count += 1
                elif pred.status == "NO_HELMET":
                    no_helmet_pred_count += 1
                else:
                    unknown_pred_count += 1

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

            # 6. Temporal violation evaluation
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

    # Generate comparison image grid
    comp_grid_path = ev_path / "helmet_roi_comparison.jpg"
    generate_roi_comparison_image(comparison_samples, comp_grid_path)

    avg_latency = round(float(np.mean(helmet_latencies)), 2) if helmet_latencies else 0.0
    end_to_end_fps = round(frames_processed / max(total_pipeline_time, 0.001), 2)

    report = {
        "video_path": str(video_file),
        "frames_processed": frames_processed,
        "total_detections": total_detections,
        "unique_persons": len(unique_persons),
        "unique_motorcycles": len(unique_motorcycles),
        "total_associations": total_associations,
        "candidate_rois": candidate_rois,
        "accepted_rois": accepted_rois,
        "rejected_rois": rejected_rois,
        "rejections_by_reason": rejections_by_reason,
        "helmet_model_calls": helmet_model_calls,
        "helmet_status_count": helmet_pred_count,
        "no_helmet_status_count": no_helmet_pred_count,
        "unknown_status_count": unknown_pred_count,
        "confirmed_violations": confirmed_violations,
        "avg_helmet_latency_ms": avg_latency,
        "end_to_end_fps": end_to_end_fps,
        "comparison_grid": str(comp_grid_path),
    }

    # Print Summary Telemetry
    print("\n" + "=" * 65)
    print(" IMPROVED REAL-WORLD ROI EXTRACTION TELEMETRY REPORT")
    print("=" * 65)
    print(f" Frames Processed          : {report['frames_processed']}")
    print(f" Unique Motorcycles Tracked: {report['unique_motorcycles']}")
    print(f" Total Rider Associations  : {report['total_associations']}")
    print(f" Candidate Rider ROIs      : {report['candidate_rois']}")
    print(f" Accepted Head ROIs        : {report['accepted_rois']}")
    print(f" Rejected Head ROIs        : {report['rejected_rois']}")
    print(f"   - Boundary Truncated    : {rejections_by_reason['FRAME_BOUNDARY_TRUNCATED']}")
    print(f"   - Below Min Size        : {rejections_by_reason['BELOW_MIN_SIZE']}")
    print(f"   - Invalid Aspect Ratio  : {rejections_by_reason['INVALID_ASPECT_RATIO']}")
    print(f"   - Low Visible Area      : {rejections_by_reason['LOW_VISIBLE_AREA']}")
    print("-" * 65)
    print(f" Helmet Model Calls        : {report['helmet_model_calls']}")
    print(f"   - HELMET status         : {report['helmet_status_count']}")
    print(f"   - NO_HELMET status      : {report['no_helmet_status_count']}")
    print(f"   - UNKNOWN status        : {report['unknown_status_count']}")
    print(f" Confirmed Violations      : {report['confirmed_violations']}")
    print("-" * 65)
    print(f" Standalone Model Latency  : {report['avg_helmet_latency_ms']} ms / crop")
    print(f" End-to-End Pipeline FPS   : {report['end_to_end_fps']} FPS")
    print("=" * 65)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Improved Helmet ROI Extraction on Video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="models/helmet_v2.pt", help="Path to helmet model weights")
    parser.add_argument("--debug", action="store_true", help="Generate annotated debug video overlay")
    parser.add_argument("--out", default=None, help="Output annotated video path")
    parser.add_argument("--evidence_dir", default="evidence/helmet_validation", help="Sample evidence output dir")
    args = parser.parse_args()

    run_helmet_pipeline_test(
        video_path=args.video,
        model_path=args.model,
        debug=args.debug,
        out_video_path=args.out,
        evidence_dir=args.evidence_dir,
    )


if __name__ == "__main__":
    main()
