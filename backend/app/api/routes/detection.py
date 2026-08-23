"""
backend/app/api/routes/detection.py
─────────────────────────────────────
Detection endpoints.

POST /api/detection/image            — Run detection on a single uploaded image.
POST /api/detection/video            — Submit an uploaded video for asynchronous processing (HTTP 202).
GET  /api/detection/video/{job_id}   — Check status/progress/results of a video processing job.
GET  /api/detection/config           — Return current detection configuration.
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import (
    get_analytics,
    get_detector,
    get_evidence_generator,
    get_tracker,
    get_violation_engine,
    reset_session,
)
from app.analytics.metrics import AnalyticsAccumulator
from app.config import get_settings
from app.cv.frame_processor import FrameProcessor
from app.cv.preprocessing import PreprocessingConfig
from app.cv.video_processor import VideoProcessor
from app.detection.detector import BaseDetector
from app.detection.job_manager import job_manager
from app.evidence.evidence_generator import EvidenceGenerator
from app.schemas.job import VideoJobResponse
from app.schemas.violation import SceneState, TrafficLightState
from app.tracking.object_tracker import ObjectTracker
from app.violations.engine import ViolationEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/detection", tags=["detection"])


@router.post("/image")
async def detect_image(
    file: UploadFile = File(...),
    detector: BaseDetector = Depends(get_detector),
    tracker: ObjectTracker = Depends(get_tracker),
    analytics: AnalyticsAccumulator = Depends(get_analytics),
    violation_engine: ViolationEngine = Depends(get_violation_engine),
) -> Dict[str, Any]:
    """
    Run YOLO detection on a single uploaded image.

    Returns detections and any identified violations.
    """
    settings = get_settings()

    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type {file.content_type} not supported. Use JPEG/PNG.",
        )

    # Read image bytes
    contents = await file.read()
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {settings.max_upload_size_mb} MB",
        )

    # Decode to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image.",
        )

    t_start = time.perf_counter()

    # Run detection
    detections = detector.detect(frame, frame_number=0, timestamp=0.0)
    tracked = tracker.update(detections, frame_number=0, timestamp=0.0)

    # Build scene state for violation evaluation
    scene_state = SceneState(
        frame_number=0,
        timestamp=0.0,
        traffic_light_state=TrafficLightState.UNKNOWN,
    )
    violations = violation_engine.evaluate(frame, tracked, scene_state)

    t_end = time.perf_counter()
    inference_ms = (t_end - t_start) * 1000

    return {
        "session_id": str(uuid.uuid4()),
        "inference_time_ms": round(inference_ms, 2),
        "frame_size": {"width": frame.shape[1], "height": frame.shape[0]},
        "detections": [
            {
                "id": d.detection_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox": {
                    "x1": d.bbox.x1, "y1": d.bbox.y1,
                    "x2": d.bbox.x2, "y2": d.bbox.y2,
                },
            }
            for d in detections
        ],
        "tracked_objects": [
            {
                "track_id": t.track_id,
                "class_name": t.class_name,
                "confidence": round(t.confidence, 4),
                "center": list(t.center),
            }
            for t in tracked
        ],
        "violations": [v.model_dump() for v in violations],
        "summary": {
            "total_detections": len(detections),
            "total_violations": len(violations),
        },
    }


def run_video_processing_job(
    job_id: str,
    tmp_path: str,
    detector: BaseDetector,
    tracker: ObjectTracker,
    analytics: AnalyticsAccumulator,
    violation_engine: ViolationEngine,
    evidence_gen: EvidenceGenerator,
) -> None:
    """
    Background worker execution for video processing jobs.
    Runs the exact video processing, tracking, violation engine, evidence generation,
    and H.264 video encoding pipeline asynchronously.
    """
    settings = get_settings()
    session_id = str(uuid.uuid4())

    from app.cv.visualization import draw_tracked_object, draw_hud
    from app.cv.video_encoder import encode_to_browser_mp4, validate_video_file

    try:
        job_manager.set_processing(job_id)
        reset_session()

        preprocessor_cfg = PreprocessingConfig(
            resize=(416, 416),
            clahe=False,
        )
        frame_processor = FrameProcessor(
            detector=detector,
            tracker=tracker,
            preprocessing_config=preprocessor_cfg,
        )

        all_violations = []
        frames_processed = 0

        # Annotated video output path setup
        videos_dir = settings.evidence_abs_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        temp_raw_path = videos_dir / f"raw_temp_{session_id}.mp4"
        video_filename = f"annotated_{session_id}.mp4"
        video_output_path = videos_dir / video_filename
        video_writer = None

        with VideoProcessor(tmp_path, frame_skip=settings.frame_skip) as vp:
            metadata = vp.metadata
            total_frames = metadata.total_frames if metadata else None
            job_manager.update_progress(job_id, 0, total_frames)

            if metadata and metadata.width and metadata.height:
                output_fps = (metadata.fps or 25.0) / settings.frame_skip
                video_writer = VideoProcessor.create_writer(
                    output_path=temp_raw_path,
                    fps=output_fps,
                    width=metadata.width,
                    height=metadata.height,
                )

            for frame_num, ts, frame in vp.frames():
                frame_result = frame_processor.process(frame, frame_num, ts)

                scene_state = SceneState(
                    frame_number=frame_num,
                    timestamp=ts,
                    traffic_light_state=TrafficLightState.UNKNOWN,
                )
                violations = violation_engine.evaluate(
                    frame,
                    frame_result.tracked_objects,
                    scene_state,
                )

                # Save evidence snapshots
                tracked_by_id = {t.track_id: t for t in frame_result.tracked_objects}
                evidence_gen.save_batch(frame, violations, tracked_by_id)

                # Render annotations for video output
                annotated = frame.copy()
                for obj in frame_result.tracked_objects:
                    viol_types = [v.violation_type for v in violations if v.vehicle_id == obj.track_id]
                    annotated = draw_tracked_object(annotated, obj, viol_types)
                
                annotated = draw_hud(
                    annotated,
                    fps=metadata.fps / settings.frame_skip if metadata else 25.0,
                    frame_number=frame_num,
                    vehicle_count=len(frame_result.tracked_objects),
                    violation_count=len(all_violations) + len(violations),
                )

                if video_writer:
                    video_writer.write(annotated)

                analytics.update(frame_result, violations)
                all_violations.extend(violations)
                frames_processed += 1

                # Update job status progress
                job_manager.update_progress(job_id, frames_processed, total_frames)

        if video_writer:
            video_writer.release()
            video_writer = None

        import gc
        gc.collect()

        # Transcode OpenCV video to browser-compatible H.264 MP4
        annotated_video_url = None
        if temp_raw_path.exists() and temp_raw_path.stat().st_size > 1000:
            try:
                final_path = encode_to_browser_mp4(temp_raw_path, video_output_path)
                valid_meta = validate_video_file(final_path)
                annotated_video_url = f"/static/evidence/videos/{final_path.name}"
                logger.info("Annotated video validated for job %s: %s", job_id, valid_meta)
            except Exception as val_err:
                logger.error("Video validation failed for job %s: %s", job_id, val_err)
                annotated_video_url = None

        summary = analytics.build_summary()

        result_payload = {
            "session_id": session_id,
            "frames_processed": frames_processed,
            "video_metadata": {
                "fps": metadata.fps if metadata else None,
                "width": metadata.width if metadata else None,
                "height": metadata.height if metadata else None,
                "total_frames": metadata.total_frames if metadata else None,
                "duration_seconds": round(metadata.total_frames / metadata.fps, 1) if (metadata and metadata.fps and metadata.total_frames) else None,
            },
            "annotated_video_url": annotated_video_url,
            "violations": [v.model_dump() for v in all_violations],
            "summary": {
                "total_vehicles": summary.total_vehicles_detected,
                "unique_vehicles": summary.unique_vehicles_tracked,
                "total_violations": summary.total_violations,
                "violations_by_type": summary.violations_by_type,
                "avg_inference_ms": summary.avg_inference_time_ms,
                "processing_fps": summary.processing_fps,
            },
        }

        job_manager.complete_job(job_id, result_payload)

    except Exception as exc:
        logger.exception("Background job %s failed with exception", job_id)
        job_manager.fail_job(job_id, str(exc))

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/video", status_code=status.HTTP_202_ACCEPTED)
async def detect_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    detector: BaseDetector = Depends(get_detector),
    tracker: ObjectTracker = Depends(get_tracker),
    analytics: AnalyticsAccumulator = Depends(get_analytics),
    violation_engine: ViolationEngine = Depends(get_violation_engine),
    evidence_gen: EvidenceGenerator = Depends(get_evidence_generator),
) -> Dict[str, Any]:
    """
    Submit a video file for asynchronous detection & violation processing.

    Returns HTTP 202 ACCEPTED with a job_id immediately.
    """
    settings = get_settings()

    allowed = {"video/mp4", "video/avi", "video/x-msvideo", "video/quicktime"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported video format. Use MP4/AVI.",
        )

    contents = await file.read()
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {settings.max_upload_size_mb} MB",
        )

    # Write to temp file for worker access
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Create background job record
    job = job_manager.create_job()

    # Schedule background execution
    background_tasks.add_task(
        run_video_processing_job,
        job.job_id,
        tmp_path,
        detector,
        tracker,
        analytics,
        violation_engine,
        evidence_gen,
    )

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "frames_processed": job.frames_processed,
    }


@router.get("/video/{job_id}", response_model=VideoJobResponse)
async def get_video_job_status(job_id: str) -> VideoJobResponse:
    """
    Retrieve current status, progress, and results for a video processing job.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video job '{job_id}' not found.",
        )
    return job


@router.get("/config")
async def get_detection_config() -> Dict[str, Any]:
    """Return the current detection configuration."""
    settings = get_settings()
    return {
        "model_path": settings.yolo_model_path,
        "model_size": settings.yolo_model_size,
        "confidence_threshold": settings.confidence_threshold,
        "iou_threshold": settings.iou_threshold,
        "inference_fps": settings.inference_fps,
        "frame_skip": settings.frame_skip,
        "device": settings.device,
    }
