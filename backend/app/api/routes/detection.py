"""
backend/app/api/routes/detection.py
─────────────────────────────────────
Detection endpoints.

POST /api/detection/image   — Run detection on a single uploaded image.
POST /api/detection/video   — Run detection on an uploaded video (async).
GET  /api/detection/config  — Return current detection configuration.
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

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
from app.evidence.evidence_generator import EvidenceGenerator
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


@router.post("/video")
async def detect_video(
    file: UploadFile = File(...),
    detector: BaseDetector = Depends(get_detector),
    tracker: ObjectTracker = Depends(get_tracker),
    analytics: AnalyticsAccumulator = Depends(get_analytics),
    violation_engine: ViolationEngine = Depends(get_violation_engine),
    evidence_gen: EvidenceGenerator = Depends(get_evidence_generator),
) -> Dict[str, Any]:
    """
    Process an uploaded video file through the full pipeline.

    Limitations:
      - This endpoint processes the entire video synchronously.
      - For production, use a background task queue (Celery/ARQ).
      - Response time depends on video length and hardware.
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

    # Write to a temp file so cv2.VideoCapture can open it
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Reset session state for fresh processing
    reset_session()
    session_id = str(uuid.uuid4())

    preprocessor_cfg = PreprocessingConfig(
        resize=(640, 640),
        clahe=True,
    )
    frame_processor = FrameProcessor(
        detector=detector,
        tracker=tracker,
        preprocessing_config=preprocessor_cfg,
    )

    all_violations = []
    frames_processed = 0

    # Annotated video output path
    videos_dir = settings.evidence_abs_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_filename = f"annotated_{session_id}.mp4"
    video_output_path = videos_dir / video_filename
    video_writer = None

    from app.cv.visualization import draw_tracked_object, draw_hud

    try:
        with VideoProcessor(tmp_path, frame_skip=settings.frame_skip) as vp:
            metadata = vp.metadata

            if metadata and metadata.width and metadata.height:
                output_fps = (metadata.fps or 25.0) / settings.frame_skip
                video_writer = VideoProcessor.create_writer(
                    output_path=video_output_path,
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

    finally:
        if video_writer:
            video_writer.release()
        Path(tmp_path).unlink(missing_ok=True)

    summary = analytics.build_summary()
    has_video = video_output_path.exists() and video_output_path.stat().st_size > 0

    return {
        "session_id": session_id,
        "frames_processed": frames_processed,
        "video_metadata": {
            "fps": metadata.fps if metadata else None,
            "width": metadata.width if metadata else None,
            "height": metadata.height if metadata else None,
            "total_frames": metadata.total_frames if metadata else None,
            "duration_seconds": round(metadata.total_frames / metadata.fps, 1) if (metadata and metadata.fps and metadata.total_frames) else None,
        },
        "annotated_video_url": f"/static/evidence/videos/{video_filename}" if has_video else None,
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
