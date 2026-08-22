"""
backend/app/detection/models.py
─────────────────────────────────
Factory functions for creating detector instances from configuration.
Isolates model instantiation logic from startup/dependency injection code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.config import Settings
from app.detection.detector import BaseDetector
from app.detection.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


def create_detector(settings: Settings) -> BaseDetector:
    """
    Create and return the configured detector.

    Currently always returns a YOLODetector.
    If settings point to a missing model file, downloads the pretrained
    weights via Ultralytics (requires internet access on first run).
    """
    model_path = settings.yolo_model_abs_path

    if not model_path.exists():
        logger.warning(
            "YOLO model not found at %s. "
            "Ultralytics will attempt to download pretrained weights.",
            model_path,
        )
        # Use Ultralytics model name string (triggers auto-download)
        model_path_str = f"yolov8{settings.yolo_model_size}.pt"
    else:
        model_path_str = str(model_path)

    return YOLODetector(
        model_path=model_path_str,
        confidence_threshold=settings.confidence_threshold,
        iou_threshold=settings.iou_threshold,
        device=settings.device,
    )
