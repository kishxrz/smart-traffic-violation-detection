"""
backend/app/cv/frame_processor.py
───────────────────────────────────
Orchestrates the per-frame pipeline:
  1. Preprocess frame
  2. Run detection
  3. Update tracker
  4. Collect and return FrameResult

This is the integration point between CV (OpenCV) and DL (YOLO/tracking).
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from app.cv.preprocessing import FramePreprocessor, PreprocessingConfig
from app.detection.detector import BaseDetector
from app.schemas.detection import FrameResult, TrackedObject
from app.tracking.object_tracker import ObjectTracker

logger = logging.getLogger(__name__)


class FrameProcessor:
    """
    Combines preprocessing → detection → tracking into a single
    callable that processes one frame and returns a FrameResult.
    """

    def __init__(
        self,
        detector: BaseDetector,
        tracker: ObjectTracker,
        preprocessing_config: Optional[PreprocessingConfig] = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.preprocessor = FramePreprocessor(preprocessing_config)

    def process(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
    ) -> FrameResult:
        """
        Run the full per-frame pipeline.

        Args:
            frame:        BGR numpy array from VideoProcessor.
            frame_number: Sequential frame index.
            timestamp:    Position in seconds relative to video start.

        Returns:
            FrameResult with both raw detections and tracked objects.
        """
        t_start = time.perf_counter()

        # Step 1: Preprocessing (resize, CLAHE, etc.)
        preprocessed = self.preprocessor.process(frame)

        # Step 2: Object detection (YOLO inference)
        detections = self.detector.detect(preprocessed, frame_number, timestamp)

        # Step 3: Object tracking (assign persistent IDs)
        tracked_objects = self.tracker.update(detections, frame_number, timestamp)

        t_end = time.perf_counter()
        processing_ms = (t_end - t_start) * 1000

        return FrameResult(
            frame_number=frame_number,
            timestamp=timestamp,
            detections=detections,
            tracked_objects=tracked_objects,
            processing_time_ms=processing_ms,
        )
