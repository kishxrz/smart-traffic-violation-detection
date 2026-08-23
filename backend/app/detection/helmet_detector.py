"""
backend/app/detection/helmet_detector.py
──────────────────────────────────────────
Custom YOLO Helmet & Rider Safety Classification Detector.

Design:
  Loads custom trained YOLO weights (default: models/helmet_v3.pt) to classify head crops into:
    - 0: helmet
    - 1: no_helmet

Known Model Limitation:
  The current helmet classifier performs reliably on helmet-positive detection but has limited NO_HELMET generalization under certain real-world conditions. The system therefore uses conservative temporal confirmation to reduce false violation reports.

Environment Variable:
  HELMET_MODEL_PATH=models/helmet_v3.pt (configurable)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from app.config import get_settings
from app.schemas.detection import BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class HelmetPrediction:
    """Prediction output for a single rider head crop."""
    status: str              # "HELMET" | "NO_HELMET" | "UNKNOWN"
    confidence: float        # Prediction confidence [0.0, 1.0]
    class_id: Optional[int]  # 0: helmet, 1: no_helmet, None: UNKNOWN
    bbox: Optional[BoundingBox] = None  # BoundingBox inside the head crop if detected
    crop_shape: Optional[Tuple[int, int]] = None
    vehicle_id: Optional[int] = None
    person_id: Optional[int] = None
    frame_number: Optional[int] = None
    obs_count: int = 0


class HelmetDetector:
    """
    Production-grade Custom Helmet Classifier wrapping Ultralytics YOLO.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        env_path = os.getenv("HELMET_MODEL_PATH", settings.helmet_model_path or "models/helmet_v3.pt")
        self.model_path = Path(model_path or env_path)
        self.conf_threshold = conf_threshold if conf_threshold is not None else getattr(settings, "helmet_confidence_threshold", 0.35)
        self.device = device or settings.device

        self._model = None
        self._is_loaded = False
        self._is_classifier = False   # True when model is YOLOv8-cls
        self._load_model_if_exists()

    def _load_model_if_exists(self) -> None:
        """Attempt to load custom YOLO helmet model if file exists."""
        if not self.model_path.exists():
            logger.info(
                "Custom helmet model file not found at %s. "
                "HelmetDetector will operate in UNKNOWN fallback mode until trained weights are provided.",
                self.model_path,
            )
            self._is_loaded = False
            return

        try:
            import torch
            import cv2
            torch.set_num_threads(1)
            cv2.setNumThreads(1)
        except Exception:
            pass

        try:
            from ultralytics import YOLO
            logger.info("Loading custom helmet YOLO model v2 from: %s", self.model_path)
            self._model = YOLO(str(self.model_path))
            self._is_loaded = True
            # Detect model task: 'classify' for cls models, 'detect' for det models
            task = getattr(self._model, 'task', None) or getattr(self._model.model, 'task', None) or ''
            self._is_classifier = (str(task).lower() == 'classify')
            logger.info(
                "Custom helmet YOLO model loaded successfully. task=%s classifier=%s",
                task, self._is_classifier,
            )
        except Exception as err:
            logger.warning("Failed to load custom helmet model: %s", err)
            self._is_loaded = False

    @property
    def is_available(self) -> bool:
        return self._is_loaded and self._model is not None

    def predict_crop(
        self,
        head_crop: np.ndarray,
        vehicle_id: Optional[int] = None,
        person_id: Optional[int] = None,
        frame_number: Optional[int] = None,
        obs_count: int = 0,
    ) -> HelmetPrediction:
        """
        Classify a cropped head ROI image.

        Returns:
            HelmetPrediction(status="HELMET" | "NO_HELMET" | "UNKNOWN", confidence=float, ...)
        """
        if not self.is_available or head_crop is None or head_crop.size == 0:
            return HelmetPrediction(
                status="UNKNOWN",
                confidence=0.0,
                class_id=None,
                crop_shape=head_crop.shape[:2] if head_crop is not None else None,
                vehicle_id=vehicle_id,
                person_id=person_id,
                frame_number=frame_number,
                obs_count=obs_count,
            )

        crop_h, crop_w = head_crop.shape[:2]

        try:
            # Run inference on crop
            results = self._model.predict(
                source=head_crop,
                device=self.device,
                verbose=False,
            )

            if not results:
                return HelmetPrediction(
                    status="UNKNOWN",
                    confidence=0.0,
                    class_id=None,
                    crop_shape=(crop_h, crop_w),
                    vehicle_id=vehicle_id,
                    person_id=person_id,
                    frame_number=frame_number,
                    obs_count=obs_count,
                )

            # ── Classification model path (YOLOv8-cls) ──────────────────────
            if self._is_classifier:
                probs = results[0].probs
                if probs is None:
                    return HelmetPrediction(
                        status="UNKNOWN", confidence=0.0, class_id=None,
                        crop_shape=(crop_h, crop_w),
                        vehicle_id=vehicle_id, person_id=person_id,
                        frame_number=frame_number, obs_count=obs_count,
                    )
                best_cls  = int(probs.top1)
                best_conf = float(probs.top1conf.cpu().item())

                if best_conf < self.conf_threshold:
                    return HelmetPrediction(
                        status="UNKNOWN", confidence=best_conf, class_id=None,
                        crop_shape=(crop_h, crop_w),
                        vehicle_id=vehicle_id, person_id=person_id,
                        frame_number=frame_number, obs_count=obs_count,
                    )

                if best_cls == 0:
                    status, class_id = "HELMET", 0
                elif best_cls == 1:
                    status, class_id = "NO_HELMET", 1
                else:
                    status, class_id = "UNKNOWN", None

                return HelmetPrediction(
                    status=status, confidence=best_conf, class_id=class_id,
                    bbox=None, crop_shape=(crop_h, crop_w),
                    vehicle_id=vehicle_id, person_id=person_id,
                    frame_number=frame_number, obs_count=obs_count,
                )

            # ── Detection model path (original YOLO bbox) ────────────────────
            if not results or len(results[0].boxes) == 0:
                return HelmetPrediction(
                    status="UNKNOWN",
                    confidence=0.0,
                    class_id=None,
                    crop_shape=(crop_h, crop_w),
                    vehicle_id=vehicle_id,
                    person_id=person_id,
                    frame_number=frame_number,
                    obs_count=obs_count,
                )

            boxes = results[0].boxes
            best_idx = int(boxes.conf.argmax())
            best_conf = float(boxes.conf[best_idx].cpu().item())
            best_cls = int(boxes.cls[best_idx].cpu().item())

            if best_conf < self.conf_threshold:
                return HelmetPrediction(
                    status="UNKNOWN", confidence=best_conf, class_id=None,
                    crop_shape=(crop_h, crop_w),
                    vehicle_id=vehicle_id, person_id=person_id,
                    frame_number=frame_number, obs_count=obs_count,
                )

            xyxy = boxes.xyxy[best_idx].cpu().numpy()
            bbox = BoundingBox(x1=int(xyxy[0]), y1=int(xyxy[1]), x2=int(xyxy[2]), y2=int(xyxy[3]))

            # Map class_id to status: 0 = helmet, 1 = no_helmet
            if best_cls == 0:
                status = "HELMET"
                class_id = 0
            elif best_cls == 1:
                status = "NO_HELMET"
                class_id = 1
            else:
                status = "UNKNOWN"
                class_id = None

            return HelmetPrediction(
                status=status,
                confidence=best_conf,
                class_id=class_id,
                bbox=bbox,
                crop_shape=(crop_h, crop_w),
                vehicle_id=vehicle_id,
                person_id=person_id,
                frame_number=frame_number,
                obs_count=obs_count,
            )

        except Exception as err:
            logger.error("Error during helmet inference: %s", err)
            return HelmetPrediction(
                status="UNKNOWN",
                confidence=0.0,
                class_id=None,
                crop_shape=(crop_h, crop_w),
                vehicle_id=vehicle_id,
                person_id=person_id,
                frame_number=frame_number,
                obs_count=obs_count,
            )
