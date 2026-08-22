"""
backend/app/detection/helmet_detector.py
──────────────────────────────────────────
Custom YOLO Helmet & Rider Safety Classification Detector.

Design:
  Loads custom trained YOLO weights (default: models/helmet_v1.pt) to detect:
    - 0: helmet
    - 1: no_helmet

Environment Variable:
  HELMET_MODEL_PATH=models/helmet_v1.pt (configurable, no hardcoded machine paths)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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


class HelmetDetector:
    """
    Production-grade Custom Helmet Classifier wrapping Ultralytics YOLO.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.50,
        device: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        env_path = os.getenv("HELMET_MODEL_PATH", "models/helmet_v1.pt")
        self.model_path = Path(model_path or env_path)
        self.conf_threshold = conf_threshold
        self.device = device or settings.device

        self._model = None
        self._is_loaded = False
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
            from ultralytics import YOLO
            logger.info("Loading custom helmet YOLO model from: %s", self.model_path)
            self._model = YOLO(str(self.model_path))
            self._is_loaded = True
            logger.info("Custom helmet YOLO model loaded successfully.")
        except Exception as err:
            logger.warning("Failed to load custom helmet model: %s", err)
            self._is_loaded = False

    @property
    def is_available(self) -> bool:
        return self._is_loaded and self._model is not None

    def predict_crop(self, head_crop: np.ndarray) -> HelmetPrediction:
        """
        Classify a cropped head ROI image.

        Returns:
            HelmetPrediction(status="HELMET" | "NO_HELMET" | "UNKNOWN", confidence=float, ...)
        """
        if not self.is_available or head_crop is None or head_crop.size == 0:
            return HelmetPrediction(status="UNKNOWN", confidence=0.0, class_id=None)

        try:
            # Run inference on crop
            results = self._model.predict(
                source=head_crop,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )

            if not results or len(results[0].boxes) == 0:
                # No helmet or no_helmet box detected with sufficient confidence
                return HelmetPrediction(status="UNKNOWN", confidence=0.0, class_id=None)

            boxes = results[0].boxes
            best_idx = int(boxes.conf.argmax())
            best_conf = float(boxes.conf[best_idx].cpu().item())
            best_cls = int(boxes.cls[best_idx].cpu().item())

            xyxy = boxes.xyxy[best_idx].cpu().numpy()
            bbox = BoundingBox(x1=int(xyxy[0]), y1=int(xyxy[1]), x2=int(xyxy[2]), y2=int(xyxy[3]))

            # Map class_id to status: 0 = helmet, 1 = no_helmet
            if best_cls == 0:
                return HelmetPrediction(status="HELMET", confidence=best_conf, class_id=0, bbox=bbox)
            elif best_cls == 1:
                return HelmetPrediction(status="NO_HELMET", confidence=best_conf, class_id=1, bbox=bbox)
            else:
                return HelmetPrediction(status="UNKNOWN", confidence=best_conf, class_id=None)

        except Exception as err:
            logger.error("Error during helmet inference: %s", err)
            return HelmetPrediction(status="UNKNOWN", confidence=0.0, class_id=None)
