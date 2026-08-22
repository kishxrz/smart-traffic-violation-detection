"""
backend/app/detection/yolo_detector.py
────────────────────────────────────────
YOLOv8 detector using the Ultralytics library (PyTorch backend).

Architecture decision:
  Ultralytics YOLO wraps PyTorch inference behind a clean Python API.
  Under the hood:
    1. Image is resized to the model's input size (640×640 by default).
    2. A forward pass through the CSP-Darknet backbone produces feature maps.
    3. The PANet neck fuses multi-scale features.
    4. The detection head outputs [batch, anchors, 4+num_classes] tensors.
    5. NMS (Non-Maximum Suppression) removes duplicate boxes.

  We intentionally do NOT scatter model.predict() calls throughout the
  codebase. Every inference goes through this class.

Interview talking points:
  - YOLO = You Only Look Once; single-pass detector (no separate proposal step).
  - YOLOv8 uses anchor-free detection (vs anchor-based YOLOv5).
  - NMS IoU threshold controls how aggressively overlapping boxes are merged.
  - Confidence threshold is a trade-off: higher → fewer false positives,
    more missed detections (lower recall).

COCO classes detected (subset relevant to traffic):
  0:  person
  1:  bicycle
  2:  car
  3:  motorcycle
  5:  bus
  7:  truck
  9:  traffic light
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set

import numpy as np

from app.detection.detector import BaseDetector
from app.schemas.detection import BoundingBox, Detection

logger = logging.getLogger(__name__)

# COCO class IDs relevant to traffic analysis
TRAFFIC_CLASS_IDS: Set[int] = {
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    9,   # traffic light
}


class YOLODetector(BaseDetector):
    """
    YOLOv8 object detector, constrained to traffic-relevant COCO classes.

    Model is loaded lazily on first detect() call to avoid blocking
    application startup.
    """

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        filter_classes: Optional[Set[int]] = None,
    ) -> None:
        """
        Args:
            model_path:           Path to .pt weights or Ultralytics model name.
            confidence_threshold: Minimum score to keep a detection.
            iou_threshold:        NMS IoU threshold for box suppression.
            device:               "cpu", "cuda", or "mps" (Apple Silicon).
            filter_classes:       Restrict results to these class IDs.
                                  Defaults to TRAFFIC_CLASS_IDS.
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.filter_classes = filter_classes if filter_classes is not None else TRAFFIC_CLASS_IDS

        self._model = None      # Loaded lazily
        self._class_names: dict = {}
        logger.info(
            "YOLODetector configured: model=%s device=%s conf=%.2f iou=%.2f",
            self.model_path, self.device,
            self.confidence_threshold, self.iou_threshold,
        )

    def _load_model(self) -> None:
        """Load YOLO model weights. Called once before first inference."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        model_path_str = str(self.model_path)
        logger.info("Loading YOLO model: %s on device: %s", model_path_str, self.device)
        self._model = YOLO(model_path_str)
        self._class_names = self._model.names  # {0: 'person', 1: 'bicycle', ...}
        logger.info(
            "YOLO model loaded. %d classes available.", len(self._class_names)
        )

    @property
    def model(self):
        """Lazy-load the model on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
    ) -> List[Detection]:
        """
        Run YOLOv8 inference on a single BGR frame.

        Steps:
          1. Ultralytics converts BGR→RGB internally.
          2. Runs the forward pass.
          3. Applies confidence + NMS filtering.
          4. We filter to traffic-relevant class IDs.
          5. Convert to our Detection schema (no raw tensors escape this method).
        """
        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,  # suppress per-frame Ultralytics logging
                classes=list(self.filter_classes) if self.filter_classes else None,
            )
        except Exception as exc:
            logger.error("YOLO inference failed on frame %d: %s", frame_number, exc)
            return []

        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if self.filter_classes and cls_id not in self.filter_classes:
                    continue

                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]

                det = Detection(
                    class_id=cls_id,
                    class_name=self._class_names.get(cls_id, str(cls_id)),
                    confidence=conf,
                    bbox=BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    ),
                    frame_number=frame_number,
                    timestamp=timestamp,
                )
                detections.append(det)

        return detections

    def warmup(self) -> None:
        """
        Pre-load GPU kernels by running one inference on a blank frame.
        The first real inference is ~2-5× faster after warmup.
        """
        logger.info("Warming up YOLO model...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, frame_number=-1, timestamp=-1.0)
        logger.info("YOLO warmup complete.")

    def model_info(self) -> dict:
        return {
            "detector": "YOLODetector",
            "model_path": str(self.model_path),
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "filtered_classes": list(self.filter_classes),
            "class_names": self._class_names,
        }
