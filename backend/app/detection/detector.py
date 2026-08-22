"""
backend/app/detection/detector.py
───────────────────────────────────
Abstract base class for object detectors.

Design principle:
  All violation logic and analytics depend on BaseDetector, NOT on
  YOLODetector directly. This means swapping YOLO for another model
  (e.g., RT-DETR, SSD) only requires writing a new subclass.

This separation also makes unit testing trivial — tests can use
MockDetector without loading any model weights.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from app.schemas.detection import Detection


class BaseDetector(ABC):
    """
    Abstract interface for object detectors.

    Any concrete detector must implement:
        detect(frame, frame_number, timestamp) -> List[Detection]

    Optional:
        warmup() — run one dummy inference to pre-load GPU kernels.
        model_info() -> dict — return model metadata for logging/API.
    """

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
    ) -> List[Detection]:
        """
        Run inference on a single frame.

        Args:
            frame:        Preprocessed BGR numpy array.
            frame_number: Index of the frame in the video sequence.
            timestamp:    Seconds since video start.

        Returns:
            List of Detection objects (may be empty if no objects found).
        """
        ...

    def warmup(self) -> None:
        """
        Optional: run one dummy inference to initialize GPU kernels.
        Call once after instantiation, before processing real frames.
        """

    def model_info(self) -> dict:
        """Return metadata about this detector for logging and the health API."""
        return {"detector": self.__class__.__name__}
