"""
backend/app/cv/video_processor.py
───────────────────────────────────
Video capture and frame extraction pipeline.

Design:
  VideoProcessor is a context manager that opens a video source,
  yields frames with metadata, and handles corrupted/unreadable
  frames gracefully. It never raises on individual bad frames.

Sources supported:
  - File path (mp4, avi, mov, etc.)
  - Integer index (webcam: 0, 1, …)
  - RTSP/HTTP stream URL

Interview talking points:
  - cv2.VideoCapture encapsulates FFmpeg under the hood.
  - We read() returns (bool, frame) — checking the bool is critical.
  - Frame skipping reduces GPU/CPU load without changing detection quality
    much at high-FPS videos (30fps → process every 3rd frame = 10 fps).
  - Writing output with cv2.VideoWriter requires knowing the target codec.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Generator, Iterator, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FrameGenerator = Generator[Tuple[int, float, np.ndarray], None, None]


class VideoMetadata:
    """Extracted properties of a video source."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc: int = int(cap.get(cv2.CAP_PROP_FOURCC))

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.fps if self.fps > 0 else 0.0

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.width, self.height

    def __repr__(self) -> str:
        return (
            f"VideoMetadata(fps={self.fps:.2f}, "
            f"resolution={self.width}×{self.height}, "
            f"total_frames={self.total_frames}, "
            f"duration={self.duration_seconds:.1f}s)"
        )


class VideoProcessor:
    """
    Context manager that wraps cv2.VideoCapture.

    Example:
        with VideoProcessor("traffic.mp4", frame_skip=2) as vp:
            for frame_num, timestamp, frame in vp.frames():
                results = detector.detect(frame)
    """

    def __init__(
        self,
        source: Union[str, int],
        frame_skip: int = 1,
        max_failures: int = 10,
    ) -> None:
        """
        Args:
            source:       Video file path, camera index, or stream URL.
            frame_skip:   Process every Nth frame. 1 = no skip.
            max_failures: Stop after this many consecutive read failures.
        """
        self.source = source
        self.frame_skip = max(1, frame_skip)
        self.max_failures = max_failures

        self._cap: Optional[cv2.VideoCapture] = None
        self.metadata: Optional[VideoMetadata] = None

    def __enter__(self) -> "VideoProcessor":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.release()

    def open(self) -> None:
        """Open the video source and populate metadata."""
        logger.info("Opening video source: %s", self.source)
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source!r}")

        self.metadata = VideoMetadata(self._cap)
        logger.info("Video opened: %s", self.metadata)

    def release(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.debug("Video capture released.")

    def frames(self) -> FrameGenerator:
        """
        Yield (frame_number, timestamp_seconds, frame_array) tuples.

        Skips frames according to frame_skip.
        Skips (but counts) corrupted frames.
        Stops after max_failures consecutive failures.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("VideoProcessor is not open. Use as context manager.")

        fps = self.metadata.fps if self.metadata else 25.0
        frame_number = 0
        consecutive_failures = 0

        while True:
            ret, frame = self._cap.read()

            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= self.max_failures:
                    logger.warning(
                        "Stopping: %d consecutive read failures.", consecutive_failures
                    )
                    break
                logger.debug("Frame %d read failed — skipping.", frame_number)
                frame_number += 1
                continue

            consecutive_failures = 0  # reset on successful read

            # Apply frame skip — only yield every Nth frame
            if frame_number % self.frame_skip == 0:
                timestamp = frame_number / fps
                yield frame_number, timestamp, frame

            frame_number += 1

    def seek(self, frame_number: int) -> bool:
        """Jump to a specific frame number."""
        if self._cap is None:
            return False
        return self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_number))

    @staticmethod
    def create_writer(
        output_path: Union[str, Path],
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ) -> cv2.VideoWriter:
        """
        Create a cv2.VideoWriter for saving annotated output video.

        Codec notes:
          - "mp4v" → .mp4 — widely compatible, decent compression.
          - "XVID" → .avi — older but reliable.
          - "avc1" → .mp4 with H.264 — best quality (requires ffmpeg build).
        """
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise IOError(f"Cannot create video writer: {output_path}")
        return writer


def read_single_frame(source: Union[str, int], frame_number: int = 0) -> np.ndarray:
    """
    Read a single frame from a video source without streaming.
    Useful for snapshot/thumbnail extraction.
    """
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise IOError(f"Cannot open: {source!r}")
        if frame_number > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_number))
        ret, frame = cap.read()
        if not ret:
            raise IOError(f"Could not read frame {frame_number} from {source!r}")
        return frame
    finally:
        cap.release()
