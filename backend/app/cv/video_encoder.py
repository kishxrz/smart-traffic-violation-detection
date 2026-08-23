"""
backend/app/cv/video_encoder.py
──────────────────────────────────
Browser-compatible H.264 MP4 video encoder and validation pipeline.

Problem solved:
  OpenCV's VideoWriter on Windows defaults to 'mp4v' (MPEG-4 Part 2), which is
  unsupported by HTML5 <video> in modern Chrome, Edge, and Safari (showing 0:00).

Solution:
  1. Write temporary annotated frames with OpenCV VideoWriter.
  2. Transcode the temporary file to H.264 (libx264) using imageio-ffmpeg's
     bundled static FFmpeg binary.
  3. Apply -pix_fmt yuv420p (for browser hardware acceleration) and
     -movflags +faststart (for instant progressive HTML5 streaming).
  4. Perform post-processing validation using OpenCV before returning API response.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Any

import cv2

logger = logging.getLogger(__name__)


def encode_to_browser_mp4(
    input_video_path: Path,
    output_mp4_path: Path,
) -> Path:
    """
    Transcode input video to browser-compatible H.264 (libx264) MP4 using imageio-ffmpeg.

    Args:
        input_video_path: Temporary video file created by OpenCV VideoWriter.
        output_mp4_path: Target path for the final browser-playable MP4 video.

    Returns:
        Path to the output_mp4_path if transcoding succeeds, or input_video_path fallback.
    """
    input_video_path = Path(input_video_path)
    output_mp4_path = Path(output_mp4_path)

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as err:
        logger.warning("imageio-ffmpeg is not available: %s. Using raw video.", err)
        return input_video_path

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_video_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-threads", "1",
        "-movflags", "+faststart",
        str(output_mp4_path),
    ]

    logger.info("Transcoding annotated video to H.264 MP4 via FFmpeg: %s", output_mp4_path.name)
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        logger.error("FFmpeg H.264 encoding failed (code %d): %s", result.returncode, result.stderr)
        return input_video_path

    if not output_mp4_path.exists() or output_mp4_path.stat().st_size < 1000:
        logger.error("FFmpeg output file missing or invalid: %s", output_mp4_path)
        return input_video_path

    logger.info(
        "H.264 encoding successful: %s (size: %d bytes)",
        output_mp4_path.name, output_mp4_path.stat().st_size
    )

    # Clean up temporary raw video file if different
    if input_video_path != output_mp4_path and input_video_path.exists():
        try:
            input_video_path.unlink(missing_ok=True)
        except Exception:
            pass

    return output_mp4_path


def validate_video_file(file_path: Path) -> Dict[str, Any]:
    """
    Validate that a generated video file exists, is non-empty, can be opened by OpenCV,
    and has valid frames, FPS, resolution, and decodable frame 1.

    Raises:
        ValueError: If file does not exist, is empty, or has bad metadata.
        RuntimeError: If OpenCV cannot decode frames from the video.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ValueError(f"Video file does not exist: {file_path}")

    size_bytes = file_path.stat().st_size
    if size_bytes < 5000:
        raise ValueError(f"Video file size is too small ({size_bytes} bytes): {file_path}")

    cap = cv2.VideoCapture(str(file_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV VideoCapture cannot open file: {file_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid video metadata for {file_path.name}: "
                f"fps={fps}, frames={frame_count}, res={width}x{height}"
            )

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            raise RuntimeError(f"Could not decode frame 1 from video: {file_path.name}")

        duration_seconds = round(frame_count / fps, 1)

        return {
            "path": str(file_path),
            "size_bytes": size_bytes,
            "fps": fps,
            "frame_count": int(frame_count),
            "width": width,
            "height": height,
            "duration_seconds": duration_seconds,
            "is_valid": True,
        }

    finally:
        cap.release()
