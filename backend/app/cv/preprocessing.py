"""
backend/app/cv/preprocessing.py
────────────────────────────────
Image preprocessing utilities.

Design philosophy:
  Every operation is optional and documented with its purpose.
  We do NOT blindly apply all techniques — each is justified.

Interview talking points:
  - Gaussian blur reduces high-frequency noise before edge detection.
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) improves
    local contrast in overexposed or underexposed traffic footage
    without washing out already-bright regions.
  - Morphological operations (erode/dilate) clean up binary masks
    used in ROI-based analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """
    Fine-grained control over each preprocessing step.
    Pass this into FramePreprocessor to enable/disable operations.
    """
    # Resize target — None means no resize
    resize: Optional[Tuple[int, int]] = None          # (width, height)

    # Gaussian blur — kernel_size must be odd; sigma=0 = auto
    gaussian_blur: bool = False
    gaussian_kernel: int = 5
    gaussian_sigma: float = 0.0

    # Contrast enhancement via CLAHE
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    # Grayscale conversion (useful for mask operations)
    grayscale: bool = False

    # Canny edge detection
    canny: bool = False
    canny_low: int = 50
    canny_high: int = 150

    # Median blur (good for salt-and-pepper noise)
    median_blur: bool = False
    median_kernel: int = 5

    # Morphological ops on binary frames
    morph_open: bool = False
    morph_close: bool = False
    morph_kernel: int = 5


class FramePreprocessor:
    """
    Stateless preprocessing pipeline for individual frames.

    Each method is independent and can be called standalone.
    The process() method chains enabled operations in a sensible order.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()
        self._clahe: Optional[cv2.CLAHE] = None

        if self.config.clahe:
            # cv2.createCLAHE is moderately expensive to instantiate — create once.
            self._clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid,
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply configured preprocessing operations in sequence.

        Typical pipeline for traffic analysis:
          1. Resize (reduce compute)
          2. CLAHE (improve contrast for detection)
          3. Gaussian blur (reduce noise before inference)

        Returns a new array — the input frame is not modified.
        """
        out = frame.copy()

        if self.config.resize:
            out = self.resize(out, self.config.resize)

        if self.config.clahe:
            out = self.apply_clahe(out)

        if self.config.gaussian_blur:
            out = self.gaussian_blur(
                out,
                kernel_size=self.config.gaussian_kernel,
                sigma=self.config.gaussian_sigma,
            )

        if self.config.median_blur:
            out = self.median_blur(out, self.config.median_kernel)

        if self.config.grayscale:
            out = self.to_grayscale(out)

        if self.config.canny:
            out = self.canny_edges(out, self.config.canny_low, self.config.canny_high)

        if self.config.morph_open:
            out = self.morphological_open(out, self.config.morph_kernel)

        if self.config.morph_close:
            out = self.morphological_close(out, self.config.morph_kernel)

        return out

    # ── Individual Operations ─────────────────────────────────────────────────

    @staticmethod
    def resize(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize frame to (width, height).
        INTER_LINEAR is the fastest interpolation mode with acceptable quality.
        """
        return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def gaussian_blur(
        frame: np.ndarray,
        kernel_size: int = 5,
        sigma: float = 0.0,
    ) -> np.ndarray:
        """
        Apply Gaussian blur.

        Purpose: Reduces high-frequency noise (sensor noise, compression
        artefacts) before running edge detection or object detection.
        sigma=0 lets OpenCV choose the standard deviation from kernel_size.
        """
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # must be odd
        return cv2.GaussianBlur(frame, (k, k), sigma)

    @staticmethod
    def median_blur(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply median blur.

        Better than Gaussian for salt-and-pepper noise (e.g., camera
        sensor issues). Preserves edges better than Gaussian at the
        cost of being slower.
        """
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.medianBlur(frame, k)

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Why CLAHE over global histogram equalization?
        - Global HE amplifies noise in already-uniform regions.
        - CLAHE works on local tiles and clips the contrast enhancement,
          preventing over-amplification.
        - Useful for traffic footage with variable lighting (tunnels, dusk).

        CLAHE operates on the luminance channel (L in L*a*b*) to avoid
        changing the hue of detected colors (important for traffic lights).
        """
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if len(frame.shape) == 2:
            # Grayscale frame
            return self._clahe.apply(frame)

        # Convert BGR → LAB, apply CLAHE to L channel, convert back
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = self._clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    @staticmethod
    def to_grayscale(frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to single-channel grayscale."""
        if len(frame.shape) == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def canny_edges(
        frame: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150,
    ) -> np.ndarray:
        """
        Canny edge detection.

        The Canny algorithm:
          1. Gaussian smoothing (built-in, mild)
          2. Sobel gradient computation
          3. Non-maximum suppression (thin edges)
          4. Hysteresis thresholding (keep strong edges, discard weak)

        Useful for lane boundary detection and stop-line localization.
        """
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, low_threshold, high_threshold)

    @staticmethod
    def morphological_open(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Morphological opening (erosion then dilation).

        Removes small noise blobs from binary masks while
        preserving the shape of larger objects.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def morphological_close(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Morphological closing (dilation then erosion).

        Fills small holes inside foreground objects — useful for
        cleaning up segmentation masks.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def normalize(frame: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] float32.
        Used before feeding into non-YOLO PyTorch models.
        """
        return frame.astype(np.float32) / 255.0
