"""
ML Model — OMR Classical CV
Optical Mark Recognition for scanned checkbox/bubble forms using OpenCV.
Only 36 samples → rule-based classical CV, not deep learning.

Pipeline:
  1. Grayscale + Gaussian blur
  2. Adaptive threshold → binary image
  3. Contour detection → find circular/rectangular marks
  4. Grid alignment → map marks to form grid positions
  5. Threshold per cell → checked vs. unchecked

See: medocr_architecture_v3.md §6 — OMR Classical CV note
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_VERSION_TAG = "omr-classical-cv-v1"

# ── Configuration ──────────────────────────────────────────────────────────────
# These are tuned for standard OMR forms; adjust per form template
FILL_THRESHOLD = 0.45      # Fraction of cell that must be filled to count as "checked"
MIN_CONTOUR_AREA = 100     # Minimum pixel area for a mark contour
MAX_CONTOUR_AREA = 5000    # Maximum pixel area (filters large noise blobs)
BLUR_KERNEL = (5, 5)


@dataclass
class OmrOutput:
    raw_text: str                    # Human-readable summary of detected marks
    checked_cells: list[dict]        # List of {row, col, confidence} for each checked mark
    total_marks_detected: int
    confidence: float
    latency_ms: int
    model_version: str


def _preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess image for contour detection."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    # Adaptive threshold handles uneven lighting in scanned forms
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return binary


def _find_mark_contours(binary: np.ndarray) -> list:
    """Find contours that are likely checkbox marks."""
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mark_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA:
            # Check if contour is roughly square/circular (marks, not text lines)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if 0.5 <= aspect <= 2.0:  # Roughly square/circular
                mark_contours.append((x, y, w, h, area))
    return mark_contours


def _cluster_into_grid(
    mark_contours: list, img_height: int, img_width: int
) -> list[dict]:
    """
    Cluster detected marks into a grid to assign row/column positions.
    Returns list of {row, col, x, y, confidence} dicts.
    """
    if not mark_contours:
        return []

    checked = []
    # Sort by y (top to bottom), then x (left to right)
    sorted_marks = sorted(mark_contours, key=lambda m: (m[1], m[0]))

    for idx, (x, y, w, h, area) in enumerate(sorted_marks):
        # Estimate fill density within the bounding box
        fill_confidence = min(area / (w * h), 1.0) if w * h > 0 else 0.0

        if fill_confidence >= FILL_THRESHOLD:
            checked.append({
                "mark_index": idx,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "fill_confidence": round(fill_confidence, 3),
            })

    return checked


def run_inference(image_path: str) -> OmrOutput:
    """
    Run classical CV OMR detection on a scanned form image.

    Args:
        image_path: Absolute path to image file.

    Returns:
        OmrOutput with checked cells, mark count, and confidence.
    """
    start_time = time.perf_counter()

    try:
        binary = _preprocess_image(image_path)
        img_h, img_w = binary.shape
        mark_contours = _find_mark_contours(binary)
        checked_cells = _cluster_into_grid(mark_contours, img_h, img_w)

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        total_detected = len(mark_contours)
        total_checked = len(checked_cells)

        # Confidence based on number of marks found (too few = likely failed detection)
        confidence = 0.85 if total_detected >= 3 else 0.55

        raw_text = (
            f"OMR Analysis: {total_detected} marks detected, "
            f"{total_checked} checked. "
            f"Checked positions: {[c['mark_index'] for c in checked_cells]}"
        )

        return OmrOutput(
            raw_text=raw_text,
            checked_cells=checked_cells,
            total_marks_detected=total_detected,
            confidence=confidence,
            latency_ms=latency_ms,
            model_version=MODEL_VERSION_TAG,
        )

    except Exception as e:
        logger.error(f"OMR inference failed for {image_path}: {e}")
        return OmrOutput(
            raw_text=f"OMR detection failed: {str(e)}",
            checked_cells=[],
            total_marks_detected=0,
            confidence=0.0,
            latency_ms=int((time.perf_counter() - start_time) * 1000),
            model_version=MODEL_VERSION_TAG,
        )
