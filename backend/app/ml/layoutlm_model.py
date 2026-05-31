"""
ML Model — LayoutLMv3 (STUB for Phase 1)
Full implementation will be added in Phase 2.

LayoutLMv3 processes lab reports with both text tokens and spatial layout information.
It requires bounding box coordinates from a PDF/image parser, making it more complex
to set up than TrOCR or Donut — hence the Phase 2 deferral.

In Phase 1, this stub returns a clearly marked placeholder output so the pipeline
does not break when a lab_report document is uploaded.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_VERSION_TAG = "layoutlmv3-labreport-stub-v0"


@dataclass
class LayoutLMOutput:
    raw_text: str
    confidence: float
    latency_ms: int
    model_version: str
    is_stub: bool = True


def load_model() -> None:
    """No-op in Phase 1 stub."""
    logger.info("LayoutLMv3 stub — full model loads in Phase 2")


def run_inference(image_path: str) -> LayoutLMOutput:
    """
    Phase 1 stub: returns placeholder output for lab report images.

    Full Phase 2 implementation will:
      - Load microsoft/layoutlmv3-base
      - Extract bounding boxes with PyMuPDF / Tesseract
      - Run token classification for biomarker field extraction
      - Return structured biomarker JSON
    """
    start_time = time.perf_counter()
    latency_ms = int((time.perf_counter() - start_time) * 1000)

    logger.warning(
        f"LayoutLMv3 stub called for {image_path}. "
        "Real model will be implemented in Phase 2."
    )

    return LayoutLMOutput(
        raw_text="[STUB] Lab report OCR not yet implemented. Phase 2 will add LayoutLMv3.",
        confidence=0.50,  # Below threshold → will be flagged 'needs_review' by confidence gate
        latency_ms=latency_ms,
        model_version=MODEL_VERSION_TAG,
        is_stub=True,
    )
