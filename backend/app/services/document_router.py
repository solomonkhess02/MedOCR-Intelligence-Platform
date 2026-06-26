"""
Service — Document Router
Classifies uploaded documents into one of four types:
  - prescription
  - lab_report
  - omr
  - invoice

Uses a lightweight heuristic classifier (filename + image analysis).
Phase 2 will upgrade this to a fine-tuned LayoutLMv3 document classifier.
"""

import re
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# ── Document type constants ───────────────────────────────────────────────────
DOC_TYPE_PRESCRIPTION = "prescription"
DOC_TYPE_LAB_REPORT = "lab_report"
DOC_TYPE_OMR = "omr"
DOC_TYPE_INVOICE = "invoice"
DOC_TYPE_UNKNOWN = "unknown"

# ── Filename keyword heuristics ───────────────────────────────────────────────
FILENAME_RULES: list[tuple[list[str], str]] = [
    (["prescription", "rx", "rx_", "presc"], DOC_TYPE_PRESCRIPTION),
    (["lab", "report", "blood", "test", "pathology", "biomarker"], DOC_TYPE_LAB_REPORT),
    (["omr", "form", "checkbox", "survey", "bubble"], DOC_TYPE_OMR),
    (["invoice", "receipt", "bill", "inv_", "payment"], DOC_TYPE_INVOICE),
]


def _classify_by_filename(filename: str) -> str | None:
    """
    Attempt classification based on filename keywords.
    Returns doc_type string or None if no match.
    """
    lower = filename.lower()
    for keywords, doc_type in FILENAME_RULES:
        if any(kw in lower for kw in keywords):
            return doc_type
    return None


def _classify_by_image_properties(image_path: str) -> str:
    """
    Fallback: use basic image properties to guess document type.
    - Aspect ratio: prescriptions are often portrait, invoices portrait
    - Color: OMR forms tend to be high-contrast B&W
    Returns a default doc_type.
    """
    try:
        img = Image.open(image_path).convert("L")  # grayscale
        width, height = img.size
        aspect = height / width

        # Very square or landscape → could be a form/OMR
        if 0.8 <= aspect <= 1.2:
            return DOC_TYPE_OMR

        # Tall portrait → prescription or lab report; default to prescription
        return DOC_TYPE_PRESCRIPTION

    except Exception as e:
        logger.warning(f"Image property classification failed: {e}")
        # Don't guess a concrete type we can't justify; 'unknown' routes to the
        # generic TrOCR fallback in the ML pipeline rather than mis-routing to Donut.
        return DOC_TYPE_UNKNOWN


def classify_document(image_path: str, original_filename: str = "") -> str:
    """
    Classify a document into one of the four supported types.

    Priority:
      1. Filename keyword heuristic (fast, no I/O)
      2. Image property analysis (fallback)

    Args:
        image_path: Absolute path to the saved image file.
        original_filename: Original uploaded filename for keyword matching.

    Returns:
        One of: 'prescription', 'lab_report', 'omr', 'invoice', 'unknown'
    """
    # Try filename first
    doc_type = _classify_by_filename(original_filename or Path(image_path).name)
    if doc_type:
        logger.info(f"Classified '{original_filename}' as '{doc_type}' via filename heuristic")
        return doc_type

    # Fallback to image properties
    doc_type = _classify_by_image_properties(image_path)
    logger.info(f"Classified '{original_filename}' as '{doc_type}' via image properties")
    return doc_type
