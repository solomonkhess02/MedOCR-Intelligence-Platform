"""
Service — Confidence Gate
Architecture-enforced quality guard: documents with OCR confidence < 0.75
are flagged as 'needs_review' and NEVER passed to the agent layer.

This is the most important quality safeguard in the ML layer.
See: medocr_architecture_v3.md §6 Confidence Gate
"""

from app.config import get_settings

settings = get_settings()


def check_confidence(confidence: float) -> tuple[bool, str]:
    """
    Evaluate whether an OCR confidence score passes the threshold.

    Args:
        confidence: Float in [0.0, 1.0] from the OCR model.

    Returns:
        (passed, status) where:
          - passed=True, status='complete'   → confidence >= threshold
          - passed=False, status='needs_review' → confidence < threshold
    """
    threshold = settings.confidence_threshold

    if confidence >= threshold:
        return True, "complete"
    else:
        return False, "needs_review"


def get_threshold() -> float:
    """Return the current confidence threshold."""
    return settings.confidence_threshold
