"""
ML Model — TrOCR
Handwritten prescription OCR using microsoft/trocr-base-handwritten.
Logs each inference run to MLflow.

See: medocr_architecture_v3.md §6 — Model Assignment by Document Type
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

import mlflow
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

import os
from pathlib import Path

MODEL_NAME = "microsoft/trocr-base-handwritten"
MODEL_VERSION_TAG = "trocr-prescription-v1"
MLFLOW_EXPERIMENT = "TrOCR-Prescription"

# Singleton: load model once, reuse across Celery task calls
_processor: Optional[TrOCRProcessor] = None
_model: Optional[VisionEncoderDecoderModel] = None
_device: Optional[torch.device] = None

# Base path for local fine-tuned models
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FINETUNED_MODEL_DIR = PROJECT_ROOT / "models" / "trocr-finetuned"


@dataclass
class OcrOutput:
    raw_text: str
    confidence: float
    cer: Optional[float]
    wer: Optional[float]
    latency_ms: int
    model_version: str


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> None:
    """Load TrOCR processor and model into memory. Called once at worker startup."""
    global _processor, _model, _device, MODEL_VERSION_TAG, MODEL_NAME
    if _model is not None:
        return  # Already loaded

    _device = _get_device()
    
    # Check if a fine-tuned model exists locally
    if FINETUNED_MODEL_DIR.exists() and (FINETUNED_MODEL_DIR / "config.json").exists():
        model_path = str(FINETUNED_MODEL_DIR)
        MODEL_NAME = "local-finetuned/trocr"
        MODEL_VERSION_TAG = "trocr-prescription-v2-finetuned"
        logger.info(f"Loading FINE-TUNED TrOCR model from {model_path}")
    else:
        model_path = "microsoft/trocr-base-handwritten"
        MODEL_NAME = "microsoft/trocr-base-handwritten"
        MODEL_VERSION_TAG = "trocr-prescription-v1-base"
        logger.info(f"Loading BASE TrOCR model: {model_path}")

    _processor = TrOCRProcessor.from_pretrained(model_path)
    _model = VisionEncoderDecoderModel.from_pretrained(model_path).to(_device)
    _model.eval()
    logger.info(f"TrOCR loaded on {_device}")


def _compute_confidence(generated_ids: torch.Tensor) -> float:
    """
    Estimate confidence from token generation.
    TrOCR doesn't natively output per-token probabilities in standard usage,
    so we use a heuristic: text length relative to expected prescription length.
    Phase 2 will add beam score extraction for true confidence.
    """
    text_len = generated_ids.shape[-1]
    # Prescriptions typically 50–400 tokens; shorter outputs get lower confidence
    if text_len < 5:
        return 0.40
    elif text_len < 20:
        return 0.65
    else:
        return 0.88  # Heuristic until beam scores are extracted


def run_inference(image_path: str, ground_truth_text: Optional[str] = None) -> OcrOutput:
    """
    Run TrOCR inference on a prescription image.

    Args:
        image_path: Absolute path to image file.
        ground_truth_text: Optional reference text for CER/WER calculation.

    Returns:
        OcrOutput dataclass with raw_text, confidence, metrics, latency.
    """
    load_model()

    start_time = time.perf_counter()

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        raise

    pixel_values = _processor(images=image, return_tensors="pt").pixel_values.to(_device)

    with torch.no_grad():
        generated_ids = _model.generate(pixel_values)

    raw_text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    latency_ms = int((time.perf_counter() - start_time) * 1000)

    confidence = _compute_confidence(generated_ids)

    # CER/WER require ground truth — only computed during evaluation runs
    cer: Optional[float] = None
    wer: Optional[float] = None
    if ground_truth_text:
        cer, wer = _compute_cer_wer(raw_text, ground_truth_text)

    output = OcrOutput(
        raw_text=raw_text,
        confidence=confidence,
        cer=cer,
        wer=wer,
        latency_ms=latency_ms,
        model_version=MODEL_VERSION_TAG,
    )

    _log_to_mlflow(output, image_path)
    return output


def _compute_cer_wer(hypothesis: str, reference: str) -> tuple[float, float]:
    """
    Compute Character Error Rate (CER) and Word Error Rate (WER).
    Uses Levenshtein distance.
    """
    # Character Error Rate
    def levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

    cer = levenshtein(hypothesis, reference) / max(len(reference), 1)

    hyp_words = hypothesis.split()
    ref_words = reference.split()
    # levenshtein() works on both str and list via duck-typing:
    # len(), enumerate(), and element != comparison all work on lists of words.
    wer = levenshtein(hyp_words, ref_words) / max(len(ref_words), 1)

    return round(cer, 4), round(wer, 4)


def _log_to_mlflow(output: OcrOutput, image_path: str) -> None:
    """Log inference metrics to MLflow."""
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name=f"inference-{MODEL_VERSION_TAG}"):
            mlflow.log_param("model_name", MODEL_NAME)
            mlflow.log_param("model_version", MODEL_VERSION_TAG)
            mlflow.log_param("image_path", image_path)
            mlflow.log_metric("latency_ms", output.latency_ms)
            mlflow.log_metric("confidence", output.confidence)
            if output.cer is not None:
                mlflow.log_metric("cer", output.cer)
            if output.wer is not None:
                mlflow.log_metric("wer", output.wer)

    except Exception as e:
        # MLflow logging failures are non-fatal
        logger.warning(f"MLflow logging failed: {e}")
