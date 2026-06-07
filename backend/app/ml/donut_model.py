"""
ML Model — Donut
Invoice/receipt document understanding using naver-clova-ix/donut-base.
Donut is an end-to-end document understanding model that outputs structured JSON
without requiring a separate OCR step.

See: medocr_architecture_v3.md §6 — Model Assignment by Document Type
"""

import time
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import mlflow
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MODEL_NAME = "naver-clova-ix/donut-base"
MODEL_VERSION_TAG = "donut-invoice-v1"
MLFLOW_EXPERIMENT = "Donut-Invoice"

_processor: Optional[DonutProcessor] = None
_model: Optional[VisionEncoderDecoderModel] = None
_device: Optional[torch.device] = None


@dataclass
class DonutOutput:
    raw_text: str           # Raw decoded text from Donut
    structured_json: dict   # Parsed structured output
    confidence: float
    latency_ms: int
    model_version: str


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> None:
    """Load Donut processor and model. Called once at worker startup."""
    global _processor, _model, _device
    if _model is not None:
        return

    logger.info(f"Loading Donut model: {MODEL_NAME}")
    _device = _get_device()
    _processor = DonutProcessor.from_pretrained(MODEL_NAME)
    _model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(_device)
    _model.eval()
    logger.info(f"Donut loaded on {_device}")


def _parse_donut_output(text: str) -> dict:
    """
    Parse Donut's raw token output into a structured dict.
    Donut outputs pseudo-XML tags: <s_invoice_no>INV-001</s_invoice_no>
    We extract key-value pairs from these tags.
    """
    result = {}
    # Match <s_key>value</s_key> patterns
    pattern = r"<s_([^>]+)>(.*?)</s_\1>"
    matches = re.findall(pattern, text, re.DOTALL)
    for key, value in matches:
        value = value.strip()
        # Attempt numeric conversion
        try:
            result[key] = float(value) if "." in value else int(value)
        except ValueError:
            result[key] = value

    if not result:
        # Fallback: return raw text in a dict
        result = {"raw_output": text}

    return result


def run_inference(image_path: str) -> DonutOutput:
    """
    Run Donut inference on an invoice/receipt image.

    Args:
        image_path: Absolute path to image file.

    Returns:
        DonutOutput with raw_text, structured JSON, confidence, and latency.
    """
    load_model()

    start_time = time.perf_counter()

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        raise

    # Task prompt tells Donut which output schema to use.
    # cord-v2 covers receipt/structured-doc fields (amounts, dates, line items).
    # TODO Phase 2: replace with a custom task prompt if fine-tuning on medical invoices.
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = _processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(_device)

    pixel_values = _processor(image, return_tensors="pt").pixel_values.to(_device)

    with torch.no_grad():
        outputs = _model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=_model.decoder.config.max_position_embeddings,
            # early_stopping is omitted intentionally: it causes a deprecation warning
            # when num_beams=1 (greedy decode). Set num_beams>=2 to use early_stopping.
            pad_token_id=_processor.tokenizer.pad_token_id,
            eos_token_id=_processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[_processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    sequence = _processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(_processor.tokenizer.eos_token, "").replace(
        _processor.tokenizer.pad_token, ""
    )
    # Remove the task prompt prefix
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    structured = _parse_donut_output(sequence)

    # Confidence heuristic: structured output with multiple fields → high confidence
    n_fields = len([v for v in structured.values() if v])
    confidence = min(0.60 + (n_fields * 0.05), 0.98)

    output = DonutOutput(
        raw_text=sequence,
        structured_json=structured,
        confidence=confidence,
        latency_ms=latency_ms,
        model_version=MODEL_VERSION_TAG,
    )

    _log_to_mlflow(output, image_path)
    return output


def _log_to_mlflow(output: DonutOutput, image_path: str) -> None:
    """Log Donut inference metrics to MLflow."""
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name=f"inference-{MODEL_VERSION_TAG}"):
            mlflow.log_param("model_name", MODEL_NAME)
            mlflow.log_param("model_version", MODEL_VERSION_TAG)
            mlflow.log_param("image_path", image_path)
            mlflow.log_metric("latency_ms", output.latency_ms)
            mlflow.log_metric("confidence", output.confidence)
            mlflow.log_metric("structured_fields_extracted", len(output.structured_json))

    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")
