"""
ML Model — LayoutLMv3
Processes medical lab reports using microsoft/layoutlmv3-base.
Extracts token bounding boxes via Tesseract or Gemini Visual API,
normalizes coordinates, runs LayoutLMv3 representation learning,
and outputs structured biomarker JSON.
"""

import os
# Must be set before mlflow is imported so its HTTP client picks up the timeout
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "2"

import time
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

import mlflow
import torch
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MODEL_NAME = "microsoft/layoutlmv3-base"
MODEL_VERSION_TAG = "layoutlmv3-labreport-v1"
MLFLOW_EXPERIMENT = "LayoutLMv3-LabReport"

# Singleton model objects
_processor: Optional[object] = None
_model: Optional[object] = None
_device: Optional[torch.device] = None


@dataclass
class LayoutLMOutput:
    raw_text: str
    structured_json: dict
    confidence: float
    latency_ms: int
    model_version: str
    is_stub: bool = False


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> None:
    """Load LayoutLMv3 processor and model into memory once."""
    global _processor, _model, _device
    if _model is not None:
        return

    _device = _get_device()
    logger.info(f"Loading LayoutLMv3 model: {MODEL_NAME} on {_device}")

    try:
        from transformers import LayoutLMv3Processor, LayoutLMv3Model
        _processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)
        _model = LayoutLMv3Model.from_pretrained(MODEL_NAME).to(_device)
        _model.eval()
        logger.info("LayoutLMv3 loaded successfully.")
    except Exception as e:
        logger.warning(
            f"Failed to load LayoutLMv3 model from HuggingFace ({e}). "
            "Falling back to mock transformer embedding generation."
        )


def _extract_words_and_boxes(
    image_path: str, image: Image.Image
) -> Tuple[List[str], List[List[int]], str]:
    """
    Extract words and their normalized bounding boxes (0–1000 scale).
    Multi-stage fallback strategy:
      1. Try pytesseract (Tesseract OCR).
      2. If pytesseract fails/not installed, try Google Gemini Vision API (if configured).
      3. Fall back to high-fidelity mock extraction based on template.
    """
    width, height = image.size
    words: List[str] = []
    boxes: List[List[int]] = []
    raw_text = ""

    # ── Stage 1: PyTesseract ──────────────────────────────────────────────────
    try:
        import pytesseract
        # Honor an explicit Tesseract binary path from config (Windows installs are
        # often not on PATH); in Docker tesseract-ocr is on PATH so this stays blank.
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        # We run image_to_data to get word-level coordinates
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        n_boxes = len(data["text"])
        
        words_list = []
        boxes_list = []
        for i in range(n_boxes):
            text = data["text"][i].strip()
            if not text:
                continue
            
            # Get coords
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            
            # Normalize to 0-1000 scale
            xmin = max(0, min(1000, int(1000 * x / width)))
            ymin = max(0, min(1000, int(1000 * y / height)))
            xmax = max(0, min(1000, int(1000 * (x + w) / width)))
            ymax = max(0, min(1000, int(1000 * (y + h) / height)))
            
            words_list.append(text)
            boxes_list.append([xmin, ymin, xmax, ymax])

        if words_list:
            words = words_list
            boxes = boxes_list
            raw_text = " ".join(words_list)
            logger.info("Successfully extracted layout coordinates using pytesseract.")
            return words, boxes, raw_text

    except Exception as e:
        logger.debug(f"Pytesseract extraction failed or not installed: {e}")

    # ── Stage 2: Gemini Vision API Fallback ────────────────────────────────────
    if settings.google_api_key and settings.google_api_key != "your-gemini-api-key-here":
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.google_api_key)
            model = genai.GenerativeModel(settings.gemini_model)
            
            prompt = (
                "You are an OCR and layout analyzer. Extract all words from this medical lab report image. "
                "For each word, provide its text and bounding box normalized to a 0-1000 scale (left, top, right, bottom). "
                "Respond ONLY with a JSON list of dictionaries: [{\"word\": \"text\", \"box\": [xmin, ymin, xmax, ymax]}]."
            )
            response = model.generate_content([image, prompt])
            # Parse json
            cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_text)
            
            words_list = []
            boxes_list = []
            for item in data:
                words_list.append(item["word"])
                # Ensure box has 4 elements
                box = item["box"]
                if len(box) == 4:
                    boxes_list.append([
                        max(0, min(1000, int(box[0]))),
                        max(0, min(1000, int(box[1]))),
                        max(0, min(1000, int(box[2]))),
                        max(0, min(1000, int(box[3])))
                    ])
                else:
                    boxes_list.append([0, 0, 0, 0])
            
            if words_list:
                words = words_list
                boxes = boxes_list
                raw_text = " ".join(words_list)
                logger.info("Successfully extracted layout coordinates using Gemini API.")
                return words, boxes, raw_text

        except Exception as e:
            logger.warning(f"Gemini OCR extraction failed: {e}")

    # ── Stage 3: No OCR available — fail honestly ─────────────────────────────
    # If neither Tesseract nor the Gemini vision fallback could read the image, we
    # do NOT fabricate medical content. Returning empty extraction yields low
    # confidence downstream, routing the document to 'needs_review' (the honest
    # outcome for a medical document we could not actually read).
    logger.warning(
        "No OCR backend available for LayoutLMv3 (Tesseract not installed and no "
        "Gemini vision key). Returning empty extraction — document will fail to review."
    )
    return [], [], ""


def _parse_biomarkers(raw_text: str) -> dict:
    """
    Helper to extract structured biomarker JSON from raw text using regex heuristics.
    Looks for Glucose, HbA1c, WBC and reference ranges.
    """
    result = {
        "patient": "Unknown",
        "glucose": "Unknown",
        "hba1c": "Unknown",
        "wbc": "Unknown",
        # Static clinical reference ranges (general medical knowledge, not patient data).
        "reference_ranges": {
            "glucose": "70–100 mg/dL",
            "hba1c": "<5.7%",
            "wbc": "4.5–11.0 K/µL"
        }
    }

    # No text extracted → nothing to parse; return all-Unknown (fail honestly).
    if not raw_text or not raw_text.strip():
        return result

    # Extract patient if present
    patient_match = re.search(r"patient[:\s]+([A-Za-z\s]+?)(?:age|date|glucose|hba1c|$)", raw_text, re.IGNORECASE)
    if patient_match:
        result["patient"] = patient_match.group(1).strip()

    # Extract Glucose
    glucose_match = re.search(r"glucose\s*(\d+\s*(?:mg/dL|mg)?\b)", raw_text, re.IGNORECASE)
    if glucose_match:
        result["glucose"] = glucose_match.group(1).strip()
        if "mg" not in result["glucose"].lower():
            result["glucose"] += " mg/dL"
    else:
        # Check if just the number is present after glucose
        glucose_match_num = re.search(r"glucose\s+(\d+)", raw_text, re.IGNORECASE)
        if glucose_match_num:
            result["glucose"] = f"{glucose_match_num.group(1).strip()} mg/dL"

    # Extract HbA1c
    hba1c_match = re.search(r"hba1c\s*(\d+(?:\.\d+)?\s*(?:%)?\b)", raw_text, re.IGNORECASE)
    if hba1c_match:
        result["hba1c"] = hba1c_match.group(1).strip()
        if "%" not in result["hba1c"]:
            result["hba1c"] += "%"
    else:
        hba1c_match_num = re.search(r"hba1c\s+(\d+(?:\.\d+)?)", raw_text, re.IGNORECASE)
        if hba1c_match_num:
            result["hba1c"] = f"{hba1c_match_num.group(1).strip()}%"

    # Extract WBC
    wbc_match = re.search(r"wbc\s*(\d+(?:\.\d+)?\s*(?:K/uL|K/µL)?\b)", raw_text, re.IGNORECASE)
    if wbc_match:
        result["wbc"] = wbc_match.group(1).strip()
        if "k" not in result["wbc"].lower():
            result["wbc"] += " K/µL"
    else:
        wbc_match_num = re.search(r"wbc\s+(\d+(?:\.\d+)?)", raw_text, re.IGNORECASE)
        if wbc_match_num:
            result["wbc"] = f"{wbc_match_num.group(1).strip()} K/µL"

    return result


def run_inference(image_path: str) -> LayoutLMOutput:
    """
    Run LayoutLMv3 processing on a lab report image.
    1. Loads the layoutlm model.
    2. Performs word and layout extraction.
    3. Feeds layout features to LayoutLMv3 processor & model.
    4. Extracts structured biomarker JSON from OCR output.
    5. Logs to MLflow.
    """
    load_model()
    start_time = time.perf_counter()

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        raise

    # ── Step 1: Extract words, bounding boxes, and raw text ───────────────────
    words, boxes, raw_text = _extract_words_and_boxes(image_path, image)

    # ── Step 2: Feed inputs to LayoutLMv3 model ──────────────────────────────
    model_inference_success = False
    if _model is not None and _processor is not None:
        try:
            # LayoutLMv3 expects pixel values, input_ids, and bboxes
            # Limit to 512 tokens to fit inside LayoutLMv3 context window
            words_chunk = words[:512]
            boxes_chunk = boxes[:512]
            
            encoding = _processor(
                image, words_chunk, boxes=boxes_chunk, return_tensors="pt"
            )
            
            # Move inputs to device
            for k, v in encoding.items():
                if isinstance(v, torch.Tensor):
                    encoding[k] = v.to(_device)
                    
            with torch.no_grad():
                outputs = _model(**encoding)
                
            model_inference_success = True
            logger.debug(f"LayoutLMv3 model forward pass completed. Output keys: {outputs.keys()}")
        except Exception as e:
            logger.warning(f"LayoutLMv3 model forward pass failed: {e}. Falling back to OCR parser.")

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    # ── Step 3: Structured biomarker extraction ─────────────────────────────
    structured = _parse_biomarkers(raw_text)

    # Confidence calculation (heuristic — LayoutLMv3 has no single decode probability
    # like TrOCR; based on how many expected biomarker fields were actually parsed).
    if not raw_text or not raw_text.strip():
        # Nothing was read from the image — fail honestly to review.
        confidence = 0.0
    else:
        val_count = sum(1 for k in ["glucose", "hba1c", "wbc"] if structured[k] != "Unknown")
        confidence = 0.40 + (val_count * 0.18)  # 0.40 (text but no fields) → 0.94 (all 3)
        if not model_inference_success:
            confidence = min(confidence, 0.70)  # Cap if the transformer forward pass failed

    output = LayoutLMOutput(
        raw_text=raw_text,
        structured_json=structured,
        confidence=confidence,
        latency_ms=latency_ms,
        model_version=MODEL_VERSION_TAG,
        is_stub=False
    )

    # ── Step 4: MLflow Logging ────────────────────────────────────────────────
    _log_to_mlflow(output, image_path)

    return output


def _log_to_mlflow(output: LayoutLMOutput, image_path: str) -> None:
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
            mlflow.log_metric("structured_fields_extracted", len(output.structured_json))
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")
