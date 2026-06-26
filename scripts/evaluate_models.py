"""
MedOCR Intelligence Platform
Model Evaluation Script

Evaluates the fine-tuned TrOCR model on the full test split (247 samples).
Computes per-sample and aggregate CER / WER.
Saves a JSON report to results/evaluation_report.json.
Logs all metrics and the report artifact to MLflow experiment: MedOCR-Evaluation.

Note on Donut evaluation:
  Donut outputs structured pseudo-XML, not raw text, so CER/WER against the
  plain-text ground truth is not directly meaningful. Donut is therefore
  evaluated by confidence heuristic only (structured field count).
  Phase 2 will add proper structured-field accuracy once field-level labels exist.

Usage:
    python scripts/evaluate_models.py
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

import mlflow
import torch
import evaluate
import numpy as np
from datasets import load_from_disk
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Shared prescription filter (same definition used by train_trocr.py).
from doc_filters import is_prescription

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

DATASET_PATH    = PROJECT_ROOT / "data"   / "medocr-vision-dataset"
TROCR_MODEL_DIR = PROJECT_ROOT / "models" / "trocr-finetuned"
DONUT_MODEL_DIR = PROJECT_ROOT / "models" / "donut-finetuned"
RESULTS_DIR     = PROJECT_ROOT / "results"
MLFLOW_URI      = "http://127.0.0.1:5000"
EXPERIMENT      = "MedOCR-Evaluation"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── TrOCR evaluation ───────────────────────────────────────────────────────────
def evaluate_trocr(test_dataset, device: torch.device) -> dict | None:
    """
    Run TrOCR inference on the full test split and compute CER + WER.

    Returns a results dict, or None if the model directory is not found.
    """
    if not (TROCR_MODEL_DIR / "config.json").exists():
        print(f"\n[SKIP] TrOCR model not found at {TROCR_MODEL_DIR}")
        print("       Run train_trocr.py first, then re-run this script.")
        return None

    print(f"\n[*] Loading fine-tuned TrOCR from {TROCR_MODEL_DIR}...")
    processor = TrOCRProcessor.from_pretrained(str(TROCR_MODEL_DIR))
    model     = VisionEncoderDecoderModel.from_pretrained(str(TROCR_MODEL_DIR)).to(device)
    model.eval()

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    predictions: list[str] = []
    references:  list[str] = []
    latencies:   list[int] = []
    per_sample:  list[dict] = []

    total = len(test_dataset)
    print(f"    Running inference on {total} test samples...")

    for i, sample in enumerate(test_dataset):
        if i % 25 == 0:
            print(f"    [{i:>3}/{total}] processing...")

        image        = sample["image"].convert("RGB")
        ground_truth = sample["text"]

        t0 = time.perf_counter()
        pixel_values   = processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids  = model.generate(pixel_values)
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        latency_ms     = int((time.perf_counter() - t0) * 1000)

        predictions.append(predicted_text)
        references.append(ground_truth)
        latencies.append(latency_ms)

        # Per-sample CER
        sample_cer = cer_metric.compute(
            predictions=[predicted_text], references=[ground_truth]
        )
        per_sample.append({
            "sample_index":  i,
            "latency_ms":    latency_ms,
            "cer":           round(sample_cer, 4),
            "text_length":   len(ground_truth),
            "predicted_len": len(predicted_text),
        })

    mean_cer     = cer_metric.compute(predictions=predictions, references=references)
    mean_wer     = wer_metric.compute(predictions=predictions, references=references)
    mean_latency = float(np.mean(latencies))

    return {
        "model":           "TrOCR",
        "model_path":      str(TROCR_MODEL_DIR),
        "sample_count":    total,
        "mean_cer":        round(mean_cer, 4),
        "mean_wer":        round(mean_wer, 4),
        "mean_latency_ms": round(mean_latency, 1),
        "min_cer":         round(float(np.min([s["cer"] for s in per_sample])), 4),
        "max_cer":         round(float(np.max([s["cer"] for s in per_sample])), 4),
        # Spot-check: first 5 predictions
        "spot_check": [
            {"reference":  references[i][:120],
             "prediction": predictions[i][:120],
             "cer":        per_sample[i]["cer"]}
            for i in range(min(5, total))
        ],
        "per_sample": per_sample,
    }


# ── Donut evaluation ───────────────────────────────────────────────────────────
def evaluate_donut(test_dataset, device: torch.device) -> dict | None:
    """
    Run Donut inference on the full test split.

    Since the dataset ground truth is plain text and Donut outputs pseudo-XML,
    we evaluate by confidence heuristic (number of structured fields extracted)
    rather than CER/WER. Phase 2 will add field-level accuracy.
    """
    try:
        from transformers import DonutProcessor
        import re
    except ImportError:
        print("[SKIP] transformers DonutProcessor not available.")
        return None

    if not (DONUT_MODEL_DIR / "config.json").exists():
        print(f"\n[SKIP] Donut model not found at {DONUT_MODEL_DIR}")
        print("       Run train_donut.py first, then re-run this script.")
        return None

    print(f"\n[*] Loading fine-tuned Donut from {DONUT_MODEL_DIR}...")
    processor = DonutProcessor.from_pretrained(str(DONUT_MODEL_DIR))
    from transformers import VisionEncoderDecoderModel as VEDM
    model     = VEDM.from_pretrained(str(DONUT_MODEL_DIR)).to(device)
    model.eval()

    task_prompt = "<s_cord-v2>"
    latencies:        list[int]   = []
    field_counts:     list[int]   = []
    confidences:      list[float] = []

    total = len(test_dataset)
    print(f"    Running inference on {total} test samples...")

    for i, sample in enumerate(test_dataset):
        if i % 25 == 0:
            print(f"    [{i:>3}/{total}] processing...")

        image = sample["image"].convert("RGB")

        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)

        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                return_dict_in_generate=True,
            )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "")
        sequence = sequence.replace(processor.tokenizer.pad_token, "")

        # Count extracted fields as a confidence proxy
        n_fields   = len(re.findall(r"<s_[^/][^>]*>.*?</s_", sequence))
        confidence = min(0.60 + (n_fields * 0.05), 0.98)

        latencies.append(latency_ms)
        field_counts.append(n_fields)
        confidences.append(confidence)

    return {
        "model":              "Donut",
        "model_path":         str(DONUT_MODEL_DIR),
        "sample_count":       total,
        "mean_latency_ms":    round(float(np.mean(latencies)), 1),
        "mean_confidence":    round(float(np.mean(confidences)), 4),
        "mean_fields_found":  round(float(np.mean(field_counts)), 2),
        "note": (
            "CER/WER not computed: Donut outputs pseudo-XML while ground truth "
            "is plain text. Use field-level accuracy in Phase 2."
        ),
    }


# ── Print summary table ────────────────────────────────────────────────────────
def print_summary(trocr: dict | None, donut: dict | None) -> None:
    print("\n" + "=" * 62)
    print("  MEDOCR EVALUATION RESULTS")
    print("=" * 62)
    print(f"  {'Model':<15} {'Samples':>8} {'CER':>8} {'WER':>8} {'Latency(ms)':>12}")
    print("  " + "-" * 55)
    if trocr:
        print(
            f"  {'TrOCR':<15} {trocr['sample_count']:>8} "
            f"{trocr['mean_cer']:>8.4f} {trocr['mean_wer']:>8.4f} "
            f"{trocr['mean_latency_ms']:>12.1f}"
        )
    if donut:
        print(
            f"  {'Donut':<15} {donut['sample_count']:>8} "
            f"{'—':>8} {'—':>8} "
            f"{donut['mean_latency_ms']:>12.1f}  (conf={donut['mean_confidence']:.2f})"
        )
    print("  " + "-" * 55)
    if trocr:
        if trocr["mean_cer"] < 0.10:
            verdict = "✅ CER < 10% — Good for production"
        elif trocr["mean_cer"] < 0.20:
            verdict = "⚠️  CER 10–20% — Needs improvement"
        else:
            verdict = "❌ CER > 20% — More training required"
        print(f"  TrOCR verdict: {verdict}")
    print("=" * 62)

    if trocr and trocr.get("spot_check"):
        print("\n  TrOCR Spot-check (first 3 samples):")
        for i, sc in enumerate(trocr["spot_check"][:3]):
            print(f"\n  [{i+1}] REF  : {sc['reference']}")
            print(f"       PRED : {sc['prediction']}")
            print(f"       CER  : {sc['cer']:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  MedOCR — Model Evaluation")
    print("=" * 62)
    print(f"\n  MLflow: {MLFLOW_URI} -> Experiments -> {EXPERIMENT}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # ── Load test split ───────────────────────────────────────────────────────
    print(f"\n[*] Loading test split from {DATASET_PATH}...")
    if not DATASET_PATH.exists():
        print("[ERROR] Dataset not found. Run: python scripts/download_dataset.py")
        sys.exit(1)
    dataset      = load_from_disk(str(DATASET_PATH))
    test_dataset = dataset["test"]
    print(f"    Test samples: {len(test_dataset)}")

    # TrOCR is a prescription model — evaluate it on the prescription test subset
    # (the same heuristic used to build its training set), not the mixed test set.
    presc_test = test_dataset.filter(lambda r: is_prescription(r["text"]))
    print(f"    Prescription test subset (for TrOCR): {len(presc_test)}")

    # ── Run evaluations ───────────────────────────────────────────────────────
    trocr_results = evaluate_trocr(presc_test, device)
    donut_results = evaluate_donut(test_dataset, device)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(trocr_results, donut_results)

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "dataset":         "naazimsnh02/medocr-vision-dataset",
        "test_split_size": len(test_dataset),
        "device":          str(device),
        "models":          {},
    }
    if trocr_results:
        report["models"]["trocr"] = trocr_results
    if donut_results:
        report["models"]["donut"] = donut_results

    report_path = RESULTS_DIR / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[*] Report saved -> {report_path}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    print("\n[*] Logging to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="medocr-evaluation"):
        mlflow.log_params({
            "trocr_model_path": str(TROCR_MODEL_DIR),
            "donut_model_path": str(DONUT_MODEL_DIR),
            "test_split_size":  len(test_dataset),
            "device":           str(device),
        })
        mlflow.set_tags({
            "phase":   "evaluation",
            "dataset": "medocr-vision-dataset",
        })

        if trocr_results:
            mlflow.log_metrics({
                "trocr/mean_cer":        trocr_results["mean_cer"],
                "trocr/mean_wer":        trocr_results["mean_wer"],
                "trocr/mean_latency_ms": trocr_results["mean_latency_ms"],
                "trocr/min_cer":         trocr_results["min_cer"],
                "trocr/max_cer":         trocr_results["max_cer"],
                "trocr/sample_count":    trocr_results["sample_count"],
            })
        if donut_results:
            mlflow.log_metrics({
                "donut/mean_latency_ms":  donut_results["mean_latency_ms"],
                "donut/mean_confidence":  donut_results["mean_confidence"],
                "donut/mean_fields_found": donut_results["mean_fields_found"],
                "donut/sample_count":     donut_results["sample_count"],
            })

        # Upload the full JSON report as a clickable artifact in MLflow UI
        mlflow.log_artifact(str(report_path))

    print("\n[SUCCESS] Evaluation complete!")
    print(f"   Report  : {report_path}")
    print(f"   MLflow  : {MLFLOW_URI} -> Experiments -> {EXPERIMENT}")


if __name__ == "__main__":
    main()
