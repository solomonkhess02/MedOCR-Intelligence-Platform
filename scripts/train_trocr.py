"""
MedOCR Intelligence Platform
TrOCR Fine-Tuning Script

Fine-tunes microsoft/trocr-base-handwritten on the MedOCR-Vision dataset.
Logs all metrics, hyperparameters, and the final model artifact to MLflow.

Model assignment (per architecture doc §6):
  TrOCR -> Handwritten prescriptions only.
  The full dataset contains prescriptions + invoices + lab reports + OMR forms.
  Since the dataset has no explicit document-type label, we note that TrOCR
  benefits most from short handwritten text (200–600 chars). In Phase 2,
  a document_type column will allow precise filtering.

Usage:
    # Fast test run — 100 train / 4 val samples, 1 epoch (~5 min on GPU)
    python scripts/train_trocr.py

    # Full training — all 1,969 train samples, 5 epochs
    python scripts/train_trocr.py --full
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

import evaluate
import mlflow
import torch
from datasets import load_from_disk
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    default_data_collator,
)

# Shared prescription filter (same definition used by evaluate_models.py).
from doc_filters import is_prescription

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

DATASET_PATH  = PROJECT_ROOT / "data" / "medocr-vision-dataset"
MODEL_NAME    = "microsoft/trocr-base-handwritten"
OUTPUT_DIR    = PROJECT_ROOT / "models" / "trocr-finetuned"
MLFLOW_URI    = "http://127.0.0.1:5000"
EXPERIMENT    = "TrOCR-FineTuning"


# ── MLflow Callback ────────────────────────────────────────────────────────────
class MLflowTagsCallback(TrainerCallback):
    """
    Logs custom params and tags to the MLflow run that the Trainer creates
    automatically when trainer.train() begins.
    Called on_train_begin so the active run is guaranteed to exist.
    """
    def __init__(self, extra_params: dict, extra_tags: dict):
        self.extra_params = extra_params
        self.extra_tags   = extra_tags

    def on_train_begin(self, args, state, control, **kwargs):
        if mlflow.active_run():
            mlflow.log_params(self.extra_params)
            mlflow.set_tags(self.extra_tags)


# ── Dataset preprocessing ──────────────────────────────────────────────────────
def prepare_dataset(batch, processor):
    """
    Converts a batch of raw dataset rows into the tensors TrOCR needs.

    TrOCR is a VisionEncoderDecoder model:
      - The ENCODER is a Vision Transformer (ViT) -> needs pixel_values
      - The DECODER is a RoBERTa language model  -> needs token IDs as labels

    IMPORTANT: HuggingFace `datasets.map()` cannot serialise raw PyTorch
    tensors. We must convert pixel_values to a plain Python list so the
    Arrow-backed dataset can store and cache it correctly.
    """
    # 1. Process images: ViT feature extractor resizes + normalises the images.
    #    Result shape: [batch_size, 3, 384, 384]
    pixel_values = processor(
        images=batch["image"], return_tensors="pt"
    ).pixel_values

    # Convert tensor -> list so Arrow/datasets can serialise it.
    batch["pixel_values"] = pixel_values.tolist()

    # 2. Tokenise the ground-truth text labels using the RoBERTa tokenizer.
    labels = processor.tokenizer(
        batch["text"],
        padding="max_length",
        max_length=128,
        truncation=True,
    ).input_ids

    # 3. Replace padding token IDs with -100.
    #    PyTorch's CrossEntropyLoss skips positions labelled -100,
    #    so the model is not penalised for predicting padding tokens.
    batch["labels"] = [
        [tok if tok != processor.tokenizer.pad_token_id else -100
         for tok in seq]
        for seq in labels
    ]

    return batch


# ── Metrics ────────────────────────────────────────────────────────────────────
def make_compute_metrics(processor, cer_metric, wer_metric):
    """
    Returns a closure that captures the processor and metric objects.
    The Trainer calls compute_metrics(pred) where pred is an
    EvalPrediction namedtuple with fields:
      - pred.predictions : raw logits, shape [N, seq_len, vocab_size]
      - pred.label_ids   : token IDs, shape [N, seq_len]
    """
    def compute_metrics(pred):
        # With predict_with_generate=True, pred.predictions are ALREADY generated
        # token IDs (shape [N, gen_len]) — NOT logits. Do not argmax them; that was
        # the bug that produced CER≈0.99 / WER=1.0 while training loss was healthy.
        pred_ids  = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = pred.label_ids

        # Restore -100 padding to the real pad token so labels decode cleanly.
        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)

        # Decode token ID sequences -> human-readable strings.
        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer, "wer": wer}

    return compute_metrics


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on MedOCR-Vision dataset")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full training run (5 epochs). "
             "Default: fast test run (100 samples, 1 epoch).",
    )
    parser.add_argument(
        "--prescriptions-only",
        action="store_true",
        help="Filter train/val to prescription samples only (the 1,000-prescription "
             "subset TrOCR is intended for). Recommended for the real model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in the output dir (keeps prior training "
             "progress; applies the current/fixed eval code from here on).",
    )
    args = parser.parse_args()
    mode = "full" if args.full else "fast"

    print("=" * 60)
    print("  MedOCR — TrOCR Fine-Tuning")
    print(f"  Mode : {'FULL (production)' if args.full else 'FAST (100-sample test)'}")
    print("=" * 60)
    print(f"\n  MLflow: {MLFLOW_URI} -> Experiments -> {EXPERIMENT}")
    print()

    # ── 1. Device check ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}")
    if device == "cuda":
        print(f"    GPU Name  : {torch.cuda.get_device_name(0)}")
        print(f"    VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif args.full:
        print("    WARNING: Full training on CPU will take several hours.")
        print("             Consider running on a machine with a GPU.")

    # ── 2. Load evaluation metrics ────────────────────────────────────────────
    print("\n[*] Loading evaluation metrics (CER, WER)...")
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    # ── 3. Load dataset ───────────────────────────────────────────────────────
    print(f"\n[*] Loading dataset from {DATASET_PATH}...")
    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset not found at {DATASET_PATH}")
        print("        Run: python scripts/download_dataset.py")
        sys.exit(1)

    dataset = load_from_disk(str(DATASET_PATH))
    print(f"    Train size : {len(dataset['train'])}")
    print(f"    Val size   : {len(dataset['validation'])}")
    print(f"    Test size  : {len(dataset['test'])}")

    # ── 4. Load processor & model ─────────────────────────────────────────────
    print(f"\n[*] Loading base model: {MODEL_NAME}")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model     = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # Configure decoder generation settings.
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    model.config.max_length             = 128
    model.config.no_repeat_ngram_size   = 3
    model.config.length_penalty         = 2.0
    model.config.num_beams              = 4

    # Freeze the ViT vision encoder.
    # Rationale: The encoder already knows how to "see" handwriting from its
    # ImageNet + IAM pretraining. We only need to fine-tune the RoBERTa
    # decoder to learn medical vocabulary and spelling.
    # This halves memory usage and speeds up training ~2×.
    print("[*] Freezing vision encoder (training decoder only)...")
    for param in model.encoder.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"    Trainable params : {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ── 5. Select samples ─────────────────────────────────────────────────────
    train_split = dataset["train"]
    val_split   = dataset["validation"]

    # Optionally filter to the prescription subset (TrOCR's actual domain).
    # The dataset has no document_type column, so we use the shared heuristic in
    # doc_filters.is_prescription, which recovers exactly the 1,000 documented
    # prescriptions (821 train + 84 val + 95 test).
    if args.prescriptions_only:
        before_tr, before_val = len(train_split), len(val_split)
        train_split = train_split.filter(lambda r: is_prescription(r["text"]))
        val_split   = val_split.filter(lambda r: is_prescription(r["text"]))
        print(f"\n[*] Prescription filter: train {before_tr}->{len(train_split)}, "
              f"val {before_val}->{len(val_split)}")

    if args.full:
        print(f"\n[*] Full mode: using {len(train_split)} train / {len(val_split)} val samples, 5 epochs.")
        train_dataset = train_split
        val_dataset   = val_split
        num_epochs    = 5
        eval_steps    = 100
        save_steps    = 100
    else:
        print("\n[*] Fast mode: using 100-sample train / 4-sample val subset.")
        print("    To use the full dataset: python scripts/train_trocr.py --full --prescriptions-only")
        train_dataset = train_split.select(range(min(100, len(train_split))))
        val_dataset   = val_split.select(range(min(4, len(val_split))))
        num_epochs    = 1
        eval_steps    = 10
        save_steps    = 10

    # ── 6. Preprocess dataset ─────────────────────────────────────────────────
    print("\n[*] Preprocessing dataset...")
    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        batched=True,
        batch_size=8,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing train set",
    )
    val_dataset = val_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        batched=True,
        batch_size=8,
        remove_columns=dataset["validation"].column_names,
        desc="Preprocessing val set",
    )

    # ── 7. Training arguments ─────────────────────────────────────────────────
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT
    os.environ["MLFLOW_TRACKING_URI"]    = MLFLOW_URI

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),

        # --- Generation (eval) ---
        predict_with_generate=True,     # generate() during eval so CER/WER are real
        generation_num_beams=1,         # greedy eval decode — ~4x faster than beams=4;
                                        # beam search stays available for production inference

        # --- Evaluation & Saving ---
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,             # Keep only 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="cer",    # Lower CER = better model
        greater_is_better=False,

        # --- Logging ---
        logging_steps=5,
        report_to=["mlflow"],           # HuggingFace Trainer auto-creates MLflow run

        # --- Optimisation ---
        learning_rate=4e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 2 × 4 = 8
        num_train_epochs=num_epochs,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,       # 0 = main process (required on Windows)
    )

    # ── 8. Build Trainer ──────────────────────────────────────────────────────
    # Extra params/tags logged via callback so they land in the same MLflow run
    # that the Trainer creates automatically on trainer.train().
    mlflow_callback = MLflowTagsCallback(
        extra_params={
            "model_name":      MODEL_NAME,
            "dataset_size":    len(train_dataset),
            "mode":            mode,
            "frozen_encoder":  True,
            "max_label_len":   128,
            "num_beams":       4,
        },
        extra_tags={
            "phase":           "1.5",
            "document_type":   "prescription",
            "dataset":         "medocr-vision-dataset",
            "architecture":    "trocr",
        },
    )

    trainer = Seq2SeqTrainer(
        model=model,
        # NOTE: Pass `processor` (not `processor.feature_extractor`) as the
        # tokenizer. The Trainer uses this for padding inside the data collator.
        tokenizer=processor,
        args=training_args,
        compute_metrics=make_compute_metrics(processor, cer_metric, wer_metric),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        callbacks=[mlflow_callback],
    )

    # ── 9. Train ──────────────────────────────────────────────────────────────
    print("\n[*] Starting training...")
    print(f"    Watch {MLFLOW_URI} -> Experiments -> {EXPERIMENT}")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── 10. Save final model ──────────────────────────────────────────────────
    print(f"\n[*] Saving fine-tuned model to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    # Log the saved model directory as an MLflow artifact
    if mlflow.active_run():
        mlflow.log_artifact(str(OUTPUT_DIR / "config.json"))

    print("\n[SUCCESS] Fine-tuning complete!")
    print(f"   Model saved to : {OUTPUT_DIR}")
    print(f"   MLflow run at  : {MLFLOW_URI}")


if __name__ == "__main__":
    main()
