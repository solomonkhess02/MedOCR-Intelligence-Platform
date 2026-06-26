"""
MedOCR Intelligence Platform
Donut Fine-Tuning Script

Fine-tunes naver-clova-ix/donut-base on the MedOCR-Vision dataset,
targeting the invoice/receipt subset.

Model assignment (per architecture doc §6):
  Donut -> Medical invoices and receipts.
  Donut is an end-to-end document understanding model that outputs structured
  JSON without a separate OCR step.

Dataset note:
  The medocr-vision-dataset has no 'document_type' label — only 'image' + 'text'.
  Invoice/receipt samples are short-to-medium structured text (200–600 chars).
  Phase 2 will add a document_type column for precise filtering.

Text format note:
  Donut training expects pseudo-XML output (e.g. <s_invoice_no>INV-001</s_invoice_no>).
  Since our dataset has plain text annotations, we wrap the full text in a generic
  <s_text> tag. Phase 2 with field-level medical invoice annotations will enable
  richer structured output (amounts, dates, drug names, etc.).

Logs all metrics and the model artifact to MLflow experiment: Donut-FineTuning.

Usage:
    # Fast test run — 50 train / 4 val samples, 1 epoch (~10 min on GPU)
    python scripts/train_donut.py

    # Full training — all ~1,969 samples, 3 epochs
    python scripts/train_donut.py --full
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import mlflow
import torch
from datasets import load_from_disk
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

# Shared invoice filter (same definition used across the project).
from doc_filters import is_invoice

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

DATASET_PATH = PROJECT_ROOT / "data" / "medocr-vision-dataset"
MODEL_NAME   = "naver-clova-ix/donut-base"
OUTPUT_DIR   = PROJECT_ROOT / "models" / "donut-finetuned"
MLFLOW_URI   = "http://127.0.0.1:5000"
EXPERIMENT   = "Donut-FineTuning"

# Donut input image size: [width, height].
# Full Donut uses 2560×1920; 960×1280 fits ~8GB VRAM. For a 4GB GPU (e.g. GTX 1650)
# we default to 768×1024 with a frozen encoder + gradient checkpointing (see below).
# Override on the CLI with --image-size WxH. If you still hit CUDA OOM, drop to 640×864.
IMAGE_SIZE   = [768, 1024]   # [width, height]
TASK_PROMPT  = "<s_cord-v2>"  # cord-v2 = receipt/structured-doc schema
MAX_LENGTH   = 512

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── MLflow Callback ────────────────────────────────────────────────────────────
class MLflowTagsCallback(TrainerCallback):
    """
    Logs custom params and tags into the MLflow run that the HuggingFace Trainer
    creates automatically when trainer.train() begins.
    """
    def __init__(self, extra_params: dict, extra_tags: dict):
        self.extra_params = extra_params
        self.extra_tags   = extra_tags

    def on_train_begin(self, args, state, control, **kwargs):
        if mlflow.active_run():
            mlflow.log_params(self.extra_params)
            mlflow.set_tags(self.extra_tags)


# ── Text -> Donut XML ──────────────────────────────────────────────────────────
def text_to_donut_xml(text: str) -> str:
    """
    Wrap plain text in Donut's pseudo-XML output format.

    Current implementation (Phase 1): generic <s_text> wrapper.
    Phase 2 target: field-level annotation:
      <s_invoice_no>INV-001</s_invoice_no>
      <s_total>1250.00</s_total>
      <s_date>2024-01-15</s_date>
    """
    # Sanitise XML-reserved characters so the tokenizer doesn't choke
    text = text.replace("&", "&amp;").replace("<", "").replace(">", "").strip()
    return f"{TASK_PROMPT}<s_text>{text}</s_text>"


# ── Dataset preprocessing ──────────────────────────────────────────────────────
def prepare_dataset(batch, processor):
    """
    Preprocess a batch for Donut training.

    Donut is a VisionEncoderDecoder model (same family as TrOCR):
      - ENCODER: Swin Transformer vision backbone -> reads pixel patches
      - DECODER: BART-style autoregressive decoder -> generates XML tokens

    Output tensors needed by the Trainer:
      pixel_values : (B, C, H, W)  normalised image tensor
      labels       : (B, seq_len)  XML token IDs, -100 on padding positions
    """
    # 1. Resize + normalise images to Donut's expected input
    images = [img.convert("RGB") for img in batch["image"]]
    pixel_values = processor(
        images, return_tensors="pt"
    ).pixel_values                        # shape: (B, 3, H, W)
    batch["pixel_values"] = pixel_values.tolist()

    # 2. Convert plain text -> pseudo-XML, then tokenise
    xml_texts = [text_to_donut_xml(t) for t in batch["text"]]
    encoding  = processor.tokenizer(
        xml_texts,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
    )

    # 3. Mask padding positions with -100 so loss ignores them
    batch["labels"] = [
        [tok if tok != processor.tokenizer.pad_token_id else -100
         for tok in seq]
        for seq in encoding.input_ids
    ]

    return batch


# ── Custom collator ────────────────────────────────────────────────────────────
def donut_collate_fn(batch):
    """
    Convert the list-of-lists stored by Arrow into proper PyTorch tensors.
    The Trainer's default_data_collator handles lists correctly for most cases,
    but Donut's large image tensors benefit from explicit stacking.
    """
    pixel_values = torch.tensor([item["pixel_values"] for item in batch],
                                 dtype=torch.float32)
    labels       = torch.tensor([item["labels"] for item in batch],
                                 dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Donut on MedOCR-Vision dataset")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full training (invoice subset, 3 epochs). Default: fast test (50 samples, 1 epoch).",
    )
    parser.add_argument(
        "--invoices-only",
        action="store_true",
        help="Filter train/val to invoice/receipt samples only (Donut's domain). Recommended.",
    )
    parser.add_argument(
        "--image-size",
        type=str,
        default=None,
        help="Override Donut input size as WxH, e.g. 640x864 (lower it if you hit CUDA OOM).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-device batch size. Default 1 (safe for 4GB GPUs).",
    )
    parser.add_argument(
        "--no-freeze-encoder",
        action="store_true",
        help="Train the Swin encoder too (needs more VRAM). Default: encoder frozen.",
    )
    args = parser.parse_args()
    mode = "full" if args.full else "fast"

    # Resolve memory-sensitive knobs (tuned for a 4GB GPU by default).
    if args.image_size:
        w, h = (int(x) for x in args.image_size.lower().split("x"))
        IMAGE_SIZE[0], IMAGE_SIZE[1] = w, h
    freeze_encoder = not args.no_freeze_encoder

    print("=" * 60)
    print("  MedOCR — Donut Fine-Tuning")
    print(f"  Mode : {'FULL (production)' if args.full else 'FAST (50-sample test)'}")
    print("=" * 60)
    print(f"\n  MLflow: {MLFLOW_URI} -> Experiments -> {EXPERIMENT}")
    print()

    # ── 1. Device check ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device}")
    if device == "cuda":
        print(f"    GPU  : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    VRAM : {vram_gb:.1f} GB")
        if vram_gb < 8 and args.full:
            print("    WARNING: < 8 GB VRAM detected. Full Donut training may OOM.")
            print("             Consider reducing IMAGE_SIZE or batch_size.")
    elif args.full:
        print("    WARNING: Full Donut training on CPU will be extremely slow.")

    # ── 2. Load dataset ───────────────────────────────────────────────────────
    print(f"\n[*] Loading dataset from {DATASET_PATH}...")
    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset not found. Run: python scripts/download_dataset.py")
        sys.exit(1)

    dataset = load_from_disk(str(DATASET_PATH))
    print(f"    Train : {len(dataset['train'])} | Val : {len(dataset['validation'])} | Test : {len(dataset['test'])}")

    # ── 3. Select samples ─────────────────────────────────────────────────────
    train_split = dataset["train"]
    val_split   = dataset["validation"]

    # Filter to the invoice/receipt subset (Donut's domain). The dataset has no
    # document_type column, so we use the shared doc_filters.is_invoice heuristic.
    if args.invoices_only:
        before_tr, before_val = len(train_split), len(val_split)
        train_split = train_split.filter(lambda r: is_invoice(r["text"]))
        val_split   = val_split.filter(lambda r: is_invoice(r["text"]))
        print(f"\n[*] Invoice filter: train {before_tr}->{len(train_split)}, "
              f"val {before_val}->{len(val_split)}")

    batch_size = args.batch_size or 1   # default 1 — safe for 4GB GPUs

    if args.full:
        train_dataset = train_split
        val_dataset   = val_split
        num_epochs    = 3
        eval_steps    = 100
        save_steps    = 100
    else:
        print("\n    Note: Using up to 50-sample train / 4-sample val for fast test.")
        print("    For the real run: python scripts/train_donut.py --full --invoices-only")
        train_dataset = train_split.select(range(min(50, len(train_split))))
        val_dataset   = val_split.select(range(min(4, len(val_split))))
        num_epochs    = 1
        eval_steps    = 10
        save_steps    = 10

    # ── 4. Load processor & model ─────────────────────────────────────────────
    print(f"\n[*] Loading Donut base model: {MODEL_NAME}")
    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model     = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # Set expected image dimensions on the processor's image processor.
    # Supports both older (feature_extractor) and newer (image_processor) APIs.
    img_proc = getattr(processor, "image_processor", None) or processor.feature_extractor
    img_proc.size = {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1]}
    img_proc.do_align_long_axis = False

    # Add the task prompt as a special token if not already present
    special_tokens = processor.tokenizer.all_special_tokens
    if TASK_PROMPT not in special_tokens:
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [TASK_PROMPT]})
        model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Configure decoder start / end tokens
    task_token_id = processor.tokenizer.convert_tokens_to_ids([TASK_PROMPT])[0]
    model.config.decoder_start_token_id = task_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.eos_token_id
    model.config.max_length             = MAX_LENGTH

    # ── Memory fit for small GPUs (default tuning for ~4GB VRAM) ───────────────
    # 1) Freeze the Swin vision encoder and train only the BART decoder. The encoder
    #    already "reads" documents well from SynthDoG pretraining; adapting our
    #    generic <s_text> output is mostly a decoder job. Freezing removes the
    #    encoder's gradients + optimizer state from memory.
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("[*] Froze Swin encoder (training BART decoder only).")

    # 2) Gradient checkpointing: recompute activations during the backward pass
    #    instead of storing them — the single biggest activation-memory saver.
    model.config.use_cache = False                 # required by gradient checkpointing
    model.gradient_checkpointing_enable()
    if freeze_encoder:
        # With a frozen encoder + checkpointing, the input embeddings must require
        # grad so the decoder's gradients can flow back through the checkpointed graph.
        model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"    Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ── 5. Preprocess dataset ─────────────────────────────────────────────────
    print("\n[*] Preprocessing dataset (image resize + XML tokenisation)...")
    train_processed = train_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        batched=True,
        batch_size=4,
        remove_columns=train_dataset.column_names,
        desc="Preprocessing train",
    )
    val_processed = val_dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        batched=True,
        batch_size=4,
        remove_columns=val_dataset.column_names,
        desc="Preprocessing val",
    )

    # ── 6. Training arguments ─────────────────────────────────────────────────
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT
    os.environ["MLFLOW_TRACKING_URI"]    = MLFLOW_URI

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),

        # Donut training uses teacher-forcing (not full generate()) for speed.
        # predict_with_generate=True would run full beam search on each eval step,
        # which is prohibitively slow for a Swin-BART model.
        predict_with_generate=False,

        # --- Evaluation & Saving ---
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # --- Logging ---
        logging_steps=5,
        report_to=["mlflow"],

        # --- Optimisation ---
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,  # Effective batch = batch_size × 8
        num_train_epochs=num_epochs,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,       # Windows-safe
        remove_unused_columns=False,    # Keep pixel_values + labels
    )

    # ── 7. MLflow callback for custom params/tags ─────────────────────────────
    mlflow_callback = MLflowTagsCallback(
        extra_params={
            "model_name":    MODEL_NAME,
            "task_prompt":   TASK_PROMPT,
            "dataset_size":     len(train_dataset),
            "mode":             mode,
            "epochs":           num_epochs,
            "effective_batch":  batch_size * 8,   # per_device_batch × grad_accum
            "learning_rate":    5e-5,
            "max_length":       MAX_LENGTH,
            "image_size":       f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
            "frozen_encoder":   freeze_encoder,
        },
        extra_tags={
            "phase":         "2",
            "document_type": "invoice",
            "dataset":       "medocr-vision-dataset",
            "architecture":  "donut",
        },
    )

    # ── 8. Build Trainer ──────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_processed,
        eval_dataset=val_processed,
        data_collator=donut_collate_fn,
        callbacks=[mlflow_callback],
    )

    # ── 9. Train ──────────────────────────────────────────────────────────────
    print("\n[*] Starting training...")
    print(f"    Watch {MLFLOW_URI} -> Experiments -> {EXPERIMENT}")
    trainer.train()

    # ── 10. Save final model + processor ─────────────────────────────────────
    print(f"\n[*] Saving fine-tuned Donut model to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    if mlflow.active_run():
        mlflow.log_artifact(str(OUTPUT_DIR / "config.json"))

    print("\n[SUCCESS] Donut fine-tuning complete!")
    print(f"   Model saved to : {OUTPUT_DIR}")
    print(f"   MLflow run at  : {MLFLOW_URI}")


if __name__ == "__main__":
    main()
