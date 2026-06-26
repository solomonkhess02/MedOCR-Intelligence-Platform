"""
Evaluate a saved TrOCR checkpoint with the CORRECTED decoding logic.

Purpose: measure the REAL CER/WER of an already-trained checkpoint (e.g. the
checkpoints the in-progress run has saved), using proper generation + decoding —
not the buggy argmax-over-generated-ids path that produced CER≈0.99.

Runs on CPU by default so it does NOT compete with a GPU training job already
using the card. CPU generation is slower, so it evaluates a sample by default.

Usage:
    .venv\\Scripts\\python.exe scripts\\eval_checkpoint.py --checkpoint models/trocr-finetuned/checkpoint-200 --n 20
"""

import sys
import argparse
from pathlib import Path

import torch
import evaluate
from datasets import load_from_disk
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from doc_filters import is_prescription

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "medocr-vision-dataset"


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a TrOCR checkpoint (corrected decode)")
    ap.add_argument("--checkpoint", required=True, help="Path to a checkpoint dir or model dir")
    ap.add_argument("--n", type=int, default=20, help="Number of validation prescriptions to score")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--device", default="cpu", help="cpu (default, GPU-safe) or cuda")
    ap.add_argument("--num-beams", type=int, default=1, help="1 = greedy (fast)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not (ckpt / "config.json").exists():
        print(f"ERROR: no model at {ckpt}")
        return 1

    device = torch.device(args.device)
    print(f"[*] Loading checkpoint {ckpt} on {device} ...")
    processor = TrOCRProcessor.from_pretrained(str(ckpt))
    model = VisionEncoderDecoderModel.from_pretrained(str(ckpt)).to(device).eval()

    ds = load_from_disk(str(DATASET_PATH))[args.split]
    ds = ds.filter(lambda r: is_prescription(r["text"]))
    n = min(args.n, len(ds))
    print(f"[*] Scoring {n} / {len(ds)} prescription {args.split} samples (beams={args.num_beams})\n")

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    preds, refs = [], []

    for i in range(n):
        sample = ds[i]
        image = sample["image"].convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            gen_ids = model.generate(pixel_values, num_beams=args.num_beams, max_length=128)
        pred = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        preds.append(pred)
        refs.append(sample["text"])
        if i < 5:
            print(f"[{i}] REF : {refs[-1][:90]}")
            print(f"    PRED: {pred[:90]}")
        elif i % 5 == 0:
            print(f"    ...scored {i}/{n}")

    cer = cer_metric.compute(predictions=preds, references=refs)
    wer = wer_metric.compute(predictions=preds, references=refs)
    print("\n" + "=" * 50)
    print(f"  Checkpoint : {ckpt.name}")
    print(f"  Samples    : {n} ({args.split} prescriptions)")
    print(f"  REAL CER   : {cer:.4f}   ({cer*100:.1f}% characters wrong)")
    print(f"  REAL WER   : {wer:.4f}   ({wer*100:.1f}% words wrong)")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())