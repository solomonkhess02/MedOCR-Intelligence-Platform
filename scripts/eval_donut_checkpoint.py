"""
Evaluate a fine-tuned Donut checkpoint on the prescription test subset.

Reports THREE things so the result is judged fairly:
  1. CER / WER  — raw character/word error vs the (sanitized) target text.
  2. Field-capture accuracy — for structured extraction the right question is
     "did it recover each field's VALUE?", not character-exactness. We parse the
     reference into `key: value` fields and check whether each value appears
     (fuzzily) in the prediction.
  3. A few REF/PRED samples for qualitative inspection.

Decoding uses beam search + no_repeat_ngram_size to suppress the greedy
degeneration (looping) that inflates CER.

Usage:
    .venv\\Scripts\\python.exe scripts\\eval_donut_checkpoint.py --n 95 --device cuda
"""

import sys
import re
import argparse
from difflib import SequenceMatcher
from pathlib import Path

import torch
import evaluate
from datasets import load_from_disk
from transformers import DonutProcessor, VisionEncoderDecoderModel

from doc_filters import is_prescription

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "medocr-vision-dataset"
DEFAULT_CKPT = PROJECT_ROOT / "models" / "donut-finetuned"
TASK_PROMPT = "<s_cord-v2>"

FIELD_KEYS = ["doctor_name", "clinic_name", "clinic_address",
              "patient_name", "patient_age", "date", "medications", "signature"]
_KEY_RE = re.compile(r"(" + "|".join(FIELD_KEYS) + r")\s*:\s*")


def sanitize_reference(text: str) -> str:
    """Match train_donut.text_to_donut_xml's inner sanitization (strip < and >)."""
    return text.replace("&", "&amp;").replace("<", "").replace(">", "").strip()


def clean_prediction(seq: str) -> str:
    """Remove Donut's structural wrapper artifacts (incl. bracket-less 'stext')."""
    for tok in (TASK_PROMPT, "</s_text>", "<s_text>", "</s>", "<s>", "s_text", "stext"):
        seq = seq.replace(tok, " ")
    return re.sub(r"\s+", " ", seq).strip()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse_fields(ref: str) -> dict:
    """Split a sanitized reference into {key: value} using the known field keys."""
    fields, matches = {}, list(_KEY_RE.finditer(ref))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(ref)
        fields[m.group(1)] = ref[start:end].strip()
    return fields


def value_captured(value: str, pred_norm: str, thresh: float = 0.8) -> bool:
    """True if the normalized field value is (mostly) present in the prediction."""
    v = _norm(value)
    if len(v) < 3:
        return v in pred_norm
    # longest contiguous matching block must cover >= thresh of the value
    match = SequenceMatcher(None, v, pred_norm).find_longest_match(0, len(v), 0, len(pred_norm))
    return (match.size / len(v)) >= thresh


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a Donut checkpoint (CER + field accuracy)")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--n", type=int, default=95)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=192)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not (ckpt / "config.json").exists():
        print(f"ERROR: no Donut model at {ckpt} (train it first).")
        return 1

    device = torch.device(args.device)
    print(f"[*] Loading Donut checkpoint {ckpt} on {device} ...")
    processor = DonutProcessor.from_pretrained(str(ckpt))
    model = VisionEncoderDecoderModel.from_pretrained(str(ckpt)).to(device).eval()

    ds = load_from_disk(str(DATASET_PATH))[args.split].filter(lambda r: is_prescription(r["text"]))
    n = min(args.n, len(ds))
    print(f"[*] Scoring {n}/{len(ds)} prescription {args.split} samples "
          f"(beams={args.num_beams}, no_repeat_ngram=3)\n")

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    task_ids = processor.tokenizer(TASK_PROMPT, add_special_tokens=False,
                                   return_tensors="pt").input_ids.to(device)

    preds, refs = [], []
    field_hits, field_total = 0, 0
    for i in range(n):
        sample = ds[i]
        pixel_values = processor(sample["image"].convert("RGB"),
                                 return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            out = model.generate(
                pixel_values,
                decoder_input_ids=task_ids,
                max_length=args.max_length,
                num_beams=args.num_beams,
                no_repeat_ngram_size=3,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )
        pred = clean_prediction(processor.batch_decode(out, skip_special_tokens=True)[0])
        ref = sanitize_reference(sample["text"])
        preds.append(pred)
        refs.append(ref)

        # field-capture accuracy
        pred_norm = _norm(pred)
        for key, val in parse_fields(ref).items():
            if not val:
                continue
            field_total += 1
            if value_captured(val, pred_norm):
                field_hits += 1

        if i < 5:
            print(f"[{i}] REF : {ref[:90]}")
            print(f"    PRED: {pred[:90]}")
        elif i % 20 == 0:
            print(f"    ...scored {i}/{n}")

    cer = cer_metric.compute(predictions=preds, references=refs)
    wer = wer_metric.compute(predictions=preds, references=refs)
    field_acc = field_hits / field_total if field_total else 0.0

    print("\n" + "=" * 56)
    print(f"  Donut checkpoint   : {ckpt.name}")
    print(f"  Samples            : {n} ({args.split} prescriptions)")
    print(f"  CER                : {cer:.4f}   ({cer*100:.1f}% chars wrong)")
    print(f"  WER                : {wer:.4f}")
    print(f"  FIELD-CAPTURE ACC  : {field_acc:.4f}   ({field_hits}/{field_total} field values recovered)")
    print("=" * 56)
    print("  TrOCR baseline (same 95): CER 0.656 / WER 0.919 (no field extraction).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
