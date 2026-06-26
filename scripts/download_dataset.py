"""
MedOCR Intelligence Platform
Dataset Download Script

Downloads naazimsnh02/medocr-vision-dataset from HuggingFace (~509 MB)
and saves all three splits to data/medocr-vision-dataset/ as Arrow files.

Usage:
    python scripts/download_dataset.py
"""

import os
import sys
import shutil
from pathlib import Path

# ── Ensure we run from project root ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

OUTPUT_DIR = PROJECT_ROOT / "data" / "medocr-vision-dataset"

# WINDOWS WORKAROUND: The HuggingFace `datasets` library uses posixpath.join()
# internally when building Arrow file paths. On Windows this produces forward-
# slash paths that fail with [Errno 22] when the path contains spaces.
# Fix: save to a short space-free temp path, then move to the real destination.
TMP_DIR = Path("C:/medocr_tmp")


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' package not found.")
        print("        Run: pip install datasets")
        sys.exit(1)

    print("=" * 60)
    print("  MedOCR Intelligence Platform — Dataset Download")
    print("=" * 60)
    print(f"  Source  : naazimsnh02/medocr-vision-dataset")
    print(f"  Target  : {OUTPUT_DIR}")
    print(f"  Size    : ~509 MB (2,462 samples)")
    print("=" * 60)
    print()

    # ── Download all splits ───────────────────────────────────────────────────
    print("[1/3] Downloading dataset from HuggingFace...")
    dataset = load_dataset("naazimsnh02/medocr-vision-dataset")

    train_data = dataset["train"]
    val_data   = dataset["validation"]
    test_data  = dataset["test"]

    # ── Print dataset structure ───────────────────────────────────────────────
    print()
    print("[2/3] Dataset structure:")
    print(dataset)
    print()

    print("  Split breakdown:")
    print(f"    Train      : {len(train_data):,} samples")
    print(f"    Validation : {len(val_data):,}   samples")
    print(f"    Test       : {len(test_data):,}   samples")
    print(f"    Total      : {len(train_data) + len(val_data) + len(test_data):,} samples")
    print()

    # ── Inspect columns ───────────────────────────────────────────────────────
    print("  Columns:", list(train_data.features.keys()))
    print()

    # ── Save via space-free temp path ─────────────────────────────────────────
    print(f"[3/3] Saving to {TMP_DIR} (temp, no spaces) ...")
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(TMP_DIR))

    # Move completed dataset from temp → real output directory
    print(f"      Moving to {OUTPUT_DIR} ...")
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    shutil.move(str(TMP_DIR), str(OUTPUT_DIR))

    print()
    print("=" * 60)
    print("  [DONE] Download complete!")
    print(f"  Saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
