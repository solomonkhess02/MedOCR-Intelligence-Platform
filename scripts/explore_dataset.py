"""
MedOCR Intelligence Platform
Dataset Explorer

Prints a full statistical summary of the MedOCR-Vision dataset,
generates two visualizations (sample image grid + text length histogram),
and logs all stats + images to MLflow experiment: Dataset-Stats.

Visualizations saved to:
    results/dataset_samples.png
    results/text_length_distribution.png

Usage:
    python scripts/explore_dataset.py
"""

import os
import sys
import random
import logging
from pathlib import Path

import mlflow
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — works in headless / server environments
import matplotlib.pyplot as plt
from datasets import load_from_disk

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

DATASET_PATH = PROJECT_ROOT / "data" / "medocr-vision-dataset"
RESULTS_DIR  = PROJECT_ROOT / "results"
MLFLOW_URI   = "http://127.0.0.1:5000"
EXPERIMENT   = "Dataset-Stats"

# Known composition from the dataset card (used for the printed breakdown table)
COMPOSITION = {
    "Medical Prescriptions": {"samples": 1000, "domain": "Medical",  "text_range": "200–400 chars"},
    "OMR Scanned Documents": {"samples": 36,   "domain": "Medical",  "text_range": "400–1,000 chars"},
    "Medical Lab Reports":   {"samples": 426,  "domain": "Medical",  "text_range": "1,000–5,000 chars"},
    "Invoices & Receipts":   {"samples": 1000, "domain": "General",  "text_range": "200–400 chars"},
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_text_stats(split) -> dict:
    """Compute character-level text length statistics for a dataset split."""
    lengths = [len(sample["text"]) for sample in split]
    return {
        "count":  len(lengths),
        "min":    int(np.min(lengths)),
        "max":    int(np.max(lengths)),
        "mean":   round(float(np.mean(lengths)), 1),
        "median": round(float(np.median(lengths)), 1),
        "std":    round(float(np.std(lengths)), 1),
        "p25":    round(float(np.percentile(lengths, 25)), 1),
        "p75":    round(float(np.percentile(lengths, 75)), 1),
    }


def save_sample_grid(train_split, output_path: Path) -> None:
    """
    Save a 3 × 4 grid of random training samples with image + text preview.
    """
    n       = min(12, len(train_split))
    indices = random.sample(range(len(train_split)), n)

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        "MedOCR-Vision Dataset — Random Training Samples",
        fontsize=15, fontweight="bold", color="white", y=1.01,
    )

    for i, idx in enumerate(indices):
        ax     = fig.add_subplot(3, 4, i + 1)
        sample = train_split[idx]
        image  = sample["image"].convert("RGB")
        text   = sample["text"][:55].replace("\n", " ")

        ax.imshow(image)
        ax.set_title(
            f"#{idx}  {text}…",
            fontsize=6.5, color="#e0e0e0", pad=3, wrap=True,
        )
        ax.axis("off")
        # Subtle border
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
            spine.set_linewidth(0.8)

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved: {output_path}")


def save_text_length_histogram(train_split, output_path: Path) -> None:
    """
    Save a styled histogram of character-level text lengths in the train split.
    Shaded regions indicate approximate ranges per document type.
    """
    lengths = [len(sample["text"]) for sample in train_split]
    mean_v  = np.mean(lengths)
    med_v   = np.median(lengths)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#12122a")

    ax.hist(lengths, bins=60, color="#7c5cbf", edgecolor="#2a2a4a", linewidth=0.4, alpha=0.9)

    # Document type range bands
    ax.axvspan(200,  400,  alpha=0.12, color="#2ecc71",
               label="Prescriptions / Invoices (200–400)")
    ax.axvspan(400,  1000, alpha=0.12, color="#3498db",
               label="OMR Docs (400–1,000)")
    ax.axvspan(1000, max(lengths) + 100, alpha=0.12, color="#9b59b6",
               label="Lab Reports (1,000+)")

    # Mean + median lines
    ax.axvline(mean_v, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_v:.0f}")
    ax.axvline(med_v,  color="#f39c12", linestyle="--", linewidth=1.5,
               label=f"Median: {med_v:.0f}")

    ax.set_xlabel("Text Length (characters)", fontsize=11, color="#cccccc")
    ax.set_ylabel("Sample Count",            fontsize=11, color="#cccccc")
    ax.set_title(
        "MedOCR-Vision — Text Length Distribution (Train Split)",
        fontsize=13, color="white", pad=10,
    )
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    legend = ax.legend(fontsize=8.5, facecolor="#1a1a2e", edgecolor="#444466",
                       labelcolor="#cccccc")
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  MedOCR — Dataset Explorer")
    print("=" * 62)
    print(f"\n  MLflow: {MLFLOW_URI} -> Experiments -> {EXPERIMENT}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"[*] Loading dataset from {DATASET_PATH}...")
    if not DATASET_PATH.exists():
        print("[ERROR] Dataset not found. Run: python scripts/download_dataset.py")
        sys.exit(1)
    dataset    = load_from_disk(str(DATASET_PATH))
    train_data = dataset["train"]
    val_data   = dataset["validation"]
    test_data  = dataset["test"]
    total      = len(train_data) + len(val_data) + len(test_data)

    # ── Split summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  SPLIT SUMMARY")
    print("=" * 50)
    print(f"  Total      : {total:,}")
    print(f"  Train      : {len(train_data):,}  ({100*len(train_data)/total:.1f}%)")
    print(f"  Validation : {len(val_data):,}  ({100*len(val_data)/total:.1f}%)")
    print(f"  Test       : {len(test_data):,}  ({100*len(test_data)/total:.1f}%)")
    print(f"  Columns    : {list(train_data.features.keys())}")

    # ── Composition breakdown ─────────────────────────────────────────────────
    print("\n  COMPOSITION (from dataset card)")
    print(f"  {'Source':<30} {'Samples':>8}  {'Domain':<10}  {'Text Range'}")
    print("  " + "-" * 65)
    for name, info in COMPOSITION.items():
        print(f"  {name:<30} {info['samples']:>8}  {info['domain']:<10}  {info['text_range']}")
    print("  " + "-" * 65)
    print(f"  {'Medical total':<30} {1462:>8}  59.4%")
    print(f"  {'General total':<30} {1000:>8}  40.6%")

    # ── Text length statistics ────────────────────────────────────────────────
    print("\n[*] Computing text length statistics (train split)...")
    stats = compute_text_stats(train_data)
    print(f"  Min    : {stats['min']:,} chars")
    print(f"  Max    : {stats['max']:,} chars")
    print(f"  Mean   : {stats['mean']:,} chars")
    print(f"  Median : {stats['median']:,} chars")
    print(f"  Std    : {stats['std']:,} chars")
    print(f"  P25    : {stats['p25']:,} chars")
    print(f"  P75    : {stats['p75']:,} chars")

    # ── Random sample previews ────────────────────────────────────────────────
    print("\n[*] Random sample previews (train split):")
    random.seed(42)
    preview_indices = random.sample(range(len(train_data)), min(3, len(train_data)))
    for idx in preview_indices:
        sample       = train_data[idx]
        text_preview = sample["text"][:200].replace("\n", " ")
        img_size     = sample["image"].size
        print(f"\n  Sample [{idx}]")
        print(f"    Image size   : {img_size[0]} × {img_size[1]} px")
        print(f"    Text length  : {len(sample['text'])} chars")
        print(f"    Text preview : {text_preview}...")

    # ── Generate visualizations ───────────────────────────────────────────────
    samples_path   = RESULTS_DIR / "dataset_samples.png"
    histogram_path = RESULTS_DIR / "text_length_distribution.png"

    print("\n[*] Generating sample image grid (3 × 4)...")
    save_sample_grid(train_data, samples_path)

    print("\n[*] Generating text length histogram...")
    save_text_length_histogram(train_data, histogram_path)

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    print("\n[*] Logging to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="medocr-vision-dataset-exploration"):
        mlflow.log_params({
            "total_samples":  total,
            "train_size":     len(train_data),
            "val_size":       len(val_data),
            "test_size":      len(test_data),
            "dataset_id":     "naazimsnh02/medocr-vision-dataset",
            "n_columns":      len(train_data.features),
        })
        mlflow.log_metrics({
            "medical_samples":           1462,
            "general_samples":           1000,
            "prescription_samples":      1000,
            "lab_report_samples":        426,
            "omr_samples":               36,
            "invoice_samples":           1000,
            "mean_text_length_train":    stats["mean"],
            "median_text_length_train":  stats["median"],
            "max_text_length_train":     stats["max"],
            "min_text_length_train":     stats["min"],
            "std_text_length_train":     stats["std"],
            "p25_text_length_train":     stats["p25"],
            "p75_text_length_train":     stats["p75"],
        })
        mlflow.set_tags({
            "phase":          "exploration",
            "dataset":        "medocr-vision-dataset",
            "domain_balance": "59.4_medical_40.6_general",
        })
        # Upload both PNG plots so they're viewable directly in the MLflow UI
        mlflow.log_artifact(str(samples_path))
        mlflow.log_artifact(str(histogram_path))

    print("\n" + "=" * 62)
    print("  [DONE] Dataset exploration complete!")
    print(f"  Sample grid  : {samples_path}")
    print(f"  Histogram    : {histogram_path}")
    print(f"  MLflow run   : {MLFLOW_URI} -> Experiments -> {EXPERIMENT}")
    print("=" * 62)


if __name__ == "__main__":
    main()
