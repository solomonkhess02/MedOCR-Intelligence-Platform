"""
Data Drift Monitoring — Evidently AI

Monitors the OCR layer for data/prediction drift. It compares a REFERENCE window of
past OCR results against the CURRENT window and produces an Evidently drift report.

Features monitored (per ocr_results row):
  - confidence   : model confidence score (numerical)
  - latency_ms   : inference latency (numerical)
  - text_length  : characters of extracted raw_text (numerical)
  - doc_type     : document type distribution (categorical)

Data source:
  Reads from the `ocr_results` / `documents` tables via SYNC_DATABASE_URL. If the
  database is unreachable or has too few rows (e.g. a fresh demo machine), it falls
  back to a CLEARLY-LABELLED synthetic sample so the report still renders. The fallback
  injects mild drift so the demo shows a non-trivial result.

Usage:
    .venv\\Scripts\\python.exe scripts\\drift_monitor.py
    .venv\\Scripts\\python.exe scripts\\drift_monitor.py --out results/drift_report.html
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

NUMERICAL = ["confidence", "latency_ms", "text_length"]
CATEGORICAL = ["doc_type"]
MIN_ROWS_PER_WINDOW = 10


def _sync_db_url() -> str:
    return os.getenv(
        "SYNC_DATABASE_URL",
        "postgresql://medocr_user:medocr_pass@localhost:5432/medocr_db",
    )


def load_from_db() -> pd.DataFrame:
    """Pull OCR results joined with doc_type. Returns empty DataFrame on any failure."""
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(_sync_db_url())
        query = text(
            """
            SELECT o.confidence,
                   o.latency_ms,
                   COALESCE(length(o.raw_text), 0) AS text_length,
                   COALESCE(d.doc_type, 'unknown') AS doc_type,
                   o.created_at
            FROM ocr_results o
            LEFT JOIN documents d ON d.id = o.document_id
            ORDER BY o.created_at ASC;
            """
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df.dropna(subset=["confidence"])
    except Exception as e:
        print(f"[!] Could not read from database ({e}).")
        return pd.DataFrame()


def synthetic_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a reference and a (mildly drifted) current window for demo/fallback."""
    rng = np.random.default_rng(42)
    doc_types = ["prescription", "invoice", "lab_report", "omr"]

    def make(n, conf_mean, lat_mean, len_mean, weights):
        return pd.DataFrame({
            "confidence": np.clip(rng.normal(conf_mean, 0.08, n), 0, 1),
            "latency_ms": np.clip(rng.normal(lat_mean, 120, n), 30, None).astype(int),
            "text_length": np.clip(rng.normal(len_mean, 80, n), 0, None).astype(int),
            "doc_type": rng.choice(doc_types, n, p=weights),
        })

    # Current window drifts: lower confidence, higher latency, shifted doc mix.
    reference = make(200, 0.86, 600, 320, [0.4, 0.3, 0.2, 0.1])
    current = make(200, 0.74, 820, 260, [0.25, 0.2, 0.4, 0.15])
    return reference, current


def split_windows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split real data chronologically into older=reference, newer=current."""
    mid = len(df) // 2
    return df.iloc[:mid].copy(), df.iloc[mid:].copy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Evidently OCR drift monitor")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "drift_report.html"))
    args = ap.parse_args()

    df = load_from_db()
    using_real = len(df) >= 2 * MIN_ROWS_PER_WINDOW
    if using_real:
        reference, current = split_windows(df)
        print(f"[*] Using REAL data: {len(reference)} reference / {len(current)} current rows.")
    else:
        if len(df) > 0:
            print(f"[!] Only {len(df)} rows in DB (<{2*MIN_ROWS_PER_WINDOW}). Using synthetic demo data.")
        else:
            print("[!] No DB data available. Using synthetic demo data.")
        reference, current = synthetic_windows()

    cols = NUMERICAL + CATEGORICAL
    reference = reference[cols]
    current = current[cols]

    data_def = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)
    ref_ds = Dataset.from_pandas(reference, data_definition=data_def)
    cur_ds = Dataset.from_pandas(current, data_definition=data_def)

    report = Report([DataDriftPreset(), DataSummaryPreset()])
    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_path))

    print("\n" + "=" * 56)
    print(f"  Evidently drift report : {out_path}")
    print(f"  Data source            : {'database' if using_real else 'synthetic (demo)'}")
    print(f"  Reference / Current     : {len(reference)} / {len(current)} rows")
    print(f"  Monitored features      : {', '.join(cols)}")
    print("=" * 56)
    print("Open the HTML in a browser to see per-feature drift + distributions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
