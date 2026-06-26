"""
LayoutLMv3 fine-tuning — Step A: annotation parsing (no OCR required).

Identifies lab-report samples in the medocr-vision-dataset and parses their
markdown-table text annotations into structured rows:
    [{"parameter": ..., "value": ..., "unit": ..., "reference": ...}, ...]

These parsed rows are the *weak labels* later aligned (Step B) against OCR
words+boxes to produce token-level BIO tags for LayoutLMv3 token classification.

This step is pure Python so we can validate weak-supervision viability before
investing in OCR/training. Run:
    .venv\\Scripts\\python.exe scripts\\prep_layoutlm_data.py
"""

import re
import json
from pathlib import Path
from collections import Counter

from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "medocr-vision-dataset"
OUT_PATH = PROJECT_ROOT / "data" / "layoutlm_labreport_parsed.json"

LAB_KEYWORDS = [
    "glucose", "hba1c", "wbc", "rbc", "hemoglobin", "haemoglobin", "cholesterol",
    "biochemistry", "pathology", "reference range", "mg/dl", "laboratory",
    "platelet", "neutrophils", "lymphocytes", "biological reference", "specimen",
]

# A numeric value, possibly decimal/scientific, optionally with thousands separators.
_VALUE_RE = re.compile(r"^[<>]?\s*\d[\d,]*(?:\.\d+)?(?:\s*[-–]\s*\d[\d,]*(?:\.\d+)?)?$")
# A reference range like "70-100", "< 0.50", "4.4 - 5.2", "150000 - 400000".
_REF_RE = re.compile(r"([<>]\s*\d|\d[\d,]*(?:\.\d+)?\s*[-–]\s*\d)")
# A unit token like mg/dL, g/dL, %, K/uL, 10³/mm³, /cu.mm, Million/ul.
_UNIT_RE = re.compile(r"(mg/dl|g/dl|%|k/u?l|10[³⁹]?/?mm[³]?|/\s*cu\.?mm|million/ul|cells?/|mmol/l|iu/l|u/l|ng/ml|pg|fl)", re.IGNORECASE)


def is_lab_report(text: str) -> bool:
    """Heuristic: a lab report mentions multiple lab keywords or has a results table."""
    t = text.lower()
    kw_hits = sum(k in t for k in LAB_KEYWORDS)
    has_results_table = ("reference" in t or "biological reference" in t) and "|" in text
    return kw_hits >= 2 or has_results_table


def _split_value_unit(cell: str) -> tuple[str, str]:
    """Split a 'value unit' cell (e.g. '11.3 g/dL', '197000 / cu.mm') into (value, unit)."""
    cell = cell.strip()
    unit_match = _UNIT_RE.search(cell)
    if unit_match:
        unit = unit_match.group(0)
        value = cell[: unit_match.start()].strip()
        return value, unit
    # No recognizable unit: whole cell is the value.
    return cell, ""


def parse_lab_tables(text: str) -> list[dict]:
    """
    Extract structured rows from markdown tables in a lab-report annotation.
    Returns a list of {parameter, value, unit, reference} dicts (best-effort).
    """
    rows: list[dict] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        cells = [c for c in cells if c != ""]
        if len(cells) < 2:
            continue
        # Skip header / separator rows.
        joined = " ".join(cells).lower()
        if set(line.strip()) <= set("|-: "):  # separator row like |---|---|
            continue
        if any(h in joined for h in ["parameter", "results", "value", "test name"]) and not re.search(r"\d", joined):
            continue

        parameter = cells[0].strip("* ")
        if not parameter or len(parameter) > 60:
            continue

        value, unit, reference = "", "", ""
        # Find the reference range (a cell that looks like a range / threshold).
        for c in cells[1:]:
            if _REF_RE.search(c) and not reference:
                reference = c
        # Find the value: first non-parameter cell containing a number that isn't the ref.
        for c in cells[1:]:
            if c == reference:
                continue
            if re.search(r"\d", c):
                value, unit = _split_value_unit(c)
                break

        if value or reference:
            rows.append({
                "parameter": parameter,
                "value": value,
                "unit": unit,
                "reference": reference,
            })
    return rows


def main() -> int:
    if not (DATASET_DIR / "dataset_dict.json").exists() and not (DATASET_DIR).exists():
        print(f"ERROR: dataset not found at {DATASET_DIR}")
        return 1

    ds = load_from_disk(str(DATASET_DIR))
    out = {}
    grand_rows = 0
    parsed_report_count = 0
    empty_parse = 0
    param_counter: Counter = Counter()

    for split in ds.keys():
        split_ds = ds[split]
        split_records = []
        for i in range(len(split_ds)):
            text = str(split_ds[i]["text"])
            if not is_lab_report(text):
                continue
            rows = parse_lab_tables(text)
            if rows:
                parsed_report_count += 1
                grand_rows += len(rows)
                for r in rows:
                    param_counter[r["parameter"].lower()] += 1
            else:
                empty_parse += 1
            split_records.append({"index": i, "n_rows": len(rows), "rows": rows})
        out[split] = split_records
        print(f"[{split}] lab reports: {len(split_records)}, with parsed rows: "
              f"{sum(1 for r in split_records if r['n_rows'])}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nTotal lab reports parsed with >=1 row: {parsed_report_count}")
    print(f"Lab reports where parsing found nothing: {empty_parse}")
    print(f"Total extracted rows: {grand_rows} "
          f"(avg {grand_rows / max(parsed_report_count,1):.1f} rows/report)")
    print(f"Top 15 parameters seen: {param_counter.most_common(15)}")
    print(f"\nWrote parsed weak-labels to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
