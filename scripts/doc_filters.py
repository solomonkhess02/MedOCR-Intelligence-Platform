"""
Shared document-type filters for the medocr-vision-dataset.

The dataset has no `document_type` column — only `image` + `text`. These
heuristics recover document types from the annotation text so training and
evaluation use a single, consistent definition.

The prescription heuristic recovers exactly the 1,000 prescriptions documented
in the dataset card (821 train + 84 val + 95 test) by keying on the structured
prescription annotation format (`<s_ocr> doctor_name: ... medications ...`).
"""


_LAB_KEYWORDS = (
    "glucose", "hba1c", "wbc", "rbc", "hemoglobin", "haemoglobin", "cholesterol",
    "biochemistry", "pathology", "reference range", "biological reference", "mg/dl",
    "laboratory", "platelet", "neutrophils", "lymphocytes", "specimen",
)
_INVOICE_KEYWORDS = (
    "invoice", "receipt", "subtotal", "vat", "gst", "tax", "qty", "quantity",
    "unit price", "amount due", "seller", "vendor", "bill to", "line items",
    "grand total", "payment",
)


def is_prescription(text: str) -> bool:
    """True if the annotation looks like a handwritten-prescription transcription."""
    t = str(text).lower()
    if "<s_ocr>" in t:
        return True
    return "doctor_name" in t and ("medication" in t or "patient_name" in t)


def is_lab_report(text: str) -> bool:
    """True if the annotation looks like a medical lab report (multiple lab terms or a results table)."""
    t = str(text).lower()
    if is_prescription(text):
        return False
    kw_hits = sum(k in t for k in _LAB_KEYWORDS)
    has_results_table = ("reference" in t or "biological reference" in t) and "|" in str(text)
    return kw_hits >= 2 or has_results_table


def is_invoice(text: str) -> bool:
    """
    True if the annotation looks like an invoice/receipt.

    Defined by exclusion + signals: it's an invoice if it isn't a prescription or a
    lab report and it carries at least one invoice/receipt indicator. This matches
    Donut's domain (the 1,000 invoice/receipt samples) without a document_type label.
    """
    if is_prescription(text) or is_lab_report(text):
        return False
    t = str(text).lower()
    return any(k in t for k in _INVOICE_KEYWORDS)


def filter_prescriptions(dataset):
    """Return a HF dataset filtered to prescription samples only."""
    return dataset.filter(lambda row: is_prescription(row["text"]))


def filter_invoices(dataset):
    """Return a HF dataset filtered to invoice/receipt samples only."""
    return dataset.filter(lambda row: is_invoice(row["text"]))
