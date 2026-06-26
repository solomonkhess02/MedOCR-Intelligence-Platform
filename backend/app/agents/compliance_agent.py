"""
Agent 07 — Compliance / PII Agent

Scans a processed medical document for Protected Health Information (PHI) and other
personally identifiable information (PII), reports what categories were found, and
produces a redacted preview of the text. This is a domain-relevant safeguard for a
medical platform (HIPAA/DPDP-style data-minimization).

It reads the OCR raw_text for the document from the database (falling back to the
extracted entities if no raw_text is available), applies deterministic detectors, and
records its activity to the agent_activity audit log.
"""

import re
import time
import logging
from typing import Dict, Any, List, Tuple
from uuid import UUID

from sqlalchemy import text

from app.config import get_settings
from app.database import SyncSessionLocal
from app.models.agent_activity import AgentActivity

logger = logging.getLogger(__name__)
settings = get_settings()

AGENT_ID = "compliance_agent"

# Deterministic PHI/PII detectors. Each entry: (category, compiled regex).
_DETECTORS: List[Tuple[str, "re.Pattern"]] = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("phone", re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b")),
    ("date", re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})\b")),
    # Medical record / patient ID style tokens, e.g. MRN-12345, ID: 99812
    ("medical_id", re.compile(r"\b(?:MRN|ID|UID|PID)[\s:#-]*\d{4,}\b", re.IGNORECASE)),
    # Long digit runs (account / Aadhaar / SSN-like) — 9+ digits
    ("id_number", re.compile(r"\b\d{9,}\b")),
]

# Entity keys whose VALUES are personal names we should treat as PII.
_NAME_KEYS = ("patient", "doctor", "patient_name", "doctor_name")


def run_compliance_agent(
    document_id: UUID,
    doc_type: str,
    extracted_entities: dict,
) -> Tuple[str, Dict[str, int]]:
    """
    Run Agent 07: Compliance / PII Agent.

    Returns:
        (summary_text, findings) where findings maps category -> count.
    """
    start_time = time.perf_counter()

    source_text = _load_raw_text(document_id) or _entities_to_text(extracted_entities)

    findings: Dict[str, int] = {}
    redacted = source_text

    # ── Pattern-based detectors ───────────────────────────────────────────────
    for category, pattern in _DETECTORS:
        matches = pattern.findall(source_text)
        if matches:
            findings[category] = findings.get(category, 0) + len(matches)
            redacted = pattern.sub(f"[REDACTED_{category.upper()}]", redacted)

    # ── Name detection from structured entities ───────────────────────────────
    name_hits = 0
    for key in _NAME_KEYS:
        val = extracted_entities.get(key)
        if val and str(val).strip().lower() not in ("", "unknown", "unknown doctor", "unknown patient", "n/a"):
            name = str(val).strip()
            name_hits += 1
            # Redact the literal name where it appears in the text.
            redacted = re.sub(re.escape(name), "[REDACTED_NAME]", redacted, flags=re.IGNORECASE)
    if name_hits:
        findings["name"] = name_hits

    total = sum(findings.values())
    is_medical = doc_type in ("prescription", "lab_report")
    severity = "high" if (is_medical and total > 0) else ("medium" if total > 0 else "none")

    if total == 0:
        summary = "Compliance scan: no PII/PHI detected."
    else:
        cats = ", ".join(f"{k}={v}" for k, v in sorted(findings.items()))
        summary = (
            f"Compliance scan: {total} PII/PHI item(s) detected ({cats}); "
            f"severity={severity}. Redacted preview: {redacted[:200]}"
        )

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    _log_agent_activity(
        agent_id=AGENT_ID,
        document_id=document_id,
        prompt_tokens=len(source_text) // 4,
        completion_tokens=len(summary) // 4,
        latency_ms=latency_ms,
        llm_model="rule-engine",
        status="success",
    )

    logger.info(f"Compliance Agent: doc {document_id} found {total} PII/PHI items (severity={severity}).")
    return summary, findings


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_raw_text(document_id: UUID) -> str:
    """Fetch the most recent OCR raw_text for this document, or '' if unavailable."""
    try:
        with SyncSessionLocal() as db:
            row = db.execute(
                text(
                    "SELECT raw_text FROM ocr_results "
                    "WHERE document_id = :doc_id ORDER BY created_at DESC LIMIT 1;"
                ),
                {"doc_id": document_id},
            ).first()
            if row and row.raw_text:
                return str(row.raw_text)
    except Exception as e:
        logger.warning(f"Compliance Agent could not load raw_text from DB: {e}")
    return ""


def _entities_to_text(entities: dict) -> str:
    """Flatten extracted entities into a scannable text blob."""
    parts: List[str] = []
    for k, v in (entities or {}).items():
        parts.append(f"{k}: {v}")
    return "\n".join(parts)


def _log_agent_activity(
    agent_id: str,
    document_id: UUID,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    llm_model: str,
    status: str,
) -> None:
    """Write agent activity log record directly to database."""
    try:
        with SyncSessionLocal() as db:
            activity = AgentActivity(
                agent_id=agent_id,
                document_id=document_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                llm_model=llm_model,
                status=status,
            )
            db.add(activity)
            db.commit()
    except Exception as e:
        logger.warning(f"Failed to write agent activity log to database: {e}")
