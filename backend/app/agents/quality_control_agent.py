"""
Agent 06 — Quality Control Agent

Validates the structured entities extracted from a document BEFORE downstream agents
(medical summary, anomaly detection, reporting) consume them. It applies per-document-type
completeness/consistency rules, computes a 0.0–1.0 quality score, and lists concrete
issues. When a DeepSeek key is configured it adds an LLM plausibility cross-check.

This is the data-quality guardrail of the agent layer: it complements the ML-layer
confidence gate (which judges OCR confidence) by judging the *extracted fields*.
"""

import re
import time
import logging
from typing import Dict, Any, List, Tuple
from uuid import UUID

from app.config import get_settings
from app.database import SyncSessionLocal
from app.models.agent_activity import AgentActivity
from app.services.llm_provider import get_llm, has_llm_api_key, get_model_name

logger = logging.getLogger(__name__)
settings = get_settings()

AGENT_ID = "quality_control_agent"

# Required fields per document type — used to compute a completeness score.
_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "prescription": ["doctor", "patient", "medications", "date"],
    "invoice": ["vendor", "amount", "invoice_no", "date"],
    "lab_report": ["patient", "glucose", "hba1c", "wbc"],
    "omr": ["total_checked"],
}

# Values that mean "the extractor produced nothing real for this field".
_PLACEHOLDERS = {"", "n/a", "na", "none", "unknown", "unknown doctor", "unknown patient", "null"}


def run_quality_control_agent(
    document_id: UUID,
    doc_type: str,
    extracted_entities: dict,
) -> Tuple[str, float, bool]:
    """
    Run Agent 06: Quality Control Agent.

    Returns:
        (summary_text, quality_score, quality_passed)
        - quality_score: completeness ratio in [0.0, 1.0]
        - quality_passed: True if score >= 0.5 and no critical issues
    """
    start_time = time.perf_counter()
    issues: List[str] = []

    required = _REQUIRED_FIELDS.get(doc_type, [])
    present = 0

    # ── Rule set 1: field completeness ────────────────────────────────────────
    for field in required:
        value = extracted_entities.get(field)
        if _is_present(value):
            present += 1
        else:
            issues.append(f"Missing or empty required field: '{field}'")

    score = present / len(required) if required else 1.0

    # ── Rule set 2: type / format consistency ─────────────────────────────────
    if doc_type == "invoice":
        amount = extracted_entities.get("amount")
        if _is_present(amount) and not _is_numeric(amount):
            issues.append(f"Invoice 'amount' is not numeric: {amount!r}")
    if doc_type == "prescription":
        meds = extracted_entities.get("medications")
        if _is_present(meds) and not isinstance(meds, list):
            issues.append("Prescription 'medications' should be a list")
    if "date" in required:
        date_val = extracted_entities.get("date")
        if _is_present(date_val) and not _looks_like_date(str(date_val)):
            issues.append(f"Field 'date' does not look like a date: {date_val!r}")

    # ── Rule set 3: optional LLM plausibility cross-check ──────────────────────
    llm_model = get_model_name()
    has_api_key = has_llm_api_key()
    used_llm = False
    if has_api_key and required:
        llm_note = _llm_plausibility_check(doc_type, extracted_entities)
        if llm_note:
            issues.append(f"LLM plausibility note: {llm_note}")
            used_llm = True

    quality_passed = score >= 0.5 and not any("not numeric" in i for i in issues)

    summary = (
        f"Quality score {score:.0%} ({present}/{len(required)} required fields present). "
        + ("No structural issues found." if not issues
           else f"{len(issues)} issue(s): " + "; ".join(issues))
    )

    latency_ms = int((time.perf_counter() - start_time) * 1000)
    _log_agent_activity(
        agent_id=AGENT_ID,
        document_id=document_id,
        prompt_tokens=len(str(extracted_entities)) // 4,
        completion_tokens=len(summary) // 4,
        latency_ms=latency_ms,
        llm_model=llm_model if used_llm else "rule-engine",
        status="success" if quality_passed else "needs_review",
    )

    logger.info(f"Quality Control: doc {document_id} score={score:.2f} passed={quality_passed}")
    return summary, round(score, 3), quality_passed


# ── Helpers ───────────────────────────────────────────────────────────────────
def _is_present(value: Any) -> bool:
    """True if the field holds real extracted content (not a placeholder/empty)."""
    if value is None:
        return False
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return str(value).strip().lower() not in _PLACEHOLDERS


def _is_numeric(value: Any) -> bool:
    try:
        float(str(value).replace(",", "").replace("INR", "").replace("$", "").strip())
        return True
    except (ValueError, TypeError):
        return False


def _looks_like_date(value: str) -> bool:
    return bool(re.search(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}", value))


def _llm_plausibility_check(doc_type: str, entities: dict) -> str:
    """Ask the LLM whether the extracted fields are internally plausible. Returns '' if fine."""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        import json

        llm = get_llm(temperature=0.0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a data quality checker for extracted document fields. "
                "Given the document type and extracted JSON, reply with a SHORT phrase "
                "describing any clearly implausible value (e.g. a negative invoice amount, "
                "a future date, a medication that is obviously not a drug). "
                "If everything looks plausible, reply with exactly 'OK'."
            )),
            ("user", "Document type: {doc_type}\nExtracted JSON:\n{json_data}")
        ])
        chain = prompt | llm
        resp = chain.invoke({
            "doc_type": doc_type,
            "json_data": json.dumps(entities, indent=2, default=str),
        }).content.strip()
        return "" if resp.upper().startswith("OK") else resp[:200]
    except Exception as e:
        logger.warning(f"Quality Control LLM check failed: {e}")
        return ""


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
