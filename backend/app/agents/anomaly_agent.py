"""
Agent 04 — Anomaly Detection Agent
Analyzes structured extraction data against database history to detect duplicate invoices,
amount outliers, unexpected vendors, and OMR layout discrepancies. Writes anomalies to the DB.
"""

import time
import logging
import math
from typing import Dict, Any, List, Optional
from uuid import UUID

from sqlalchemy import text

from app.config import get_settings
from app.database import SyncSessionLocal
from app.models.anomaly import Anomaly
from app.models.agent_activity import AgentActivity

logger = logging.getLogger(__name__)
settings = get_settings()

AGENT_ID = "anomaly_agent"


def run_anomaly_agent(
    document_id: UUID,
    doc_type: str,
    extracted_entities: dict
) -> List[dict]:
    """
    Run Agent 04: Anomaly Detection Agent.
    Evaluates extraction results against historical database context and records anomalies.
    """
    start_time = time.perf_counter()
    anomalies_detected: List[dict] = []

    # ── Rule 1: Duplicate Invoices ────────────────────────────────────────────
    if doc_type == "invoice" and "invoice_no" in extracted_entities:
        inv_no = str(extracted_entities["invoice_no"])
        if inv_no and inv_no != "N/A":
            dup = _check_duplicate_invoice(document_id, inv_no)
            if dup:
                anomalies_detected.append(dup)

    # ── Rule 2: Cost Outliers (Amount > mean + 3σ) ────────────────────────────
    if doc_type == "invoice" and "amount" in extracted_entities:
        try:
            amt = float(extracted_entities["amount"])
            outlier = _check_cost_outlier(document_id, amt)
            if outlier:
                anomalies_detected.append(outlier)
        except ValueError:
            pass

    # ── Rule 3: Unexpected Vendor Accounts ───────────────────────────────────
    if doc_type == "invoice" and "vendor" in extracted_entities:
        vendor = str(extracted_entities["vendor"])
        if vendor:
            new_vendor = _check_new_vendor(document_id, vendor)
            if new_vendor:
                anomalies_detected.append(new_vendor)

    # ── Rule 4: OMR Grid Inconsistency ────────────────────────────────────────
    if doc_type == "omr":
        total_checked = extracted_entities.get("total_checked", 0)
        # Standard form has 1 to 20 checkboxes; if 0 are marked, it is flagged as outlier/unfilled
        if total_checked == 0:
            anomalies_detected.append({
                "anomaly_type": "omr_inconsistency",
                "severity": "low",
                "confidence": 0.80,
                "reasoning": "OMR form was processed but zero checked bubbles were registered. Verify form is filled."
            })

    # ── Step 5: Save Detected Anomalies to DB ────────────────────────────────
    _save_anomalies_to_db(document_id, anomalies_detected)

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    # ── Log Activity to Database ─────────────────────────────────────────────
    _log_agent_activity(
        agent_id=AGENT_ID,
        document_id=document_id,
        prompt_tokens=len(str(extracted_entities)) // 4,
        completion_tokens=len(str(anomalies_detected)) // 4,
        latency_ms=latency_ms,
        llm_model="heuristic-engine",
        status="success"
    )

    return anomalies_detected


def _check_duplicate_invoice(document_id: UUID, invoice_no: str) -> Optional[dict]:
    """Checks database if another invoice has the same invoice_no."""
    sql = (
        "SELECT document_id FROM extracted_entities "
        "WHERE entity_type = 'invoice' "
        "AND entity_data->>'invoice_no' = :invoice_no "
        "AND document_id != :doc_id LIMIT 1;"
    )
    try:
        with SyncSessionLocal() as db:
            row = db.execute(text(sql), {"invoice_no": invoice_no, "doc_id": document_id}).first()
            if row:
                return {
                    "anomaly_type": "duplicate_invoice",
                    "severity": "high",
                    "confidence": 0.95,
                    "reasoning": f"Invoice number '{invoice_no}' was already processed in document {row.document_id}."
                }
    except Exception as e:
        logger.warning(f"Duplicate invoice check failed: {e}")
    return None


def _check_cost_outlier(document_id: UUID, amount: float) -> Optional[dict]:
    """Checks if invoice amount is a statistical outlier (> mean + 3σ)."""
    sql = (
        "SELECT (entity_data->>'amount')::float as amt "
        "FROM extracted_entities "
        "WHERE entity_type = 'invoice' AND document_id != :doc_id "
        "AND (entity_data->>'amount') IS NOT NULL;"
    )
    try:
        with SyncSessionLocal() as db:
            rows = db.execute(text(sql), {"doc_id": document_id}).fetchall()
            amounts = [r.amt for r in rows if r.amt is not None]
            
            if len(amounts) >= 3:
                mean = sum(amounts) / len(amounts)
                variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
                std_dev = math.sqrt(variance)
                
                # Check 3-sigma rule
                threshold = mean + 3 * std_dev
                if amount > threshold and std_dev > 0:
                    sigma_val = (amount - mean) / std_dev
                    return {
                        "anomaly_type": "amount_outlier",
                        "severity": "high" if sigma_val > 4 else "medium",
                        "confidence": 0.90,
                        "reasoning": f"Invoice amount INR {amount:,.2f} is a cost outlier. "
                                     f"It is {sigma_val:.1f}σ above the historical mean of INR {mean:,.2f} (StdDev: {std_dev:.1f})."
                    }
    except Exception as e:
        logger.warning(f"Cost outlier check failed: {e}")
    return None


def _check_new_vendor(document_id: UUID, vendor: str) -> Optional[dict]:
    """Flags if vendor has not been seen in the historical database."""
    sql = (
        "SELECT count(*) as count FROM extracted_entities "
        "WHERE entity_type = 'invoice' AND document_id != :doc_id "
        "AND entity_data->>'vendor' = :vendor;"
    )
    try:
        with SyncSessionLocal() as db:
            count = db.execute(text(sql), {"vendor": vendor, "doc_id": document_id}).scalar() or 0
            if count == 0:
                return {
                    "anomaly_type": "vendor_mismatch",
                    "severity": "low",
                    "confidence": 0.85,
                    "reasoning": f"First invoice from vendor '{vendor}' registered on this platform. Marked for validation."
                }
    except Exception as e:
        logger.warning(f"New vendor check failed: {e}")
    return None


def _save_anomalies_to_db(document_id: UUID, anomalies: List[dict]) -> None:
    """Commit detected anomalies into anomalies database table."""
    if not anomalies:
        return
    try:
        with SyncSessionLocal() as db:
            for item in anomalies:
                anomaly = Anomaly(
                    document_id=document_id,
                    anomaly_type=item["anomaly_type"],
                    severity=item["severity"],
                    confidence=item["confidence"],
                    reasoning=item["reasoning"]
                )
                db.add(anomaly)
            db.commit()
            logger.info(f"Anomaly Agent: committed {len(anomalies)} anomalies for document {document_id}.")
    except Exception as e:
        logger.error(f"Failed to save anomalies to database: {e}")


def _log_agent_activity(
    agent_id: str,
    document_id: UUID,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    llm_model: str,
    status: str
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
                status=status
            )
            db.add(activity)
            db.commit()
    except Exception as e:
        logger.warning(f"Failed to write agent activity log to database: {e}")
