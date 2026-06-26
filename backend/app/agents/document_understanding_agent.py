"""
Agent 01 — Document Understanding Agent
Interprets structured JSON from extracted_entities into a natural language description.
Logs token usage and execution details to agent_activity.
"""

import json
import time
import logging
from typing import Dict, Any, Optional
from uuid import UUID

from langchain_core.prompts import ChatPromptTemplate

from app.config import get_settings
from app.database import SyncSessionLocal
from app.models.agent_activity import AgentActivity
from app.services.llm_provider import get_llm, has_llm_api_key, get_model_name

logger = logging.getLogger(__name__)
settings = get_settings()

AGENT_ID = "document_understanding_agent"


def run_document_understanding_agent(
    document_id: UUID,
    doc_type: str,
    extracted_entities: dict
) -> str:
    """
    Run Agent 01: Document Understanding Agent.
    Translates structured JSON entity keys into a plain English narrative.
    """
    start_time = time.perf_counter()
    llm_model = get_model_name()
    prompt_tokens = 0
    completion_tokens = 0
    status = "success"
    explanation = ""

    # Check for valid LLM API key
    has_api_key = has_llm_api_key()

    if not has_api_key:
        # ── Fallback Simulation Mode (Deterministic) ─────────────────────────
        logger.info("No active DEEPSEEK_API_KEY found. Running Agent-01 in simulation mode.")
        time.sleep(0.1)  # Simulate small network latency
        explanation = _generate_mock_explanation(doc_type, extracted_entities)
        prompt_tokens = len(str(extracted_entities)) // 4
        completion_tokens = len(explanation) // 4
    else:
        # ── Real LLM Execution Mode ──────────────────────────────────────────
        try:
            llm = get_llm(temperature=0.2)

            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a helpful assistant. You take a structured JSON representation "
                    "of an OCR'd document (type: {doc_type}) and explain its key details in simple, "
                    "professional, natural-sounding English sentences. Do not add any extra inferences "
                    "beyond what is explicitly given in the input JSON."
                )),
                ("user", "Document JSON:\n{json_data}")
            ])

            chain = prompt | llm
            response = chain.invoke({
                "doc_type": doc_type,
                "json_data": json.dumps(extracted_entities, indent=2)
            })

            explanation = response.content.strip()

            # Attempt to extract token usage if returned by LangChain/DeepSeek
            if response.response_metadata and "token_usage" in response.response_metadata:
                usage = response.response_metadata["token_usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
            else:
                prompt_tokens = len(str(extracted_entities)) // 4
                completion_tokens = len(explanation) // 4

        except Exception as e:
            logger.error(f"Error calling LLM for Document Understanding Agent: {e}")
            status = "failed"
            explanation = (
                f"Document Understanding failure: could not generate narrative from "
                f"structured data. Raw details: {json.dumps(extracted_entities)}"
            )

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    # ── Log Activity to Database ─────────────────────────────────────────────
    _log_agent_activity(
        agent_id=AGENT_ID,
        document_id=document_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        llm_model=llm_model if has_api_key else f"{llm_model}-simulated",
        status=status
    )

    return explanation


def _generate_mock_explanation(doc_type: str, entities: dict) -> str:
    """Generates standard clean narratives for testing purposes when API key is unset."""
    if doc_type == "invoice":
        vendor = entities.get("vendor", "an unknown vendor")
        inv_no = entities.get("invoice_no", "N/A")
        amount = entities.get("amount", "unknown")
        currency = entities.get("currency", "INR")
        date = entities.get("date", "N/A")
        return (
            f"This is an invoice (no. {inv_no}) issued by {vendor} on {date}. "
            f"The total amount due is {amount} {currency}."
        )
    elif doc_type == "prescription":
        doctor = entities.get("doctor", "an unknown physician")
        patient = entities.get("patient", "an unknown patient")
        meds = ", ".join(entities.get("medications", []))
        instructions = entities.get("dosage_instructions", "N/A")
        date = entities.get("date", "N/A")
        return (
            f"This is a medical prescription written by {doctor} for patient {patient} "
            f"on {date}. The prescribed medications are: {meds}. "
            f"Instructions state: {instructions}."
        )
    elif doc_type == "lab_report":
        patient = entities.get("patient", "Unknown")
        glucose = entities.get("glucose", "Unknown")
        hba1c = entities.get("hba1c", "Unknown")
        wbc = entities.get("wbc", "Unknown")
        return (
            f"This is a medical lab report for {patient}. "
            f"Biomarkers extracted show Glucose at {glucose}, HbA1c at {hba1c}, "
            f"and White Blood Cell (WBC) count at {wbc}."
        )
    elif doc_type == "omr":
        checked = entities.get("checked_cells", [])
        total = entities.get("total_checked", len(checked))
        return (
            f"This is a scanned OMR form. Visual analysis detected a total of {total} checked fields. "
            f"The marks were registered at bubble indices: {[c.get('mark_index') for c in checked]}."
        )
    else:
        return f"Document of type '{doc_type}' contains the following structured data: {json.dumps(entities)}."


def _log_agent_activity(
    agent_id: str,
    document_id: UUID,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    llm_model: str,
    status: str
) -> None:
    """Write activity log record directly to database using sync session."""
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
