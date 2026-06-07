"""
Agent 05 — Medical Summary Agent
Summarizes medical document data (prescriptions and lab reports).
Enforces architectural safety guardrails to block diagnostics, predictions, or recommendations.
"""

import time
import logging
import re
import json
from typing import Dict, Any, Tuple
from uuid import UUID

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.database import SyncSessionLocal
from app.models.agent_activity import AgentActivity

logger = logging.getLogger(__name__)
settings = get_settings()

AGENT_ID = "medical_summary_agent"

# Deterministic safety filters (regex checking for common diagnostic/recommendation phrasing)
GUARDRAIL_REGEXES = [
    r"\b(?:you\s+have|diagnose|suffer\s+from|diabetic|indicates\s+that\s+you|indicates\s+you\s+have)\b",
    r"\b(?:should\s+take|recommend\s+taking|increase\s+dose|decrease\s+dose|prescribe\s+you)\b",
    r"\b(?:condition\s+(?:may|will)\s+(?:worsen|improve|get\s+better|deteriorate))\b",
    r"\b(?:likely\s+indicates\s+disease|has\s+developed|signs\s+of\s+cancer|signs\s+of\s+diabetes)\b"
]


def run_medical_summary_agent(
    document_id: UUID,
    doc_type: str,
    extracted_entities: dict
) -> Tuple[str, bool]:
    """
    Run Agent 05: Medical Summary Agent.
    Summarizes medical documents in plain English while enforcing strict guardrails.
    
    Returns:
        (summary_text, guardrail_blocked_boolean)
    """
    start_time = time.perf_counter()
    llm_model = settings.gemini_model or "gemini-2.0-flash"
    prompt_tokens = 0
    completion_tokens = 0
    status = "success"
    summary = ""
    guardrail_blocked = False

    # Guard: only run for medical documents
    if doc_type not in ["prescription", "lab_report"]:
        return "Non-medical document. Medical summary skipped.", False

    # Check for valid Google API Key
    has_api_key = (
        settings.google_api_key and 
        "your-gemini-api" not in settings.google_api_key
    )

    if not has_api_key:
        # ── Fallback Simulation Mode ─────────────────────────────────────────
        logger.info("No active GOOGLE_API_KEY found. Running Agent-05 in simulation mode.")
        time.sleep(0.1)
        summary = _generate_mock_medical_summary(doc_type, extracted_entities)
        prompt_tokens = len(str(extracted_entities)) // 4
        completion_tokens = len(summary) // 4
    else:
        # ── Real LLM Execution Mode ──────────────────────────────────────────
        try:
            llm = ChatGoogleGenerativeAI(
                model=llm_model,
                google_api_key=settings.google_api_key,
                temperature=0.0  # Zero temperature for maximum reliability
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a medical summarizer. Summarize the provided document JSON. "
                    "Rule: ONLY list the extracted values, dosages, medications, and reference ranges "
                    "literally. DO NOT write any diagnostic statements, doctor advice, predictive claims "
                    "about condition progression, or custom dosage recommendations."
                )),
                ("user", "Document JSON:\n{json_data}")
            ])

            chain = prompt | llm
            response = chain.invoke({
                "json_data": json.dumps(extracted_entities, indent=2)
            })

            summary = response.content.strip()

            if response.response_metadata and "token_usage" in response.response_metadata:
                usage = response.response_metadata["token_usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
            else:
                prompt_tokens = len(str(extracted_entities)) // 4
                completion_tokens = len(summary) // 4

        except Exception as e:
            logger.error(f"Error calling LLM for Medical Summary Agent: {e}")
            status = "failed"
            summary = "Medical Summary failure: could not connect to summary generator."

    # ── Safety Guardrail Validation Layer ─────────────────────────────────────
    # Tier 1: Regex checks (deterministic)
    for pattern in GUARDRAIL_REGEXES:
        if re.search(pattern, summary, re.IGNORECASE):
            logger.warning(f"Medical Summary guardrail triggered: summary matched regex rule '{pattern}'")
            guardrail_blocked = True
            break

    # Tier 2: LLM Auditor check (if API key is active and not already blocked)
    if not guardrail_blocked and has_api_key:
        try:
            auditor_llm = ChatGoogleGenerativeAI(
                model=llm_model,
                google_api_key=settings.google_api_key,
                temperature=0.0
            )
            
            auditor_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a strict medical safety auditor. Your job is to check summaries of medical tests "
                    "or prescriptions. Ensure the summary is safe. "
                    "It is UNSAFE (BLOCKED) if it contains: "
                    "1. A diagnosis (e.g. telling the patient they have a condition, 'you have diabetes'). "
                    "2. Future health projections or warnings ('your condition might worsen'). "
                    "3. Dosage advice/recommendations not found literally in the document. "
                    "If the summary violates any safety rules, reply with 'BLOCKED'. "
                    "If the summary is completely safe and only lists literal findings, reply with 'PASSED'."
                )),
                ("user", "Medical Summary to Audit:\n{summary_text}")
            ])
            
            auditor_chain = auditor_prompt | auditor_llm
            audit_result = auditor_chain.invoke({"summary_text": summary}).content.strip()
            
            if "BLOCKED" in audit_result.upper():
                logger.warning("Medical Summary guardrail triggered: secondary LLM auditor flagged summary as unsafe.")
                guardrail_blocked = True

        except Exception as audit_err:
            logger.warning(f"Secondary LLM guardrail check failed to execute: {audit_err}. Falling back to strict regex filter only.")

    # ── Handle Blocked Outputs ────────────────────────────────────────────────
    if guardrail_blocked:
        summary = "Summary withheld: guardrail triggered."
        status = "guardrail_blocked"

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

    return summary, guardrail_blocked


def _generate_mock_medical_summary(doc_type: str, entities: dict) -> str:
    """Generates a clean mock medical summary for verification."""
    if doc_type == "prescription":
        meds = ", ".join(entities.get("medications", ["None"]))
        dosage = entities.get("dosage_instructions", "None")
        doctor = entities.get("doctor", "Unknown")
        return (
            f"Prescription summary details:\n"
            f"- Prescribing Physician: {doctor}\n"
            f"- Active Medications: {meds}\n"
            f"- Instruction: {dosage}\n"
            f"No medical warnings or diagnostic claims are inferred."
        )
    elif doc_type == "lab_report":
        glucose = entities.get("glucose", "Unknown")
        hba1c = entities.get("hba1c", "Unknown")
        wbc = entities.get("wbc", "Unknown")
        ref = entities.get("reference_ranges", {})
        return (
            f"Laboratory report summary details:\n"
            f"- Glucose: {glucose} (Reference: {ref.get('glucose')})\n"
            f"- HbA1c: {hba1c} (Reference: {ref.get('hba1c')})\n"
            f"- WBC: {wbc} (Reference: {ref.get('wbc')})\n"
            f"Summary includes literal results only. No clinical diagnosis is provided."
        )
    return "No medical data to summarize."


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
