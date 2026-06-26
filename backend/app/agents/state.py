"""
Agent State Definition
Defines the shared state dictionary / schema used by LangGraph nodes.
"""

from typing import TypedDict, Dict, List, Optional
from uuid import UUID


class DocumentIntelligenceState(TypedDict):
    """
    Schema for the LangGraph orchestrator state.
    Allows agents to accumulate outputs, track status, and trigger safety guardrails.
    """
    document_id: UUID
    doc_type: str                       # 'prescription' | 'lab_report' | 'omr' | 'invoice'
    ocr_confidence: float
    extracted_entities: dict            # Structured entity data from database
    agent_outputs: Dict[str, str]       # Keyed by agent ID, e.g. {'understanding': '...', 'medical_summary': '...'}
    guardrail_blocked: bool             # Set to True if medical safety guardrail triggers
    quality_score: float                # Data-quality completeness score from Agent 06 (0.0–1.0)
    quality_passed: bool                # Whether extracted entities passed quality control
    pii_findings: Dict[str, int]        # PII/PHI categories→counts from Agent 07 (Compliance)
    errors: List[str]                   # Running list of execution errors
