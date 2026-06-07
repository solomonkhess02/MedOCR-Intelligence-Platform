"""
LangGraph Orchestrator
Sets up the StateGraph to route documents through agent nodes.
Exposes a simple synchronous runner function for the multi-agent pipeline,
and a gateway function for the Database Agent.
"""

import logging
from typing import Dict, Any, List
from uuid import UUID

from langgraph.graph import StateGraph, START, END

from app.agents.state import DocumentIntelligenceState
from app.agents.document_understanding_agent import run_document_understanding_agent
from app.agents.medical_summary_agent import run_medical_summary_agent
from app.agents.anomaly_agent import run_anomaly_agent
from app.agents.analytics_agent import run_analytics_agent
from app.agents.report_agent import run_report_agent
from app.agents.database_agent import run_database_agent

logger = logging.getLogger(__name__)


# ── Node Functions ────────────────────────────────────────────────────────────
def document_understanding_node(
    state: DocumentIntelligenceState
) -> Dict[str, Any]:
    """Invoke the Document Understanding Agent to create a natural language summary."""
    logger.info(f"LangGraph: running document_understanding node for {state['document_id']}")
    try:
        output = run_document_understanding_agent(
            document_id=state["document_id"],
            doc_type=state["doc_type"],
            extracted_entities=state["extracted_entities"]
        )
        
        agent_outputs = dict(state.get("agent_outputs") or {})
        agent_outputs["understanding"] = output
        
        return {"agent_outputs": agent_outputs}
    except Exception as e:
        logger.error(f"Error in document_understanding node: {e}")
        errors = list(state.get("errors") or [])
        errors.append(f"document_understanding node failed: {str(e)}")
        return {"errors": errors}


def medical_summary_node(
    state: DocumentIntelligenceState
) -> Dict[str, Any]:
    """Invoke the Medical Summary Agent (which applies medical safety guardrails)."""
    logger.info(f"LangGraph: running medical_summary node for {state['document_id']}")
    try:
        summary, blocked = run_medical_summary_agent(
            document_id=state["document_id"],
            doc_type=state["doc_type"],
            extracted_entities=state["extracted_entities"]
        )
        
        agent_outputs = dict(state.get("agent_outputs") or {})
        agent_outputs["medical_summary"] = summary
        
        return {
            "agent_outputs": agent_outputs,
            "guardrail_blocked": blocked
        }
    except Exception as e:
        logger.error(f"Error in medical_summary node: {e}")
        errors = list(state.get("errors") or [])
        errors.append(f"medical_summary node failed: {str(e)}")
        return {"errors": errors}


def anomaly_detection_node(
    state: DocumentIntelligenceState
) -> Dict[str, Any]:
    """Invoke the Anomaly Detection Agent (Agent 04) to audit invoices and OMR layouts."""
    logger.info(f"LangGraph: running anomaly_detection node for {state['document_id']}")
    try:
        anomalies = run_anomaly_agent(
            document_id=state["document_id"],
            doc_type=state["doc_type"],
            extracted_entities=state["extracted_entities"]
        )
        
        agent_outputs = dict(state.get("agent_outputs") or {})
        agent_outputs["anomalies"] = f"Detected {len(anomalies)} anomalies. Raw: {str(anomalies)}"
        
        return {"agent_outputs": agent_outputs}
    except Exception as e:
        logger.error(f"Error in anomaly_detection node: {e}")
        errors = list(state.get("errors") or [])
        errors.append(f"anomaly_detection node failed: {str(e)}")
        return {"errors": errors}


def analytics_node(
    state: DocumentIntelligenceState
) -> Dict[str, Any]:
    """Invoke the Analytics Agent (Agent 03) to run platform-wide statistics aggregates."""
    logger.info(f"LangGraph: running analytics node for {state['document_id']}")
    try:
        report = run_analytics_agent(document_id=state["document_id"])
        
        agent_outputs = dict(state.get("agent_outputs") or {})
        agent_outputs["analytics"] = report
        
        return {"agent_outputs": agent_outputs}
    except Exception as e:
        logger.error(f"Error in analytics node: {e}")
        errors = list(state.get("errors") or [])
        errors.append(f"analytics node failed: {str(e)}")
        return {"errors": errors}


def executive_report_node(
    state: DocumentIntelligenceState
) -> Dict[str, Any]:
    """Invoke the Executive Report Agent (Agent 08) to compile historical PDF summaries."""
    logger.info(f"LangGraph: running executive_report node for {state['document_id']}")
    try:
        report_summary = run_report_agent(document_id=state["document_id"])
        
        agent_outputs = dict(state.get("agent_outputs") or {})
        agent_outputs["report"] = report_summary
        
        return {"agent_outputs": agent_outputs}
    except Exception as e:
        logger.error(f"Error in executive_report node: {e}")
        errors = list(state.get("errors") or [])
        errors.append(f"executive_report node failed: {str(e)}")
        return {"errors": errors}


# ── Routing Logic ─────────────────────────────────────────────────────────────
def route_after_understanding(state: DocumentIntelligenceState) -> str:
    """
    Conditional routing:
    - If the document is medical ('prescription' or 'lab_report'), route to medical_summary.
    - Otherwise, bypass medical summary and route directly to anomaly_detection.
    """
    doc_type = state.get("doc_type")
    logger.info(f"LangGraph routing: doc_type='{doc_type}'")
    if doc_type in ["prescription", "lab_report"]:
        return "medical_summary"
    return "anomaly_detection"


# ── Build & Compile the StateGraph ────────────────────────────────────────────
builder = StateGraph(DocumentIntelligenceState)

# Add nodes
builder.add_node("document_understanding", document_understanding_node)
builder.add_node("medical_summary", medical_summary_node)
builder.add_node("anomaly_detection", anomaly_detection_node)
builder.add_node("analytics", analytics_node)
builder.add_node("executive_report", executive_report_node)

# Add edges
builder.add_edge(START, "document_understanding")

# Add conditional edge from understanding to summary/anomaly
builder.add_conditional_edges(
    "document_understanding",
    route_after_understanding,
    {
        "medical_summary": "medical_summary",
        "anomaly_detection": "anomaly_detection"
    }
)

# Continue sequential pipeline
builder.add_edge("medical_summary", "anomaly_detection")
builder.add_edge("anomaly_detection", "analytics")
builder.add_edge("analytics", "executive_report")
builder.add_edge("executive_report", END)

# Compile into a runnable application
orchestrator_graph = builder.compile()


# ── Runner Interface ──────────────────────────────────────────────────────────
def run_orchestrator(
    document_id: UUID,
    doc_type: str,
    ocr_confidence: float,
    extracted_entities: dict
) -> Dict[str, Any]:
    """
    Synchronous entry point to run the agent graph for a document.
    Executes all nodes and returns the final state dictionary.
    """
    initial_state: DocumentIntelligenceState = {
        "document_id": document_id,
        "doc_type": doc_type,
        "ocr_confidence": ocr_confidence,
        "extracted_entities": extracted_entities,
        "agent_outputs": {},
        "guardrail_blocked": False,
        "errors": []
    }

    logger.info(f"Starting LangGraph Orchestration for document {document_id}")
    try:
        final_state = orchestrator_graph.invoke(initial_state)
        logger.info(f"LangGraph Orchestration completed for document {document_id}")
        return final_state
    except Exception as e:
        logger.error(f"LangGraph Orchestration failed for document {document_id}: {e}")
        initial_state["errors"].append(f"Orchestrator invocation failed: {str(e)}")
        return initial_state


# ── Database Agent Natural Language Query Gateway ─────────────────────────────
def query_database_agent(query: str) -> str:
    """
    Natural Language Query Gateway.
    Passes a user query to Agent 02 to retrieve info via SQL, pgvector search, or both.
    """
    return run_database_agent(query)
