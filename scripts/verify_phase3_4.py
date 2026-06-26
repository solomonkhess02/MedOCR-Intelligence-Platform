"""
Verification Script — Phase 3 & 4: RAG Foundation + Analytics Agents
Tests:
  - 768-dimensional embedding generation (all-mpnet-base-v2)
  - Text chunking & vector search
  - Database Agent SQL/RAG intent routing & SELECT-only safety gate
  - Anomaly Detection Agent outlier & duplicate audits
  - Executive Report Agent PDF compilation using ReportLab

Usage:
    python scripts/verify_phase3_4.py
"""

import os
import sys
import uuid
import re
from pathlib import Path

# Setup path so we can import from backend app
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "backend"))

# Set env vars for testing
os.environ["APP_ENV"] = "testing"
os.environ["DEBUG"] = "true"
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "2"

from app.config import get_settings
from sqlalchemy import text
from app.database import SyncSessionLocal
from app.services import embedding_service
from app.agents.database_agent import run_database_agent, _is_safe_select_query, _classify_intent
from app.agents.anomaly_agent import run_anomaly_agent
from app.agents.report_agent import run_report_agent
from app.models.extracted_entity import ExtractedEntity
from app.models.anomaly import Anomaly

settings = get_settings()


def test_embedding_service() -> None:
    """Verifies embedding dimensions and text chunking parameters."""
    print("\n--- 1. Testing Embedding Service ---")
    test_phrase = "Glucose level was measured at 180 mg/dL in patient lab report."
    
    # ── Embedding Dimension check ─────────────────────────────────────────────
    print("[*] Generating text embedding...")
    vector = embedding_service.embed_text(test_phrase)
    dim = len(vector)
    print(f"   Embedding dimension: {dim} (Expected: 768)")
    assert dim == 768, f"Dimension mismatch: got {dim}, expected 768. Check pgvector schema."
    print("[OK] Embedding dimensions are correct.")

    # ── Chunking validation ──────────────────────────────────────────────────
    print("[*] Testing chunking text...")
    long_text = " ".join([f"word{i}" for i in range(500)])  # 500 words
    chunks = embedding_service.chunk_text(long_text, chunk_size=200, overlap=30)
    print(f"   Created {len(chunks)} chunks from 500 words.")
    assert len(chunks) == 3, f"Unexpected chunk count: {len(chunks)}"
    print("[OK] Text chunking algorithm working correctly.")


def test_database_agent_sql_safety() -> None:
    """Verifies that the SQL parser blocks dangerous modifying queries."""
    print("\n--- 2. Testing Database Agent SQL Safety Gate ---")
    
    safe_queries = [
        "SELECT count(*) FROM documents;",
        "SELECT entity_data->>'vendor', (entity_data->>'amount')::float FROM extracted_entities WHERE entity_type = 'invoice';",
        "SELECT filename, status FROM documents WHERE doc_type = 'prescription' ORDER BY uploaded_at DESC LIMIT 5;"
    ]
    
    unsafe_queries = [
        "DELETE FROM documents;",
        "DROP TABLE ocr_results;",
        "UPDATE documents SET status = 'complete';",
        "SELECT count(*) FROM documents; DROP TABLE ocr_results;",
        "INSERT INTO anomalies (anomaly_type, severity) VALUES ('fake', 'high');"
    ]
    
    for sq in safe_queries:
        is_safe = _is_safe_select_query(sq)
        print(f"   Safe query check: [{is_safe}] for: \"{sq}\"")
        assert is_safe, f"Safe query was blocked: {sq}"
        
    for uq in unsafe_queries:
        is_safe = _is_safe_select_query(uq)
        print(f"   Unsafe query check: [{is_safe}] for: \"{uq}\"")
        assert not is_safe, f"Unsafe query was permitted: {uq}"
        
    print("[OK] SQL Safety Gate successfully blocks modifying / injection statements.")


def test_database_agent_intent_routing() -> None:
    """Tests the Database Agent query intent classifier."""
    print("\n--- 3. Testing Database Agent Intent Routing ---")
    
    sql_query = "What is the total invoice spend from MedSupply Corp?"
    rag_query = "Are there any reports describing patient glucose or diabetic conditions?"
    hybrid_query = "Show count of invoices mentioning Metformin medication."
    
    has_api_key = settings.google_api_key and "your-gemini-api" not in settings.google_api_key
    llm_model = settings.gemini_model or "gemini-2.0-flash"
    
    intent_sql = _classify_intent(sql_query, has_api_key, llm_model)
    intent_rag = _classify_intent(rag_query, has_api_key, llm_model)
    intent_hybrid = _classify_intent(hybrid_query, has_api_key, llm_model)
    
    print(f"   Query: \"{sql_query}\" -> Intent: {intent_sql} (Expected: sql or hybrid)")
    print(f"   Query: \"{rag_query}\" -> Intent: {intent_rag} (Expected: rag or hybrid)")
    print(f"   Query: \"{hybrid_query}\" -> Intent: {intent_hybrid} (Expected: hybrid or sql/rag)")
    
    assert intent_sql in ["sql", "hybrid"]
    assert intent_rag in ["rag", "hybrid"]
    print("[OK] Intent classification correctly routes query classes.")


def test_anomaly_agent_detection() -> None:
    """Tests Anomaly Detection Agent against database invoice metrics."""
    print("\n--- 4. Testing Anomaly Detection Agent ---")
    doc_id_1 = uuid.uuid4()
    doc_id_2 = uuid.uuid4()
    
    with SyncSessionLocal() as db:
        # Clear existing test extractions
        db.execute(text("DELETE FROM extracted_entities WHERE entity_type = 'invoice';"))
        db.execute(text("DELETE FROM anomalies;"))
        db.commit()
        
        # Insert historical invoices for statistical mean/stddev
        inv1 = ExtractedEntity(
            id=uuid.uuid4(),
            document_id=doc_id_1,
            entity_type="invoice",
            entity_data={"vendor": "MedSupply Corp", "amount": 10000.0, "invoice_no": "INV-101"}
        )
        inv2 = ExtractedEntity(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            entity_type="invoice",
            entity_data={"vendor": "MedSupply Corp", "amount": 12000.0, "invoice_no": "INV-102"}
        )
        inv3 = ExtractedEntity(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            entity_type="invoice",
            entity_data={"vendor": "Global Pharma", "amount": 11000.0, "invoice_no": "INV-103"}
        )
        db.add_all([inv1, inv2, inv3])
        db.commit()

    # Case A: Duplicate Invoice detection
    print("[*] Case A: Testing Duplicate Invoice Number Audit...")
    duplicate_entities = {
        "vendor": "MedSupply Corp",
        "amount": 9000.0,
        "invoice_no": "INV-101"
    }
    
    anoms = run_anomaly_agent(doc_id_2, "invoice", duplicate_entities)
    print(f"   Flagged anomalies: {anoms}")
    assert any(a["anomaly_type"] == "duplicate_invoice" for a in anoms)
    print("[OK] Case A Successful: Duplicate invoice number flagged.")

    # Case B: Outlier Amount detection (> mean + 3σ)
    print("\n[*] Case B: Testing Invoice Cost Outlier Audit...")
    outlier_entities = {
        "vendor": "MedSupply Corp",
        "amount": 90000.0,
        "invoice_no": "INV-200"
    }
    
    anoms_b = run_anomaly_agent(uuid.uuid4(), "invoice", outlier_entities)
    print(f"   Flagged anomalies: {anoms_b}")
    assert any(a["anomaly_type"] == "amount_outlier" for a in anoms_b)
    print("[OK] Case B Successful: Outlier invoice amount flagged.")


def test_executive_report_generation() -> None:
    """Verifies Report Agent pdf compiling."""
    print("\n--- 5. Testing Executive Report Agent PDF Compilation ---")
    doc_id = uuid.uuid4()
    
    report_summary = run_report_agent(doc_id)
    print(f"   Report Agent Summary output:\n{report_summary}")
    
    # Check if file was created
    results_dir = PROJECT_ROOT / "results"
    pdf_path = results_dir / f"executive_report_{doc_id}.pdf"
    
    print(f"   Checking file path: {pdf_path}")
    assert pdf_path.exists(), f"Executive Report PDF was not generated!"
    file_size = pdf_path.stat().st_size
    print(f"   PDF File Size: {file_size} bytes")
    assert file_size > 100, "Generated PDF is empty / corrupt!"
    print("[OK] Report Agent successfully compiled the PDF executive report.")

    # Cleanup temp pdf
    try:
        os.remove(pdf_path)
    except Exception:
        pass


def main():
    print("==================================================")
    print("  MedOCR - Verification of Phase 3 & 4 Implementation")
    print("==================================================")
    
    try:
        test_embedding_service()
        test_database_agent_sql_safety()
        test_database_agent_intent_routing()
        test_anomaly_agent_detection()
        test_executive_report_generation()
        print("\n[SUCCESS] ALL PHASE 3 & 4 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\n[FAIL] VERIFICATION TEST SUITE FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
