"""
Demo helper — seed one clean document and run the full 8-agent pipeline on it.

Why this exists:
  The OCR models (TrOCR/Donut/LayoutLMv3) are still being improved, so real uploads
  often score below the confidence gate (0.75) and are correctly flagged
  'needs_review' — which means the agent layer is skipped. That gate is a feature,
  but it makes the *agentic AI* hard to show live.

  This script seeds ONE representative high-confidence extraction (as a properly
  trained model would produce) directly into the database, then runs the real
  LangGraph orchestrator on it with the real LLM. It demonstrates all 8 agents end
  to end and also gives the Database Agent + drift monitor real data to work with.

Usage:
    .venv\\Scripts\\python.exe scripts\\demo_agents.py
"""

import sys
import uuid
from pathlib import Path

# Windows consoles default to cp1252 and crash on any non-Latin-1 char the LLM may
# emit (em-dashes, bullets, emoji). Force UTF-8 output so the demo never dies on a print.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Make `app.*` importable (this script lives in scripts/, app lives in backend/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.database import SyncSessionLocal
from app.models.document import Document
from app.models.ocr_result import OcrResult
from app.models.extracted_entity import ExtractedEntity
from app.models.document_chunk import DocumentChunk
from app.services import embedding_service
from app.agents import run_orchestrator, query_database_agent


CLEAN_INVOICE = {
    "vendor": "Acme Medical Supplies",
    "amount": "15400.00",
    "invoice_no": "INV-2026-0042",
    "date": "2026-06-15",
    "email": "billing@acme.com",
    "phone": "555-123-4567",
}
RAW_TEXT = (
    "INVOICE  Acme Medical Supplies  Invoice No: INV-2026-0042  Date: 2026-06-15  "
    "Bill To: Riverside Clinic  Item: Disposable Syringes (x500)  Amount Due: 15400.00  "
    "Contact: billing@acme.com  555-123-4567"
)


def seed_document() -> uuid.UUID:
    """Insert a clean invoice (document + ocr_result + chunks + entity). Returns doc id."""
    doc_id = uuid.uuid4()
    embedding = embedding_service.embed_text(RAW_TEXT)
    chunks = embedding_service.chunk_text(RAW_TEXT)
    chunk_embeddings = embedding_service.embed_chunks(chunks)

    with SyncSessionLocal() as db:
        db.add(Document(
            id=doc_id,
            filename=f"demo_invoice_{doc_id.hex[:8]}.png",
            file_hash=doc_id.hex,            # unique per run so dedup never blocks
            status="complete",
            source_path="(demo-seeded)",
            doc_type="invoice",
        ))
        ocr = OcrResult(
            document_id=doc_id,
            model_version="donut-invoice-demo",
            raw_text=RAW_TEXT,
            confidence=0.95,
            latency_ms=120,
            embedding=embedding,
        )
        db.add(ocr)
        db.flush()  # assign ocr.id before chunks reference it
        for idx, (txt, emb) in enumerate(zip(chunks, chunk_embeddings)):
            db.add(DocumentChunk(
                document_id=doc_id, ocr_result_id=ocr.id,
                chunk_index=idx, chunk_text=txt, embedding=emb,
            ))
        db.add(ExtractedEntity(
            document_id=doc_id, entity_type="invoice",
            entity_data=CLEAN_INVOICE, model_version="donut-invoice-demo", confidence=0.95,
        ))
        db.commit()
    return doc_id


def main() -> int:
    print("=" * 64)
    print("  MedOCR - 8-Agent Pipeline Demo (clean seeded invoice)")
    print("=" * 64)

    print("\n[1/3] Seeding a clean invoice into the database ...")
    doc_id = seed_document()
    print(f"      document_id = {doc_id}")

    print("\n[2/3] Running the LangGraph orchestrator (real LLM) ...\n")
    result = run_orchestrator(
        document_id=doc_id,
        doc_type="invoice",
        ocr_confidence=0.95,
        extracted_entities=CLEAN_INVOICE,
    )
    for name, output in result.get("agent_outputs", {}).items():
        print(f"  -- agent: {name} " + "-" * (48 - len(name)))
        print("   " + str(output)[:400].replace("\n", "\n   "))
        print()
    print(f"  quality_passed={result.get('quality_passed')}  "
          f"pii_findings={result.get('pii_findings')}  "
          f"guardrail_blocked={result.get('guardrail_blocked')}")
    if result.get("errors"):
        print(f"  errors={result['errors']}")

    print("\n[3/3] Database Agent — natural-language queries ...\n")
    for q in ["How many invoices are there?",
              "What is the total invoice amount on the platform?"]:
        print(f"  Q: {q}")
        print("  A: " + query_database_agent(q)[:300].replace("\n", " "))
        print()

    print("=" * 64)
    print("  Demo complete. Document, OCR result, chunks, entity, anomalies,")
    print("  and agent_activity rows are now in the database.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
