"""
Celery Tasks — Document Processing Pipeline

Main task: process_document(document_id)

Pipeline:
  1. Load document record from DB
  2. Classify document type (document_router)
  3. Route to appropriate ML model (TrOCR / Donut / LayoutLMv3 / OMR)
  4. Run confidence gate
  5. Store OCR result to ocr_results table
  6. Update documents.status
  7. Log run reference to mlflow_runs table

See: medocr_architecture_v3.md §5 L0 — Ingestion Pipeline
"""

import logging
import uuid
import re
from datetime import datetime, timezone

from celery import Task
from sqlalchemy.orm import Session

from app.celery_app.celery_config import celery_app
from app.config import get_settings
from app.database import SyncSessionLocal
from app.models.document import Document
from app.models.ocr_result import OcrResult
from app.models.extracted_entity import ExtractedEntity
from app.models.document_chunk import DocumentChunk
from app.services.document_router import classify_document
from app.services.confidence_gate import check_confidence
from app.services import embedding_service
from app.agents import run_orchestrator

# ML model imports
from app.ml import trocr_model, donut_model, layoutlm_model, omr_model

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentProcessingTask(Task):
    """Custom Celery Task base class — ensures models are loaded once per worker process."""

    _models_loaded = False

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Mark document as failed on unhandled exception."""
        document_id = args[0] if args else kwargs.get("document_id")
        if document_id:
            _update_document_status(str(document_id), "failed")
        logger.error(f"Task {task_id} failed for document {document_id}: {exc}")

    def _load_models(self):
        """Lazy-load all models on first task execution in this worker process."""
        if not self.__class__._models_loaded:
            logger.info("Loading ML models into worker process...")
            trocr_model.load_model()
            donut_model.load_model()
            layoutlm_model.load_model()
            self.__class__._models_loaded = True
            logger.info("All ML models loaded.")


def _update_document_status(document_id: str, status: str) -> None:
    """Helper: update document status in the database."""
    with SyncSessionLocal() as db:
        doc = db.query(Document).filter(Document.id == uuid.UUID(document_id)).first()
        if doc:
            doc.status = status
            db.commit()


@celery_app.task(
    bind=True,
    base=DocumentProcessingTask,
    name="medocr.process_document",
    max_retries=3,
    default_retry_delay=5,
)
def process_document(self: DocumentProcessingTask, document_id: str) -> dict:
    """
    Main document processing Celery task.

    Args:
        document_id: UUID string of the document to process.

    Returns:
        dict with processing results (stored in Celery result backend).
    """
    logger.info(f"Starting processing for document: {document_id}")

    # ── Step 0: Lazy-load models ───────────────────────────────────────────────
    self._load_models()

    with SyncSessionLocal() as db:
        # ── Step 1: Load document record ───────────────────────────────────────
        doc = db.query(Document).filter(
            Document.id == uuid.UUID(document_id)
        ).first()

        if not doc:
            raise ValueError(f"Document {document_id} not found in database")

        if not doc.source_path:
            raise ValueError(f"Document {document_id} has no source_path")

        # ── Step 2: Update status to 'processing' ──────────────────────────────
        doc.status = "processing"
        db.commit()

        # ── Step 3: Classify document type ─────────────────────────────────────
        doc_type = classify_document(doc.source_path, doc.filename)
        doc.doc_type = doc_type
        db.commit()

        logger.info(f"Document {document_id} classified as: {doc_type}")

        # ── Step 4: Route to ML model ──────────────────────────────────────────
        try:
            raw_text, confidence, cer, wer, latency_ms, model_version, structured_json = _run_ml_pipeline(
                doc_type, doc.source_path
            )
        except Exception as exc:
            logger.error(f"ML pipeline failed for {document_id}: {exc}")
            doc.status = "failed"
            db.commit()
            raise self.retry(exc=exc)

        # ── Step 5: Confidence gate ────────────────────────────────────────────
        passed, new_status = check_confidence(confidence)

        if not passed:
            logger.warning(
                f"Document {document_id} flagged 'needs_review' "
                f"(confidence={confidence:.2f} < threshold={settings.confidence_threshold})"
            )

        # ── Step 6: Store OCR result with embedding ───────────────────────────
        full_embedding = embedding_service.embed_text(raw_text)
        ocr_result = OcrResult(
            document_id=uuid.UUID(document_id),
            model_version=model_version,
            raw_text=raw_text,
            confidence=confidence,
            cer=cer,
            wer=wer,
            latency_ms=latency_ms,
            embedding=full_embedding,
        )
        db.add(ocr_result)
        # Flush so the DB assigns ocr_result.id (its UUID default is applied at
        # flush time). Without this, the chunk FK below is set from a None id and
        # violates the document_chunks.ocr_result_id NOT NULL constraint.
        db.flush()

        # ── Step 6.2: Chunk and Embed Document for RAG ────────────────────────
        chunks = embedding_service.chunk_text(raw_text)
        if chunks:
            chunk_embeddings = embedding_service.embed_chunks(chunks)
            for idx, (chunk_txt, chunk_emb) in enumerate(zip(chunks, chunk_embeddings)):
                doc_chunk = DocumentChunk(
                    document_id=uuid.UUID(document_id),
                    ocr_result_id=ocr_result.id,
                    chunk_index=idx,
                    chunk_text=chunk_txt,
                    embedding=chunk_emb,
                )
                db.add(doc_chunk)

        # ── Step 6.5: Store Extracted Entities ────────────────────────────────
        extracted_entity = ExtractedEntity(
            document_id=uuid.UUID(document_id),
            entity_type=doc_type,
            entity_data=structured_json,
            model_version=model_version,
            confidence=confidence,
        )
        db.add(extracted_entity)
        db.commit()

        # ── Step 7: Update document status ────────────────────────────────────
        doc.status = new_status
        db.commit()
        db.refresh(ocr_result)

        # ── Step 8: Run LangGraph Agent Orchestrator ─────────────────────────
        agent_outputs = {}
        guardrail_blocked = False
        if passed:
            try:
                orchestration_result = run_orchestrator(
                    document_id=uuid.UUID(document_id),
                    doc_type=doc_type,
                    ocr_confidence=confidence,
                    extracted_entities=structured_json
                )
                agent_outputs = orchestration_result.get("agent_outputs", {})
                guardrail_blocked = orchestration_result.get("guardrail_blocked", False)
                if guardrail_blocked:
                    # Update status to needs_review if medical summary got blocked by safety guardrails
                    doc.status = "needs_review"
                    db.commit()
                    new_status = "needs_review"
            except Exception as agent_exc:
                logger.error(f"LangGraph Agent Orchestrator failed for {document_id}: {agent_exc}")

        result = {
            "document_id": document_id,
            "doc_type": doc_type,
            "status": new_status,
            "confidence": confidence,
            "raw_text": raw_text[:200] + "..." if len(raw_text) > 200 else raw_text,
            "model_version": model_version,
            "latency_ms": latency_ms,
            "passed_confidence_gate": passed,
            "agent_outputs": agent_outputs,
            "guardrail_blocked": guardrail_blocked,
        }

        logger.info(
            f"Document {document_id} processed: status={new_status}, "
            f"confidence={confidence:.2f}, latency={latency_ms}ms"
        )
        return result


def _run_ml_pipeline(
    doc_type: str, image_path: str
) -> tuple[str, float, float | None, float | None, int, str, dict]:
    """
    Route to the correct ML model based on doc_type.

    Returns:
        (raw_text, confidence, cer, wer, latency_ms, model_version, structured_json)
    """
    if doc_type == "prescription":
        # Prescriptions are routed to the fine-tuned Donut model: the prescription
        # labels are structured extractions (Donut's domain), and our head-to-head
        # evaluation showed Donut (CER 0.51) beats TrOCR (CER 0.66) here while
        # actually recovering fields. The LLM entity extractor then structures
        # Donut's text output into the prescription schema.
        output = donut_model.run_inference(image_path)
        structured_json = _extract_prescription_entities(output.raw_text)
        return (
            output.raw_text,
            output.confidence,
            None,  # CER/WER computed offline in evaluation, not at serving time
            None,
            output.latency_ms,
            output.model_version,
            structured_json,
        )

    elif doc_type == "invoice":
        output = donut_model.run_inference(image_path)
        return (
            output.raw_text,
            output.confidence,
            None,  # CER/WER not applicable to structured Donut output
            None,
            output.latency_ms,
            output.model_version,
            output.structured_json,
        )

    elif doc_type == "lab_report":
        output = layoutlm_model.run_inference(image_path)
        return (
            output.raw_text,
            output.confidence,
            None,
            None,
            output.latency_ms,
            output.model_version,
            output.structured_json,
        )

    elif doc_type == "omr":
        output = omr_model.run_inference(image_path)
        structured_json = {
            "checked_cells": output.checked_cells,
            "total_checked": len(output.checked_cells),
            "total_marks_detected": output.total_marks_detected,
        }
        return (
            output.raw_text,
            output.confidence,
            None,
            None,
            output.latency_ms,
            output.model_version,
            structured_json,
        )

    else:
        # Unknown document type — default to TrOCR as generic OCR fallback
        logger.warning(f"Unknown doc_type '{doc_type}', falling back to TrOCR")
        output = trocr_model.run_inference(image_path)
        structured_json = _extract_prescription_entities(output.raw_text)
        return (
            output.raw_text,
            output.confidence,
            output.cer,
            output.wer,
            output.latency_ms,
            output.model_version,
            structured_json,
        )


def _extract_prescription_entities(raw_text: str) -> dict:
    """Extract structured fields from prescription raw text."""
    result = {
        "doctor": "Unknown Doctor",
        "patient": "Unknown Patient",
        "medications": [],
        "dosage_instructions": "N/A",
        "date": "N/A",
    }
    
    # Check if we can use the configured LLM to extract structured fields
    from app.services.llm_provider import get_llm, has_llm_api_key
    has_api_key = has_llm_api_key()

    if has_api_key:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            import json

            llm = get_llm(temperature=0.0)

            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a clinical entity extractor. Extract the following fields from the prescription text:\n"
                    "1. doctor: name of the doctor (e.g., 'Dr. Smith')\n"
                    "2. patient: name of the patient (e.g., 'John Doe')\n"
                    "3. medications: a JSON list of medications with strength (e.g. ['Amoxicillin 500mg'])\n"
                    "4. dosage_instructions: instructions (e.g., 'Twice daily after meals')\n"
                    "5. date: date of prescription (YYYY-MM-DD format if possible)\n"
                    "Respond ONLY with a JSON object containing these keys."
                )),
                ("user", "Prescription Text:\n{text}")
            ])
            
            chain = prompt | llm
            resp = chain.invoke({"text": raw_text})
            cleaned = resp.content.strip().replace("```json", "").replace("```", "").strip()
            extracted = json.loads(cleaned)

            # Validate shape before trusting it: the LLM can return valid JSON of the
            # wrong type (a list) or omit keys. Merge only known keys onto the safe
            # defaults so malformed responses can't corrupt downstream entities.
            if not isinstance(extracted, dict):
                raise ValueError(f"LLM returned non-object JSON: {type(extracted).__name__}")
            for key in result:
                if key in extracted and extracted[key] is not None:
                    result[key] = extracted[key]
            if not isinstance(result["medications"], list):
                result["medications"] = [str(result["medications"])]
            return result
        except Exception as e:
            logger.warning(f"LLM prescription entity extraction failed: {e}. Falling back to regex.")

    # Fallback: Regex parsing
    # Doctor name
    doc_match = re.search(r"\b(?:dr\.?|doctor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", raw_text, re.IGNORECASE)
    if doc_match:
        result["doctor"] = f"Dr. {doc_match.group(1).strip()}"
        
    # Patient name
    pat_match = re.search(r"\b(?:patient|patient name|for|mr\.?|mrs\.?|ms\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", raw_text, re.IGNORECASE)
    if pat_match:
        result["patient"] = pat_match.group(1).strip()
        
    # Date
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})\b", raw_text)
    if date_match:
        result["date"] = date_match.group(1)
        
    # Medications & instructions — parse only what is present; never fabricate.
    if "prescribed" in raw_text.lower():
        parts = raw_text.split("prescribed")
        if len(parts) > 1:
            med_part = parts[1].split("take")
            med = med_part[0].strip()
            if med:
                result["medications"] = [med]
            if len(med_part) > 1:
                result["dosage_instructions"] = f"take {med_part[1].strip()}"
    # If nothing parseable, medications stays [] and dosage stays "N/A" (honest).

    return result
