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
from datetime import datetime, timezone

from celery import Task
from sqlalchemy.orm import Session

from app.celery_app.celery_config import celery_app
from app.config import get_settings
from app.database import SyncSessionLocal
from app.models.document import Document
from app.models.ocr_result import OcrResult
from app.services.document_router import classify_document
from app.services.confidence_gate import check_confidence

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
            raw_text, confidence, cer, wer, latency_ms, model_version = _run_ml_pipeline(
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

        # ── Step 6: Store OCR result ───────────────────────────────────────────
        ocr_result = OcrResult(
            document_id=uuid.UUID(document_id),
            model_version=model_version,
            raw_text=raw_text,
            confidence=confidence,
            cer=cer,
            wer=wer,
            latency_ms=latency_ms,
            # embedding will be added in Phase 3 by the embedding sub-task
        )
        db.add(ocr_result)

        # ── Step 7: Update document status ────────────────────────────────────
        doc.status = new_status
        db.commit()
        db.refresh(ocr_result)

        result = {
            "document_id": document_id,
            "doc_type": doc_type,
            "status": new_status,
            "confidence": confidence,
            "raw_text": raw_text[:200] + "..." if len(raw_text) > 200 else raw_text,
            "model_version": model_version,
            "latency_ms": latency_ms,
            "passed_confidence_gate": passed,
        }

        logger.info(
            f"Document {document_id} processed: status={new_status}, "
            f"confidence={confidence:.2f}, latency={latency_ms}ms"
        )
        return result


def _run_ml_pipeline(
    doc_type: str, image_path: str
) -> tuple[str, float, float | None, float | None, int, str]:
    """
    Route to the correct ML model based on doc_type.

    Returns:
        (raw_text, confidence, cer, wer, latency_ms, model_version)
    """
    if doc_type == "prescription":
        output = trocr_model.run_inference(image_path)
        return (
            output.raw_text,
            output.confidence,
            output.cer,
            output.wer,
            output.latency_ms,
            output.model_version,
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
        )

    elif doc_type == "omr":
        output = omr_model.run_inference(image_path)
        return (
            output.raw_text,
            output.confidence,
            None,
            None,
            output.latency_ms,
            output.model_version,
        )

    else:
        # Unknown document type — default to TrOCR as generic OCR fallback
        logger.warning(f"Unknown doc_type '{doc_type}', falling back to TrOCR")
        output = trocr_model.run_inference(image_path)
        return (
            output.raw_text,
            output.confidence,
            output.cer,
            output.wer,
            output.latency_ms,
            output.model_version,
        )
