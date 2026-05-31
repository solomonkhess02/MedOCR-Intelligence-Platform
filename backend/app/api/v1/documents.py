"""
API Route — Documents
POST /api/v1/documents — Upload a document, get task_id back immediately
GET  /api/v1/documents — List all documents (paginated)
GET  /api/v1/documents/{document_id} — Get single document with OCR results
"""

import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.document import Document
from app.models.ocr_result import OcrResult
from app.schemas.document import (
    DocumentUploadResponse,
    DocumentStatusResponse,
    DocumentListResponse,
)
from app.services.storage import save_upload_file, get_max_size_bytes
from app.celery_app.tasks import process_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/tiff",
    "image/bmp", "image/webp", "application/pdf",
}


@router.post("", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document for processing.
    Returns task_id immediately — use /tasks/{task_id}/status to poll progress.
    """
    # ── Validate MIME type ──────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {', '.join(ALLOWED_MIME_TYPES)}",
        )

    # ── Validate file size ─────────────────────────────────────────────────────
    max_bytes = get_max_size_bytes()
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {max_bytes // (1024*1024)}MB",
        )
    # Reset file pointer after reading
    await file.seek(0)

    # ── Save file + compute hash ───────────────────────────────────────────────
    saved_path, file_hash = await save_upload_file(file)

    # ── Check for duplicate (SHA-256 dedup) ────────────────────────────────────
    existing = await db.execute(
        select(Document).where(Document.file_hash == file_hash)
    )
    existing_doc = existing.scalar_one_or_none()
    if existing_doc:
        logger.info(f"Duplicate file detected: {file.filename} matches {existing_doc.id}")
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Duplicate file detected (same SHA-256 hash already processed).",
                "existing_document_id": str(existing_doc.id),
                "existing_status": existing_doc.status,
            },
        )

    # ── Create document record ─────────────────────────────────────────────────
    doc = Document(
        filename=file.filename or "unknown",
        file_hash=file_hash,
        status="pending",
        source_path=saved_path,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # ── Dispatch Celery task ───────────────────────────────────────────────────
    task = process_document.delay(str(doc.id))

    logger.info(f"Document {doc.id} queued for processing. Task: {task.id}")

    return DocumentUploadResponse(
        task_id=task.id,
        document_id=doc.id,
        filename=doc.filename,
        message="Document queued for processing",
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    doc_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """List documents with optional filtering by doc_type and status."""
    query = select(Document).order_by(Document.uploaded_at.desc())

    if doc_type:
        query = query.where(Document.doc_type == doc_type)
    if status:
        query = query.where(Document.status == status)

    # Count total
    count_result = await db.execute(
        select(func.count()).select_from(query.subquery())
    )
    total = count_result.scalar_one()

    # Paginate
    offset = (page - 1) * page_size
    result = await db.execute(query.offset(offset).limit(page_size))
    docs = result.scalars().all()

    return DocumentListResponse(
        items=[DocumentStatusResponse.model_validate(d) for d in docs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a single document by ID."""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse.model_validate(doc)
