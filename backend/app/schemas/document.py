"""
Pydantic schemas for document-related API request/response bodies.
"""

import uuid
from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class DocumentUploadResponse(BaseModel):
    """Returned immediately on POST /api/v1/documents."""
    task_id: str
    document_id: uuid.UUID
    filename: str
    message: str = "Document queued for processing"


class DocumentStatusResponse(BaseModel):
    """Document record with current processing status."""
    id: uuid.UUID
    filename: str
    doc_type: Optional[str]
    status: str
    uploaded_at: datetime
    source_path: Optional[str]

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""
    items: list[DocumentStatusResponse]
    total: int
    page: int
    page_size: int
