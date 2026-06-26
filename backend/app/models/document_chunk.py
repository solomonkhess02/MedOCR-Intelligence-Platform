"""
ORM Model — document_chunks table
Chunked text for long-document RAG (lab reports, multi-page invoices).
Each chunk has its own pgvector embedding for chunk-level semantic search.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

from app.database import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ocr_result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ocr_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer, nullable=False
    )  # Positional order of this chunk
    chunk_text: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # ~256-token window with 32-token overlap
    embedding: Mapped[list | None] = mapped_column(
        Vector(768), nullable=True
    )  # pgvector: chunk-level semantic embedding
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    document: Mapped["Document"] = relationship(  # noqa: F821
        "Document", back_populates="document_chunks"
    )
    ocr_result: Mapped["OcrResult"] = relationship(  # noqa: F821
        "OcrResult", back_populates="document_chunks"
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk id={self.id} doc={self.document_id} chunk={self.chunk_index}>"
