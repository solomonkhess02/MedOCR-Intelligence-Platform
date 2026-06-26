"""
ORM Model — ocr_results table
Stores OCR model outputs including the pgvector embedding for RAG.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, Text, Float, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

from app.database import Base


class OcrResult(Base):
    __tablename__ = "ocr_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_version: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # e.g. 'trocr-prescription-v4'
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0.0–1.0
    cer: Mapped[float | None] = mapped_column(Float, nullable=True)  # Character Error Rate
    wer: Mapped[float | None] = mapped_column(Float, nullable=True)  # Word Error Rate
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding: Mapped[list | None] = mapped_column(
        Vector(768), nullable=True
    )  # pgvector: full-document semantic embedding
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    document: Mapped["Document"] = relationship(  # noqa: F821
        "Document", back_populates="ocr_results"
    )
    document_chunks: Mapped[list["DocumentChunk"]] = relationship(  # noqa: F821
        "DocumentChunk", back_populates="ocr_result", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<OcrResult id={self.id} doc={self.document_id} conf={self.confidence:.2f}>"
