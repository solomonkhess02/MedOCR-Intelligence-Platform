"""
ORM Model — documents table
Master record for every uploaded document.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    doc_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # 'prescription' | 'lab_report' | 'omr' | 'invoice'
    file_hash: Mapped[str | None] = mapped_column(
        Text, unique=True, nullable=True
    )  # SHA-256, used for deduplication
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # 'pending' | 'processing' | 'complete' | 'failed' | 'needs_review'
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    source_path: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # Local path or S3 URI

    # ── Relationships ──────────────────────────────────────────────────────────
    ocr_results: Mapped[list["OcrResult"]] = relationship(  # noqa: F821
        "OcrResult", back_populates="document", cascade="all, delete-orphan"
    )
    extracted_entities: Mapped[list["ExtractedEntity"]] = relationship(  # noqa: F821
        "ExtractedEntity", back_populates="document", cascade="all, delete-orphan"
    )
    document_chunks: Mapped[list["DocumentChunk"]] = relationship(  # noqa: F821
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )
    anomalies: Mapped[list["Anomaly"]] = relationship(  # noqa: F821
        "Anomaly", back_populates="document", cascade="all, delete-orphan"
    )
    agent_activities: Mapped[list["AgentActivity"]] = relationship(  # noqa: F821
        "AgentActivity", back_populates="document"
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename} status={self.status}>"
