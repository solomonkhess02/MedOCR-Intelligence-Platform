"""
ORM Model — extracted_entities table
Structured JSON extraction results per document (JSONB for schema flexibility).
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, Text, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.database import Base


class ExtractedEntity(Base):
    __tablename__ = "extracted_entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # 'prescription' | 'lab_report' | 'omr' | 'invoice'
    entity_data: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict
    )  # Full structured ML extraction output
    model_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    document: Mapped["Document"] = relationship(  # noqa: F821
        "Document", back_populates="extracted_entities"
    )

    def __repr__(self) -> str:
        return f"<ExtractedEntity id={self.id} type={self.entity_type} doc={self.document_id}>"
