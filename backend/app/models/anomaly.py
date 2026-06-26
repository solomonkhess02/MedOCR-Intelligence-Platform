"""
ORM Model — anomalies table
Anomalies detected by Agent-04 (Anomaly Detection Agent).
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, Text, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class Anomaly(Base):
    __tablename__ = "anomalies"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    anomaly_type: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # 'duplicate_invoice' | 'amount_outlier' | 'vendor_mismatch' | 'omr_inconsistency'
    severity: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True
    )  # 'low' | 'medium' | 'high' | 'critical'
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    document: Mapped["Document"] = relationship(  # noqa: F821
        "Document", back_populates="anomalies"
    )

    def __repr__(self) -> str:
        return f"<Anomaly id={self.id} type={self.anomaly_type} severity={self.severity}>"
