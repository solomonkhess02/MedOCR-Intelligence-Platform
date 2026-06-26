"""
ORM Model — agent_activity table
Per-invocation log for every agent call: token counts, latency, LLM model, status.
Used for cost tracking, observability, and guardrail audit trail.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, Text, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class AgentActivity(Base):
    __tablename__ = "agent_activity"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    agent_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )  # e.g. 'document_understanding_agent' | 'medical_summary_agent'
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_model: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # e.g. 'gemini-2.0-flash'
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="success"
    )  # 'success' | 'failed' | 'guardrail_blocked'
    invoked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    document: Mapped["Document"] = relationship(  # noqa: F821
        "Document", back_populates="agent_activities"
    )

    def __repr__(self) -> str:
        return f"<AgentActivity id={self.id} agent={self.agent_id} status={self.status}>"
