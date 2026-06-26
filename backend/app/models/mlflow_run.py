"""
ORM Model — mlflow_runs table
References to MLflow experiment runs, read by the MLOps Agent (Agent-07).
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.database import Base


class MlflowRun(Base):
    __tablename__ = "mlflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[str] = mapped_column(
        Text, unique=True, nullable=False
    )  # MLflow run ID
    experiment_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict
    )  # CER, WER, F1, latency, etc.
    params: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    stage: Mapped[str | None] = mapped_column(
        String(30), nullable=True
    )  # 'development' | 'staging' | 'production' | 'archived'
    logged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:
        return f"<MlflowRun id={self.id} run_id={self.run_id} model={self.model_name}>"
