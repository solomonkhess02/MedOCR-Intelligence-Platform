"""
Models package — import all ORM models here so Alembic autogenerate can discover them.
"""

from app.models.document import Document
from app.models.ocr_result import OcrResult
from app.models.document_chunk import DocumentChunk
from app.models.extracted_entity import ExtractedEntity
from app.models.anomaly import Anomaly
from app.models.agent_activity import AgentActivity
from app.models.mlflow_run import MlflowRun

__all__ = [
    "Document",
    "OcrResult",
    "DocumentChunk",
    "ExtractedEntity",
    "Anomaly",
    "AgentActivity",
    "MlflowRun",
]
