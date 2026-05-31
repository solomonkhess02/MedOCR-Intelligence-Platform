"""
Pydantic schemas for Celery task status polling.
"""

from pydantic import BaseModel
from typing import Optional, Any


class TaskStatusResponse(BaseModel):
    """Returned by GET /api/v1/tasks/{task_id}/status"""
    task_id: str
    status: str          # 'PENDING' | 'STARTED' | 'SUCCESS' | 'FAILURE' | 'RETRY'
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: Optional[int] = None   # 0–100 percent
