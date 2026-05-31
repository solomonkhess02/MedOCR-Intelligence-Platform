"""
API Route — Task Status
GET /api/v1/tasks/{task_id}/status — Poll Celery task status
"""

from fastapi import APIRouter, HTTPException
from celery.result import AsyncResult

from app.celery_app.celery_config import celery_app
from app.schemas.task import TaskStatusResponse

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Poll the status of a Celery document processing task.

    Status values:
      PENDING  — Task queued, not yet picked up by a worker
      STARTED  — Worker has picked up the task
      SUCCESS  — Processing complete
      FAILURE  — Processing failed
      RETRY    — Task is being retried
    """
    result = AsyncResult(task_id, app=celery_app)

    if result.state == "FAILURE":
        return TaskStatusResponse(
            task_id=task_id,
            status="FAILURE",
            error=str(result.result),
        )

    if result.state == "SUCCESS":
        return TaskStatusResponse(
            task_id=task_id,
            status="SUCCESS",
            result=result.result,
            progress=100,
        )

    return TaskStatusResponse(
        task_id=task_id,
        status=result.state,
        progress=50 if result.state == "STARTED" else 0,
    )
