"""
Celery Application Configuration
"""

from celery import Celery
from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "medocr",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.celery_app.tasks"],
)

celery_app.conf.update(
    # ── Serialization ──────────────────────────────────────────────────────────
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # ── Timezone ───────────────────────────────────────────────────────────────
    timezone="UTC",
    enable_utc=True,
    # ── Task behavior ─────────────────────────────────────────────────────────
    task_track_started=True,
    task_acks_late=True,           # Only ack after task completes (prevents loss)
    worker_prefetch_multiplier=1,  # One task at a time per worker (ML models are heavy)
    task_soft_time_limit=120,      # 2 min soft limit (warn)
    task_time_limit=180,           # 3 min hard limit (kill)
    # ── Result expiry ─────────────────────────────────────────────────────────
    result_expires=3600,           # Keep results for 1 hour
    # ── Retry defaults ────────────────────────────────────────────────────────
    task_max_retries=3,
    task_default_retry_delay=5,    # 5 seconds between retries
)
