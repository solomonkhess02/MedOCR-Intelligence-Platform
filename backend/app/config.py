"""
MedOCR Intelligence Platform — Application Configuration
Reads all settings from environment variables / .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ──────────────────────────────────────────────
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "change-me-in-production"
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    # ── Database ─────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://medocr_user:medocr_pass@localhost:5432/medocr_db"
    sync_database_url: str = "postgresql://medocr_user:medocr_pass@localhost:5432/medocr_db"

    # ── Redis / Celery ────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ── MLflow ───────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"

    # ── Google Gemini ─────────────────────────────────────
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # ── Storage ──────────────────────────────────────────
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50

    # ── Confidence Gate ───────────────────────────────────
    confidence_threshold: float = 0.75


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — import this everywhere."""
    return Settings()
