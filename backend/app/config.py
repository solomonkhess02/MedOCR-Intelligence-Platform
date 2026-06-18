"""
MedOCR Intelligence Platform — Application Configuration
Reads all settings from environment variables / .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

# Resolve the .env path absolutely so settings load regardless of the process's
# working directory (uvicorn/celery are often launched from backend/, not the repo
# root where .env lives). config.py is at backend/app/config.py → parents[2] = repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILES = (_PROJECT_ROOT / ".env", _PROJECT_ROOT / "backend" / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILES, extra="ignore")

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

    # ── LLM Provider (DeepSeek — agent layer) ─────────────
    # DeepSeek is OpenAI-compatible; the agent layer talks to it via langchain_openai.
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com"

    # ── Google Gemini (vision-OCR fallback only) ──────────
    # Retained solely for the LayoutLMv3 bounding-box vision fallback,
    # which DeepSeek (text-only) cannot perform.
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # ── OCR (LayoutLMv3 lab-report pipeline) ──────────────
    # Absolute path to the Tesseract binary. On Windows it is typically
    # "C:/Program Files/Tesseract-OCR/tesseract.exe". Leave blank to rely on PATH.
    # In Docker, tesseract-ocr is installed system-wide so PATH works and this stays blank.
    tesseract_cmd: str = ""

    # ── Storage ──────────────────────────────────────────
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50

    # ── Confidence Gate ───────────────────────────────────
    confidence_threshold: float = 0.75


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — import this everywhere."""
    return Settings()
