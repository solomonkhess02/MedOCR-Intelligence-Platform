"""
MedOCR Intelligence Platform — FastAPI Application Entry Point

Layers managed here:
  - CORS middleware
  - API router registration
  - Startup lifespan (DB pool warm-up, upload dir creation)
  - Health check endpoint
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.config import get_settings
from app.database import async_engine
from app.api.v1.documents import router as documents_router
from app.api.v1.tasks import router as tasks_router

settings = get_settings()
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # ── Startup ────────────────────────────────────────────────────────────────
    logger.info("MedOCR API starting up...")

    # Ensure upload directory exists
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory ready: {settings.upload_dir}")

    # Verify DB connection (will raise on misconfiguration)
    async with async_engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info("Database connection verified.")

    yield  # ← Application runs here

    # ── Shutdown ───────────────────────────────────────────────────────────────
    await async_engine.dispose()
    logger.info("MedOCR API shut down.")


# ── Create FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(
    title="MedOCR Intelligence Platform",
    description=(
        "Multi-Agent Document Intelligence Platform · "
        "OCR · MLflow · RAG · LangGraph"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(documents_router, prefix="/api/v1")
app.include_router(tasks_router, prefix="/api/v1")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health_check():
    """Liveness probe — returns 200 when the API is running."""
    return {
        "status": "ok",
        "service": "MedOCR Intelligence Platform",
        "version": "1.0.0",
        "environment": settings.app_env,
    }


@app.get("/", tags=["health"])
async def root():
    """Root endpoint — redirects to API docs."""
    return {
        "message": "MedOCR Intelligence Platform API",
        "docs": "/docs",
        "health": "/health",
    }
