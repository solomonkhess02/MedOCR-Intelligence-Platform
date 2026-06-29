# MedOCR Intelligence Platform

A multi-agent document intelligence platform for medical and financial documents. It
ingests an uploaded image, classifies it, runs the right OCR / document-understanding
model, gates the result on **real model confidence**, embeds it for semantic search,
and then runs a **LangGraph multi-agent pipeline** to summarize, audit, and answer
questions about the document.

The project is built end-to-end with production tooling — async FastAPI, a Celery/Redis
task queue, PostgreSQL + pgvector, MLflow experiment tracking, and Docker Compose — so
it doubles as a hands-on study of Computer Vision + ML + Agentic AI + MLOps.

---

## Demo

[![MedOCR demo walkthrough](https://img.youtube.com/vi/lq-w2N-NXOA/hqdefault.jpg)](https://youtu.be/lq-w2N-NXOA)

A short walkthrough: the imaging dataset, the CV models (TrOCR / Donut / LayoutLMv3 / OpenCV),
the end-to-end pipeline, and the TrOCR → Donut fine-tuning result. *(Click the thumbnail to watch.)*

---

## Architecture

```
                       ┌──────────────────────────────────────────────┐
   Upload (image/pdf)  │  FastAPI  /api/v1/documents                   │
        ───────────────▶  • validate · dedup (SHA-256) · save          │
                       │  • enqueue Celery task ─────────────┐         │
                       └─────────────────────────────────────┼─────────┘
                                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │  Celery worker — process_document             │
                       │  1. classify doc_type (router)                │
                       │  2. route to ML model:                        │
                       │       prescription → TrOCR                    │
                       │       invoice      → Donut                    │
                       │       lab_report   → LayoutLMv3               │
                       │       omr          → OpenCV OMR               │
                       │  3. CONFIDENCE GATE (≥0.75 or needs_review)   │
                       │  4. embed + chunk → pgvector                  │
                       │  5. store OCR result + extracted entities     │
                       └───────────────────┬──────────────────────────┘
                                           ▼ (only if gate passed)
                       ┌──────────────────────────────────────────────┐
                       │  LangGraph Orchestrator (StateGraph)          │
                       │   document_understanding                      │
                       │     └→ quality_control (entity validation)    │
                       │          └→ compliance (PII/PHI redaction)    │
                       │     └─(medical?)→ medical_summary (guardrails)│
                       │            └→ anomaly_detection               │
                       │                  └→ analytics                 │
                       │                        └→ executive_report    │
                       │   + Database Agent (NL→SQL / pgvector RAG)    │
                       └──────────────────────────────────────────────┘

   Cross-cutting:  PostgreSQL+pgvector · Redis · MLflow · DeepSeek V4 Flash (agents)
```

### Why these pieces
- **Confidence gate** — low-confidence OCR never reaches the agent layer; it's flagged
  `needs_review` instead. The gate uses *real* model confidence (geometric mean of
  per-token probabilities via `compute_transition_scores`), not a fabricated score.
- **LangGraph** — the agent pipeline is a `StateGraph` with conditional routing
  (medical vs non-medical) and a medical safety guardrail (regex + LLM auditor) that can
  block unsafe summaries.
- **pgvector RAG** — every document is chunked and embedded (`all-mpnet-base-v2`, 768-d)
  so the Database Agent can answer semantic questions and cite sources.
- **DeepSeek V4 Flash** is the agent-layer LLM, accessed through a single
  `llm_provider.get_llm()` factory (OpenAI-compatible), so the provider is swappable
  from one place.

---

## Tech stack

| Layer            | Tools                                                            |
|------------------|-----------------------------------------------------------------|
| API              | FastAPI (async), Pydantic                                       |
| Task queue       | Celery + Redis (Flower for monitoring)                          |
| Database         | PostgreSQL 16 + pgvector, SQLAlchemy 2.0, Alembic               |
| CV / OCR models  | TrOCR, Donut, LayoutLMv3, OpenCV OMR (HuggingFace Transformers) |
| Embeddings / RAG | sentence-transformers (`all-mpnet-base-v2`)                     |
| Agents           | LangGraph, LangChain, DeepSeek V4 Flash                         |
| MLOps            | MLflow (tracking + registry), Evidently                        |
| Infra            | Docker Compose                                                  |

---

## Quick start

```bash
# 1. Infrastructure (Postgres+pgvector, Redis, MLflow)
docker compose up -d

# 2. Python env
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r backend/requirements.txt

# 3. Config
cp .env.example .env          # then fill in DEEPSEEK_API_KEY

# 4. Database schema
cd backend && alembic upgrade head && cd ..

# 5. Run API + worker (two terminals)
uvicorn app.main:app --reload --app-dir backend
celery -A app.celery_app.celery_config.celery_app worker --loglevel=info --pool=solo   # --app-dir backend
```

API docs: http://localhost:8000/docs · MLflow: http://localhost:5000

---

## Repository layout

```
backend/app/
  api/v1/          FastAPI routes (documents, tasks)
  agents/          LangGraph orchestrator + 8 agents
  ml/              TrOCR / Donut / LayoutLMv3 / OMR inference
  services/        router, confidence gate, embeddings, llm_provider, storage
  models/          SQLAlchemy ORM (documents, ocr_results, chunks, entities, …)
  celery_app/      task queue + process_document pipeline
  schemas/         Pydantic request/response models
  alembic/         database migrations
scripts/           dataset prep, training (TrOCR/Donut), evaluation, MLflow registry
docker-compose.yml infra (Postgres+pgvector, Redis, MLflow)
```

---

## Project status (honest)

This is a learning/portfolio project; not every component is production-grade yet.

| Component                       | Status                                                        |
|---------------------------------|--------------------------------------------------------------|
| API + Celery pipeline           | ✅ Working end-to-end                                         |
| Confidence gate (real scores)   | ✅ Implemented with `compute_transition_scores`              |
| pgvector RAG + Database Agent   | ✅ Working (NL→SQL with SELECT-only safety, + similarity)    |
| LangGraph orchestrator (8 agents) | ✅ Working; medical guardrails enforced                     |
| Quality Control + Compliance/PII agents | ✅ Entity validation + PHI/PII detection & redaction   |
| Evidently drift monitoring      | ✅ `scripts/drift_monitor.py` → HTML drift report           |
| TrOCR (prescriptions)           | 🔧 Fine-tuned, but accuracy still being improved — the target is structured extraction, which is a better fit for a Donut-style model; under active iteration |
| Donut (invoices)                | 🔧 Base model wired in; fine-tune pending                    |
| LayoutLMv3 (lab reports)        | 🔧 Base model wired in; weak-supervision fine-tune pending   |
| OMR (OpenCV)                    | ✅ Working                                                    |

When a model isn't confident or isn't trained, the platform **fails honestly**
(flags `needs_review` / returns empty) rather than fabricating data.
