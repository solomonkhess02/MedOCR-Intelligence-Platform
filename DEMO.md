# MedOCR — Demo Runbook

A tested, step-by-step script for demoing the platform live. Every command here was
verified end-to-end. Follow the order and you won't be surprised.

> **Narrative in one line:** "Upload a medical/financial document → it's classified,
> run through the right CV/OCR model, gated on real confidence, embedded for search,
> then analyzed by an 8-agent LangGraph pipeline — with drift monitoring and MLflow
> tracking around it."

---

## 0. Prerequisites (do this once, before the audience arrives)

- **Docker Desktop running** (whale icon steady, not animating).
- Python venv ready: `.venv` with `pip install -r backend/requirements.txt` done.
- `.env` present at repo root with a real `DEEPSEEK_API_KEY`.
- **Recommended for a clean console:** set `DEBUG=false` in `.env` — this turns off
  SQLAlchemy SQL echo so your terminals aren't flooded with SQL.
- *(Optional)* Tesseract installed if you want the **lab_report** path to produce text.
  Without it, lab reports return empty and are flagged `needs_review` (honest behavior).

---

## 1. Start the stack (3 terminals)

**Terminal 1 — infrastructure**
```bash
docker compose up -d
# wait ~20s for postgres healthcheck, then create the schema:
cd backend && alembic upgrade head && cd ..
```

**Terminal 2 — API**
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir backend
```

**Terminal 3 — Celery worker** (Windows needs `--pool=solo`)
```bash
cd backend
celery -A app.celery_app.celery_config.celery_app worker --loglevel=info --pool=solo
```

Sanity check (Terminal 1):
```bash
curl http://127.0.0.1:8000/health        # -> {"status":"ok",...}
```
Open these tabs: **API docs** http://localhost:8000/docs · **MLflow** http://localhost:5000

---

## 2. Demo flow

### (a) Live upload → full pipeline → 8 agents  *(the headline — this runs end-to-end live)*
Upload a prescription via the Swagger UI (`POST /api/v1/documents`) or:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents -F "file=@temp_test/demo_prescription.png;type=image/png"
# copy the task_id, then poll:
curl http://127.0.0.1:8000/api/v1/tasks/<TASK_ID>/status
```
**What happens:** classified as `prescription` → routed to the **fine-tuned Donut** model →
scored with *real* confidence (geometric mean of per-token probs) ≈ **0.90** → **passes the
0.75 gate** → status `complete` → the **8-agent LangGraph pipeline runs**:
- **document_understanding** — plain-English summary
- **quality_control** — completeness score + LLM plausibility (flags missing fields / a
  hallucinated medication — a great safety beat)
- **compliance** — detects & redacts PHI (patient/doctor names), severity high
- **medical_summary** — guardrailed clinical summary
- **anomaly_detection**, **analytics**, **executive_report** (PDF)
- plus the **Database Agent** (NL→SQL + RAG).

**Talking points:** (1) the confidence gate is a clinical-safety valve — low-confidence reads
are flagged `needs_review` and skip the agents; (2) the extraction is imperfect, and the
quality-control + compliance agents *catch* the problems — defense in depth.

> Try an **invoice** (`temp_test/demo_invoice.png`) too: its confidence is lower, so it gets
> gated to `needs_review` — a clean way to *show the gate working* both ways.

### (b) Backup: deterministic agent showcase on clean data
If you want a guaranteed-clean run (e.g. weak Wi-Fi to the LLM, or to show ideal output):
```bash
python scripts/demo_agents.py
```
Seeds one clean invoice and runs the same orchestrator — useful as a fallback.

### (c) Database Agent — natural language → SQL + RAG
Already shown at the end of `demo_agents.py`. To run ad hoc:
```bash
python -c "import sys; sys.path.insert(0,'backend'); from app.agents import query_database_agent; print(query_database_agent('How many invoices are there?'))"
```

### (d) Drift monitoring (Evidently)
```bash
python scripts/drift_monitor.py
# then open results/drift_report.html in a browser
```
**Talking point:** monitors OCR confidence/latency/text-length/doc-type distributions
for drift between a reference and current window. (Uses synthetic demo data until the
DB has ≥20 processed documents.)

### (e) MLflow — training history
Open http://localhost:5000 → **TrOCR-FineTuning** experiment → show the CER/WER curves
and the registered **TrOCR-Prescription** model in the Model Registry.

---

## 3. Troubleshooting (issues we already hit + fixes)

| Symptom | Cause | Fix |
|---|---|---|
| `docker ps` fails / DB connection refused | Docker Desktop not started | Launch Docker Desktop, wait for steady whale icon |
| Worker: `Tesseract not installed` on lab reports | No OCR backend | Expected; demo invoices/prescriptions, or install Tesseract |
| Console flooded with SQL | `DEBUG=true` | Set `DEBUG=false` in `.env`, restart API/worker |
| `UnicodeEncodeError` printing | Windows cp1252 console | Scripts force UTF-8; if running your own, `set PYTHONIOENCODING=utf-8` |
| Upload returns 409 | Same file already uploaded (SHA-256 dedup) | Use a different file, or `TRUNCATE` the data tables |

Reset the data tables for a clean run:
```bash
docker exec medocr_postgres psql -U medocr_user -d medocr_db -c "TRUNCATE documents, ocr_results, document_chunks, extracted_entities, anomalies, agent_activity RESTART IDENTITY CASCADE;"
```

---

## 4. Shutdown
```bash
# Ctrl+C the API and worker terminals, then:
docker compose down          # add -v to also wipe the database volume
```
