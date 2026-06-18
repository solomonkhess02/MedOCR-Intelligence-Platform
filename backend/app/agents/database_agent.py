"""
Agent 02 — Database Agent
Interprets natural language queries, routes them to SQL or pgvector RAG search (or both),
and synthesizes answers with source citations. Enforces SELECT-only safety.
"""

import time
import logging
import json
import re
from typing import Dict, Any, List, Tuple, Optional
from uuid import UUID

import sqlparse
from sqlalchemy import text
from langchain_core.prompts import ChatPromptTemplate

from app.config import get_settings
from app.database import SyncSessionLocal
from app.models.agent_activity import AgentActivity
from app.services import embedding_service
from app.services.llm_provider import get_llm, has_llm_api_key, get_model_name

logger = logging.getLogger(__name__)
settings = get_settings()

AGENT_ID = "database_agent"

# Database Schema details given to LLM for SQL generation
DB_SCHEMA_PROMPT = """
You have the following tables:

1. documents:
   - id: UUID PRIMARY KEY
   - filename: TEXT
   - doc_type: TEXT ('prescription' | 'lab_report' | 'omr' | 'invoice')
   - status: TEXT ('pending' | 'processing' | 'complete' | 'failed' | 'needs_review')
   - uploaded_at: TIMESTAMPTZ

2. extracted_entities:
   - id: UUID PRIMARY KEY
   - document_id: UUID (foreign key to documents.id)
   - entity_type: TEXT
   - entity_data: JSONB (contains extracted fields like:
       - for invoice: {"vendor": "Name", "amount": 15000, "invoice_no": "INV-1", "date": "2026-06-01"}
       - for prescription: {"doctor": "Dr. X", "patient": "John", "medications": ["Med1"], "dosage_instructions": "X"}
       - for lab_report: {"patient": "Jane", "glucose": "180 mg/dL", "hba1c": "8.1%", "wbc": "11.2 K/µL"}
       - for omr: {"checked_cells": [...], "total_checked": 5}
     )

3. ocr_results:
   - id: UUID PRIMARY KEY
   - document_id: UUID
   - raw_text: TEXT
   - confidence: FLOAT

4. anomalies:
   - id: UUID PRIMARY KEY
   - document_id: UUID
   - anomaly_type: TEXT
   - severity: TEXT ('low' | 'medium' | 'high' | 'critical')
   - reasoning: TEXT
"""


def run_database_agent(user_query: str) -> str:
    """
    Main entry point for Agent 02: Database Agent.
    Routes queries to SQL, pgvector similarity search, or hybrid search.
    """
    start_time = time.perf_counter()
    llm_model = get_model_name()
    prompt_tokens = 0
    completion_tokens = 0
    status = "success"

    # Check for valid LLM API key
    has_api_key = has_llm_api_key()

    # ── Step 1: Intent Classification ─────────────────────────────────────────
    intent = _classify_intent(user_query, has_api_key, llm_model)
    logger.info(f"Database Agent: classified intent for '{user_query}' as '{intent}'")

    sql_results: List[Dict[str, Any]] = []
    rag_results: List[Dict[str, Any]] = []
    sql_query_used = ""

    # ── Step 2: SQL Execution (if SQL or Hybrid) ─────────────────────────────
    if intent in ["sql", "hybrid"]:
        generated_sql = _generate_sql(user_query, has_api_key, llm_model)
        if generated_sql:
            sql_query_used = generated_sql
            # Safety Gate check:
            if _is_safe_select_query(generated_sql):
                sql_results = _execute_sql_query(generated_sql)
            else:
                logger.warning(f"SQL safety check blocked query: {generated_sql}")
                sql_results = [{"error": "Security Block: SQL statement was not a read-only SELECT query."}]
                status = "failed"

    # ── Step 3: pgvector Similarity Search (if RAG or Hybrid) ─────────────────
    if intent in ["rag", "hybrid"]:
        rag_results = _execute_similarity_search(user_query)

    # ── Step 4: Synthesize Final Response ────────────────────────────────────
    response_text = _synthesize_response(
        query=user_query,
        intent=intent,
        sql_results=sql_results,
        rag_results=rag_results,
        sql_query=sql_query_used,
        has_api_key=has_api_key,
        llm_model=llm_model
    )

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    # ── Log Activity to Database ─────────────────────────────────────────────
    _log_agent_activity(
        agent_id=AGENT_ID,
        document_id=None,
        prompt_tokens=prompt_tokens or (len(user_query) // 4),
        completion_tokens=len(response_text) // 4,
        latency_ms=latency_ms,
        llm_model=llm_model if has_api_key else f"{llm_model}-simulated",
        status=status
    )

    return response_text


def _classify_intent(query: str, has_api_key: bool, llm_model: str) -> str:
    """Classifies user query as 'sql', 'rag', or 'hybrid'."""
    if not has_api_key:
        # Rule-based fallback classification
        q_lower = query.lower()
        if any(w in q_lower for w in ["sum", "average", "count", "invoice", "vendor", "how many", "total"]):
            if any(w in q_lower for w in ["similar", "mentioning", "about", "describe"]):
                return "hybrid"
            return "sql"
        return "rag"

    try:
        llm = get_llm(temperature=0.0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an intent classifier. Categorize the user's database query into one of three intents:\n"
                "- 'sql': Queries requesting exact filters, structured sums, averages, lists of vendors, or counts.\n"
                "- 'rag': Queries seeking semantic meaning, similarities, mentions of topics/medications, or open-ended document lookup.\n"
                "- 'hybrid': Queries combining both (e.g., finding the total spend for documents mentioning a specific medication).\n"
                "Respond ONLY with one of the three lowercase strings: 'sql', 'rag', or 'hybrid'."
            )),
            ("user", "Query: {query}")
        ])
        chain = prompt | llm
        resp = chain.invoke({"query": query}).content.strip().lower()
        if resp in ["sql", "rag", "hybrid"]:
            return resp
    except Exception as e:
        logger.warning(f"LLM Intent classification failed: {e}")
    
    return "rag"  # safe fallback


def _generate_sql(query: str, has_api_key: bool, llm_model: str) -> Optional[str]:
    """Uses LLM to generate PostgreSQL-compatible SQL query."""
    if not has_api_key:
        # Check query keywords and return mock SELECT queries
        q_lower = query.lower()
        if "invoice" in q_lower and "above" in q_lower:
            return "SELECT * FROM extracted_entities WHERE entity_type = 'invoice' AND (entity_data->>'amount')::float > 50000;"
        return "SELECT count(*) FROM documents;"

    try:
        llm = get_llm(temperature=0.0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a PostgreSQL expert. Generate a SQL query that retrieves data to answer the user request. "
                "CRITICAL: Generate ONLY a read-only SELECT statement. No insert, update, or delete statements. "
                "Here is the database schema:\n"
                f"{DB_SCHEMA_PROMPT}\n"
                "Return ONLY the SQL query. Do not wrap it in markdown code blocks or add explanations."
            )),
            ("user", "Request: {query}")
        ])
        chain = prompt | llm
        sql = chain.invoke({"query": query}).content.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
    return None


def _is_safe_select_query(sql: str) -> bool:
    """Uses sqlparse to check if SQL contains only read-only SELECT statements."""
    try:
        parsed = sqlparse.parse(sql)
        if not parsed:
            return False
        
        for statement in parsed:
            if statement.get_type() != "SELECT":
                return False
            # Extra safety check for dangerous words inside query (injection)
            stmt_str = str(statement).upper()
            dangerous_words = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "RENAME", "GRANT", "REVOKE"]
            for word in dangerous_words:
                # Word boundary check for dangerous sql statements
                if re.search(r"\b" + word + r"\b", stmt_str):
                    return False
        return True
    except Exception as e:
        logger.error(f"SQL safety parsing error: {e}")
        return False


def _execute_sql_query(sql: str) -> List[Dict[str, Any]]:
    """Runs read-only SELECT query on PostgreSQL."""
    logger.info(f"Database Agent: executing SQL query: {sql}")
    results = []
    try:
        with SyncSessionLocal() as db:
            result = db.execute(text(sql))
            # Check if query returned rows
            if result.returns_rows:
                keys = result.keys()
                for row in result:
                    results.append(dict(zip(keys, row)))
            else:
                results.append({"status": "Success", "rows_affected": result.rowcount})
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        results.append({"error": str(e)})
    return results


def _execute_similarity_search(query: str) -> List[Dict[str, Any]]:
    """Runs pgvector cosine similarity search on document chunks."""
    logger.info(f"Database Agent: executing pgvector search for query: {query}")
    results = []
    try:
        # Embed query text (768 dimensions)
        query_emb = embedding_service.embed_text(query)
        
        sql = """
        SELECT 
            c.document_id, 
            d.filename, 
            c.chunk_index, 
            c.chunk_text,
            (1 - (c.embedding <=> :query_emb)) as similarity
        FROM document_chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE d.status != 'needs_review' AND (1 - (c.embedding <=> :query_emb)) > 0.60
        ORDER BY similarity DESC
        LIMIT 5;
        """
        
        with SyncSessionLocal() as db:
            # pgvector requires "[v1,v2,...]" format — str() produces Python repr with spaces
            emb_str = "[" + ",".join(str(v) for v in query_emb) + "]"
            result = db.execute(text(sql), {"query_emb": emb_str})
            for row in result:
                results.append({
                    "document_id": str(row.document_id),
                    "filename": row.filename,
                    "chunk_index": row.chunk_index,
                    "chunk_text": row.chunk_text,
                    "similarity": float(row.similarity)
                })
    except Exception as e:
        logger.error(f"pgvector similarity search failed: {e}")
        results.append({"error": str(e)})
    return results


def _synthesize_response(
    query: str,
    intent: str,
    sql_results: list,
    rag_results: list,
    sql_query: str,
    has_api_key: bool,
    llm_model: str
) -> str:
    """Synthesizes final answer from search result rows and chunks."""
    if not has_api_key:
        # Simulated responses for testing
        if intent == "sql":
            return (
                f"SQL Query used: `{sql_query}`\n"
                f"Result: Found {len(sql_results)} records. Details:\n"
                f"{json.dumps(sql_results, indent=2)}"
            )
        elif intent == "rag":
            top_hit = rag_results[0] if rag_results else None
            if top_hit:
                return (
                    f"RAG search found matching documents. Top reference: {top_hit['filename']} (chunk {top_hit['chunk_index']})\n"
                    f"Extract: \"{top_hit['chunk_text']}\" (Similarity: {top_hit['similarity']:.2%})"
                )
            return "No matching documents found in vector store."
        else:
            return f"Hybrid search result: SQL returned {len(sql_results)} rows, RAG returned {len(rag_results)} chunks."

    try:
        llm = get_llm(temperature=0.2)

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are the Database Agent (Agent 02) for the MedOCR Intelligence Platform. "
                "Your role is to answer user queries using search results retrieved from SQL query outputs "
                "and/or pgvector similarity RAG chunks.\n\n"
                "CRITICAL Rules:\n"
                "1. Base your answer ONLY on the provided context (SQL rows or RAG chunks).\n"
                "2. When using chunk text, cite the source filename and chunk index (e.g. '[report_1.png (chunk 2)]').\n"
                "3. If context has database errors, report them gracefully.\n"
                "4. Respect medical boundaries: do not diagnose conditions yourself."
            )),
            ("user", (
                "User Query: {query}\n\n"
                "Intent: {intent}\n"
                "SQL Query Used: {sql_query}\n"
                "SQL Results:\n{sql_data}\n\n"
                "RAG Chunks:\n{rag_data}\n\n"
                "Please formulate a comprehensive response."
            ))
        ])

        chain = prompt | llm
        response = chain.invoke({
            "query": query,
            "intent": intent,
            "sql_query": sql_query or "None",
            "sql_data": json.dumps(sql_results, indent=2),
            "rag_data": json.dumps(rag_results, indent=2)
        })

        return response.content.strip()
    except Exception as e:
        logger.error(f"Response synthesis failed: {e}")
        return f"Database query complete. SQL rows count: {len(sql_results)}, RAG chunk matches: {len(rag_results)}."


def _log_agent_activity(
    agent_id: str,
    document_id: Optional[UUID],
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    llm_model: str,
    status: str
) -> None:
    """Write agent activity log record directly to database using sync session."""
    try:
        with SyncSessionLocal() as db:
            activity = AgentActivity(
                agent_id=agent_id,
                document_id=document_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                llm_model=llm_model,
                status=status
            )
            db.add(activity)
            db.commit()
    except Exception as e:
        logger.warning(f"Failed to write agent activity log to database: {e}")
