"""
Service — Embedding Service
Handles text chunking and 768-dimensional embedding generation.
Uses sentence-transformers/all-mpnet-base-v2 to match PostgreSQL pgvector column size.
"""

import time
import logging
from typing import List, Optional
import torch

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

_model: Optional[object] = None
_device: Optional[torch.device] = None


def load_model() -> None:
    """Load the sentence-transformer model once into CPU/GPU memory."""
    global _model, _device
    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading embedding model: {MODEL_NAME} on {_device}")

    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device=str(_device))
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.warning(
            f"Failed to load embedding model ({e}). "
            "Falling back to a deterministic mock embedding vector generator."
        )


def embed_text(text: str) -> List[float]:
    """
    Generate a 768-dimensional embedding vector for a single text block.
    """
    load_model()

    if _model is not None:
        try:
            # sentence-transformers encode returns a numpy array
            embedding = _model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding encoding failed: {e}. Generating mock embedding.")

    # ── Fallback mock embedding (768 dimensions) ─────────────────────────────
    # Generates a deterministic vector based on the hash of the text
    import hashlib
    h = hashlib.sha256(text.encode("utf-8")).digest()
    mock_vec = []
    for i in range(768):
        # Derive values between -1.0 and 1.0 from the hash bytes
        val = ((h[i % len(h)] + i) % 256) / 128.0 - 1.0
        mock_vec.append(round(val, 6))
    return mock_vec


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Batch generate 768-dimensional embeddings for multiple text chunks.
    """
    if not chunks:
        return []

    load_model()

    if _model is not None:
        try:
            embeddings = _model.encode(chunks, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Batch embedding encoding failed: {e}. Generating mock embeddings.")

    return [embed_text(chunk) for chunk in chunks]


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 30) -> List[str]:
    """
    Split a long string of text into overlapping word-level chunks.
    Approximate: 200 words is ~256 tokens.
    
    Args:
        text: Raw document text to slice.
        chunk_size: Target number of words per chunk.
        overlap: Word overlap size between consecutive chunks.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        step = max(1, chunk_size - overlap)
        start += step

    logger.debug(f"Chunked text of length {len(words)} words into {len(chunks)} chunks.")
    return chunks
