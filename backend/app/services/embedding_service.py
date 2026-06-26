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
    """
    Load the sentence-transformer model once into CPU/GPU memory.

    Raises:
        RuntimeError: if the model cannot be loaded. We deliberately do NOT fall
        back to a fake embedding — writing meaningless vectors into pgvector would
        silently corrupt the RAG index and return plausible-but-wrong results.
        Failing loudly is the honest behavior.
    """
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
        raise RuntimeError(
            f"Failed to load embedding model '{MODEL_NAME}': {e}. "
            "Install sentence-transformers and ensure the model can be downloaded."
        ) from e


def embed_text(text: str) -> List[float]:
    """
    Generate a 768-dimensional embedding vector for a single text block.

    Raises on failure rather than returning a fabricated vector.
    """
    load_model()
    # sentence-transformers encode returns a numpy array
    embedding = _model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Batch generate 768-dimensional embeddings for multiple text chunks.

    Raises on failure rather than returning fabricated vectors.
    """
    if not chunks:
        return []

    load_model()
    embeddings = _model.encode(chunks, convert_to_numpy=True)
    return embeddings.tolist()


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
