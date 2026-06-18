"""
Service — LLM Provider Factory

Single source of truth for the agent layer's LLM client. The platform standardizes
on DeepSeek (V4 Flash), which is OpenAI-compatible, so we use langchain_openai's
ChatOpenAI pointed at DeepSeek's base URL. Centralizing construction here keeps the
provider swappable from one place (change config / this factory, not six agents).
"""

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

# Placeholder value shipped in .env.example — treated as "no key configured".
_PLACEHOLDER_FRAGMENT = "your-deepseek"


def has_llm_api_key() -> bool:
    """True when a real DeepSeek API key is configured (not the placeholder)."""
    key = get_settings().deepseek_api_key
    return bool(key) and _PLACEHOLDER_FRAGMENT not in key


def get_model_name() -> str:
    """Return the configured DeepSeek model id (used for activity logging)."""
    return get_settings().deepseek_model or "deepseek-v4-flash"


def get_llm(temperature: float = 0.2, **kwargs) -> ChatOpenAI:
    """
    Construct a chat LLM client for the agent layer.

    Args:
        temperature: Sampling temperature (0.0 for deterministic safety-critical calls).
        **kwargs: Extra args forwarded to ChatOpenAI (e.g. model_kwargs, extra_body).

    Returns:
        A ChatOpenAI instance configured for DeepSeek.
    """
    settings = get_settings()
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        **kwargs,
    )