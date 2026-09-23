"""The chat and embedding models. This is the only module that knows which provider we use."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openrouter import ChatOpenRouter
from openrouter import OpenRouter

from .config import Settings
from .rag.embeddings import OpenRouterEmbeddings


def build_model(settings: Settings) -> BaseChatModel:
    """Construct the OpenRouter-backed chat model from settings.

    ``ChatOpenRouter`` (rather than ``ChatOpenAI`` with a ``base_url``) is
    deliberate: it round-trips OpenRouter's ``reasoning_details`` on
    multi-turn conversations, which the generic OpenAI client drops.
    """
    reasoning = (
        None if settings.reasoning_effort == "none" else {"effort": settings.reasoning_effort}
    )
    return ChatOpenRouter(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        reasoning=reasoning,
        # ChatOpenRouter's timeout is in milliseconds.
        timeout=int(settings.request_timeout_s * 1000),
        max_retries=settings.max_retries,
        app_title="lgraph",
    )


def build_embeddings(settings: Settings) -> Embeddings:
    """Construct the OpenRouter-backed embeddings from settings."""
    client = OpenRouter(
        api_key=settings.api_key,
        server_url=settings.base_url,
        timeout_ms=int(settings.request_timeout_s * 1000),
    )
    return OpenRouterEmbeddings(
        client,
        model=settings.embedding_model,
        expected_dimensions=settings.embedding_dimensions,
    )
