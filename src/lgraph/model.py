"""The chat and embedding models. This is the only module that knows which provider we use."""

from __future__ import annotations

from typing import Any

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


def _sdk(settings: Settings, api_key: str | None = None) -> OpenRouter:
    return OpenRouter(
        api_key=api_key or settings.api_key,
        server_url=settings.base_url,
        timeout_ms=int(settings.request_timeout_s * 1000),
    )


async def list_tool_models(settings: Settings) -> list[dict[str, Any]]:
    """OpenRouter chat models that support tool calling, which retrieval needs."""
    body = await _sdk(settings).models.list_async(supported_parameters="tools")
    return [
        {
            "id": m.id,
            "name": m.name,
            "context_length": m.context_length,
            "prompt_price": _per_million(m.pricing.prompt),
            "completion_price": _per_million(m.pricing.completion),
            # Whether the model accepts a reasoning effort.
            "reasoning": "reasoning" in {getattr(p, "value", p) for p in m.supported_parameters},
        }
        for m in body.data
    ]


def _per_million(per_token: str | None) -> float | None:
    """USD per token (a decimal string) -> USD per million tokens. Negative means variable."""
    try:
        value = float(per_token) * 1_000_000 if per_token is not None else None
    except ValueError:
        return None
    return round(value, 4) if value is not None and value >= 0 else None


async def key_usage(settings: Settings) -> dict[str, Any]:
    """Spend and limit of the API key the service runs on, in USD."""
    key = (await _sdk(settings).api_keys.get_current_key_metadata_async()).data
    return {
        "usage": key.usage,
        "usage_daily": key.usage_daily,
        "usage_weekly": key.usage_weekly,
        "usage_monthly": key.usage_monthly,
        "limit": key.limit,
        "limit_remaining": key.limit_remaining,
        "limit_reset": key.limit_reset,
        "is_free_tier": key.is_free_tier,
    }


async def account_credits(settings: Settings) -> dict[str, float]:
    """Account balance in USD. OpenRouter serves this only to a management key."""
    credits = (await _sdk(settings, settings.management_key).credits.get_credits_async()).data
    return {
        "total": credits.total_credits,
        "used": credits.total_usage,
        "remaining": credits.total_credits - credits.total_usage,
    }


def build_embeddings(settings: Settings) -> Embeddings:
    """Construct the OpenRouter-backed embeddings from settings."""
    return OpenRouterEmbeddings(
        _sdk(settings),
        model=settings.embedding_model,
        expected_dimensions=settings.embedding_dimensions,
    )
