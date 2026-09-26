from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_openrouter import ChatOpenRouter

import lgraph.model as model_module
from lgraph.config import Settings
from lgraph.model import _per_million, account_credits, build_model, key_usage


def test_build_model_returns_configured_chat_model() -> None:
    s = Settings(
        api_key="k",
        model="vendor/model",
        base_url="https://example.test/api/v1",
        reasoning_effort="high",
        request_timeout_s=12.5,
        max_retries=1,
    )

    m = build_model(s)

    assert isinstance(m, BaseChatModel)
    assert isinstance(m, ChatOpenRouter)
    assert m.model_name == "vendor/model"
    assert m.openrouter_api_base == "https://example.test/api/v1"
    assert m.openrouter_api_key is not None
    assert m.openrouter_api_key.get_secret_value() == "k"
    assert m.reasoning == {"effort": "high"}
    assert m.request_timeout == 12500  # ChatOpenRouter takes milliseconds
    assert m.max_retries == 1


def test_reasoning_none_disables_reasoning() -> None:
    m = build_model(Settings(api_key="k", reasoning_effort="none"))
    assert m.reasoning is None


def test_prices_are_converted_to_dollars_per_million_tokens() -> None:
    assert _per_million("0.000003") == 3.0
    assert _per_million("0") == 0.0
    assert _per_million("-1") is None  # OpenRouter's marker for variable pricing
    assert _per_million("n/a") is None
    assert _per_million(None) is None


class _FakeSDK:
    """Stands in for openrouter.OpenRouter; records which key each client got."""

    def __init__(self, keys: list[str | None], api_key: str | None) -> None:
        keys.append(api_key)
        key = SimpleNamespace(usage=12.5, usage_daily=0.4, usage_weekly=3.1, usage_monthly=9.8,
                              limit=50.0, limit_remaining=37.5, limit_reset="monthly",
                              is_free_tier=False, label="secret label", creator_user_id="u_1")
        credits = SimpleNamespace(total_credits=100.0, total_usage=62.5)

        async def key_meta():
            return SimpleNamespace(data=key)

        async def get_credits():
            return SimpleNamespace(data=credits)

        self.api_keys = SimpleNamespace(get_current_key_metadata_async=key_meta)
        self.credits = SimpleNamespace(get_credits_async=get_credits)


async def test_key_usage_returns_only_spend_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    keys: list[str | None] = []
    monkeypatch.setattr(model_module, "_sdk", lambda s, api_key=None: _FakeSDK(keys, api_key or s.api_key))
    usage = await key_usage(Settings(api_key="k"))
    assert usage["limit_remaining"] == 37.5 and usage["usage_monthly"] == 9.8
    assert "label" not in usage and "creator_user_id" not in usage
    assert keys == ["k"]


async def test_account_credits_uses_the_management_key(monkeypatch: pytest.MonkeyPatch) -> None:
    keys: list[str | None] = []
    monkeypatch.setattr(model_module, "_sdk", lambda s, api_key=None: _FakeSDK(keys, api_key or s.api_key))
    credits = await account_credits(Settings(api_key="k", management_key="mgmt"))
    assert credits == {"total": 100.0, "used": 62.5, "remaining": 37.5}
    assert keys == ["mgmt"]
