from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openrouter import ChatOpenRouter

from lgraph.config import Settings
from lgraph.model import build_model


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
