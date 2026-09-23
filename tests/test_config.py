from __future__ import annotations

import pytest

from pathlib import Path

from lgraph.config import (
    DEFAULT_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_REASONING_EFFORT,
    ConfigError,
    Settings,
)

RAG_ENV = (
    "LGRAPH_DATA_DIR",
    "LGRAPH_RAG_ENABLED",
    "OPENROUTER_EMBEDDING_MODEL",
    "OPENROUTER_EMBEDDING_DIMENSIONS",
    "LGRAPH_RETRIEVAL_K",
    "LGRAPH_CONTACT_EMAIL",
    "LGRAPH_MINERU_TIER",
    "LGRAPH_MINERU_API_URL",
)


def test_missing_api_key_is_a_config_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    for name in (
        "OPENROUTER_BASE_URL",
        "OPENROUTER_MODEL",
        "OPENROUTER_REASONING_EFFORT",
        "LGRAPH_REQUEST_TIMEOUT_S",
        "LGRAPH_MAX_RETRIES",
        "LGRAPH_SYSTEM_PROMPT",
    ):
        monkeypatch.delenv(name, raising=False)

    s = Settings.from_env()

    assert s.base_url == DEFAULT_BASE_URL == "https://openrouter.ai/api/v1"
    assert s.reasoning_effort == DEFAULT_REASONING_EFFORT
    assert s.request_timeout_s == 120.0
    assert s.max_retries == 2
    assert s.system_prompt is None


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "high")
    monkeypatch.setenv("LGRAPH_REQUEST_TIMEOUT_S", "30.5")
    monkeypatch.setenv("LGRAPH_MAX_RETRIES", "0")
    monkeypatch.setenv("LGRAPH_SYSTEM_PROMPT", "Be terse.")

    s = Settings.from_env()

    assert s.reasoning_effort == "high"
    assert s.request_timeout_s == 30.5
    assert s.max_retries == 0
    assert s.system_prompt == "Be terse."


def test_invalid_reasoning_effort_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "turbo")
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_invalid_timeout_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("LGRAPH_REQUEST_TIMEOUT_S", "soon")
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_repr_masks_key() -> None:
    s = Settings(api_key="super-secret")
    assert "super-secret" not in repr(s)
    assert "***" in repr(s)


def _clear_rag_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    for name in RAG_ENV:
        monkeypatch.delenv(name, raising=False)


def test_rag_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_rag_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    s = Settings.from_env()

    assert s.data_dir == Path("data")
    assert s.lancedb_dir == Path("data") / "lancedb"
    assert s.pdf_dir == Path("data") / "pdfs"
    assert s.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert s.embedding_dimensions == 1536
    assert s.retrieval_k == 6
    assert s.contact_email is None
    assert s.mineru_tier == "flash"
    assert s.mineru_api_url is None
    # No store on disk, nothing explicit: retrieval stays off.
    assert s.rag_enabled is None
    assert s.retrieval_enabled is False


def test_rag_auto_enables_when_store_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_rag_env(monkeypatch)
    monkeypatch.setenv("LGRAPH_DATA_DIR", str(tmp_path))
    (tmp_path / "lancedb").mkdir()

    assert Settings.from_env().retrieval_enabled is True


def test_rag_explicit_flag_overrides_detection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_rag_env(monkeypatch)
    monkeypatch.setenv("LGRAPH_DATA_DIR", str(tmp_path))
    (tmp_path / "lancedb").mkdir()
    monkeypatch.setenv("LGRAPH_RAG_ENABLED", "false")
    assert Settings.from_env().retrieval_enabled is False

    monkeypatch.setenv("LGRAPH_RAG_ENABLED", "true")
    monkeypatch.setenv("LGRAPH_DATA_DIR", str(tmp_path / "missing"))
    assert Settings.from_env().retrieval_enabled is True


def test_rag_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_rag_env(monkeypatch)
    monkeypatch.setenv("LGRAPH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-large")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_DIMENSIONS", "3072")
    monkeypatch.setenv("LGRAPH_RETRIEVAL_K", "10")
    monkeypatch.setenv("LGRAPH_CONTACT_EMAIL", "someone@example.org")
    monkeypatch.setenv("LGRAPH_MINERU_TIER", "flash")
    monkeypatch.setenv("LGRAPH_MINERU_API_URL", "http://127.0.0.1:8001")

    s = Settings.from_env()

    assert s.data_dir == tmp_path
    assert s.embedding_model == "openai/text-embedding-3-large"
    assert s.embedding_dimensions == 3072
    assert s.retrieval_k == 10
    assert s.contact_email == "someone@example.org"
    assert s.mineru_tier == "flash"
    assert s.mineru_api_url == "http://127.0.0.1:8001"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LGRAPH_MINERU_TIER", "ultra"),
        ("LGRAPH_RAG_ENABLED", "maybe"),
        ("OPENROUTER_EMBEDDING_DIMENSIONS", "0"),
        ("LGRAPH_RETRIEVAL_K", "-1"),
    ],
)
def test_invalid_rag_settings_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    _clear_rag_env(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_dotenv_is_loaded_without_overriding_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_rag_env(monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_MODEL", "from-env")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "# comment\n"
        "OPENROUTER_API_KEY='from-file'\n"
        'OPENROUTER_MODEL="from-file"\n'
        "export LGRAPH_CONTACT_EMAIL=me@example.org\n"
        "not a valid line\n"
        "9BAD=skip\n"
    )

    s = Settings.from_env()

    assert s.api_key == "from-file"
    assert s.model == "from-env"  # real environment wins
    assert s.contact_email == "me@example.org"
    assert "9BAD" not in __import__("os").environ


def test_missing_key_message_mentions_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match=".env"):
        Settings.from_env()
