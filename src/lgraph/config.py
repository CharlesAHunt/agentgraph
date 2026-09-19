"""Runtime settings, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import httpx

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.8-max-0902"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    # LLM calls are slow to first byte but should not hang forever; a generous
    # read timeout with a tight connect timeout is the useful split.
    connect_timeout: float = 5.0
    read_timeout: float = 120.0
    write_timeout: float = 10.0
    pool_timeout: float = 5.0

    @property
    def timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=self.write_timeout,
            pool=self.pool_timeout,
        )

    @classmethod
    def from_env(cls) -> Settings:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ConfigError("OPENROUTER_API_KEY environment variable is missing.")

        return cls(
            api_key=api_key,
            base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
            host=os.environ.get("LGRAPH_HOST", DEFAULT_HOST),
            port=_int_from_env("LGRAPH_PORT", DEFAULT_PORT),
        )

    def __repr__(self) -> str:
        # Keep the key out of tracebacks and log lines.
        return (
            f"Settings(base_url={self.base_url!r}, model={self.model!r}, "
            f"host={self.host!r}, port={self.port!r}, api_key=***)"
        )


def _int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, resolved on first use."""
    return Settings.from_env()
