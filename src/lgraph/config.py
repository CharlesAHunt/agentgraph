"""Runtime settings, read once from the environment."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DOTENV_PATH = Path(".env")
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# API root. ChatOpenRouter appends the endpoint paths itself.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3.8-max-0902"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_REQUEST_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# Retrieval.
DEFAULT_DATA_DIR = Path("data")
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536
DEFAULT_RETRIEVAL_K = 6
DEFAULT_MINERU_TIER = "flash"

# OpenRouter's unified reasoning levels. "none" turns reasoning off.
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
MINERU_TIERS = frozenset({"flash", "standard", "advanced"})


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    # LLM calls are slow to first byte; the SDK handles connect/retry itself,
    # so a single generous end-to-end budget is the useful knob.
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    max_retries: int = DEFAULT_MAX_RETRIES
    system_prompt: str | None = None
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    # Retrieval over the paper corpus.
    data_dir: Path = DEFAULT_DATA_DIR
    # None means "auto": on when a store exists under data_dir.
    rag_enabled: bool | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    retrieval_k: int = DEFAULT_RETRIEVAL_K
    # Needed only for ingesting DOIs (Crossref polite pool, Unpaywall).
    contact_email: str | None = None
    mineru_tier: str = DEFAULT_MINERU_TIER
    # When set, parsing is delegated to a self-hosted MinerU API server.
    mineru_api_url: str | None = None

    def __post_init__(self) -> None:
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ConfigError(
                f"reasoning_effort must be one of {_choices(REASONING_EFFORTS)}; "
                f"got {self.reasoning_effort!r}."
            )
        if self.mineru_tier not in MINERU_TIERS:
            raise ConfigError(
                f"mineru_tier must be one of {_choices(MINERU_TIERS)}; got {self.mineru_tier!r}."
            )
        if self.embedding_dimensions <= 0:
            raise ConfigError("embedding_dimensions must be a positive integer.")
        if self.retrieval_k <= 0:
            raise ConfigError("retrieval_k must be a positive integer.")

    @property
    def lancedb_dir(self) -> Path:
        return self.data_dir / "lancedb"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def retrieval_enabled(self) -> bool:
        """Whether the agent gets the retrieval tools."""
        if self.rag_enabled is not None:
            return self.rag_enabled
        return self.lancedb_dir.exists()

    @classmethod
    def from_env(cls, dotenv: Path | None = DOTENV_PATH) -> Settings:
        if dotenv is not None:
            load_dotenv(dotenv)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ConfigError(
                "OPENROUTER_API_KEY is missing. Export it or put OPENROUTER_API_KEY=... in a .env file."
            )

        return cls(
            api_key=api_key,
            base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
            reasoning_effort=os.environ.get(
                "OPENROUTER_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
            ),
            request_timeout_s=_float_from_env(
                "LGRAPH_REQUEST_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S
            ),
            max_retries=_int_from_env("LGRAPH_MAX_RETRIES", DEFAULT_MAX_RETRIES),
            system_prompt=os.environ.get("LGRAPH_SYSTEM_PROMPT") or None,
            host=os.environ.get("LGRAPH_HOST", DEFAULT_HOST),
            port=_int_from_env("LGRAPH_PORT", DEFAULT_PORT),
            data_dir=Path(os.environ.get("LGRAPH_DATA_DIR") or DEFAULT_DATA_DIR),
            rag_enabled=_bool_from_env("LGRAPH_RAG_ENABLED"),
            embedding_model=os.environ.get("OPENROUTER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_dimensions=_int_from_env(
                "OPENROUTER_EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS
            ),
            retrieval_k=_int_from_env("LGRAPH_RETRIEVAL_K", DEFAULT_RETRIEVAL_K),
            contact_email=os.environ.get("LGRAPH_CONTACT_EMAIL") or None,
            mineru_tier=os.environ.get("LGRAPH_MINERU_TIER", DEFAULT_MINERU_TIER),
            mineru_api_url=os.environ.get("LGRAPH_MINERU_API_URL") or None,
        )

    def __repr__(self) -> str:
        # Keep the key out of tracebacks and log lines.
        return (
            f"Settings(base_url={self.base_url!r}, model={self.model!r}, "
            f"reasoning_effort={self.reasoning_effort!r}, "
            f"data_dir={str(self.data_dir)!r}, retrieval_enabled={self.retrieval_enabled}, "
            f"host={self.host!r}, port={self.port!r}, api_key=***)"
        )


def load_dotenv(path: Path) -> int:
    """Load ``KEY=VALUE`` lines from ``path`` into the environment.

    Real environment variables win over the file. Blank lines and ``#``
    comments are ignored; surrounding quotes on values are stripped. Returns
    the number of variables set. A missing file is not an error.
    """
    if not path.is_file():
        return 0
    loaded = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not _ENV_KEY.match(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def _choices(values: frozenset[str]) -> str:
    return ", ".join(sorted(values))


def _int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from exc


def _float_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}.") from exc


def _bool_from_env(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, resolved on first use."""
    return Settings.from_env()
