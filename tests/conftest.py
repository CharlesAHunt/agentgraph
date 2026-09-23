from __future__ import annotations

import hashlib
import os
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult

from lgraph.config import Settings
from lgraph.rag.documents import Chunk, Paper, chunk_id


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Every test runs in an empty cwd with the environment restored afterwards.

    ``Settings.from_env`` reads ``./.env`` and ``load_dotenv`` writes into
    ``os.environ`` directly, so without this a developer's real ``.env`` or a
    previous test's file could leak into later tests.
    """
    snapshot = dict(os.environ)
    monkeypatch.chdir(tmp_path)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


@pytest.fixture
def settings() -> Settings:
    return Settings(api_key="test-key", model="test/model")


class RecordingFakeChatModel(GenericFakeChatModel):
    """Fake model that remembers every prompt it was asked to answer."""

    calls: list[list[BaseMessage]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> RecordingFakeChatModel:
        # create_agent binds tools onto the model; the fake ignores them and
        # replays scripted replies (which may themselves contain tool calls).
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class RaisingChatModel(BaseChatModel):
    """Fake model whose every call raises the configured exception."""

    error: Exception

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "raising-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> RaisingChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise self.error


def fake_model(*replies: AIMessage | str) -> RecordingFakeChatModel:
    """Build a recording fake that returns ``replies`` in order."""
    return RecordingFakeChatModel(messages=iter(replies), calls=[])



# --- Retrieval fakes -------------------------------------------------------


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: same words -> nearby vectors."""

    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for word in text.lower().split():
            digest = hashlib.md5(word.encode()).digest()
            vec[digest[0] % self.dimensions] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def make_paper(key: str = "arxiv:1706.03762", **overrides: Any) -> Paper:
    fields: dict[str, Any] = {
        "key": key,
        "title": "Attention Is All You Need",
        "authors": ("Ashish Vaswani", "Noam Shazeer"),
        "year": 2017,
        "arxiv_id": "1706.03762",
        "source": "arxiv",
    }
    fields.update(overrides)
    return Paper(**fields)


def make_chunks(paper: Paper, *texts: str, section: str = "Intro") -> list[Chunk]:
    return [
        Chunk(
            chunk_id=chunk_id(paper.key, i),
            paper_key=paper.key,
            section=section,
            page=i + 1,
            order=i,
            text=text,
        )
        for i, text in enumerate(texts)
    ]
