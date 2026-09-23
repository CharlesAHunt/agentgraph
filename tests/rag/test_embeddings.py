from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from lgraph.rag.embeddings import EmbeddingDimensionError, OpenRouterEmbeddings


@dataclass
class _Datum:
    embedding: list[float]
    index: int


@dataclass
class _Body:
    data: list[_Datum]


@dataclass
class _FakeEmbeddingsApi:
    dimensions: int = 4
    calls: list[dict[str, Any]] = field(default_factory=list)

    def _respond(self, texts: list[str]) -> _Body:
        # Return items in reverse order to prove the client re-sorts by index.
        data = [_Datum([float(i)] * self.dimensions, i) for i in range(len(texts))]
        return _Body(list(reversed(data)))

    def generate(self, *, input: list[str], model: str, dimensions: int | None = None, **kwargs: Any) -> _Body:
        self.calls.append({"input": list(input), "model": model, "dimensions": dimensions, "async": False})
        return self._respond(input)

    async def generate_async(self, *, input: list[str], model: str, dimensions: int | None = None, **kwargs: Any) -> _Body:
        self.calls.append({"input": list(input), "model": model, "dimensions": dimensions, "async": True})
        return self._respond(input)


@dataclass
class _FakeClient:
    embeddings: _FakeEmbeddingsApi


def _make(batch_size: int = 64, expected: int = 4) -> tuple[OpenRouterEmbeddings, _FakeEmbeddingsApi]:
    api = _FakeEmbeddingsApi()
    emb = OpenRouterEmbeddings(_FakeClient(api), model="m", expected_dimensions=expected, batch_size=batch_size)
    return emb, api


def test_embed_documents_batches_and_orders_by_index() -> None:
    emb, api = _make(batch_size=2)
    out = emb.embed_documents(["a", "b", "c", "d", "e"])
    assert [c["input"] for c in api.calls] == [["a", "b"], ["c", "d"], ["e"]]
    assert all(c["model"] == "m" and c["dimensions"] == 4 for c in api.calls)
    assert out == [[0.0] * 4, [1.0] * 4, [0.0] * 4, [1.0] * 4, [0.0] * 4]


def test_embed_query_uses_sync_api() -> None:
    emb, api = _make()
    assert emb.embed_query("q") == [0.0] * 4
    assert api.calls[0]["async"] is False


async def test_async_paths_use_async_api() -> None:
    emb, api = _make(batch_size=2)
    out = await emb.aembed_documents(["a", "b", "c"])
    assert len(out) == 3
    assert all(c["async"] for c in api.calls)
    assert await emb.aembed_query("q") == [0.0] * 4


def test_dimension_mismatch_raises() -> None:
    emb, _ = _make(expected=8)
    with pytest.raises(EmbeddingDimensionError):
        emb.embed_documents(["a"])


def test_empty_input_makes_no_call() -> None:
    emb, api = _make()
    assert emb.embed_documents([]) == []
    assert api.calls == []
