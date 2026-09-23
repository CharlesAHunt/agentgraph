"""LangChain ``Embeddings`` over OpenRouter's embeddings endpoint.

Uses the ``openrouter`` SDK directly; ``langchain-openrouter`` ships no
embeddings class. SDK and transport errors propagate unchanged so callers
can map them with :func:`lgraph.errors.translate`.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings


class EmbeddingDimensionError(ValueError):
    """The provider returned vectors of a different width than configured."""


class OpenRouterEmbeddings(Embeddings):
    def __init__(
        self,
        client: Any,
        *,
        model: str,
        expected_dimensions: int,
        batch_size: int = 64,
    ) -> None:
        self._client = client
        self.model = model
        self.expected_dimensions = expected_dimensions
        self.batch_size = batch_size

    # -- sync ---------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for batch in _batches(texts, self.batch_size):
            body = self._client.embeddings.generate(
                input=batch, model=self.model, dimensions=self.expected_dimensions
            )
            out.extend(self._unpack(body, len(batch)))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    # -- async --------------------------------------------------------------

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for batch in _batches(texts, self.batch_size):
            body = await self._client.embeddings.generate_async(
                input=batch, model=self.model, dimensions=self.expected_dimensions
            )
            out.extend(self._unpack(body, len(batch)))
        return out

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]

    # -- helpers ------------------------------------------------------------

    def _unpack(self, body: Any, expected_count: int) -> list[list[float]]:
        data = sorted(body.data, key=lambda d: d.index if d.index is not None else 0)
        if len(data) != expected_count:
            raise ValueError(
                f"OpenRouter returned {len(data)} embeddings for {expected_count} inputs."
            )
        vectors: list[list[float]] = []
        for datum in data:
            vector = datum.embedding
            if not isinstance(vector, list):
                raise ValueError("OpenRouter returned a non-list embedding; request float format.")
            if len(vector) != self.expected_dimensions:
                raise EmbeddingDimensionError(
                    f"Model {self.model!r} returned {len(vector)}-dimensional vectors; "
                    f"embedding_dimensions is {self.expected_dimensions}."
                )
            vectors.append([float(x) for x in vector])
        return vectors


def _batches(items: Sequence[str], size: int) -> list[list[str]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]
