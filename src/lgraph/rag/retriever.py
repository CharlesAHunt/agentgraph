"""Query embedding plus store lookup, as one async call."""

from __future__ import annotations

import asyncio

from langchain_core.embeddings import Embeddings

from .documents import Hit
from .store import PaperStore


class Retriever:
    def __init__(self, embeddings: Embeddings, store: PaperStore) -> None:
        self._embeddings = embeddings
        self._store = store

    @property
    def store(self) -> PaperStore:
        return self._store

    async def search(
        self,
        query: str,
        *,
        k: int,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[Hit]:
        vector = await self._embeddings.aembed_query(query)
        # LanceDB's Python API is synchronous; keep it off the event loop.
        return await asyncio.to_thread(
            self._store.search, query, vector, k, year_from=year_from, year_to=year_to
        )
