from __future__ import annotations

from pathlib import Path

from lgraph.rag.retriever import Retriever
from lgraph.rag.store import PaperStore

from ..conftest import FakeEmbeddings, make_chunks, make_paper


async def test_retriever_embeds_query_and_searches(tmp_path: Path) -> None:
    emb = FakeEmbeddings(dimensions=8)
    store = PaperStore.open(tmp_path / "lancedb", dimensions=8)
    paper = make_paper()
    chunks = make_chunks(paper, "multi-head attention mechanism", "sinusoidal positional encoding")
    store.upsert(paper, chunks, emb.embed_documents([c.text for c in chunks]))
    store.rebuild_fts()

    hits = await Retriever(emb, store).search("positional encoding", k=1)

    assert [h.chunk.chunk_id for h in hits] == ["arxiv:1706.03762#0001"]


async def test_retriever_passes_year_filters(tmp_path: Path) -> None:
    emb = FakeEmbeddings(dimensions=8)
    store = PaperStore.open(tmp_path / "lancedb", dimensions=8)
    paper = make_paper()
    chunks = make_chunks(paper, "attention")
    store.upsert(paper, chunks, emb.embed_documents([c.text for c in chunks]))

    assert await Retriever(emb, store).search("attention", k=3, year_from=2018) == []
    assert len(await Retriever(emb, store).search("attention", k=3, year_to=2018)) == 1
