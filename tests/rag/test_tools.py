from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage

from lgraph.rag.documents import Hit, Source
from lgraph.rag.retriever import Retriever
from lgraph.rag.store import PaperStore
from lgraph.rag.tools import make_tools

from ..conftest import FakeEmbeddings, make_chunks, make_paper


class _StubRetriever:
    def __init__(self, hits: list[Hit] | None = None, error: Exception | None = None, store=None) -> None:
        self.hits = hits or []
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.store = store

    async def search(self, query: str, *, k: int, year_from: int | None = None, year_to: int | None = None):
        self.calls.append({"query": query, "k": k, "year_from": year_from, "year_to": year_to})
        if self.error:
            raise self.error
        return self.hits


def _hit(order: int, text: str, page: int = 4) -> Hit:
    paper = make_paper()
    [chunk] = make_chunks(paper, text, section="3.2 Attention")
    chunk = replace(chunk, order=order, page=page, chunk_id=f"{paper.key}#{order:04d}")
    return Hit(chunk=chunk, title=paper.title, authors=paper.authors, year=paper.year,
               doi=None, arxiv_id=paper.arxiv_id, score=0.5)


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


async def test_search_tool_formats_excerpts_and_returns_sources() -> None:
    retriever = _StubRetriever(hits=[_hit(3, "Multi-head attention allows..."), _hit(4, "Positional encodings...", page=5)])
    search = _tool(make_tools(retriever, k=6), "search_publications")

    msg = await search.ainvoke({"name": "search_publications", "type": "tool_call", "id": "c1",
                                "args": {"query": "multi-head attention"}})

    assert isinstance(msg, ToolMessage)
    assert retriever.calls == [{"query": "multi-head attention", "k": 6, "year_from": None, "year_to": None}]
    text = msg.content
    assert "[1] Vaswani et al. (2017). Attention Is All You Need. §3.2 Attention, p. 4. arXiv:1706.03762" in text
    assert "Multi-head attention allows..." in text
    assert "[2] Vaswani et al. (2017). Attention Is All You Need. §3.2 Attention, p. 5. arXiv:1706.03762" in text
    sources = [Source.model_validate(s) for s in msg.artifact]
    assert [s.chunk_id for s in sources] == ["arxiv:1706.03762#0003", "arxiv:1706.03762#0004"]
    assert sources[0].page == 4 and sources[0].section == "3.2 Attention"


async def test_search_tool_passes_year_filters() -> None:
    retriever = _StubRetriever()
    search = _tool(make_tools(retriever, k=3), "search_publications")
    msg = await search.ainvoke({"name": "search_publications", "type": "tool_call", "id": "c1",
                                "args": {"query": "q", "year_from": 2020, "year_to": 2022}})
    assert retriever.calls[0]["year_from"] == 2020 and retriever.calls[0]["year_to"] == 2022
    assert "No matching excerpts" in msg.content
    assert msg.artifact == []


async def test_search_tool_reports_unavailable_on_error(caplog) -> None:
    retriever = _StubRetriever(error=RuntimeError("embeddings down"))
    search = _tool(make_tools(retriever, k=3), "search_publications")
    msg = await search.ainvoke({"name": "search_publications", "type": "tool_call", "id": "c1", "args": {"query": "q"}})
    assert msg.content.startswith("Search is temporarily unavailable")
    assert msg.artifact == []
    assert "embeddings down" in caplog.text


async def test_list_publications_lists_corpus(tmp_path: Path) -> None:
    emb = FakeEmbeddings(8)
    store = PaperStore.open(tmp_path / "lancedb", dimensions=8)
    paper = make_paper()
    chunks = make_chunks(paper, "a", "b")
    store.upsert(paper, chunks, emb.embed_documents([c.text for c in chunks]))
    listing = _tool(make_tools(Retriever(emb, store), k=3), "list_publications")

    msg = await listing.ainvoke({"name": "list_publications", "type": "tool_call", "id": "c1", "args": {}})

    assert "Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762" in msg.content
    assert "2 excerpts" in msg.content


async def test_list_publications_on_empty_corpus(tmp_path: Path) -> None:
    store = PaperStore.open(tmp_path / "lancedb", dimensions=8)
    listing = _tool(make_tools(Retriever(FakeEmbeddings(8), store), k=3), "list_publications")
    msg = await listing.ainvoke({"name": "list_publications", "type": "tool_call", "id": "c1", "args": {}})
    assert "empty" in msg.content.lower()


def test_tool_descriptions_mention_when_to_use() -> None:
    tools = make_tools(_StubRetriever(), k=3)
    assert {t.name for t in tools} == {"search_publications", "list_publications"}
    assert "year_from" in _tool(tools, "search_publications").args
