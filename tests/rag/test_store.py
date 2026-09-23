from __future__ import annotations

from pathlib import Path

import pytest

from lgraph.config import ConfigError
from lgraph.rag.store import PaperStore

from ..conftest import FakeEmbeddings, make_chunks, make_paper


@pytest.fixture
def emb() -> FakeEmbeddings:
    return FakeEmbeddings(dimensions=8)


@pytest.fixture
def store(tmp_path: Path, emb: FakeEmbeddings) -> PaperStore:
    return PaperStore.open(tmp_path / "lancedb", dimensions=emb.dimensions)


def _ingest(store: PaperStore, emb: FakeEmbeddings, paper, *texts, section="Intro"):
    chunks = make_chunks(paper, *texts, section=section)
    store.upsert(paper, chunks, emb.embed_documents([c.text for c in chunks]))
    return chunks


def test_open_creates_tables_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "lancedb"
    PaperStore.open(path, dimensions=8)
    again = PaperStore.open(path, dimensions=8)
    assert path.exists()
    assert again.count_chunks() == 0
    assert again.list_papers() == []


def test_open_rejects_dimension_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "lancedb"
    PaperStore.open(path, dimensions=8)
    with pytest.raises(ConfigError):
        PaperStore.open(path, dimensions=16)


def test_upsert_then_hybrid_search_returns_hits(store: PaperStore, emb: FakeEmbeddings) -> None:
    paper = make_paper()
    _ingest(store, emb, paper, "multi-head attention mechanism", "sinusoidal positional encoding")
    store.rebuild_fts()

    hits = store.search("multi-head attention", emb.embed_query("multi-head attention"), k=2)

    assert [h.chunk.chunk_id for h in hits][0] == "arxiv:1706.03762#0000"
    top = hits[0]
    assert top.title == paper.title
    assert top.authors == paper.authors
    assert top.year == 2017
    assert top.arxiv_id == "1706.03762"
    assert top.doi is None
    assert top.chunk.section == "Intro"
    assert top.chunk.page == 1
    assert top.score > 0


def test_search_empty_store_returns_nothing(store: PaperStore, emb: FakeEmbeddings) -> None:
    assert store.search("anything", emb.embed_query("anything"), k=3) == []


def test_year_filter(store: PaperStore, emb: FakeEmbeddings) -> None:
    old = make_paper()
    new = make_paper(key="10.1038/s41586-021-03819-2", title="AlphaFold", year=2021,
                     doi="10.1038/s41586-021-03819-2", arxiv_id=None, source="crossref")
    _ingest(store, emb, old, "attention mechanism for translation")
    _ingest(store, emb, new, "attention mechanism for protein structure")
    store.rebuild_fts()
    q = emb.embed_query("attention mechanism")

    assert {h.chunk.paper_key for h in store.search("attention mechanism", q, k=5)} == {old.key, new.key}
    assert [h.chunk.paper_key for h in store.search("attention mechanism", q, k=5, year_from=2020)] == [new.key]
    assert [h.chunk.paper_key for h in store.search("attention mechanism", q, k=5, year_to=2018)] == [old.key]
    assert store.search("attention mechanism", q, k=5, year_from=2018, year_to=2020) == []


def test_year_filter_rejects_non_integers(store: PaperStore, emb: FakeEmbeddings) -> None:
    with pytest.raises(TypeError):
        store.search("x", emb.embed_query("x"), k=1, year_from="2020 OR 1=1")  # type: ignore[arg-type]


def test_reingest_replaces_chunks_and_updates_paper(store: PaperStore, emb: FakeEmbeddings) -> None:
    paper = make_paper()
    _ingest(store, emb, paper, "one", "two", "three")
    assert store.count_chunks() == 3

    _ingest(store, emb, paper, "uno", "dos")

    assert store.count_chunks() == 2
    [listed] = store.list_papers()
    assert listed.key == paper.key
    assert store.chunk_count(paper.key) == 2


def test_list_papers_round_trips_metadata(store: PaperStore, emb: FakeEmbeddings) -> None:
    from dataclasses import replace

    paper = make_paper(venue="NeurIPS", abstract="We propose...", pdf_url="https://arxiv.org/pdf/1706.03762")
    _ingest(store, emb, paper, "text", "more")
    [listed] = store.list_papers()
    assert listed.chunk_count == 2  # read from the stored column, no per-paper query
    assert replace(listed, chunk_count=0) == paper


def test_keys_with_quotes_are_escaped(store: PaperStore, emb: FakeEmbeddings) -> None:
    paper = make_paper(key="10.1000/it's", doi="10.1000/it's", arxiv_id=None)
    _ingest(store, emb, paper, "a", "b")
    _ingest(store, emb, paper, "c")
    assert store.count_chunks() == 1


def test_has_paper(store: PaperStore, emb: FakeEmbeddings) -> None:
    paper = make_paper()
    assert store.has_paper(paper.key) is False
    _ingest(store, emb, paper, "text")
    assert store.has_paper(paper.key) is True
    assert store.has_paper("10.1000/it's") is False


def test_reads_see_writes_from_another_connection(tmp_path: Path, emb: FakeEmbeddings) -> None:
    reader = PaperStore.open(tmp_path / "lancedb", dimensions=8)   # e.g. the running server
    writer = PaperStore.open(tmp_path / "lancedb", dimensions=8)   # e.g. an ingest run
    assert reader.count_chunks() == 0

    _ingest(writer, emb, make_paper(), "multi-head attention", "positional encoding")

    assert reader.count_chunks() == 2
    assert len(reader.list_papers()) == 1
    assert reader.has_paper("arxiv:1706.03762")
    hits = reader.search("attention", emb.embed_query("attention"), k=2)
    assert hits and hits[0].chunk.paper_key == "arxiv:1706.03762"
