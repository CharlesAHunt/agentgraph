from __future__ import annotations

import json
from pathlib import Path

import pytest

from lgraph.rag.chunk import chunk_document
from lgraph.rag.parse import ContentItem, ParsedDocument, items_from_structured_content

from ..conftest import make_paper

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "structured_content.json"


@pytest.fixture
def doc() -> ParsedDocument:
    return ParsedDocument(items=items_from_structured_content(json.loads(FIXTURE.read_text())))


def test_fixture_normalises_blocks(doc: ParsedDocument) -> None:
    kinds = [i.type for i in doc.items]
    assert kinds.count("table") == 1 and kinds.count("image") == 1 and kinds.count("equation") == 1
    table = next(i for i in doc.items if i.type == "table")
    assert table.text.startswith("Table 1: Maximum path lengths")
    assert "| Layer Type |" in table.text
    image = next(i for i in doc.items if i.type == "image")
    assert image.text == "Figure 1: The Transformer - model architecture."
    assert doc.pages == 10


def test_chunks_follow_sections_and_skip_references(doc: ParsedDocument) -> None:
    paper = make_paper(abstract=None)
    chunks = chunk_document(paper, doc, target_chars=400, min_chars=50)

    sections = [c.section for c in chunks]
    assert "References" not in sections
    assert "Acknowledgements" not in sections
    assert sections[0] == "Attention Is All You Need"  # title heading holds the author line
    assert "Abstract" in sections
    assert "1 Introduction" in sections
    assert "3.2 Attention" in sections

    for c in chunks:
        assert "Layer normalization" not in c.text
        assert "fruitful comments" not in c.text

    attention = [c for c in chunks if c.section == "3.2 Attention"]
    joined = "\n".join(c.text for c in attention)
    assert "softmax" in joined  # equation kept inline
    assert "Multi-head attention" in joined
    assert "Table 1" in joined
    assert attention[0].page == 3  # page_idx 2 -> 1-based


def test_chunk_ids_are_dense_and_ordered(doc: ParsedDocument) -> None:
    chunks = chunk_document(make_paper(abstract=None), doc, target_chars=300, min_chars=50)
    assert [c.order for c in chunks] == list(range(len(chunks)))
    assert [c.chunk_id for c in chunks] == [f"arxiv:1706.03762#{i:04d}" for i in range(len(chunks))]


def test_metadata_abstract_becomes_first_chunk_and_parsed_abstract_is_skipped(doc: ParsedDocument) -> None:
    chunks = chunk_document(make_paper(abstract="We propose the Transformer."), doc)
    assert chunks[0].section == "Abstract"
    assert chunks[0].text == "We propose the Transformer."
    assert chunks[0].page == 1
    # The PDF's own Abstract section is not indexed a second time.
    assert [c for c in chunks if c.section == "Abstract"] == chunks[:1]
    assert not any("dominant sequence transduction" in c.text for c in chunks)


def test_parsed_abstract_is_kept_when_metadata_has_none(doc: ParsedDocument) -> None:
    chunks = chunk_document(make_paper(abstract=None), doc)
    assert any(c.section == "Abstract" and "dominant sequence transduction" in c.text for c in chunks)


def test_target_size_splits_long_sections() -> None:
    items = [ContentItem("text", "Method", 0, 1)] + [
        ContentItem("text", f"Paragraph {i} " + "x" * 300, i // 3, 0) for i in range(9)
    ]
    chunks = chunk_document(make_paper(abstract=None), ParsedDocument(items=tuple(items)), target_chars=700, min_chars=100)
    assert len(chunks) > 1
    assert all(c.section == "Method" for c in chunks)
    assert all(len(c.text) <= 1000 for c in chunks)
    # Each chunk's page is that of its first paragraph.
    assert chunks[0].page == 1


def test_tiny_tail_is_merged_into_previous_chunk() -> None:
    items = [
        ContentItem("text", "Results", 0, 1),
        ContentItem("text", "A" * 500, 0, 0),
        ContentItem("text", "short tail", 1, 0),
    ]
    chunks = chunk_document(make_paper(abstract=None), ParsedDocument(items=tuple(items)), target_chars=450, min_chars=200)
    assert len(chunks) == 1
    assert chunks[0].text.endswith("short tail")


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_document(make_paper(abstract=None), ParsedDocument(items=())) == []
