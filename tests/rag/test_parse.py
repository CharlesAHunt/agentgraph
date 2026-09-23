from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lgraph.rag.parse import (
    ContentItem,
    IngestError,
    MinerUParser,
    items_from_structured_content,
    parsed_document_from_result,
)

STRUCTURED = {
    "pages": [
        {"page_idx": 0, "blocks": [
            {"type": "aside_text", "content": "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023"},
            {"type": "doc_title", "level": 1, "content": "Attention Is All You Need"},
            {"type": "text", "content": "**Ashish Vaswani** Google Brain"},
            {"type": "paragraph_title", "level": 2, "content": "Abstract"},
            {"type": "text", "content": "The dominant sequence transduction models are based on"},
            {"type": "page_number", "content": "1"},
        ]},
        {"page_idx": 1, "blocks": [
            {"type": "text", "continues_prev": True, "content": "complex recurrent or convolutional neural networks."},
            {"type": "paragraph_title", "level": 2, "content": "3.2 Attention"},
            {"type": "equation", "content": "$$\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}(QK^T/\\sqrt{d_k})V$$", "image_source": "data:..."},
            {"type": "equation", "content": "", "image_source": "data:..."},
            {"type": "image", "content": "Scaled Dot-Product Attention", "captions": [{"content": "Figure 2: Multi-Head Attention."}], "footnotes": []},
            {"type": "table", "content": "| Layer | Complexity |\n| --- | --- |\n| Self-Attention | O(n^2 d) |",
             "captions": [{"content": "Table 1: Maximum path lengths."}], "footnotes": [{"content": "n is the sequence length."}]},
            {"type": "page_footnote", "content": "4 To illustrate why the dot products get large..."},
        ]},
    ],
    "metadata": {}, "is_full_document": True,
}


def test_structured_content_is_normalised() -> None:
    items = items_from_structured_content(STRUCTURED)
    kinds = [(i.type, i.text_level, i.page_idx) for i in items]
    assert kinds == [
        ("doc_title", 1, 0),
        ("text", 0, 0),
        ("paragraph_title", 2, 0),
        ("text", 0, 0),
        ("paragraph_title", 2, 1),
        ("equation", 0, 1),
        ("image", 0, 1),
        ("table", 0, 1),
        ("page_footnote", 0, 1),
    ]
    # A continued paragraph is merged into its predecessor and keeps the first page.
    assert items[3].text == "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks."
    # Empty equations, page numbers and margin text are dropped.
    assert all(i.text for i in items)
    assert not any("arXiv:1706.03762v7" in i.text for i in items)
    assert items[6].text == "Figure 2: Multi-Head Attention."
    assert items[7].text == "Table 1: Maximum path lengths.\n| Layer | Complexity |\n| --- | --- |\n| Self-Attention | O(n^2 d) |\nn is the sequence length."


def test_headings_without_level_get_defaults() -> None:
    items = items_from_structured_content({"pages": [{"page_idx": 0, "blocks": [
        {"type": "doc_title", "content": "T"}, {"type": "paragraph_title", "content": "S"}]}]})
    assert [i.text_level for i in items] == [1, 2]


class _FakeResult:
    def __init__(self, structured: dict) -> None:
        self._structured = structured

    def structured_content(self) -> dict:
        return self._structured

    def markdown(self) -> str:
        return "# Attention Is All You Need"


def test_parsed_document_from_result() -> None:
    doc = parsed_document_from_result(_FakeResult(STRUCTURED))
    assert doc.pages == 2
    assert doc.markdown.startswith("# Attention")
    assert isinstance(doc.items[0], ContentItem)


def test_result_without_pages_is_an_error() -> None:
    with pytest.raises(IngestError):
        parsed_document_from_result(_FakeResult({"pages": []}))


async def test_missing_mineru_gives_install_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "mineru", None)  # makes `from mineru import parser` raise ImportError
    with pytest.raises(IngestError, match="uv sync --extra ingest"):
        await MinerUParser(tier="flash").parse(tmp_path / "x.pdf")
