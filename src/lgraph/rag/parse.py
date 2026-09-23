"""PDF parsing behind a small protocol.

``MinerUParser`` is the only place that imports MinerU, and it does so
lazily: the API server never loads it, and ``lgraph ingest`` gives a clear
message when the optional dependency is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class IngestError(RuntimeError):
    """A paper could not be ingested. The message is shown to the operator."""


@dataclass(frozen=True, slots=True)
class ContentItem:
    """One block from a parsed document, in reading order."""

    type: str  # text | equation | table | image | ...
    text: str
    page_idx: int  # zero-based
    text_level: int = 0  # >= 1 marks a heading; 0 is body text


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    items: tuple[ContentItem, ...]
    markdown: str = ""

    @property
    def pages(self) -> int:
        return max((i.page_idx for i in self.items), default=-1) + 1


class Parser(Protocol):
    async def parse(self, pdf_path: Path) -> ParsedDocument: ...


# Block types MinerU emits that carry no content worth indexing.
_SKIPPED_BLOCK_TYPES = frozenset({"page_number", "header", "footer", "aside_text", "discarded"})


def items_from_structured_content(structured: dict[str, Any]) -> tuple[ContentItem, ...]:
    """Normalise MinerU 4's ``ParseResult.structured_content()`` output.

    Shape: ``{"pages": [{"page_idx": int, "blocks": [block, ...]}, ...]}`` where
    a block has ``type``, ``content``, and for titles ``level``, for tables and
    figures ``captions``/``footnotes`` (lists of ``{"content": str}``), and
    ``continues_prev`` when it continues the previous paragraph across a
    column or page break.
    """
    items: list[ContentItem] = []
    for page in structured.get("pages") or []:
        page_idx = int(page.get("page_idx") or 0)
        for block in page.get("blocks") or []:
            kind = str(block.get("type") or "text")
            if kind in _SKIPPED_BLOCK_TYPES:
                continue
            content = str(block.get("content") or "")
            captions = _contents(block.get("captions"))
            footnotes = _contents(block.get("footnotes"))
            level = 0
            if kind == "doc_title":
                level = int(block.get("level") or 1)
            elif kind == "paragraph_title":
                level = int(block.get("level") or 2)
            elif kind == "image":
                # The block content is OCR of the figure itself; captions are
                # what people search for.
                content = _join(captions, footnotes)
                kind = "image"
            elif kind == "table":
                content = _join(captions, content, footnotes)
            elif kind in {"equation", "interline_equation"}:
                kind = "equation"
            text = content.strip()
            if not text:
                continue
            if block.get("continues_prev") and items and items[-1].text_level == 0 and level == 0:
                prev = items[-1]
                items[-1] = ContentItem(prev.type, f"{prev.text} {text}", prev.page_idx, 0)
                continue
            items.append(ContentItem(type=kind, text=text, page_idx=page_idx, text_level=level))
    return tuple(items)


def _contents(blocks: Any) -> list[str]:
    if not isinstance(blocks, list):
        return []
    return [str(b.get("content") or "").strip() for b in blocks if isinstance(b, dict) and b.get("content")]


def _join(*parts: Any) -> str:
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if isinstance(part, (list, tuple)):
            out.extend(str(p) for p in part if p)
        else:
            out.append(str(part))
    return "\n".join(out)


@dataclass
class MinerUParser:
    """Parse PDFs with MinerU, locally or through a self-hosted API server."""

    tier: str = "standard"
    api_url: str | None = None
    _api_parser: Any = field(default=None, init=False, repr=False)

    async def parse(self, pdf_path: Path) -> ParsedDocument:
        try:
            from mineru import parser as mineru_parser  # noqa: PLC0415
        except ImportError as exc:
            raise IngestError(
                "MinerU is not installed. Install the ingest extra: uv sync --extra ingest"
            ) from exc

        if self.api_url:
            if self._api_parser is None:
                self._api_parser = mineru_parser.MinerUApiParser(
                    api_url=self.api_url, tier=self.tier, ocr_mode="txt" if self.tier == "flash" else "auto"
                )
            import asyncio  # noqa: PLC0415

            result = await asyncio.to_thread(self._api_parser.parse, str(pdf_path))
        else:
            # Flash tier means "use the PDF's own text layer": say so explicitly
            # so MinerU does not reach for OCR models on image-heavy pages.
            ocr_mode = "txt" if self.tier == "flash" else "auto"
            result = await mineru_parser.parse_async(str(pdf_path), tier=self.tier, ocr_mode=ocr_mode)
        return parsed_document_from_result(result)


def parsed_document_from_result(result: Any) -> ParsedDocument:
    """Adapt a MinerU ``ParseResult`` to :class:`ParsedDocument`."""
    structured = result.structured_content()
    if not isinstance(structured, dict) or not structured.get("pages"):
        raise IngestError("MinerU returned no pages for this document.")
    markdown = result.markdown() if hasattr(result, "markdown") else ""
    return ParsedDocument(items=items_from_structured_content(structured), markdown=markdown)
