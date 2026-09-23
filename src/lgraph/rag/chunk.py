"""Section-aware chunking of a parsed paper.

Headings open sections; body blocks accumulate until a size target is met.
Reference lists and acknowledgements are skipped: they add noise to
retrieval and are never what a question is about. The abstract from the
paper's metadata, when known, becomes the first chunk.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .documents import Chunk, Paper, chunk_id
from .parse import ParsedDocument

SKIPPED_SECTIONS = re.compile(
    r"^\s*(\d+[\.\d]*\s*)?(references?|bibliography|acknowledg(e)?ments?|author contributions|"
    r"conflicts? of interest|funding)\b",
    re.IGNORECASE,
)
FRONT_MATTER = "Front matter"


def chunk_document(
    paper: Paper,
    doc: ParsedDocument,
    *,
    target_chars: int = 2400,
    min_chars: int = 200,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(section: str, page_idx: int, parts: Sequence[str]) -> None:
        text = "\n\n".join(parts).strip()
        if not text:
            return
        chunks.append(
            Chunk(
                chunk_id=chunk_id(paper.key, len(chunks)),
                paper_key=paper.key,
                section=section,
                page=page_idx + 1,
                order=len(chunks),
                text=text,
            )
        )

    if paper.abstract:
        emit("Abstract", 0, [paper.abstract])

    section = FRONT_MATTER
    skipping = False
    buffer: list[str] = []
    buffer_page = 0
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if buffer and not skipping:
            emit(section, buffer_page, buffer)
        buffer, buffer_len = [], 0

    for item in doc.items:
        if item.text_level >= 1:
            flush()
            section = _clean_heading(item.text)
            skipping = bool(SKIPPED_SECTIONS.match(section)) or (
                # The metadata abstract already became chunk 0.
                bool(paper.abstract) and section.lower() == "abstract"
            )
            continue
        if skipping:
            continue
        # A block on its own that would overflow the target goes in whole; we
        # never split a paragraph or equation.
        if buffer and buffer_len + len(item.text) > target_chars and buffer_len >= min_chars:
            flush()
        if not buffer:
            buffer_page = item.page_idx
        buffer.append(item.text)
        buffer_len += len(item.text)
    flush()

    return _merge_tiny_tail(chunks, min_chars)


def _clean_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip(":")


def _merge_tiny_tail(chunks: list[Chunk], min_chars: int) -> list[Chunk]:
    """Fold a very short final chunk of a section into its predecessor."""
    merged: list[Chunk] = []
    for c in chunks:
        prev = merged[-1] if merged else None
        if prev and prev.section == c.section and len(c.text) < min_chars:
            merged[-1] = Chunk(
                chunk_id=prev.chunk_id,
                paper_key=prev.paper_key,
                section=prev.section,
                page=prev.page,
                order=prev.order,
                text=f"{prev.text}\n\n{c.text}",
            )
            continue
        merged.append(c)
    # Re-number so ids stay dense after merges.
    return [
        Chunk(
            chunk_id=chunk_id(c.paper_key, i),
            paper_key=c.paper_key,
            section=c.section,
            page=c.page,
            order=i,
            text=c.text,
        )
        for i, c in enumerate(merged)
    ]
