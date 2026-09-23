"""Agent tools over the paper corpus.

Both tools are async so the agent's ``ainvoke`` path never blocks. The
search tool returns numbered excerpts as its content (for the model) and
``Source`` dicts as its artifact (for the API client). Retrieval failures
become a short message rather than an exception, so the model can still
answer or explain.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

from langchain_core.tools import BaseTool, StructuredTool

from .documents import Hit, Source

logger = logging.getLogger(__name__)

UNAVAILABLE = "Search is temporarily unavailable. Answer from what you already know and say that the corpus could not be consulted."
NO_MATCHES = "No matching excerpts in the corpus for that query and year range."


class SearchLike(Protocol):
    store: Any

    async def search(
        self, query: str, *, k: int, year_from: int | None = None, year_to: int | None = None
    ) -> list[Hit]: ...


def make_tools(retriever: SearchLike, *, k: int) -> list[BaseTool]:
    async def search_publications(
        query: str, year_from: int | None = None, year_to: int | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        """Search the corpus of academic papers for passages relevant to a query.

        Returns numbered excerpts with author, year, section, page and
        identifier so they can be cited. Use a focused natural-language query;
        narrow by publication year with year_from and year_to (inclusive).
        Call it again with a different query to cover another aspect.
        """
        try:
            hits = await retriever.search(query, k=k, year_from=year_from, year_to=year_to)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, logged here
            logger.warning("search_publications failed: %s", exc)
            return UNAVAILABLE, []
        if not hits:
            return NO_MATCHES, []
        sources = [Source.from_hit(h, n) for n, h in enumerate(hits, start=1)]
        content = "\n\n".join(format_excerpt(s, h) for s, h in zip(sources, hits, strict=True))
        return content, [s.model_dump() for s in sources]

    async def list_publications() -> str:
        """List every paper in the corpus with author, year, title and identifier.

        Use it when asked what the corpus contains or which papers are available.
        """
        # LanceDB reads are synchronous; keep them off the event loop.
        papers = await asyncio.to_thread(retriever.store.list_papers)
        if not papers:
            return "The corpus is empty."
        return "\n".join(f"- {p.citation_line()} [{p.chunk_count} excerpts]" for p in papers)

    return [
        StructuredTool.from_function(
            coroutine=search_publications,
            name="search_publications",
            response_format="content_and_artifact",
        ),
        StructuredTool.from_function(coroutine=list_publications, name="list_publications"),
    ]


def format_excerpt(source: Source, hit: Hit) -> str:
    paper = hit.paper
    year = paper.year if paper.year is not None else "n.d."
    header = (
        f"[{source.n}] {paper.short_authors} ({year}). {paper.title}. "
        f"§{source.section}, p. {source.page}. {paper.citation_id}"
    )
    return f"{header}\n{hit.chunk.text}"
