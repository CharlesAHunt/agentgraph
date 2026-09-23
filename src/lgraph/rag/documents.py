"""Plain data types shared by the RAG modules."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel

_VERSION_SUFFIX = re.compile(r"v\d+$")


def paper_key(*, doi: str | None = None, arxiv_id: str | None = None) -> str:
    """Stable corpus key: the DOI when there is one, else the arXiv id.

    arXiv versions are stripped so v1 and v2 of a preprint share a key.
    """
    if doi:
        return doi.strip().lower()
    if arxiv_id:
        return f"arxiv:{_VERSION_SUFFIX.sub('', arxiv_id.strip().lower())}"
    raise ValueError("A paper needs a DOI or an arXiv id.")


def chunk_id(key: str, order: int) -> str:
    return f"{key}#{order:04d}"


@dataclass(frozen=True, slots=True)
class Paper:
    key: str
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None
    abstract: str | None = None
    pdf_url: str | None = None
    source: str = ""
    # Chunks currently stored for this paper; 0 for a paper not yet ingested.
    chunk_count: int = 0

    def to_record(self) -> dict[str, Any]:
        """JSON-friendly dict (authors as a list). Inverse of :meth:`from_record`."""
        record = asdict(self)
        record["authors"] = list(self.authors)
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Paper:
        return cls(
            key=record["key"],
            title=record.get("title") or record["key"],
            authors=tuple(record.get("authors") or ()),
            year=record.get("year"),
            doi=record.get("doi"),
            arxiv_id=record.get("arxiv_id"),
            venue=record.get("venue"),
            abstract=record.get("abstract"),
            pdf_url=record.get("pdf_url"),
            source=record.get("source") or "",
            chunk_count=int(record.get("chunk_count") or 0),
        )

    def citation_line(self) -> str:
        """``Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762``"""
        year = self.year if self.year is not None else "n.d."
        return f"{self.short_authors} ({year}). {self.title}. {self.citation_id}"

    @property
    def citation_id(self) -> str:
        """Identifier to print in citations: DOI first, else arXiv."""
        if self.doi:
            return self.doi
        if self.arxiv_id:
            return f"arXiv:{self.arxiv_id}"
        return self.key

    @property
    def short_authors(self) -> str:
        if not self.authors:
            return "Unknown"
        first = self.authors[0].split()[-1] if self.authors[0] else "Unknown"
        return first if len(self.authors) == 1 else f"{first} et al."


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    paper_key: str
    section: str
    page: int
    order: int
    text: str


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    title: str
    authors: tuple[str, ...]
    year: int | None
    doi: str | None
    arxiv_id: str | None
    score: float = field(default=0.0)

    @property
    def paper(self) -> Paper:
        return Paper(
            key=self.chunk.paper_key,
            title=self.title,
            authors=self.authors,
            year=self.year,
            doi=self.doi,
            arxiv_id=self.arxiv_id,
        )


class Source(BaseModel):
    """A retrieved excerpt as reported to API clients.

    ``n`` is the position in the response's ``sources`` list, not a citation
    key; citations in the answer text use author, year and page.
    """

    n: int
    title: str
    authors: list[str]
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    section: str
    page: int
    chunk_id: str

    @classmethod
    def from_hit(cls, hit: Hit, n: int) -> Source:
        return cls(
            n=n,
            title=hit.title,
            authors=list(hit.authors),
            year=hit.year,
            doi=hit.doi,
            arxiv_id=hit.arxiv_id,
            section=hit.chunk.section,
            page=hit.chunk.page,
            chunk_id=hit.chunk.chunk_id,
        )
