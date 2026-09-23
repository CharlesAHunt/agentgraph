"""Embedded LanceDB storage for papers and their chunks.

Two tables live under one directory:

* ``papers``: one row per paper, keyed by :func:`~lgraph.rag.documents.paper_key`.
* ``chunks``: one row per chunk with the paper's citation metadata copied in,
  so a search hit needs no join.

Search is hybrid (dense vector + full text, fused by reciprocal rank). Filter
clauses are built only from validated integers; free text never reaches SQL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Sequence

import lancedb
import pyarrow as pa
from lancedb.index import FTS

from ..config import ConfigError
from .documents import Chunk, Hit, Paper

PAPERS = "papers"
CHUNKS = "chunks"


def _papers_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("key", pa.string()),
            pa.field("title", pa.string()),
            pa.field("authors", pa.list_(pa.string())),
            pa.field("year", pa.int32()),
            pa.field("doi", pa.string()),
            pa.field("arxiv_id", pa.string()),
            pa.field("venue", pa.string()),
            pa.field("abstract", pa.string()),
            pa.field("pdf_url", pa.string()),
            pa.field("source", pa.string()),
            pa.field("chunk_count", pa.int32()),
        ]
    )


def _chunks_schema(dimensions: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("paper_key", pa.string()),
            pa.field("title", pa.string()),
            pa.field("authors", pa.list_(pa.string())),
            pa.field("year", pa.int32()),
            pa.field("doi", pa.string()),
            pa.field("arxiv_id", pa.string()),
            pa.field("section", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("order", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dimensions)),
        ]
    )


class PaperStore:
    def __init__(self, papers: lancedb.table.Table, chunks: lancedb.table.Table, dimensions: int) -> None:
        self._papers = papers
        self._chunks = chunks
        self.dimensions = dimensions

    @classmethod
    def open(cls, path: Path, *, dimensions: int) -> PaperStore:
        """Connect to ``path``, creating the tables on first use.

        Raises :class:`ConfigError` if an existing chunk table was built with
        a different embedding width: the corpus would need re-ingesting.
        """
        db = lancedb.connect(str(path))
        existing = set(db.list_tables().tables)

        papers = (
            db.open_table(PAPERS)
            if PAPERS in existing
            else db.create_table(PAPERS, schema=_papers_schema())
        )
        if CHUNKS in existing:
            chunks = db.open_table(CHUNKS)
            width = chunks.schema.field("vector").type.list_size
            if width != dimensions:
                raise ConfigError(
                    f"Store at {path} has {width}-dimensional embeddings but "
                    f"embedding_dimensions is {dimensions}. Re-ingest or change the setting."
                )
        else:
            chunks = db.create_table(CHUNKS, schema=_chunks_schema(dimensions))
            # Hybrid search needs a full-text index to exist, even on an
            # empty table. Rows added later are searched without reindexing.
            chunks.create_index("text", config=FTS(), replace=True)
        return cls(papers, chunks, dimensions)

    # -- writes -------------------------------------------------------------

    def upsert(self, paper: Paper, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Replace ``paper``'s chunks and metadata."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._chunks.delete(f"paper_key = '{_escape(paper.key)}'")
        if chunks:
            self._chunks.add(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "paper_key": c.paper_key,
                        "title": paper.title,
                        "authors": list(paper.authors),
                        "year": paper.year,
                        "doi": paper.doi,
                        "arxiv_id": paper.arxiv_id,
                        "section": c.section,
                        "page": c.page,
                        "order": c.order,
                        "text": c.text,
                        "vector": [float(x) for x in v],
                    }
                    for c, v in zip(chunks, vectors, strict=True)
                ]
            )
        row = {**paper.to_record(), "chunk_count": len(chunks)}
        (
            self._papers.merge_insert("key")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute([row])
        )

    def rebuild_fts(self) -> None:
        """(Re)build the full-text index over chunk text.

        Unindexed rows are still searched, so this is about keeping the
        index compact after an ingest run rather than correctness.
        """
        # LanceDB's sync API takes the column as the first positional argument
        # when an FTS config is given.
        self._chunks.create_index("text", config=FTS(), replace=True)

    # -- reads --------------------------------------------------------------

    def refresh(self) -> None:
        """See writes made by other processes (an ingest run beside a server).

        A LanceDB table handle stays on the version it opened; this moves
        both handles to the latest version. Cheap: metadata only.
        """
        self._papers.checkout_latest()
        self._chunks.checkout_latest()

    def search(
        self,
        text: str,
        vector: Sequence[float],
        k: int,
        *,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[Hit]:
        where = _year_clause(year_from, year_to)
        self.refresh()
        if self._chunks.count_rows() == 0:
            return []
        query = self._chunks.search(query_type="hybrid").vector(list(vector)).text(text)
        if where:
            query = query.where(where, prefilter=True)
        rows = query.limit(k).to_list()
        return [_row_to_hit(r) for r in rows]

    def list_papers(self) -> list[Paper]:
        """All papers with their stored ``chunk_count``, oldest first."""
        self.refresh()
        rows = self._papers.to_arrow().to_pylist()
        rows.sort(key=lambda r: ((r.get("year") or 0), r["title"]))
        return [Paper.from_record(r) for r in rows]

    def chunk_count(self, key: str) -> int:
        self.refresh()
        return self._chunks.count_rows(f"paper_key = '{_escape(key)}'")

    def has_paper(self, key: str) -> bool:
        self.refresh()
        return self._papers.count_rows(f"key = '{_escape(key)}'") > 0

    def count_chunks(self) -> int:
        self.refresh()
        return self._chunks.count_rows()


def _year_clause(year_from: int | None, year_to: int | None) -> str:
    parts: list[str] = []
    for name, op, value in (("year_from", ">=", year_from), ("year_to", "<=", year_to)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        parts.append(f"year {op} {value}")
    return " AND ".join(parts)


def _escape(value: str) -> str:
    return value.replace("'", "''")


def _row_to_hit(row: dict[str, Any]) -> Hit:
    return Hit(
        chunk=Chunk(
            chunk_id=row["chunk_id"],
            paper_key=row["paper_key"],
            section=row["section"],
            page=row["page"],
            order=row["order"],
            text=row["text"],
        ),
        title=row["title"],
        authors=tuple(row["authors"] or ()),
        year=row["year"],
        doi=row["doi"],
        arxiv_id=row["arxiv_id"],
        score=float(row.get("_relevance_score") or 0.0),
    )
