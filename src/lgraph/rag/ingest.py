"""The ingestion pipeline: identifier -> metadata -> PDF -> chunks -> store."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Awaitable, Callable, Sequence

import httpx
from langchain_core.embeddings import Embeddings
from openrouter.errors import NoResponseError, OpenRouterError

from .chunk import chunk_document
from .documents import Paper
from .embeddings import EmbeddingDimensionError
from .parse import IngestError, Parser
from .sources import ArxivLookup, arxiv_lookup, classify, download_pdf, pdf_filename, resolve_doi
from .store import PaperStore

logger = logging.getLogger(__name__)

CONNECTIVITY_PREFIX = "OpenRouter/network error:"
IngestItem = str | Paper
Progress = Callable[["IngestReport", int, int], None]
Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class IngestReport:
    identifier: str
    ok: bool
    key: str | None = None
    title: str | None = None
    chunks: int = 0
    error: str | None = None
    skipped: bool = False


@dataclass
class Ingestor:
    parser: Parser
    embeddings: Embeddings
    store: PaperStore
    http: httpx.AsyncClient
    pdf_dir: Path
    contact_email: str | None = None
    lookup_arxiv: ArxivLookup = arxiv_lookup
    # Bulk-run behaviour.
    skip_existing: bool = True
    download_delay_s: float = 0.0
    # Stop the batch after this many consecutive network/provider failures:
    # the machine is offline or the provider is down, and every further item
    # would fail the same way. 0 disables.
    max_consecutive_failures: int = 10
    sleep: Sleep = field(default=asyncio.sleep, repr=False)

    async def ingest(
        self, items: Sequence[IngestItem], *, on_progress: Progress | None = None
    ) -> list[IngestReport]:
        """Ingest ``items`` in order.

        Returns one report per item attempted. Fewer reports than items means
        the batch was stopped early (see ``max_consecutive_failures``); the
        untouched items are simply not reported and a re-run picks them up.
        """
        reports: list[IngestReport] = []
        streak = 0
        for index, item in enumerate(items, start=1):
            report = await self.ingest_one(item)
            reports.append(report)
            if on_progress:
                on_progress(report, index, len(items))
            streak = streak + 1 if (not report.ok and _is_connectivity_error(report)) else 0
            if self.max_consecutive_failures and streak >= self.max_consecutive_failures:
                logger.error(
                    "stopping after %d consecutive network/provider failures; %d items not attempted",
                    streak, len(items) - index,
                )
                break
        if any(r.ok and not r.skipped for r in reports):
            self.store.rebuild_fts()
        return reports

    async def ingest_one(self, item: IngestItem) -> IngestReport:
        identifier = item if isinstance(item, str) else item.key
        try:
            paper = item if isinstance(item, Paper) else await self.resolve(item)
            # A paper row with zero chunks (an earlier upsert died between its
            # delete and add steps) must be treated as absent, not done.
            existing = self.store.chunk_count(paper.key) if self.skip_existing else 0
            if existing > 0:
                return IngestReport(
                    identifier=identifier, ok=True, key=paper.key, title=paper.title,
                    chunks=existing, skipped=True,
                )
            pdf = await self.fetch_pdf(paper)
            doc = await self.parser.parse(pdf)
            chunks = chunk_document(paper, doc)
            if not chunks:
                raise IngestError("Parsed document produced no text")
            vectors = await self.embeddings.aembed_documents([c.text for c in chunks])
            self.store.upsert(paper, chunks, vectors)
        except IngestError as exc:
            logger.warning("ingest %s failed: %s", identifier, exc)
            return IngestReport(identifier=identifier, ok=False, error=str(exc))
        except (OpenRouterError, NoResponseError, httpx.HTTPError, EmbeddingDimensionError) as exc:
            # Provider or network trouble: report cleanly, no traceback.
            message = f"{CONNECTIVITY_PREFIX} {_short(exc)}"
            logger.warning("ingest %s failed: %s", identifier, message)
            return IngestReport(identifier=identifier, ok=False, error=message)
        except Exception as exc:  # noqa: BLE001 - one bad paper must not stop the batch
            logger.exception("ingest %s failed unexpectedly", identifier)
            return IngestReport(identifier=identifier, ok=False, error=f"{type(exc).__name__}: {exc}")
        return IngestReport(
            identifier=identifier, ok=True, key=paper.key, title=paper.title, chunks=len(chunks)
        )

    async def resolve(self, identifier: str) -> Paper:
        ident = classify(identifier)
        if ident.kind == "arxiv":
            return self.lookup_arxiv(ident.value)
        if not self.contact_email:
            raise IngestError(
                "Resolving a DOI needs LGRAPH_CONTACT_EMAIL (Crossref and Unpaywall ask for it)."
            )
        return await resolve_doi(ident.value, email=self.contact_email, http=self.http)

    async def fetch_pdf(self, paper: Paper) -> Path:
        dest = self.pdf_dir / pdf_filename(paper.key)
        if dest.exists():
            return dest
        if not paper.pdf_url:
            raise IngestError(f"No PDF url for {paper.key}")
        path = await download_pdf(paper.pdf_url, dest, http=self.http)
        if self.download_delay_s > 0:
            # Courtesy delay between downloads from the same publisher (arXiv
            # in particular throttles and blocks bulk fetchers).
            await self.sleep(self.download_delay_s)
        return path


def _is_connectivity_error(report: IngestReport) -> bool:
    return bool(report.error and report.error.startswith(CONNECTIVITY_PREFIX))


async def preflight_embeddings(embeddings: Embeddings) -> str | None:
    """Embed one short string; return an error message if the provider is unusable.

    Run before a bulk ingest so a bad key or a wrong embedding width fails in
    seconds instead of after hours of downloads.
    """
    try:
        await embeddings.aembed_query("connectivity check")
    except EmbeddingDimensionError as exc:
        return str(exc)
    except (OpenRouterError, NoResponseError, httpx.HTTPError, ValueError) as exc:
        return f"OpenRouter embeddings request failed: {_short(exc)}"
    return None


def _short(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    status = getattr(exc, "status_code", None)
    if status is not None:
        text = f"HTTP {status}: {text}"
    return text[:300]
