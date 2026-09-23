from __future__ import annotations

import json
from pathlib import Path

import httpx

from lgraph.rag.ingest import Ingestor
from lgraph.rag.parse import IngestError, ParsedDocument, items_from_structured_content
from lgraph.rag.store import PaperStore

from ..conftest import FakeEmbeddings, make_paper

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "structured_content.json"


class FakeParser:
    def __init__(self, fail_for: set[str] | None = None) -> None:
        self.parsed: list[Path] = []
        self.fail_for = fail_for or set()

    async def parse(self, pdf_path: Path) -> ParsedDocument:
        self.parsed.append(pdf_path)
        if pdf_path.name in self.fail_for:
            raise IngestError("MinerU choked")
        return ParsedDocument(items=items_from_structured_content(json.loads(FIXTURE.read_text())))


def _pdf_http() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".pdf"):
            return httpx.Response(200, content=b"%PDF-1.4 fake")
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ingestor(tmp_path: Path, parser: FakeParser, lookups: dict[str, object]) -> tuple[Ingestor, PaperStore, FakeEmbeddings]:
    emb = FakeEmbeddings(8)
    store = PaperStore.open(tmp_path / "lancedb", dimensions=8)

    def lookup(arxiv_id: str):
        paper = lookups[arxiv_id]
        if isinstance(paper, Exception):
            raise paper
        return paper

    ing = Ingestor(parser=parser, embeddings=emb, store=store, http=_pdf_http(),
                   pdf_dir=tmp_path / "pdfs", contact_email=None, lookup_arxiv=lookup)
    return ing, store, emb


async def test_ingest_arxiv_end_to_end(tmp_path: Path) -> None:
    paper = make_paper(pdf_url="https://arxiv.org/pdf/1706.03762.pdf", abstract="We propose the Transformer.")
    parser = FakeParser()
    ing, store, emb = _ingestor(tmp_path, parser, {"1706.03762": paper})

    [report] = await ing.ingest(["1706.03762"])

    assert report.ok and report.key == paper.key and report.chunks > 1
    assert store.chunk_count(paper.key) == report.chunks
    assert (tmp_path / "pdfs" / "arxiv_1706.03762.pdf").exists()
    assert parser.parsed == [tmp_path / "pdfs" / "arxiv_1706.03762.pdf"]
    assert sum(len(c) for c in emb.calls) == report.chunks
    hits = store.search("multi-head attention", emb.embed_query("multi-head attention"), k=1)
    assert hits and hits[0].chunk.paper_key == paper.key


async def test_reingest_uses_cached_pdf_and_replaces_chunks(tmp_path: Path) -> None:
    paper = make_paper(pdf_url="https://arxiv.org/pdf/1706.03762.pdf")
    parser = FakeParser()
    ing, store, _ = _ingestor(tmp_path, parser, {"1706.03762": paper})
    first = (await ing.ingest(["1706.03762"]))[0]
    pdf = tmp_path / "pdfs" / "arxiv_1706.03762.pdf"
    pdf.write_bytes(b"%PDF-cached")

    second = (await ing.ingest(["arXiv:1706.03762"]))[0]

    assert second.ok and second.chunks == first.chunks
    assert pdf.read_bytes() == b"%PDF-cached"  # not re-downloaded
    assert store.chunk_count(paper.key) == first.chunks
    assert len(store.list_papers()) == 1


async def test_failures_are_reported_per_paper_and_do_not_stop_the_batch(tmp_path: Path) -> None:
    good = make_paper(pdf_url="https://arxiv.org/pdf/1706.03762.pdf")
    bad = make_paper(key="arxiv:2000.00001", arxiv_id="2000.00001", title="Bad",
                     pdf_url="https://arxiv.org/pdf/2000.00001.pdf")
    parser = FakeParser(fail_for={"arxiv_2000.00001.pdf"})
    ing, store, _ = _ingestor(tmp_path, parser, {
        "1706.03762": good, "2000.00001": bad, "1999.99999": IngestError("arXiv has no record"),
    })

    reports = await ing.ingest(["1706.03762", "2000.00001", "1999.99999", "not-an-id", "10.1000/x"])

    assert [r.ok for r in reports] == [True, False, False, False, False]
    assert "MinerU choked" in reports[1].error
    assert "no record" in reports[2].error
    assert "Not an arXiv id or DOI" in reports[3].error
    assert "LGRAPH_CONTACT_EMAIL" in reports[4].error
    assert len(store.list_papers()) == 1


async def test_unexpected_exception_is_captured(tmp_path: Path) -> None:
    class ExplodingParser(FakeParser):
        async def parse(self, pdf_path: Path) -> ParsedDocument:
            raise RuntimeError("segfault-ish")

    paper = make_paper(pdf_url="https://arxiv.org/pdf/1706.03762.pdf")
    ing, _, _ = _ingestor(tmp_path, ExplodingParser(), {"1706.03762": paper})
    [report] = await ing.ingest(["1706.03762"])
    assert not report.ok and report.error == "RuntimeError: segfault-ish"


async def test_ingest_accepts_paper_records_without_lookup(tmp_path: Path) -> None:
    paper = make_paper(pdf_url="https://arxiv.org/pdf/1706.03762.pdf")
    ing, store, _ = _ingestor(tmp_path, FakeParser(), {})  # no lookups available
    [report] = await ing.ingest([paper])
    assert report.ok and report.identifier == paper.key and store.has_paper(paper.key)


async def test_skip_existing_and_progress(tmp_path: Path) -> None:
    paper = make_paper(pdf_url="https://arxiv.org/pdf/1706.03762.pdf")
    parser = FakeParser()
    ing, store, _ = _ingestor(tmp_path, parser, {"1706.03762": paper})
    await ing.ingest(["1706.03762"])
    seen: list[tuple[str, int, int]] = []

    reports = await ing.ingest([paper, "1706.03762"], on_progress=lambda r, i, n: seen.append((r.identifier, i, n)))

    assert all(r.ok and r.skipped for r in reports)
    assert reports[0].chunks == store.chunk_count(paper.key) > 0
    assert len(parser.parsed) == 1  # nothing re-parsed
    assert seen == [(paper.key, 1, 2), ("1706.03762", 2, 2)]

    ing.skip_existing = False
    [again] = await ing.ingest([paper])
    assert again.ok and not again.skipped and len(parser.parsed) == 2


async def test_download_delay_applies_only_to_real_downloads(tmp_path: Path) -> None:
    paper = make_paper(pdf_url="https://arxiv.org/pdf/1706.03762.pdf")
    ing, _, _ = _ingestor(tmp_path, FakeParser(), {})
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    ing.download_delay_s = 3.0
    ing.sleep = fake_sleep
    ing.skip_existing = False
    await ing.ingest([paper])
    await ing.ingest([paper])  # PDF now cached
    assert slept == [3.0]



class _FailingEmbeddings(FakeEmbeddings):
    def __init__(self, error: Exception) -> None:
        super().__init__(8)
        self.error = error

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise self.error

    def embed_query(self, text: str) -> list[float]:
        raise self.error


async def test_provider_errors_are_reported_cleanly(tmp_path: Path) -> None:
    import httpx as _httpx
    from openrouter.errors import OpenRouterError

    from lgraph.rag.ingest import Ingestor as _Ingestor
    from lgraph.rag.store import PaperStore as _Store

    paper = make_paper(pdf_url="https://arxiv.org/pdf/1706.03762.pdf")
    error = OpenRouterError("User not found.", raw_response=_httpx.Response(401, text="User not found."))
    ing = _Ingestor(parser=FakeParser(), embeddings=_FailingEmbeddings(error), store=_Store.open(tmp_path / "db", dimensions=8),
                    http=_pdf_http(), pdf_dir=tmp_path / "pdfs")
    [report] = await ing.ingest([paper])
    assert not report.ok
    assert report.error == "OpenRouter/network error: HTTP 401: User not found."


async def test_preflight_embeddings() -> None:
    import httpx as _httpx
    from openrouter.errors import OpenRouterError

    from lgraph.rag.embeddings import EmbeddingDimensionError
    from lgraph.rag.ingest import preflight_embeddings

    assert await preflight_embeddings(FakeEmbeddings(8)) is None
    bad_key = OpenRouterError("User not found.", raw_response=_httpx.Response(401, text="x"))
    msg = await preflight_embeddings(_FailingEmbeddings(bad_key))
    assert msg and "HTTP 401" in msg
    msg = await preflight_embeddings(_FailingEmbeddings(EmbeddingDimensionError("model returned 3072-dimensional vectors")))
    assert msg and "3072" in msg
    msg = await preflight_embeddings(_FailingEmbeddings(_httpx.ConnectError("refused")))
    assert msg and "refused" in msg



async def test_batch_stops_after_consecutive_connectivity_failures(tmp_path: Path) -> None:
    import httpx as _httpx

    from lgraph.rag.ingest import Ingestor as _Ingestor
    from lgraph.rag.store import PaperStore as _Store

    papers = [make_paper(key=f"arxiv:2001.0000{i}", arxiv_id=f"2001.0000{i}", pdf_url=f"https://arxiv.org/pdf/2001.0000{i}.pdf")
              for i in range(6)]
    ing = _Ingestor(parser=FakeParser(), embeddings=_FailingEmbeddings(_httpx.ConnectError("offline")),
                    store=_Store.open(tmp_path / "db", dimensions=8), http=_pdf_http(), pdf_dir=tmp_path / "pdfs",
                    max_consecutive_failures=3)
    reports = await ing.ingest(papers)
    assert len(reports) == 3 and all(not r.ok for r in reports)

    ing.max_consecutive_failures = 0  # disabled: every item is attempted
    assert len(await ing.ingest(papers)) == 6


async def test_non_connectivity_failures_do_not_trip_the_breaker(tmp_path: Path) -> None:
    papers = [make_paper(key=f"arxiv:2001.0000{i}", arxiv_id=f"2001.0000{i}", pdf_url=f"https://arxiv.org/pdf/2001.0000{i}.pdf")
              for i in range(4)]
    parser = FakeParser(fail_for={f"arxiv_2001.0000{i}.pdf" for i in range(4)})  # every parse fails
    ing, _, _ = _ingestor(tmp_path, parser, {})
    ing.max_consecutive_failures = 2
    reports = await ing.ingest(papers)
    assert len(reports) == 4  # parse errors are per-paper problems, not connectivity



async def test_paper_row_without_chunks_is_reingested(tmp_path: Path) -> None:
    paper = make_paper(pdf_url="https://arxiv.org/pdf/1706.03762.pdf")
    parser = FakeParser()
    ing, store, emb = _ingestor(tmp_path, parser, {})
    # Simulate an upsert that died after deleting chunks: paper row present, no chunks.
    store.upsert(paper, [], [])
    assert store.has_paper(paper.key) and store.chunk_count(paper.key) == 0

    [report] = await ing.ingest([paper])

    assert report.ok and not report.skipped and report.chunks > 0
