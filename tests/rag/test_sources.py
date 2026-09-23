from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from lgraph.rag.parse import IngestError
from lgraph.rag.sources import (
    Identifier,
    classify,
    download_pdf,
    pdf_filename,
    resolve_doi,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1706.03762", Identifier("arxiv", "1706.03762")),
        ("1706.03762v5", Identifier("arxiv", "1706.03762v5")),
        ("arXiv:1706.03762", Identifier("arxiv", "1706.03762")),
        ("https://arxiv.org/abs/1706.03762", Identifier("arxiv", "1706.03762")),
        ("https://arxiv.org/pdf/1706.03762.pdf", Identifier("arxiv", "1706.03762")),
        ("hep-th/9901001", Identifier("arxiv", "hep-th/9901001")),
        ("10.48550/arXiv.1706.03762", Identifier("arxiv", "1706.03762")),
        ("10.1038/s41586-021-03819-2", Identifier("doi", "10.1038/s41586-021-03819-2")),
        ("https://doi.org/10.1038/s41586-021-03819-2", Identifier("doi", "10.1038/s41586-021-03819-2")),
        ("doi:10.1000/ABC", Identifier("doi", "10.1000/ABC")),
    ],
)
def test_classify(raw: str, expected: Identifier) -> None:
    assert classify(raw) == expected


@pytest.mark.parametrize("raw", ["", "attention paper", "12345", "10.12/x", "https://example.com"])
def test_classify_rejects_garbage(raw: str) -> None:
    with pytest.raises(IngestError):
        classify(raw)


CROSSREF_OK = {
    "message": {
        "DOI": "10.1038/s41586-021-03819-2",
        "title": ["Highly accurate protein structure prediction with AlphaFold"],
        "author": [{"given": "John", "family": "Jumper"}, {"given": "Richard", "family": "Evans"}],
        "container-title": ["Nature"],
        "published-print": {"date-parts": [[2021, 8, 26]]},
        "abstract": "<jats:p>Proteins are <jats:italic>essential</jats:italic> to life.</jats:p>",
    }
}
UNPAYWALL_OK = {
    "best_oa_location": {"url_for_pdf": "https://www.nature.com/articles/s41586-021-03819-2.pdf"},
    "oa_locations": [],
}


def _http(routes: dict[str, httpx.Response]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        for prefix, response in routes.items():
            if str(request.url).startswith(prefix):
                handler.seen.append(request)  # type: ignore[attr-defined]
                return response
        return httpx.Response(500, text="unexpected " + str(request.url))

    handler.seen = []  # type: ignore[attr-defined]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client.seen = handler.seen  # type: ignore[attr-defined]
    return client


async def test_resolve_doi_combines_crossref_and_unpaywall() -> None:
    http = _http({
        "https://api.crossref.org/works/": httpx.Response(200, json=CROSSREF_OK),
        "https://api.unpaywall.org/v2/": httpx.Response(200, json=UNPAYWALL_OK),
    })
    paper = await resolve_doi("10.1038/s41586-021-03819-2", email="me@example.org", http=http)

    assert paper.key == "10.1038/s41586-021-03819-2"
    assert paper.title.startswith("Highly accurate protein structure")
    assert paper.authors == ("John Jumper", "Richard Evans")
    assert paper.year == 2021
    assert paper.venue == "Nature"
    assert paper.abstract == "Proteins are essential to life."
    assert paper.pdf_url == "https://www.nature.com/articles/s41586-021-03819-2.pdf"
    assert paper.source == "crossref"
    urls = [str(r.url) for r in http.seen]
    assert "mailto=me%40example.org" in urls[0]
    assert "email=me%40example.org" in urls[1]
    # The DOI is percent-encoded in the path.
    assert "10.1038%2Fs41586-021-03819-2" in urls[0]


async def test_resolve_doi_without_oa_pdf_fails() -> None:
    http = _http({
        "https://api.crossref.org/works/": httpx.Response(200, json=CROSSREF_OK),
        "https://api.unpaywall.org/v2/": httpx.Response(200, json={"best_oa_location": None, "oa_locations": []}),
    })
    with pytest.raises(IngestError, match="open-access"):
        await resolve_doi("10.1038/s41586-021-03819-2", email="me@example.org", http=http)


async def test_resolve_doi_unknown_to_crossref_fails() -> None:
    http = _http({"https://api.crossref.org/works/": httpx.Response(404, text="Resource not found.")})
    with pytest.raises(IngestError, match="Crossref"):
        await resolve_doi("10.1000/nope", email="me@example.org", http=http)


async def test_download_pdf_streams_and_verifies(tmp_path: Path) -> None:
    http = _http({"https://host.example/paper.pdf": httpx.Response(200, content=b"%PDF-1.7\n...")})
    dest = tmp_path / "pdfs" / "x.pdf"
    out = await download_pdf("https://host.example/paper.pdf", dest, http=http)
    assert out == dest and dest.read_bytes().startswith(b"%PDF-")
    assert not dest.with_suffix(".pdf.part").exists()


async def test_download_pdf_rejects_http_scheme(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="https"):
        await download_pdf("http://host.example/paper.pdf", tmp_path / "x.pdf", http=_http({}))


async def test_download_pdf_rejects_oversize(tmp_path: Path) -> None:
    http = _http({"https://host.example/big.pdf": httpx.Response(200, content=b"%PDF-" + b"0" * 100)})
    with pytest.raises(IngestError, match="exceeds"):
        await download_pdf("https://host.example/big.pdf", tmp_path / "x.pdf", http=http, max_bytes=50)
    assert list(tmp_path.iterdir()) == []


async def test_download_pdf_rejects_non_pdf(tmp_path: Path) -> None:
    http = _http({"https://host.example/page": httpx.Response(200, content=b"<html>paywall</html>")})
    with pytest.raises(IngestError, match="not a PDF"):
        await download_pdf("https://host.example/page", tmp_path / "x.pdf", http=http)
    assert not (tmp_path / "x.pdf").exists()


def test_pdf_filename_is_sanitised() -> None:
    assert pdf_filename("10.1038/s41586-021-03819-2") == "10.1038_s41586-021-03819-2.pdf"
    assert pdf_filename("arxiv:1706.03762") == "arxiv_1706.03762.pdf"
    assert pdf_filename("../../etc/passwd") == "etc_passwd.pdf"


async def test_download_pdf_removes_partial_file_on_transport_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("connection reset", request=request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(IngestError, match="download failed"):
        await download_pdf("https://host.example/x.pdf", tmp_path / "x.pdf", http=http)
    assert list(tmp_path.iterdir()) == []
