"""Resolve arXiv ids and DOIs to paper metadata and an open-access PDF.

* arXiv ids go through the arXiv API (via the ``arxiv`` package, imported
  lazily) which gives metadata and the PDF url directly.
* DOIs go to Crossref for metadata and to Unpaywall for an open-access PDF
  location. Both services ask for a contact email.

Downloads are https-only, streamed with a size cap, and written under the
configured PDF directory using a sanitised file name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

import httpx

from .documents import Paper, paper_key
from .parse import IngestError

ARXIV_NEW = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
ARXIV_OLD = re.compile(r"^([a-z\-]+(\.[A-Z]{2})?/\d{7})(v\d+)?$")
ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)
DOI = re.compile(r"^10\.\d{4,9}/\S+$")
MAX_PDF_BYTES = 50 * 1024 * 1024
CROSSREF = "https://api.crossref.org/works/"
UNPAYWALL = "https://api.unpaywall.org/v2/"


@dataclass(frozen=True, slots=True)
class Identifier:
    kind: str  # "arxiv" | "doi"
    value: str


def classify(raw: str) -> Identifier:
    """Recognise an arXiv id or a DOI, tolerating common prefixes and urls."""
    s = raw.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "DOI:"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/", "https://arxiv.org/pdf/", "arXiv:", "arxiv:"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    if s.endswith(".pdf"):
        s = s[: -len(".pdf")]

    if m := ARXIV_DOI.match(s):
        return Identifier("arxiv", m.group(1))
    if ARXIV_NEW.match(s) or ARXIV_OLD.match(s):
        return Identifier("arxiv", s)
    if DOI.match(s):
        return Identifier("doi", s)
    raise IngestError(f"Not an arXiv id or DOI: {raw!r}")


class ArxivLookup(Protocol):
    def __call__(self, arxiv_id: str) -> Paper: ...


def arxiv_lookup(arxiv_id: str) -> Paper:
    """Fetch metadata for one arXiv id using the ``arxiv`` package."""
    try:
        import arxiv  # noqa: PLC0415
    except ImportError as exc:
        raise IngestError("The arxiv package is not installed. Run: uv sync --extra ingest") from exc

    results = list(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
    if not results:
        raise IngestError(f"arXiv has no record for {arxiv_id}")
    r = results[0]
    short_id = r.get_short_id()
    return Paper(
        key=paper_key(doi=None, arxiv_id=short_id),
        title=" ".join(str(r.title).split()),
        authors=tuple(a.name for a in r.authors),
        year=r.published.year if r.published else None,
        doi=(r.doi or None),
        arxiv_id=short_id,
        venue=r.journal_ref or None,
        abstract=" ".join(str(r.summary).split()) or None,
        pdf_url=r.pdf_url,
        source="arxiv",
    )


async def resolve_doi(doi: str, *, email: str, http: httpx.AsyncClient) -> Paper:
    """Crossref for metadata, Unpaywall for an open-access PDF url."""
    meta = await _get_json(http, f"{CROSSREF}{quote(doi, safe='')}", params={"mailto": email})
    message = meta.get("message") or {}
    if not message:
        raise IngestError(f"Crossref has no record for {doi}")

    oa = await _get_json(http, f"{UNPAYWALL}{quote(doi, safe='')}", params={"email": email})
    pdf_url = _best_pdf_url(oa)
    if not pdf_url:
        raise IngestError(f"No open-access PDF found for {doi}")

    return Paper(
        key=paper_key(doi=doi),
        title=" ".join(str(_first(message.get("title")) or "").split()) or doi,
        authors=tuple(_author_name(a) for a in message.get("author") or []),
        year=_year(message),
        doi=message.get("DOI") or doi,
        arxiv_id=None,
        venue=_first(message.get("container-title")),
        abstract=_strip_tags(message.get("abstract")) or None,
        pdf_url=pdf_url,
        source="crossref",
    )


async def download_pdf(
    url: str, dest: Path, *, http: httpx.AsyncClient, max_bytes: int = MAX_PDF_BYTES
) -> Path:
    """Stream ``url`` to ``dest`` with scheme, size and content checks."""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.netloc:
        raise IngestError(f"Refusing to download non-https url: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    size = 0
    try:
        async with http.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
            if response.status_code != 200:
                raise IngestError(f"PDF download failed with HTTP {response.status_code}: {url}")
            with tmp.open("wb") as fh:
                async for block in response.aiter_bytes():
                    size += len(block)
                    if size > max_bytes:
                        raise IngestError(f"PDF exceeds {max_bytes} bytes: {url}")
                    fh.write(block)
        with tmp.open("rb") as fh:
            head = fh.read(5)
        if head != b"%PDF-":
            raise IngestError(f"Downloaded file is not a PDF: {url}")
        tmp.replace(dest)
    except httpx.HTTPError as exc:
        raise IngestError(f"PDF download failed: {exc}") from exc
    finally:
        # Whatever happened (error, cancellation, kill signal that Python
        # still gets to handle), never leave a partial file behind.
        tmp.unlink(missing_ok=True)
    return dest


def pdf_filename(key: str) -> str:
    """A file name derived from the paper key, restricted to safe characters."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("._") + ".pdf"


# -- helpers ------------------------------------------------------------------


async def _get_json(http: httpx.AsyncClient, url: str, *, params: dict[str, str]) -> dict[str, Any]:
    try:
        response = await http.get(url, params=params, timeout=30.0, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise IngestError(f"Request to {url} failed: {exc}") from exc
    if response.status_code == 404:
        return {}
    if response.status_code != 200:
        raise IngestError(f"{url} returned HTTP {response.status_code}")
    try:
        data = response.json()
    except ValueError as exc:
        raise IngestError(f"{url} returned invalid JSON") from exc
    return data if isinstance(data, dict) else {}


def _best_pdf_url(oa: dict[str, Any]) -> str | None:
    best = oa.get("best_oa_location") or {}
    if url := best.get("url_for_pdf"):
        return str(url)
    for loc in oa.get("oa_locations") or []:
        if url := (loc or {}).get("url_for_pdf"):
            return str(url)
    return None


def _first(value: Any) -> str | None:
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value else None


def _author_name(author: dict[str, Any]) -> str:
    if name := author.get("name"):
        return str(name)
    return " ".join(p for p in (author.get("given"), author.get("family")) if p)


def _year(message: dict[str, Any]) -> int | None:
    for field_name in ("published-print", "published-online", "issued", "created"):
        parts = (message.get(field_name) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def _strip_tags(text: Any) -> str:
    if not text:
        return ""
    return " ".join(re.sub(r"<[^>]+>", " ", str(text)).split())
