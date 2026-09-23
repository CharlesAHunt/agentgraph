"""Enumerate an arXiv category over a date range via arXiv's OAI-PMH service.

OAI-PMH (https://oaipmh.arxiv.org/oai) is arXiv's bulk-harvesting interface:
it walks a whole set with resumption tokens, returns full metadata for each
record, and signals throttling with 503 plus ``Retry-After``. The search API
is not used: it rejects large pages and penalises clients at its cache edge.

``from``/``until`` on OAI filter by *last-updated* datestamp, so a harvest
also returns older papers that were revised in the window. Records are
therefore filtered again on their ``created`` (first submission) date.
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from collections.abc import Awaitable, Callable, Sequence

import httpx

from .documents import Paper, paper_key
from .parse import IngestError

logger = logging.getLogger(__name__)

OAI_URL = "https://oaipmh.arxiv.org/oai"
USER_AGENT = "lgraph/0.1 (corpus builder)"
PAGE_DELAY_S = 3.0
MAX_RETRIES = 5
MAX_RETRY_AFTER_S = 120.0

_OAI = "{http://www.openarchives.org/OAI/2.0/}"
_ARXIV = "{http://arxiv.org/OAI/arXiv/}"
_CATEGORY = re.compile(r"^[a-z\-]+(\.[A-Za-z\-]+)?$")
# arXiv archives that are their own OAI top-level group; everything else
# (hep-th, cond-mat, astro-ph, physics.*, ...) lives under "physics".
_NON_PHYSICS_GROUPS = frozenset({"cs", "math", "q-bio", "q-fin", "stat", "eess", "econ"})

Sleep = Callable[[float], Awaitable[None]]


def oai_set_for_category(category: str) -> str:
    """Map an arXiv category to its OAI set spec, e.g. physics.plasm-ph -> physics:physics:plasm-ph."""
    if not _CATEGORY.match(category):
        raise IngestError(f"Not an arXiv category: {category!r}")
    if "." in category:
        archive, subject = category.split(".", 1)
        group = archive if archive in _NON_PHYSICS_GROUPS else "physics"
        return f"{group}:{archive}:{subject}"
    return f"physics:{category}"


@dataclass(frozen=True, slots=True)
class OaiRecord:
    paper: Paper
    created: date | None  # first-submission date


@dataclass(frozen=True, slots=True)
class OaiPage:
    records: tuple[OaiRecord, ...]
    resumption_token: str | None
    complete_list_size: int | None

    @property
    def papers(self) -> tuple[Paper, ...]:
        return tuple(r.paper for r in self.records)


def parse_oai_page(xml_text: str) -> OaiPage:
    """Parse one ``ListRecords`` response."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise IngestError(f"arXiv OAI returned malformed XML: {exc}") from exc
    error = root.find(f"{_OAI}error")
    if error is not None:
        code = error.get("code") or "unknown"
        if code == "noRecordsMatch":
            return OaiPage((), None, 0)
        raise IngestError(f"arXiv OAI error {code}: {(error.text or '').strip()}")
    records = root.find(f"{_OAI}ListRecords")
    if records is None:
        raise IngestError("arXiv OAI response has no ListRecords element")
    parsed = tuple(p for p in (_record_to_paper(r) for r in records.findall(f"{_OAI}record")) if p)
    token_el = records.find(f"{_OAI}resumptionToken")
    token = (token_el.text or "").strip() if token_el is not None else ""
    size = None
    if token_el is not None and (token_el.get("completeListSize") or "").isdigit():
        size = int(token_el.get("completeListSize"))  # type: ignore[arg-type]
    return OaiPage(parsed, token or None, size)


def _record_to_paper(record: ET.Element) -> OaiRecord | None:
    header = record.find(f"{_OAI}header")
    if header is not None and header.get("status") == "deleted":
        return None
    meta = record.find(f"{_OAI}metadata/{_ARXIV}arXiv")
    if meta is None:
        return None
    arxiv_id = _text(meta, f"{_ARXIV}id").strip()
    if not arxiv_id:
        return None
    created = _text(meta, f"{_ARXIV}created").strip()
    authors = []
    for author in meta.findall(f"{_ARXIV}authors/{_ARXIV}author"):
        name = " ".join(
            part for part in (_text(author, f"{_ARXIV}forenames"), _text(author, f"{_ARXIV}keyname")) if part
        )
        if name:
            authors.append(_squash(name))
    created_date = _parse_date(created)
    paper = Paper(
        key=paper_key(arxiv_id=arxiv_id),
        title=_squash(_text(meta, f"{_ARXIV}title")),
        authors=tuple(authors),
        year=created_date.year if created_date else None,
        doi=_squash(_text(meta, f"{_ARXIV}doi")).split(" ")[0] or None,
        arxiv_id=arxiv_id,
        venue=_squash(_text(meta, f"{_ARXIV}journal-ref")) or None,
        abstract=_squash(_text(meta, f"{_ARXIV}abstract")) or None,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        source="arxiv",
    )
    return OaiRecord(paper=paper, created=created_date)


def _parse_date(text: str) -> date | None:
    try:
        return date.fromisoformat(text.strip()[:10])
    except ValueError:
        return None


def _text(el: ET.Element, path: str) -> str:
    found = el.find(path)
    return (found.text or "") if found is not None else ""


def _squash(text: str) -> str:
    return " ".join(text.split())


def build_matcher(terms: Sequence[str] | None) -> re.Pattern[str] | None:
    cleaned = [t.strip() for t in (terms or []) if t and t.strip()]
    if not cleaned:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in cleaned) + r")\b", re.IGNORECASE)


def matches(paper: Paper, pattern: re.Pattern[str] | None) -> bool:
    if pattern is None:
        return True
    return bool(pattern.search(f"{paper.title}\n{paper.abstract or ''}"))


async def discover_arxiv(
    category: str,
    *,
    since: date,
    until: date,
    match: Sequence[str] | None = None,
    http: httpx.AsyncClient,
    sleep: Sleep = asyncio.sleep,
    on_page: Callable[[int, int, int | None], None] | None = None,
) -> list[Paper]:
    """Papers in ``category`` first submitted between ``since`` and ``until``.

    ``on_page(seen, kept, complete_list_size)`` is called after each page.
    """
    if until < since:
        raise ValueError("until must not be before since")
    set_spec = oai_set_for_category(category)
    pattern = build_matcher(match)
    found: dict[str, Paper] = {}
    seen = 0

    params: dict[str, str] = {
        "verb": "ListRecords",
        "metadataPrefix": "arXiv",
        "set": set_spec,
        "from": since.isoformat(),
        "until": until.isoformat(),
    }
    first = True
    while True:
        if not first:
            await sleep(PAGE_DELAY_S)
        first = False
        page = await _fetch_page(http, params, sleep)
        for record in page.records:
            seen += 1
            paper = record.paper
            # The harvest window is by last-updated date; keep only papers
            # whose first submission falls inside the requested range.
            if record.created is None or not (since <= record.created <= until):
                continue
            if paper.key in found or not matches(paper, pattern):
                continue
            found[paper.key] = paper
        if on_page:
            on_page(seen, len(found), page.complete_list_size)
        if not page.resumption_token:
            break
        params = {"verb": "ListRecords", "resumptionToken": page.resumption_token}

    return list(found.values())


async def _fetch_page(http: httpx.AsyncClient, params: dict[str, str], sleep: Sleep) -> OaiPage:
    last_error: str | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await http.get(OAI_URL, params=params, timeout=180.0)
        except httpx.HTTPError as exc:
            last_error = f"request failed: {exc}"
        else:
            if response.status_code == 200:
                return parse_oai_page(response.text)
            last_error = f"HTTP {response.status_code}"
            if response.status_code == 503:
                # The OAI protocol's flow control: wait exactly as long as asked.
                wait = _retry_after(response.headers.get("Retry-After"))
                logger.info("arXiv OAI asked us to wait %.0fs", wait)
                await sleep(wait)
                continue
        logger.warning("arXiv OAI attempt %d/%d failed (%s)", attempt, MAX_RETRIES, last_error)
        if attempt < MAX_RETRIES:
            await sleep(PAGE_DELAY_S * attempt)
    raise IngestError(f"arXiv OAI failed after {MAX_RETRIES} attempts: {last_error}")


def _retry_after(value: str | None) -> float:
    try:
        seconds = float(value) if value else PAGE_DELAY_S
    except ValueError:
        seconds = PAGE_DELAY_S
    return max(1.0, min(seconds, MAX_RETRY_AFTER_S))
