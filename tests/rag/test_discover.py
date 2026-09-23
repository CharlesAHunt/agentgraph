from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest

from lgraph.rag.discover import (
    build_matcher,
    discover_arxiv,
    matches,
    oai_set_for_category,
    parse_oai_page,
)
from lgraph.rag.documents import Paper
from lgraph.rag.parse import IngestError

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "arxiv_oai.xml"


def test_parse_oai_page_extracts_papers() -> None:
    page = parse_oai_page(FIXTURE.read_text())
    assert page.resumption_token is None
    assert page.records[0].created == date(2025, 4, 15)
    a, b, c = page.papers
    assert a.arxiv_id == "2504.11237" and a.key == "arxiv:2504.11237"
    assert a.title == "Periodic table for highly charged ions"
    assert a.year == 2025  # from <created>, i.e. first submission
    assert a.doi == "10.1103/bppp-53g1"
    assert a.venue == "Phys. Rev. Research 7, L042071 (2025)"
    assert a.authors[0].endswith("Lyu") and " " in a.authors[0]  # "Forenames Keyname"
    assert a.pdf_url == "https://arxiv.org/pdf/2504.11237"
    assert a.abstract and "  " not in a.abstract
    assert a.source == "arxiv"
    assert b.doi is None and b.venue is None
    assert c.arxiv_id == "2510.07502"


def test_parse_oai_page_handles_no_records_and_errors() -> None:
    ns = 'xmlns="http://www.openarchives.org/OAI/2.0/"'
    empty = f'<OAI-PMH {ns}><error code="noRecordsMatch">none</error></OAI-PMH>'
    assert parse_oai_page(empty).papers == ()
    with pytest.raises(IngestError, match="badArgument"):
        parse_oai_page(f'<OAI-PMH {ns}><error code="badArgument">bad set</error></OAI-PMH>')
    with pytest.raises(IngestError):
        parse_oai_page("<not xml")


def test_deleted_records_are_skipped() -> None:
    ns = 'xmlns="http://www.openarchives.org/OAI/2.0/"'
    xml = (
        f'<OAI-PMH {ns}><ListRecords><record><header status="deleted"><identifier>oai:arXiv.org:1</identifier>'
        "</header></record></ListRecords></OAI-PMH>"
    )
    assert parse_oai_page(xml).papers == ()


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("physics.plasm-ph", "physics:physics:plasm-ph"),
        ("hep-th", "physics:hep-th"),
        ("cond-mat.soft", "physics:cond-mat:soft"),
        ("cs.AI", "cs:cs:AI"),
        ("math.AG", "math:math:AG"),
        ("q-bio.BM", "q-bio:q-bio:BM"),
    ],
)
def test_oai_set_for_category(category: str, expected: str) -> None:
    assert oai_set_for_category(category) == expected


@pytest.mark.parametrize("category", ["physics.plasm-ph OR all", "cat:x", "", "../x", "a b"])
def test_bad_category_rejected(category: str) -> None:
    with pytest.raises(IngestError, match="category"):
        oai_set_for_category(category)


def test_matcher_is_case_insensitive_and_word_bounded() -> None:
    page = parse_oai_page(FIXTURE.read_text())
    reconnection = build_matcher(["magnetic reconnection", "tokamak"])
    assert [matches(p, reconnection) for p in page.papers] == [False, False, True]
    assert build_matcher(None) is None and build_matcher(["", " "]) is None
    iter_ = build_matcher(["ITER"])
    assert iter_ is not None and not iter_.search("after many iterations") and iter_.search("the ITER device")


def test_record_round_trip() -> None:
    paper = parse_oai_page(FIXTURE.read_text()).papers[0]
    assert Paper.from_record(paper.to_record()) == paper


NS = ('xmlns="http://www.openarchives.org/OAI/2.0/"', "http://arxiv.org/OAI/arXiv/")


def _record(arxiv_id: str, created: str, title: str) -> str:
    return (
        f"<record><header><identifier>oai:arXiv.org:{arxiv_id}</identifier></header><metadata>"
        f'<arXiv xmlns="{NS[1]}"><id>{arxiv_id}</id><created>{created}</created>'
        f"<authors><author><keyname>Doe</keyname><forenames>J.</forenames></author></authors>"
        f"<title>{title}</title><abstract>abstract</abstract></arXiv></metadata></record>"
    )


def _page(*records: str, token: str | None = None, size: int | None = None) -> str:
    tok = ""
    if token is not None:
        attrs = f' completeListSize="{size}"' if size is not None else ""
        tok = f"<resumptionToken{attrs}>{token}</resumptionToken>"
    return f"<OAI-PMH {NS[0]}><ListRecords>{''.join(records)}{tok}</ListRecords></OAI-PMH>"


async def test_discover_follows_resumption_tokens_and_filters() -> None:
    requests: list[httpx.Request] = []
    responses = {
        None: _page(_record("2501.00001", "2025-01-10", "Tokamak A"), _record("1901.00002", "2019-01-10", "Old tokamak"),
                    _record("2403.00004", "2024-03-01", "Tokamak same year but before cutoff"),
                    token="tok1", size=3),
        "tok1": _page(_record("2502.00003", "2025-02-10", "Laser B"), _record("2501.00001", "2025-01-10", "Tokamak A"),
                      token="", size=3),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=responses[request.url.params.get("resumptionToken")])

    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    pages: list[tuple[int, int, int | None]] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        papers = await discover_arxiv(
            "physics.plasm-ph", since=date(2024, 6, 1), until=date(2026, 6, 30),
            match=["tokamak"], http=http, sleep=fake_sleep, on_page=lambda *a: pages.append(a),
        )

    # Papers first submitted before ``since`` (even in the same year) are
    # dropped although they were revised in the window; the duplicate on page
    # two is not double counted; the non-match is out.
    assert [p.arxiv_id for p in papers] == ["2501.00001"]
    first = requests[0].url.params
    assert first["verb"] == "ListRecords" and first["metadataPrefix"] == "arXiv"
    assert first["set"] == "physics:physics:plasm-ph"
    assert first["from"] == "2024-06-01" and first["until"] == "2026-06-30"
    assert requests[1].url.params["resumptionToken"] == "tok1" and "set" not in requests[1].url.params
    assert sleeps == [3.0]
    assert pages == [(3, 1, 3), (5, 1, 3)]


async def test_discover_honours_retry_after_and_retries_errors() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, headers={"Retry-After": "7"}, text="Retry after 7 seconds")
        if attempts == 2:
            return httpx.Response(500, text="oops")
        return httpx.Response(200, text=_page(_record("2505.00009", "2025-05-01", "Fusion Z")))

    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        papers = await discover_arxiv("physics.plasm-ph", since=date(2025, 1, 1), until=date(2025, 12, 31),
                                      http=http, sleep=fake_sleep)
    assert [p.arxiv_id for p in papers] == ["2505.00009"]
    assert attempts == 3
    assert sleeps[0] == 7.0  # Retry-After respected exactly


async def test_discover_gives_up_after_retries() -> None:
    async def no_sleep(_: float) -> None:
        pass

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as http:
        with pytest.raises(IngestError, match="after 5 attempts"):
            await discover_arxiv("physics.plasm-ph", since=date(2025, 1, 1), until=date(2025, 1, 2),
                                 http=http, sleep=no_sleep)


async def test_discover_rejects_inverted_range() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as http:
        with pytest.raises(ValueError):
            await discover_arxiv("physics.plasm-ph", since=date(2026, 1, 1), until=date(2025, 1, 1), http=http)
