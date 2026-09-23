from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from lgraph.__main__ import _parser, load_items, main
from lgraph.rag.documents import Paper


def test_default_command_is_serve() -> None:
    assert _parser().parse_args([]).command is None  # resolved to serve in main()
    assert _parser().parse_args(["serve"]).command == "serve"


def test_ingest_arguments() -> None:
    args = _parser().parse_args(["ingest", "1706.03762", "10.1000/x"])
    assert args.command == "ingest" and args.identifiers == ["1706.03762", "10.1000/x"]
    assert args.from_file is None and args.limit is None and args.delay == 3.0 and not args.no_skip_existing

    args = _parser().parse_args(["ingest", "--from", "p.jsonl", "--limit", "5", "--delay", "0", "--no-skip-existing"])
    assert args.identifiers == [] and args.from_file == Path("p.jsonl") and args.limit == 5
    assert args.delay == 0.0 and args.no_skip_existing


def test_discover_arguments() -> None:
    args = _parser().parse_args(["discover", "--since", "2019-09-19", "--match", "fusion,tokamak", "-o", "out.jsonl"])
    assert args.command == "discover" and args.category == "physics.plasm-ph"
    assert args.since == date(2019, 9, 19) and args.until is None
    assert args.match == "fusion,tokamak" and args.output == Path("out.jsonl")
    assert _parser().parse_args(["discover", "--since", "2019-09-19"]).output == Path("data/papers.jsonl")
    with pytest.raises(SystemExit):
        _parser().parse_args(["discover"])  # --since is required


def test_discover_does_not_require_an_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import lgraph.__main__ as cli

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    seen = {}

    async def fake_discover(args):
        seen["output"] = args.output
        return 0

    monkeypatch.setattr(cli, "_discover", fake_discover)
    with pytest.raises(SystemExit) as exc:
        main(["discover", "--since", "2025-01-01", "-o", str(tmp_path / "p.jsonl")])
    assert exc.value.code == 0 and seen["output"] == tmp_path / "p.jsonl"


def test_papers_requires_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["papers"])
    assert exc.value.code == 2


def test_papers_command() -> None:
    assert _parser().parse_args(["papers"]).command == "papers"


def test_load_items_mixes_ids_and_records(tmp_path: Path) -> None:
    f = tmp_path / "items.txt"
    f.write_text(
        "# comment\n1706.03762\n\n"
        '{"key": "arxiv:2505.00001", "title": "T", "authors": ["A B"], "year": 2025, "arxiv_id": "2505.00001", '
        '"pdf_url": "https://arxiv.org/pdf/2505.00001v1", "source": "arxiv"}\n'
        "10.1000/x\n"
    )
    items = load_items(f)
    assert items[0] == "1706.03762" and items[2] == "10.1000/x"
    assert isinstance(items[1], Paper) and items[1].authors == ("A B",) and items[1].year == 2025
