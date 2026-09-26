"""Console entry point.

``lgraph`` (or ``python -m lgraph``) with no arguments serves the API, as it
always has. Subcommands manage the paper corpus:

    lgraph serve
    lgraph discover --since 2019-09-19 --match fusion,tokamak   # writes data/papers.jsonl
    lgraph ingest                                               # ingests data/papers.jsonl
    lgraph ingest 1706.03762 10.1038/s41586-021-03819-2
    lgraph papers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import date
from pathlib import Path
from collections.abc import Sequence

import uvicorn

from .config import ConfigError, Settings, get_settings

DEFAULT_CATEGORY = "physics.plasm-ph"
DEFAULT_DOWNLOAD_DELAY_S = 3.0
DEFAULT_PAPERS_FILE = Path("data/papers.jsonl")


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args(argv)
    command = args.command or "serve"

    # `discover` only talks to arXiv and needs no OpenRouter key.
    if command == "discover":
        raise SystemExit(asyncio.run(_discover(args)))

    try:
        settings = get_settings()
    except ConfigError as exc:
        # A missing API key is an operator error, not a crash worth a traceback.
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if command == "serve":
        _serve(settings)
    elif command == "ingest":
        raise SystemExit(asyncio.run(_ingest(settings, args)))
    elif command == "papers":
        _papers(settings)
    elif command == "sandbox":
        raise SystemExit(asyncio.run(_sandbox(settings)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lgraph", description="LangGraph OpenRouter endpoint server")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="run the HTTP API (default)")

    discover = sub.add_parser("discover", help="list papers in an arXiv category over a date range")
    discover.add_argument("--category", default=DEFAULT_CATEGORY, help=f"arXiv category (default {DEFAULT_CATEGORY})")
    discover.add_argument("--since", type=date.fromisoformat, required=True, help="earliest first-submission date, YYYY-MM-DD")
    discover.add_argument("--until", type=date.fromisoformat, default=None, help="latest submission date (default today)")
    discover.add_argument("--match", default=None, help="comma-separated terms; keep papers whose title or abstract contains any")
    discover.add_argument("-o", "--output", type=Path, default=DEFAULT_PAPERS_FILE,
                          help=f"JSONL file to write, one paper per line (default {DEFAULT_PAPERS_FILE})")

    ingest = sub.add_parser("ingest", help="add papers to the corpus by arXiv id or DOI")
    ingest.add_argument("identifiers", nargs="*", help="arXiv ids or DOIs")
    ingest.add_argument("--from", dest="from_file", type=Path, default=None,
                        help=f"file with one identifier or one discover JSON record per line "
                             f"(default {DEFAULT_PAPERS_FILE} when no identifiers are given)")
    ingest.add_argument("--limit", type=int, default=None, help="stop after this many items")
    ingest.add_argument("--delay", type=float, default=DEFAULT_DOWNLOAD_DELAY_S,
                        help=f"seconds to wait between PDF downloads (default {DEFAULT_DOWNLOAD_DELAY_S})")
    ingest.add_argument("--no-skip-existing", action="store_true", help="re-ingest papers already in the corpus")

    sub.add_parser("papers", help="list the corpus")
    sub.add_parser("sandbox", help="check that the run_python sandbox starts and is isolated")
    return parser


def _serve(settings: Settings) -> None:
    # Passing an import string (rather than the app object) is what lets
    # uvicorn support --reload/workers-style respawning later on.
    uvicorn.run("lgraph.api:create_app", factory=True, host=settings.host, port=settings.port)


async def _discover(args: argparse.Namespace) -> int:
    import httpx  # noqa: PLC0415

    from .rag.discover import USER_AGENT, discover_arxiv  # noqa: PLC0415

    until = args.until or date.today()
    match = [t for t in (args.match or "").split(",") if t.strip()] or None

    def on_page(seen: int, kept: int, total: int | None) -> None:
        of = f" of {total}" if total else ""
        print(f"harvested {seen}{of} records, {kept} kept so far", flush=True)

    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as http:
        papers = await discover_arxiv(
            args.category, since=args.since, until=until, match=match, http=http, on_page=on_page
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for paper in papers:
            fh.write(json.dumps(paper.to_record(), ensure_ascii=False) + "\n")
    print(f"wrote {len(papers)} papers to {args.output}")
    return 0


def load_items(path: Path) -> list:
    """Read ingest items: bare identifiers or discover JSON records, one per line."""
    from .rag.documents import Paper  # noqa: PLC0415

    items = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        items.append(Paper.from_record(json.loads(line)) if line.startswith("{") else line)
    return items


async def _ingest(settings: Settings, args: argparse.Namespace) -> int:
    import httpx  # noqa: PLC0415

    from .model import build_embeddings  # noqa: PLC0415
    from .rag.ingest import IngestReport, Ingestor, preflight_embeddings  # noqa: PLC0415
    from .rag.parse import MinerUParser  # noqa: PLC0415
    from .rag.store import PaperStore  # noqa: PLC0415

    items: list = list(args.identifiers)
    from_file = args.from_file
    if from_file is None and not items and DEFAULT_PAPERS_FILE.is_file():
        from_file = DEFAULT_PAPERS_FILE
    if from_file:
        print(f"reading papers from {from_file}", flush=True)
        items.extend(load_items(from_file))
    if not items:
        print(f"Nothing to ingest: pass identifiers, --from FILE, or run `lgraph discover` to create {DEFAULT_PAPERS_FILE}",
              file=sys.stderr)
        return 2
    if args.limit is not None:
        items = items[: args.limit]

    embeddings = build_embeddings(settings)
    problem = await preflight_embeddings(embeddings)
    if problem:
        print(f"Cannot reach OpenRouter embeddings ({settings.embedding_model}): {problem}", file=sys.stderr)
        print("Check OPENROUTER_API_KEY (environment or .env) and OPENROUTER_EMBEDDING_MODEL.", file=sys.stderr)
        return 2
    print(f"embeddings ok: {settings.embedding_model}, {settings.embedding_dimensions} dimensions", flush=True)

    store = PaperStore.open(settings.lancedb_dir, dimensions=settings.embedding_dimensions)

    def on_progress(r: IngestReport, i: int, n: int) -> None:
        status = "skip" if r.skipped else ("ok  " if r.ok else "FAIL")
        detail = r.error if not r.ok else f"{r.title}  ({r.chunks} chunks)"
        print(f"[{i}/{n}] {status}  {r.identifier}  {detail}", flush=True)

    async with httpx.AsyncClient(headers={"User-Agent": f"lgraph ({settings.contact_email or 'no-email'})"}) as http:
        ingestor = Ingestor(
            parser=MinerUParser(tier=settings.mineru_tier, api_url=settings.mineru_api_url),
            embeddings=embeddings,
            store=store,
            http=http,
            pdf_dir=settings.pdf_dir,
            contact_email=settings.contact_email,
            skip_existing=not args.no_skip_existing,
            download_delay_s=args.delay,
        )
        reports = await ingestor.ingest(items, on_progress=on_progress)

    ok = sum(1 for r in reports if r.ok and not r.skipped)
    skipped = sum(1 for r in reports if r.skipped)
    failed = sum(1 for r in reports if not r.ok)
    not_attempted = len(items) - len(reports)
    print(f"{ok} ingested, {skipped} already present, {failed} failed; corpus now has "
          f"{len(store.list_papers())} papers and {store.count_chunks()} chunks")
    if not_attempted:
        print(f"Stopped early: {not_attempted} papers not attempted after repeated network/provider "
              f"failures. Check connectivity and re-run the same command to resume.", file=sys.stderr)
    return 1 if failed or not_attempted else 0


def _papers(settings: Settings) -> None:
    from .rag.store import PaperStore  # noqa: PLC0415

    if not settings.lancedb_dir.exists():
        print(f"No corpus at {settings.lancedb_dir}. Run: lgraph ingest <arxiv-id|doi> ...")
        return
    store = PaperStore.open(settings.lancedb_dir, dimensions=settings.embedding_dimensions)
    papers = store.list_papers()
    if not papers:
        print("The corpus is empty.")
        return
    for p in papers:
        print(f"{p.citation_line()}  [{p.chunk_count} chunks]")



SANDBOX_CHECKS = [
    ("kernel runs code", "print(1 + 1)", "2"),
    ("SymPy works", "print(sp.integrate(sp.exp(-sp.Symbol('x')**2), (sp.Symbol('x'), -sp.oo, sp.oo)))", "sqrt(pi)"),
    ("network is blocked",
     "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), timeout=3)\n    print('OPEN')\n"
     "except OSError:\n    print('BLOCKED')", "BLOCKED"),
    # Read the mount table rather than try a write: the sandbox user could not write
    # to most paths anyway, so a failed write would not prove --read-only.
    ("filesystem is read-only",
     "try:\n    roots = [l.split() for l in open('/proc/mounts') if l.split()[1:2] == ['/']]\n"
     "    print('READ-ONLY' if roots and 'ro' in roots[-1][3].split(',') else 'WRITABLE')\n"
     "except OSError:\n    print('unknown')", "READ-ONLY"),
    ("memory limited to 1 GiB",
     "try:\n    print(open('/sys/fs/cgroup/memory.max').read().strip())\nexcept OSError:\n    print('unknown')",
     "1073741824"),
]


async def _sandbox(settings: Settings) -> int:
    import uuid  # noqa: PLC0415

    from .notebook import SandboxError, open_kernel_pool  # noqa: PLC0415

    if settings.python == "off":
        print("LGRAPH_PYTHON is off. Set LGRAPH_PYTHON=container (or unsafe-local for development).", file=sys.stderr)
        return 2
    pool = await open_kernel_pool(settings)
    if pool is None:
        print("The sandbox is not available; see the messages above.", file=sys.stderr)
        return 1
    session = str(uuid.uuid4())
    failures = 0
    try:
        print(f"starting a sandbox ({settings.python})…", flush=True)
        for label, code, expected in SANDBOX_CHECKS:
            try:
                cell = await pool.execute(session, code)
            except SandboxError as exc:
                print(f"FAIL  {label}: {exc}")
                return 1
            out = "".join(o.get("text", "") for o in cell["outputs"] if o["type"] == "stream").strip()
            ok = expected in out
            failures += not ok
            detail = "" if ok else f" (got {out or cell['status']!r})"
            print(f"{'ok  ' if ok else 'FAIL'}  {label}{detail}")
    finally:
        await pool.close_all()
    if failures and settings.python == "unsafe-local":
        print("unsafe-local has no isolation, so the isolation checks are expected to fail.")
    return 1 if failures else 0


if __name__ == "__main__":
    main()
