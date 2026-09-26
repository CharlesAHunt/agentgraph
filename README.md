# lgraph

A FastAPI service that runs a LangGraph agent against OpenRouter, with
retrieval-augmented generation over a corpus of academic papers.

The agent is built with LangChain 1.x `create_agent` on top of
`langchain-openrouter`'s `ChatOpenRouter`. That model class is used instead
of `ChatOpenAI` with a custom `base_url` because it round-trips OpenRouter's
`reasoning_details` field across turns; the generic OpenAI client drops it.

When a corpus exists, the agent gets two tools: `search_publications`, a
hybrid dense + full-text search over paper chunks, and `list_publications`.
Retrieved excerpts come back to the API client as structured `sources`.

The service is stateless: the client sends the full conversation on every
request and receives the full conversation back. Assistant messages may
carry `reasoning_details` and `tool_calls`, and tool results appear as
`tool` messages. Send all of them back unmodified on the next turn.

## Quick start

End to end on one machine, using the model-free `flash` parsing tier:

```bash
uv sync --extra ingest --group dev
echo 'OPENROUTER_API_KEY=sk-or-...' > .env

# 1. Pick the papers: seven years of arXiv plasma physics, fusion-related.
uv run lgraph discover --since 2019-09-19 \
  --match "fusion,tokamak,stellarator,ITER,inertial confinement,magnetic confinement"

# 2. Build the corpus (hours; resumable by re-running).
uv run lgraph ingest --limit 3      # trial: must print "embeddings ok"
uv run lgraph ingest 2>&1 | tee ingest.log

# 3. Serve and ask.
uv run lgraph papers | wc -l
uv run lgraph
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What limits divertor heat flux in tokamaks?"}]}'
```

For a GPU machine and better parsing, see "Ingesting on a GPU machine".

## Install

```bash
uv sync                               # API server only
uv sync --group dev                   # plus pytest
uv sync --extra ingest --group dev    # plus MinerU and arxiv, needed for `lgraph ingest`
uv sync --extra ingest-gpu --group dev   # Linux/Windows + NVIDIA: MinerU's GPU engines
```

The `ingest` extras are large (MinerU pulls in torch, onnxruntime and
gradio; the GPU extra adds vLLM or LMDeploy). The server never imports them,
so deploy the API without the extras and run ingestion from a machine that
has them. The corpus directory is plain files and can be copied between
machines.

## Configure

Variables can be exported or placed in a `.env` file in the working
directory (`KEY=value` per line; real environment variables take
precedence). `.env` is gitignored.

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | required | OpenRouter API key |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API root (not the chat-completions path) |
| `OPENROUTER_MODEL` | `qwen/qwen3.8-max-0902` | Chat model slug |
| `OPENROUTER_REASONING_EFFORT` | `medium` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `LGRAPH_REQUEST_TIMEOUT_S` | `120` | Per-request timeout to OpenRouter, in seconds |
| `LGRAPH_MAX_RETRIES` | `2` | Upstream retries; `0` disables |
| `LGRAPH_SYSTEM_PROMPT` | unset | System prompt. When unset and retrieval is on, a citation-oriented default is used |
| `LGRAPH_HOST` / `LGRAPH_PORT` | `127.0.0.1` / `8000` | Bind address. Use `0.0.0.0` to reach the server from other machines |
| `LGRAPH_DATA_DIR` | `./data` | Corpus location: LanceDB at `data/lancedb`, PDFs at `data/pdfs`, static reports at `data/results` |
| `LGRAPH_RAG_ENABLED` | auto | `true`/`false`. Auto means on when `data/lancedb` exists |
| `OPENROUTER_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model slug |
| `OPENROUTER_EMBEDDING_DIMENSIONS` | `1536` | Must match the model; checked against an existing store |
| `LGRAPH_RETRIEVAL_K` | `6` | Excerpts returned per search |
| `LGRAPH_CONTACT_EMAIL` | unset | Required to ingest DOIs (Crossref polite pool and Unpaywall ask for it) |
| `LGRAPH_MINERU_TIER` | `flash` | `flash` reads the PDF text layer with no model download; `standard` and `advanced` add layout models and OCR |
| `LGRAPH_MINERU_API_URL` | unset | Use a self-hosted `mineru-kit api-server` instead of parsing locally |
| `LGRAPH_WEB_DIR` | `web/dist` | Built web UI, served at `/` when the directory exists |

## Build a corpus

### Individual papers

```bash
echo 'LGRAPH_CONTACT_EMAIL=you@example.org' >> .env   # only needed for DOIs
uv run lgraph ingest 1706.03762 10.1038/s41586-021-03819-2
uv run lgraph papers
```

`ingest` accepts arXiv ids (with or without version, `arXiv:` prefix or
URL) and DOIs (bare, `doi:` prefix or `https://doi.org/` URL). For each
paper it resolves metadata (arXiv API, or Crossref plus Unpaywall for
DOIs), downloads the open-access PDF once into `data/pdfs`, parses it with
MinerU, splits it into section-aware chunks, embeds them through OpenRouter
and writes them to LanceDB. Re-ingesting a paper replaces its chunks.
Papers without an open-access PDF are reported as failures; there is no
paywall bypass.

### Bulk: a whole arXiv category

`discover` harvests a category through arXiv's OAI-PMH bulk interface and
writes one JSON record per paper, metadata included, so the bulk ingest
needs no metadata call per paper. `--since` and `--until` bound the paper's
first-submission date. Cross-listed papers count. An optional keyword
filter is applied to title and abstract. A seven-year harvest of
`physics.plasm-ph` takes a few minutes and yields a few thousand papers.

```bash
uv run lgraph discover --since 2019-09-19 \
  --match "fusion,tokamak,stellarator,ITER,inertial confinement,magnetic confinement"
uv run lgraph ingest
```

`discover` writes `data/papers.jsonl` and `ingest` with no arguments reads
it. Both accept a path (`-o` and `--from`) to use another file, and
`ingest --from` also accepts a plain list of identifiers, one per line.

Ingest is built for unattended runs:

- It checks the embeddings API with one tiny request before touching any
  paper, so a missing or wrong key fails in seconds, not hours.
- It skips papers already in the corpus, so an interrupted run resumes by
  re-running the same command. Downloaded PDFs are cached under `data/pdfs`.
- It stops after ten consecutive network or provider failures (machine
  offline, provider down) instead of failing every remaining paper.
- It waits three seconds between PDF downloads because arXiv blocks bulk
  fetchers that do not. `--delay` changes the pause; `--limit N` does a trial.

Expect about ten seconds per paper at the `flash` tier, dominated by the
download pause, so a few thousand papers is several hours and a few GB of
PDFs. A long run needs the machine awake and online:

- macOS: `caffeinate -i uv run lgraph ingest 2>&1 | tee ingest.log`
- Linux: run inside `tmux`, and disable suspend in the desktop's power
  settings (or `systemd-inhibit --what=idle:sleep ...` on the host).
- Windows: set the power plan to never sleep for the duration.

### Parsing tiers

The default `flash` tier uses the PDF's own text layer and needs no models.
It is fast and works anywhere, but scanned or image-only papers come
through thin and equations are not captured. The `standard` tier runs
MinerU's layout, formula and OCR models, so equations come back as LaTeX,
tables are located from the page image, and scanned pages are read. It
needs a GPU to be practical; see the next section. The tier is per run:
build at `flash` now and re-run later with `LGRAPH_MINERU_TIER=standard`
and `--no-skip-existing` to upgrade the corpus in place.

## Ingesting on a GPU machine

On Linux or Windows with an NVIDIA GPU, install the GPU extra. It adds
MinerU's served-model engines (vLLM on Linux, LMDeploy on Windows) so the
`standard` and `advanced` tiers run on the GPU.

Requirements: an NVIDIA GPU of Volta generation or newer with at least 8 GB
of VRAM, and a driver from the 580 series or newer (the bundled vLLM
expects a CUDA 13 capable driver). RTX 50-series cards work with the torch
build the lockfile pins. The GPU extra is not resolvable on macOS.

```bash
uv python install 3.12                               # MinerU's recommended interpreter
uv sync --python 3.12 --extra ingest-gpu --group dev
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
uv run mineru-kit models download --tier standard    # a few GB, once
uv run mineru-kit models show                        # should report cuda and vllm/lmdeploy
```

Then set the tier and run the ingest as above:

```
LGRAPH_MINERU_TIER=standard
```

MinerU picks CUDA automatically when torch sees a GPU; `MINERU_DEVICE_MODE`
set to `cuda` or `cpu` overrides that. Model weights live under `~/.mineru`
unless `MINERU_HOME` says otherwise. Watch `nvidia-smi` during a trial
`ingest --limit 3` to confirm the GPU is doing the parsing.

### Immutable distributions (Bazzite, Silverblue, Kinoite)

The host image is read-only and ships no compilers, so everything runs in a
distrobox that shares your home directory and borrows the host's NVIDIA
driver. Nothing is layered onto the host.

```bash
# On the host, once. An RTX 40/50 card needs the -nvidia-open image.
rpm-ostree status | head -3
nvidia-smi                                # driver 580+, CUDA 13.x

distrobox create --name mineru --image ubuntu:24.04 --nvidia
distrobox enter mineru

# Inside the box.
sudo apt update && sudo apt install -y build-essential git tmux curl
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
nvidia-smi                                # must work here too
cd ~/agentgraph                           # the repo in your home directory
```

Then follow the GPU steps above inside the box. `build-essential` matters:
Triton compiles small kernel launchers at runtime and needs a C compiler.
No CUDA toolkit install is needed; the torch and vLLM wheels bundle their
runtime. Run the long ingest inside `tmux` (detach with Ctrl-B, D) and set
the desktop's power settings to never suspend while plugged in.

If vLLM misbehaves in the box, `LGRAPH_MINERU_TIER=flash` skips the models
entirely, or run MinerU's own container with GPU access and point
`LGRAPH_MINERU_API_URL` at it; the rest of the setup is unchanged.

## Run the service

```bash
uv run lgraph            # same as: uv run lgraph serve
```

Retrieval turns on automatically when `data/lancedb` exists (override with
`LGRAPH_RAG_ENABLED`). The store is read on every search, so papers added
while the server runs are searchable immediately; but a server started
before the corpus existed has no tools bound, so restart it once after the
first ingest. `GET /papers` shows whether retrieval is enabled and what is
in the corpus.

The server needs only the base install and the `.env` key. To serve from
the GPU machine to other computers on your network, set
`LGRAPH_HOST=0.0.0.0`. To serve from a different machine than the one that
ingested, copy `data/lancedb` (and optionally `data/pdfs`) over.

## Use

Any `.html` file dropped into `data/results` is served as-is under `/results`
(e.g. `data/results/summary.html` -> `GET /results/summary.html`); the
directory is created automatically if it doesn't exist. `GET /results/` shows
`index.html` when one is present. This is unauthenticated, like every other
route here, so don't put anything sensitive there if the server is reachable
beyond localhost.

```bash
curl -s localhost:8000/health
curl -s localhost:8000/papers

curl -s localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "How does multi-head attention differ from single-head attention?"}]}'
```

The response is:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "", "tool_calls": [...], "reasoning_details": [...]},
    {"role": "tool", "name": "search_publications", "tool_call_id": "...", "content": "[1] Vaswani et al. (2017). ..."},
    {"role": "assistant", "content": "Multi-head attention ... (Vaswani et al., 2017, p. 4).", "reasoning_details": [...]}
  ],
  "sources": [
    {"n": 1, "title": "Attention Is All You Need", "authors": ["..."], "year": 2017,
     "arxiv_id": "1706.03762", "section": "3.2 Attention", "page": 4, "chunk_id": "arxiv:1706.03762#0007"}
  ]
}
```

`sources` lists every excerpt retrieved during the turn, deduplicated. The
answer cites papers as (First author et al., Year, p. N); match them to
`sources` by author, year and page. To continue the conversation, post the
whole returned `messages` list plus your next user message.

Upstream failures map to `504` (timeout), `429` (rate limited, passed
through) or `502` (anything else), with body
`{"detail": "...", "upstream_status": <int|null>}`. If retrieval itself
fails mid-turn, the tool tells the model the corpus is unavailable and the
request still succeeds.

### Streaming

`POST /chat/stream` takes the same body and streams the turn as server-sent
events. Each frame is `event: <name>` plus one JSON `data:` line:

| Event | Data | When |
|---|---|---|
| `reasoning` | `{"text"}` | model thinking, as it arrives |
| `token` | `{"text"}` | answer text, as it arrives |
| `tool_start` | `{"id", "name", "args"}` | the model starts a search |
| `tool_end` | `{"id", "name", "status", "sources"}` | that search returned |
| `done` | `{"messages", "sources"}` | same payload as `/chat`; send `messages` back next turn |
| `error` | `{"detail", "status", "upstream_status"}` | the turn failed; no `done` follows |

```bash
curl -N localhost:8000/chat/stream -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What limits divertor heat flux?"}]}'
```

## Web interface

A Svelte chat UI lives in `web/`. It streams answers from `/chat/stream`,
shows each corpus search live, renders the answer as a brief (summary
callout, key-figure tiles, mermaid diagrams), and turns citations into
chips that jump to a numbered source list with arXiv/DOI links.

Build it once and `uv run lgraph` serves it at `/` alongside the API:

```bash
cd web && npm install && npm run build   # writes web/dist
cd .. && uv run lgraph                   # open http://127.0.0.1:8000
```

For frontend development, run the API and Vite side by side; Vite proxies
API calls to port 8000 (override with `LGRAPH_API=http://host:port`):

```bash
uv run lgraph            # terminal 1
cd web && npm run dev    # terminal 2, open http://localhost:5173
```

Model output is untrusted (a retrieved paper could contain injected HTML),
so the UI sanitizes rendered Markdown with DOMPurify and renders diagrams
with mermaid's `strict` security level.

## Test

```bash
uv run pytest
```

Tests use fake chat and embedding models, a real LanceDB in a temporary
directory, and mocked HTTP for Crossref, Unpaywall and arXiv. They make no
network calls and do not need MinerU installed.

## Layout

```
src/lgraph/
  config.py         Settings from the environment and .env
  model.py          build_model() -> ChatOpenRouter, build_embeddings() -> OpenRouter embeddings
  agent.py          build_agent(model, tools, system_prompt) -> compiled graph
  prompts.py        default citation-oriented system prompt
  messages.py       wire dict <-> LangChain message conversion
  errors.py         UpstreamError and exception -> status translation
  api.py            FastAPI app: /health, /papers, /chat, /chat/stream, /results, web UI
  __main__.py       CLI: serve (default), discover, ingest, papers
  rag/
    documents.py    Paper, Chunk, Hit, Source
    store.py        PaperStore over LanceDB (hybrid search, upsert)
    embeddings.py   OpenRouterEmbeddings
    retriever.py    query embedding + store lookup
    tools.py        search_publications, list_publications
    sources.py      arXiv / DOI resolution and PDF download
    discover.py     arXiv category harvesting (OAI-PMH) for bulk runs
    parse.py        Parser protocol, MinerU adapter
    chunk.py        section-aware chunking
    ingest.py       the pipeline, preflight check, early stop
web/                Svelte 5 + Vite chat UI (npm run build -> web/dist)
  src/lib/
    api.ts          /chat/stream client (SSE over fetch), /papers
    chat.svelte.ts  conversation state: turns, searches, history round-trip
    markdown.ts     marked -> DOMPurify -> citation chips, stats and mermaid blocks
    Answer.svelte   the rendered brief; Sources.svelte, Activity.svelte, Composer.svelte
```

Nothing under `rag/` imports FastAPI, and only `rag/parse.py` imports
MinerU, lazily.
