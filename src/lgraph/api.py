"""FastAPI layer: request models, routes and application wiring."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any
from uuid import UUID
from collections.abc import Awaitable, Callable, Sequence

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .agent import TurnContext, build_agent
from .catalog import Agents, ModelCatalog
from .config import ReasoningEffort, Settings, get_settings
from .errors import UpstreamError, raise_upstream
from .messages import to_langchain
from .model import account_credits, build_embeddings, build_model, key_usage, list_tool_models
from .notebook import KernelPool, SandboxError, make_python_tool, open_kernel_pool
from .prompts import DEFAULT_RAG_SYSTEM_PROMPT, PYTHON_GUIDE
from .rag.documents import Source
from .rag.retriever import Retriever
from .rag.store import PaperStore
from .rag.tools import make_tools
from .turns import stream_turn, turn_result

logger = logging.getLogger(__name__)

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str | None = None
    reasoning_details: Any | None = None
    # Tool use, OpenAI wire shape. Assistant messages carry ``tool_calls``;
    # tool messages carry ``tool_call_id`` and ``name``.
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    # An id from GET /models; omitted means the configured default.
    model: str | None = Field(default=None, max_length=200)
    # A role for the model (tone, depth, emphasis), added after the server's prompt.
    instructions: str | None = Field(default=None, max_length=1000)
    # Reasoning effort for this request; omitted means OPENROUTER_REASONING_EFFORT.
    effort: ReasoningEffort | None = None
    # Conversation id (a UUID the client generates); keeps run_python state between turns.
    session: UUID | None = None

    def turn_context(self) -> TurnContext:
        return TurnContext(
            instructions=self.instructions,
            effort=self.effort,
            session=str(self.session) if self.session else None,
        )


class ChatResponse(BaseModel):
    messages: list[Message]
    # Excerpts retrieved by tools during this turn, deduplicated. Empty when
    # no retrieval happened.
    sources: list[Source] = Field(default_factory=list)
    # run_python cells executed during this turn, in order.
    cells: list[dict[str, Any]] = Field(default_factory=list)


class ExecuteRequest(BaseModel):
    code: str = Field(min_length=1, max_length=20_000)


# --- chat --------------------------------------------------------------------------

SERVER_ROLES = frozenset({"system", "developer"})


def _parse_history(payload: ChatRequest) -> list[BaseMessage]:
    if any(m.role in SERVER_ROLES for m in payload.messages):
        raise HTTPException(
            status_code=422,
            detail="System messages are set by the server; send a role as `instructions` instead.",
        )
    try:
        return to_langchain([m.model_dump(exclude_unset=True) for m in payload.messages])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        # Pydantic checks field types; message *semantics* (known roles,
        # well-formed tool_calls, tool_call_id on tool messages) surface here.
        raise HTTPException(status_code=422, detail=f"Invalid message list: {exc}") from exc


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(request: Request, payload: ChatRequest) -> dict[str, Any]:
    history = _parse_history(payload)
    agent = await request.app.state.agents.get(payload.model)
    try:
        output_state = await agent.ainvoke({"messages": history}, context=payload.turn_context())
    except Exception as exc:
        raise_upstream(exc)
    return turn_result(output_state["messages"], history)


@router.post("/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    """The /chat turn as server-sent events.

    Events: ``reasoning`` and ``token`` (text deltas), ``tool_start`` and
    ``tool_end`` (each tool call and what it produced), then exactly one of
    ``done`` (the same ``messages``, ``sources`` and ``cells`` /chat returns)
    or ``error``.
    """
    history = _parse_history(payload)
    agent = await request.app.state.agents.get(payload.model)
    return StreamingResponse(
        stream_turn(agent, history, payload.turn_context()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- notebook sessions ------------------------------------------------------------------

NOTEBOOK_OFF = "The Python notebook is not enabled on this server (LGRAPH_PYTHON)."


@router.post("/sessions/{session}/execute")
async def execute(request: Request, session: UUID, payload: ExecuteRequest) -> dict[str, Any]:
    """Run code in a conversation's kernel directly, e.g. a cell the user edited."""
    pool: KernelPool | None = request.app.state.kernels
    if pool is None:
        raise HTTPException(status_code=404, detail=NOTEBOOK_OFF)
    try:
        return await pool.execute(str(session), payload.code)
    except SandboxError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/sessions/{session}", status_code=204)
async def end_session(request: Request, session: UUID) -> Response:
    """Shut a conversation's kernel down. Idempotent."""
    pool: KernelPool | None = request.app.state.kernels
    if pool is not None:
        await pool.close(str(session))
    return Response(status_code=204)


# --- information ----------------------------------------------------------------------


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Liveness only — deliberately does not call OpenRouter."""
    return {"status": "ok", "model": request.app.state.settings.model}


@router.get("/papers")
async def papers(request: Request, count_only: bool = False) -> dict[str, Any]:
    """The ingested corpus, or with ``count_only`` just its size. ``enabled`` is false when retrieval is off."""
    store: PaperStore | None = request.app.state.store
    if count_only:
        count = await asyncio.to_thread(store.count_papers) if store else 0
        return {"enabled": store is not None, "count": count}
    if store is None:
        return {"enabled": False, "papers": []}
    papers = await asyncio.to_thread(store.list_papers)
    return {"enabled": True, "papers": [p.to_record() for p in papers]}


@router.get("/models")
async def models(request: Request) -> dict[str, Any]:
    """Chat models a request's ``model`` may name; ``default`` is used when it is omitted."""
    return {
        "default": request.app.state.settings.model,
        "default_effort": request.app.state.settings.reasoning_effort,
        "models": await request.app.state.catalog.models(),
    }


NO_MANAGEMENT_KEY = "Set OPENROUTER_MANAGEMENT_KEY to show the account balance."
CREDITS_UNAVAILABLE = "OpenRouter did not return the account balance; check OPENROUTER_MANAGEMENT_KEY."


@router.get("/usage")
async def usage(request: Request) -> dict[str, Any]:
    """Spend and limit for the API key, plus the account balance when a management key is set."""
    state = request.app.state
    fetches = [state.key_usage()] + ([state.credits()] if state.credits is not None else [])
    key, *rest = await asyncio.gather(*fetches, return_exceptions=True)
    if isinstance(key, BaseException):
        raise_upstream(key)
    if not rest:
        return {"key": key, "credits": None, "credits_note": NO_MANAGEMENT_KEY}
    credits = rest[0]
    if isinstance(credits, BaseException):
        # The key figures are still worth returning.
        logger.warning("Could not load OpenRouter credits: %s", credits)
        return {"key": key, "credits": None, "credits_note": CREDITS_UNAVAILABLE}
    return {"key": key, "credits": credits, "credits_note": None}


# --- application ----------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    tools: Sequence[BaseTool] | None = None,
    model_factory: Callable[[str], BaseChatModel] | None = None,
    model_catalog: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
    key_usage_fetch: Callable[[], Awaitable[dict[str, Any]]] | None = None,
    credits_fetch: Callable[[], Awaitable[dict[str, float]]] | None = None,
    kernel_pool: KernelPool | None = None,
) -> FastAPI:
    """Application factory.

    Pass ``settings`` to override the environment. ``model`` and
    ``embeddings`` replace the real provider clients; ``tools`` replaces the
    retrieval tools entirely. ``model_factory`` builds the chat model for a
    requested model id and ``model_catalog`` lists the offered models.
    ``key_usage_fetch`` and ``credits_fetch`` read spend and balance; the
    balance is only read when a management key is configured.
    ``kernel_pool`` replaces the run_python sandboxes set by LGRAPH_PYTHON.
    Tests inject fakes through these.
    """
    resolved = settings or get_settings()
    make_model = model_factory or (lambda slug: build_model(replace(resolved, model=slug)))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store: PaperStore | None = None
        system_prompt = resolved.system_prompt

        if tools is not None:
            agent_tools: list[BaseTool] = list(tools)
        elif resolved.retrieval_enabled:
            store = PaperStore.open(resolved.lancedb_dir, dimensions=resolved.embedding_dimensions)
            retriever = Retriever(embeddings or build_embeddings(resolved), store)
            agent_tools = make_tools(retriever, k=resolved.retrieval_k)
            system_prompt = system_prompt or DEFAULT_RAG_SYSTEM_PROMPT
            logger.info("Retrieval enabled: %d chunks in %s", store.count_chunks(), resolved.lancedb_dir)
        else:
            agent_tools = []

        kernels = kernel_pool if kernel_pool is not None else await open_kernel_pool(resolved)
        if kernels is not None:
            agent_tools = [*agent_tools, make_python_tool(kernels)]
            system_prompt = f"{system_prompt}\n\n{PYTHON_GUIDE}" if system_prompt else PYTHON_GUIDE
        reaper = asyncio.create_task(kernels.reap_forever()) if kernels is not None else None

        def build(slug: str):
            chat_model = model if model is not None and slug == resolved.model else make_model(slug)
            return build_agent(chat_model, tools=agent_tools, system_prompt=system_prompt)

        catalog = ModelCatalog(
            model_catalog or (lambda: list_tool_models(resolved)),
            default=resolved.model,
            allow=resolved.models,
        )
        app.state.settings = resolved
        app.state.store = store
        app.state.catalog = catalog
        app.state.agents = Agents(resolved.model, build, catalog)
        app.state.key_usage = key_usage_fetch or (lambda: key_usage(resolved))
        app.state.credits = (
            (credits_fetch or (lambda: account_credits(resolved))) if resolved.management_key else None
        )
        app.state.kernels = kernels
        try:
            yield
        finally:
            if reaper is not None:
                reaper.cancel()
            if kernels is not None:
                await kernels.close_all()

    app = FastAPI(title="LangGraph OpenRouter Endpoint Server", lifespan=lifespan)

    @app.exception_handler(UpstreamError)
    async def _handle_upstream_error(_: Request, exc: UpstreamError) -> JSONResponse:
        logger.warning(
            "OpenRouter call failed (upstream=%s): %s", exc.upstream_status, exc.message
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "upstream_status": exc.upstream_status},
        )

    app.include_router(router)

    resolved.results_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/results", StaticFiles(directory=str(resolved.results_dir), html=True), name="results")

    # Mounted last: "/" matches everything, so API routes must be registered first.
    if resolved.web_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(resolved.web_dir), html=True), name="web")

    return app
