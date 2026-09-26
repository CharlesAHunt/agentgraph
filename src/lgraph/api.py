"""FastAPI layer: wiring, routes, and error translation."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any
from collections.abc import AsyncIterator, Sequence

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

from .agent import build_agent
from .config import Settings, get_settings
from .errors import UpstreamError, translate
from .messages import to_langchain, to_wire
from .model import build_embeddings, build_model
from .prompts import DEFAULT_RAG_SYSTEM_PROMPT
from .rag.documents import Source
from .rag.retriever import Retriever
from .rag.store import PaperStore
from .rag.tools import make_tools

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


class ChatResponse(BaseModel):
    messages: list[Message]
    # Excerpts retrieved by tools during this turn, deduplicated. Empty when
    # no retrieval happened.
    sources: list[Source] = Field(default_factory=list)


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Liveness only — deliberately does not call OpenRouter."""
    return {"status": "ok", "model": request.app.state.settings.model}


@router.get("/papers")
async def papers(request: Request) -> dict[str, Any]:
    """The ingested corpus. ``enabled`` is false when retrieval is off."""
    store: PaperStore | None = request.app.state.store
    if store is None:
        return {"enabled": False, "papers": []}
    papers = await asyncio.to_thread(store.list_papers)
    return {"enabled": True, "papers": [p.to_record() for p in papers]}


def _parse_history(payload: ChatRequest) -> list[BaseMessage]:
    try:
        return to_langchain([m.model_dump(exclude_unset=True) for m in payload.messages])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        # Pydantic checks field types; message *semantics* (known roles,
        # well-formed tool_calls, tool_call_id on tool messages) surface here.
        raise HTTPException(status_code=422, detail=f"Invalid message list: {exc}") from exc


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(request: Request, payload: ChatRequest) -> dict[str, Any]:
    history = _parse_history(payload)
    try:
        output_state = await request.app.state.agent.ainvoke({"messages": history})
    except Exception as exc:
        upstream = translate(exc)
        if upstream is None:
            raise
        raise upstream from exc
    messages: list[BaseMessage] = output_state["messages"]
    return {
        "messages": to_wire(messages),
        # Only tool results produced in this turn count as retrievals; echoed
        # history carries no artifacts anyway.
        "sources": collect_sources(messages[len(history) :]),
    }


@router.post("/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    """The /chat turn as server-sent events.

    Events: ``reasoning`` and ``token`` (text deltas), ``tool_start`` and
    ``tool_end`` (each search and its sources), then exactly one of ``done``
    (the same ``messages`` and ``sources`` /chat returns) or ``error``.
    """
    history = _parse_history(payload)
    return StreamingResponse(
        stream_turn(request.app.state.agent, history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def stream_turn(agent: Any, history: list[BaseMessage]) -> AsyncIterator[str]:
    final: list[BaseMessage] = history
    try:
        async for mode, data in agent.astream(
            {"messages": history}, stream_mode=["messages", "updates", "values"]
        ):
            if mode == "messages":
                chunk, meta = data
                if meta.get("langgraph_node") != "model" or not isinstance(chunk, AIMessageChunk):
                    continue
                reasoning = chunk.additional_kwargs.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    yield _sse("reasoning", {"text": reasoning})
                if text := _chunk_text(chunk):
                    yield _sse("token", {"text": text})
            elif mode == "updates":
                for event in _update_events(data):
                    yield event
            elif mode == "values":
                final = data["messages"]
    except Exception as exc:  # noqa: BLE001 - the response has already started
        upstream = translate(exc)
        if upstream is None:
            logger.exception("Streaming chat turn failed")
            yield _sse("error", {"detail": "Internal error while generating the answer.",
                                 "status": 500, "upstream_status": None})
        else:
            logger.warning("OpenRouter call failed (upstream=%s): %s",
                           upstream.upstream_status, upstream.message)
            yield _sse("error", {"detail": upstream.message, "status": upstream.status_code,
                                 "upstream_status": upstream.upstream_status})
        return

    new = final[len(history) :]
    yield _sse("done", {
        "messages": to_wire(final),
        "sources": [s.model_dump() for s in collect_sources(new)],
    })


def _update_events(update: Any) -> list[str]:
    """Tool activity from one ``updates`` item: which searches ran, what they found."""
    events: list[str] = []
    for node, delta in (update or {}).items():
        messages = delta.get("messages", []) if isinstance(delta, dict) else []
        for msg in messages:
            if node == "model" and isinstance(msg, AIMessage):
                for call in msg.tool_calls:
                    events.append(_sse("tool_start", {
                        "id": call["id"], "name": call["name"], "args": call["args"],
                    }))
            elif node == "tools" and isinstance(msg, ToolMessage):
                events.append(_sse("tool_end", {
                    "id": msg.tool_call_id,
                    "name": msg.name,
                    "status": msg.status,
                    "sources": [s.model_dump() for s in _artifact_sources(msg)],
                }))
    return events


def _chunk_text(chunk: AIMessageChunk) -> str:
    if isinstance(chunk.content, str):
        return chunk.content
    return "".join(
        block.get("text", "")
        for block in chunk.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    # json.dumps escapes newlines, so a payload can never break SSE framing.
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _artifact_sources(msg: ToolMessage) -> list[Source]:
    if not isinstance(msg.artifact, list):
        return []
    sources: list[Source] = []
    for item in msg.artifact:
        try:
            sources.append(Source.model_validate(item))
        except ValidationError:
            continue
    return sources


def collect_sources(messages: Sequence[BaseMessage]) -> list[Source]:
    """Gather ``Source`` artifacts from tool messages, deduplicated by chunk."""
    seen: set[str] = set()
    sources: list[Source] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        for source in _artifact_sources(msg):
            if source.chunk_id in seen:
                continue
            seen.add(source.chunk_id)
            sources.append(source.model_copy(update={"n": len(sources) + 1}))
    return sources


def create_app(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    tools: Sequence[BaseTool] | None = None,
) -> FastAPI:
    """Application factory.

    Pass ``settings`` to override the environment. ``model`` and
    ``embeddings`` replace the real provider clients; ``tools`` replaces the
    retrieval tools entirely. Tests inject fakes through these.
    """
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        chat_model = model or build_model(resolved)
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

        app.state.settings = resolved
        app.state.model = chat_model
        app.state.store = store
        app.state.agent = build_agent(chat_model, tools=agent_tools, system_prompt=system_prompt)
        yield

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
