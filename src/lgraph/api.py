"""FastAPI layer: wiring, routes, and error translation."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any
from collections.abc import Sequence

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage
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


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(request: Request, payload: ChatRequest) -> dict[str, Any]:
    try:
        history = to_langchain([m.model_dump(exclude_unset=True) for m in payload.messages])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        # Pydantic checks field types; message *semantics* (known roles,
        # well-formed tool_calls, tool_call_id on tool messages) surface here.
        raise HTTPException(status_code=422, detail=f"Invalid message list: {exc}") from exc
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


def collect_sources(messages: Sequence[BaseMessage]) -> list[Source]:
    """Gather ``Source`` artifacts from tool messages, deduplicated by chunk."""
    seen: set[str] = set()
    sources: list[Source] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage) or not isinstance(msg.artifact, list):
            continue
        for item in msg.artifact:
            try:
                source = Source.model_validate(item)
            except ValidationError:
                continue
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
    return app
