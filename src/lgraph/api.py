"""FastAPI layer: wiring, routes, and error translation."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .client import OpenRouterClient, OpenRouterError
from .config import Settings, get_settings
from .graph import build_graph

logger = logging.getLogger(__name__)

router = APIRouter()


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    reasoning_details: Optional[Any] = None


class ChatRequest(BaseModel):
    messages: List[Message] = Field(min_length=1)


class ChatResponse(BaseModel):
    messages: List[Message]


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Liveness only — deliberately does not call OpenRouter."""
    return {"status": "ok", "model": request.app.state.client.model}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> dict[str, Any]:
    initial_state = {
        "messages": [m.model_dump(exclude_unset=True) for m in payload.messages]
    }
    output_state = await request.app.state.graph.ainvoke(initial_state)
    return {"messages": output_state["messages"]}


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Application factory. Pass ``settings`` to override the environment."""
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = OpenRouterClient.create(resolved)
        app.state.settings = resolved
        app.state.client = client
        app.state.graph = build_graph(client)
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(
        title="LangGraph Custom OpenRouter Endpoint Server",
        lifespan=lifespan,
    )

    @app.exception_handler(OpenRouterError)
    async def _handle_openrouter_error(_: Request, exc: OpenRouterError) -> JSONResponse:
        logger.warning(
            "OpenRouter call failed (upstream=%s): %s", exc.upstream_status, exc.message
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "upstream_status": exc.upstream_status},
        )

    app.include_router(router)
    return app
