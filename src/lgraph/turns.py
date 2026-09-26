"""One agent turn as a result (for /chat) or as server-sent events (for /chat/stream)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from pydantic import ValidationError

from .agent import TurnContext
from .errors import translate
from .messages import to_wire
from .notebook import TOOL_NAME as PYTHON_TOOL
from .rag.documents import Source

logger = logging.getLogger(__name__)


def turn_result(final: Sequence[BaseMessage], history: Sequence[BaseMessage]) -> dict[str, Any]:
    """The whole conversation back, plus what tools produced during this turn.

    Only tool results after ``history`` count; echoed history carries no artifacts.
    """
    new = final[len(history) :]
    return {
        "messages": to_wire(final),
        "sources": [s.model_dump() for s in collect_sources(new)],
        "cells": collect_cells(new),
    }


async def stream_turn(agent: Any, history: list[BaseMessage], context: TurnContext) -> AsyncIterator[str]:
    final: list[BaseMessage] = history
    try:
        async for mode, data in agent.astream(
            {"messages": history}, context=context, stream_mode=["messages", "updates", "values"]
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
    yield _sse("done", turn_result(final, history))


def _update_events(update: Any) -> list[str]:
    """Tool activity from one ``updates`` item: which tools ran, and what they produced."""
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
                    "cell": _artifact_cell(msg),
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


# --- tool artifacts ------------------------------------------------------------------


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


def _artifact_cell(msg: ToolMessage) -> dict[str, Any] | None:
    if msg.name == PYTHON_TOOL and isinstance(msg.artifact, dict) and "code" in msg.artifact:
        return msg.artifact
    return None


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


def collect_cells(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """run_python cells produced by these messages, in order."""
    return [cell for m in messages if isinstance(m, ToolMessage) and (cell := _artifact_cell(m))]
