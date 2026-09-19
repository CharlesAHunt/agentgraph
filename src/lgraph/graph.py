"""The LangGraph workflow.

No web-framework imports here either — the node talks to
:class:`~lgraph.client.OpenRouterClient` and lets ``OpenRouterError``
propagate.
"""

from __future__ import annotations

from typing import Annotated, Any, List, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from .client import OpenRouterClient


class OpenRouterMessage(TypedDict):
    """One conversation turn, shaped the way OpenRouter expects it."""

    role: str
    content: Optional[str]
    # Carries the model's reasoning state across turns; absent on plain input.
    reasoning_details: NotRequired[Optional[Any]]


def append_messages(
    left: List[OpenRouterMessage], right: List[OpenRouterMessage]
) -> List[OpenRouterMessage]:
    """Reducer: append new messages to the running history."""
    return left + right


class GraphState(TypedDict):
    messages: Annotated[List[OpenRouterMessage], append_messages]


def sanitize(messages: List[OpenRouterMessage]) -> List[dict[str, Any]]:
    """Strip graph metadata, keeping the pure JSON OpenRouter accepts."""
    sanitized: List[dict[str, Any]] = []
    for msg in messages:
        clean: dict[str, Any] = {"role": msg.get("role"), "content": msg.get("content")}
        # Pass reasoning_details back unmodified on multi-turn interactions.
        reasoning = msg.get("reasoning_details")
        if reasoning:
            clean["reasoning_details"] = reasoning
        sanitized.append(clean)
    return sanitized


def make_model_node(client: OpenRouterClient):
    """Build the ``agent`` node, bound to a client instance."""

    async def call_model(state: GraphState) -> dict[str, List[OpenRouterMessage]]:
        assistant = await client.chat_completion(sanitize(state["messages"]))

        new_message: OpenRouterMessage = {
            "role": assistant.get("role") or "assistant",
            "content": assistant.get("content"),
            "reasoning_details": assistant.get("reasoning_details"),
        }
        # The reducer folds this into the history.
        return {"messages": [new_message]}

    return call_model


def build_graph(client: OpenRouterClient):
    """Compile the workflow. Call once per client, at startup."""
    workflow = StateGraph(GraphState)
    workflow.add_node("agent", make_model_node(client))
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    return workflow.compile()
