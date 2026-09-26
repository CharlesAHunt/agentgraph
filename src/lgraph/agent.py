"""The LangGraph workflow, built with LangChain's prebuilt agent.

No web-framework imports here. Model failures propagate to the caller,
which decides how they become responses.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from .prompts import compose_system_prompt


@dataclass(frozen=True)
class TurnContext:
    """Per-request values the graph reads at run time."""

    instructions: str | None = None
    # One of config.REASONING_EFFORTS; None keeps the model's configured effort.
    effort: str | None = None
    # Conversation id; selects the run_python kernel.
    session: str | None = None


class TurnOptions(AgentMiddleware):
    """Apply per-request options to every model call in the turn.

    Builds the one system message (server rules, then the user's role) and
    sets the reasoning effort, so one compiled agent serves every request.
    """

    def __init__(self, base: str | None) -> None:
        super().__init__()
        self.base = base

    def _apply(self, request: ModelRequest) -> ModelRequest:
        context = request.runtime.context
        changes: dict = {}
        prompt = compose_system_prompt(self.base, getattr(context, "instructions", None))
        if prompt:
            changes["system_message"] = SystemMessage(content=prompt)
        if effort := getattr(context, "effort", None):
            changes["model_settings"] = {**request.model_settings, "reasoning": {"effort": effort}}
        return request.override(**changes) if changes else request

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        return handler(self._apply(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(self._apply(request))


def build_agent(
    model: BaseChatModel,
    *,
    tools: Sequence[BaseTool] = (),
    system_prompt: str | None = None,
):
    """Compile the agent graph. Call once per model, at startup.

    With no tools this is a single model call per request. Passing tools
    turns it into a full tool-calling loop with no other changes. Invoke with
    ``context=TurnContext(...)`` to set a user-chosen role or reasoning effort.
    """
    return create_agent(
        model,
        tools=list(tools),
        middleware=[TurnOptions(system_prompt)],
        context_schema=TurnContext,
    )
