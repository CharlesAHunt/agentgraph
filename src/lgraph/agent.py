"""The LangGraph workflow, built with LangChain's prebuilt agent.

No web-framework imports here. Model failures propagate to the caller,
which decides how they become responses.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool


def build_agent(
    model: BaseChatModel,
    *,
    tools: Sequence[BaseTool] = (),
    system_prompt: str | None = None,
):
    """Compile the agent graph. Call once per model, at startup.

    With no tools this is a single model call per request. Passing tools
    turns it into a full tool-calling loop with no other changes.
    """
    return create_agent(model, tools=list(tools), system_prompt=system_prompt)
