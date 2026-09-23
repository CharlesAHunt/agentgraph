"""Conversion between the HTTP wire format and LangChain messages.

The wire format is the OpenAI-style message dict the service has always
accepted: ``role``, ``content``, optional ``reasoning_details``, and for
tool use ``tool_calls`` (assistant) or ``tool_call_id``/``name`` (tool).
Everything inside the graph is a ``BaseMessage``.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Sequence

from langchain_core.messages import (
    BaseMessage,
    convert_to_messages,
    convert_to_openai_messages,
)


def to_langchain(messages: Sequence[Mapping[str, Any]]) -> list[BaseMessage]:
    """Wire dicts -> LangChain messages.

    ``convert_to_messages`` understands OpenAI-format ``tool_calls`` and
    ``tool_call_id`` and places unknown keys such as ``reasoning_details``
    into ``additional_kwargs``, which is exactly where ``ChatOpenRouter``
    looks when forwarding them back upstream. Optional keys with ``None``
    values are dropped so an absent field is not forwarded as an explicit
    null; ``content`` is kept because the converter requires it and maps
    ``None`` to an empty string.
    """
    cleaned = [
        {k: v for k, v in m.items() if v is not None or k == "content"} for m in messages
    ]
    return convert_to_messages(cleaned)


def to_wire(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """LangChain messages -> wire dicts.

    Delegates role, content, ``tool_calls`` and ``tool_call_id`` to
    LangChain's OpenAI converter, then re-attaches ``reasoning_details``,
    which that converter does not know about.
    """
    converted = convert_to_openai_messages(list(messages), text_format="string")
    wire: list[dict[str, Any]] = []
    for msg, entry in zip(messages, converted, strict=True):
        reasoning_details = msg.additional_kwargs.get("reasoning_details")
        if reasoning_details is not None:
            entry["reasoning_details"] = reasoning_details
        wire.append(entry)
    return wire
