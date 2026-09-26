"""Small text helpers shared across modules."""

from __future__ import annotations


def truncate(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` characters, saying how much was dropped."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [{len(text) - limit} more characters]"
