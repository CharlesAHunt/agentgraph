"""Which chat models a request may name, and one compiled agent per model."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from fastapi import HTTPException

logger = logging.getLogger(__name__)

CATALOG_TTL_S = 3600.0


class ModelCatalog:
    """Chat models a request may name: OpenRouter's tool-capable models, cached.

    With an allowlist (``LGRAPH_MODELS``) only those ids are offered. The
    configured default is always offered, first.
    """

    def __init__(
        self,
        fetch: Callable[[], Awaitable[list[dict[str, Any]]]],
        *,
        default: str,
        allow: Sequence[str] = (),
    ) -> None:
        self._fetch = fetch
        self._default = default
        self._allow = set(allow)
        self._cache: list[dict[str, Any]] | None = None
        self._fetched_at = 0.0
        self._lock = asyncio.Lock()

    async def models(self) -> list[dict[str, Any]]:
        async with self._lock:
            if self._cache is None or time.monotonic() - self._fetched_at > CATALOG_TTL_S:
                try:
                    self._cache = self._offer(await self._fetch())
                    self._fetched_at = time.monotonic()
                except Exception as exc:  # noqa: BLE001 - any failure degrades to the default model
                    logger.warning("Could not load the OpenRouter model catalog: %s", exc)
                    return self._cache or [self._placeholder()]
            return self._cache

    async def ids(self) -> set[str]:
        return {m["id"] for m in await self.models()}

    def _offer(self, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._allow:
            missing = self._allow - {m["id"] for m in catalog}
            if missing:
                logger.warning("LGRAPH_MODELS ids not offered (unknown, or no tool support): %s",
                               ", ".join(sorted(missing)))
            catalog = [m for m in catalog if m["id"] in self._allow or m["id"] == self._default]
        default = next((m for m in catalog if m["id"] == self._default), None) or self._placeholder()
        rest = sorted((m for m in catalog if m["id"] != self._default), key=lambda m: m["name"].lower())
        return [default, *rest]

    def _placeholder(self) -> dict[str, Any]:
        # Unknown from here; the server already sends an effort to its default model.
        return {"id": self._default, "name": self._default, "context_length": None,
                "prompt_price": None, "completion_price": None, "reasoning": True}


class Agents:
    """One compiled agent per chat model, built on first use."""

    def __init__(self, default: str, build: Callable[[str], Any], catalog: ModelCatalog) -> None:
        self.default = default
        self._build = build
        self._catalog = catalog
        self._agents: dict[str, Any] = {default: build(default)}

    async def get(self, model: str | None) -> Any:
        slug = model or self.default
        if slug not in self._agents:
            if slug not in await self._catalog.ids():
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown or unsupported model {slug!r}. GET /models lists the choices.",
                )
            self._agents[slug] = self._build(slug)
        return self._agents[slug]
