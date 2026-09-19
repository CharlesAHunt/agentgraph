"""OpenRouter HTTP client.

Deliberately free of any web-framework imports: this layer raises
:class:`OpenRouterError`, and the API layer decides how that becomes a
response.
"""

from __future__ import annotations

from typing import Any, List, Optional

import httpx

from .config import Settings


class OpenRouterError(Exception):
    """An OpenRouter call failed.

    ``status_code`` is the HTTP status this should surface as, already
    translated into gateway terms. ``upstream_status`` is what OpenRouter
    actually returned, when there was a response at all.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        upstream_status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.upstream_status = upstream_status


class OpenRouterClient:
    """Thin async wrapper over the chat-completions endpoint.

    One instance per process: it owns a pooled ``httpx.AsyncClient`` so
    connections and TLS handshakes are reused across requests.
    """

    def __init__(self, settings: Settings, http: httpx.AsyncClient) -> None:
        self._settings = settings
        self._http = http

    @classmethod
    def create(cls, settings: Settings) -> OpenRouterClient:
        return cls(settings, httpx.AsyncClient(timeout=settings.timeout))

    async def aclose(self) -> None:
        await self._http.aclose()

    @property
    def model(self) -> str:
        return self._settings.model

    async def chat_completion(self, messages: List[dict[str, Any]]) -> dict[str, Any]:
        """Send ``messages`` and return the assistant message object.

        Raises :class:`OpenRouterError` for every failure mode.
        """
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._settings.model,
            "messages": messages,
            "reasoning": {"enabled": True},
        }

        try:
            response = await self._http.post(
                self._settings.base_url, headers=headers, json=payload
            )
        except httpx.TimeoutException as exc:
            raise OpenRouterError(
                f"OpenRouter request timed out: {exc}", status_code=504
            ) from exc
        except httpx.HTTPError as exc:
            raise OpenRouterError(
                f"OpenRouter request failed: {exc}", status_code=502
            ) from exc

        if response.status_code != 200:
            # Pass 429 through so clients can back off; everything else is a
            # gateway problem from the caller's point of view, not theirs.
            status_code = 429 if response.status_code == 429 else 502
            raise OpenRouterError(
                f"OpenRouter returned {response.status_code}: {_truncate(response.text)}",
                status_code=status_code,
                upstream_status=response.status_code,
            )

        try:
            body = response.json()
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise OpenRouterError(
                f"Could not parse OpenRouter response: {exc}. "
                f"Raw body: {_truncate(response.text)}",
                status_code=502,
                upstream_status=response.status_code,
            ) from exc

        if not isinstance(message, dict):
            raise OpenRouterError(
                f"OpenRouter returned a non-object message: {type(message).__name__}",
                status_code=502,
                upstream_status=response.status_code,
            )

        return message


def _truncate(text: str, limit: int = 500) -> str:
    """Keep an unexpected upstream body from flooding logs and error payloads."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text) - limit} more chars]"
