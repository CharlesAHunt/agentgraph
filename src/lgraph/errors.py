"""Translate upstream failures into a single gateway-level error.

Framework-free: this module decides *which* HTTP status a failure should
surface as, and the API layer turns it into a response.
"""

from __future__ import annotations


from typing import NoReturn

import httpx
from openrouter.errors import NoResponseError, OpenRouterError

from .text import truncate

# ChatOpenRouter raises a plain ValueError for API-level errors it detects in
# an otherwise successful HTTP response. This is the prefix it uses.
_CHAT_OPENROUTER_ERROR_PREFIX = "OpenRouter API returned an error"


class UpstreamError(Exception):
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
        upstream_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.upstream_status = upstream_status


def translate(exc: BaseException) -> UpstreamError | None:
    """Map a model-call exception to an :class:`UpstreamError`, or ``None``.

    ``None`` means the exception is not an upstream failure we recognise and
    should propagate unchanged.
    """
    if isinstance(exc, httpx.TimeoutException):
        return UpstreamError(f"OpenRouter request timed out: {exc}", status_code=504)

    if isinstance(exc, OpenRouterError):
        # Pass 429 through so clients can back off; everything else is a
        # gateway problem from the caller's point of view, not theirs.
        status_code = 429 if exc.status_code == 429 else 502
        return UpstreamError(
            f"OpenRouter returned {exc.status_code}: {_truncate(exc.body or exc.message)}",
            status_code=status_code,
            upstream_status=exc.status_code,
        )

    if isinstance(exc, (NoResponseError, httpx.HTTPError)):
        return UpstreamError(f"OpenRouter request failed: {exc}", status_code=502)

    if isinstance(exc, ValueError) and str(exc).startswith(_CHAT_OPENROUTER_ERROR_PREFIX):
        return UpstreamError(_truncate(str(exc)), status_code=502)

    return None


def _truncate(text: str) -> str:
    """Keep an unexpected upstream body from flooding logs and error payloads."""
    return truncate(text, 500)


def raise_upstream(exc: BaseException) -> NoReturn:
    """Raise ``exc`` as an :class:`UpstreamError` when it is an upstream failure, else unchanged."""
    upstream = translate(exc)
    if upstream is None:
        raise exc
    raise upstream from exc
