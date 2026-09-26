from __future__ import annotations

import httpx
import pytest
from openrouter.errors import NoResponseError, OpenRouterError

from lgraph.errors import UpstreamError, translate


def _sdk_error(status: int, body: str = "boom") -> OpenRouterError:
    return OpenRouterError("API error occurred", raw_response=httpx.Response(status, text=body))


def test_timeout_maps_to_504() -> None:
    err = translate(httpx.ReadTimeout("slow"))
    assert isinstance(err, UpstreamError)
    assert err.status_code == 504
    assert err.upstream_status is None


def test_rate_limit_passes_through_as_429() -> None:
    err = translate(_sdk_error(429, "slow down"))
    assert err is not None
    assert err.status_code == 429
    assert err.upstream_status == 429
    assert "slow down" in err.message


def test_other_sdk_errors_are_502_with_upstream_status() -> None:
    err = translate(_sdk_error(500))
    assert err is not None
    assert err.status_code == 502
    assert err.upstream_status == 500


def test_sdk_error_body_is_truncated() -> None:
    err = translate(_sdk_error(500, "x" * 5000))
    assert err is not None
    assert len(err.message) < 1000
    assert "more characters" in err.message


def test_no_response_maps_to_502() -> None:
    err = translate(NoResponseError())
    assert err is not None
    assert err.status_code == 502


def test_transport_errors_map_to_502() -> None:
    err = translate(httpx.ConnectError("refused"))
    assert err is not None
    assert err.status_code == 502


def test_chat_openrouter_value_error_maps_to_502() -> None:
    err = translate(ValueError("OpenRouter API returned an error: bad model"))
    assert err is not None
    assert err.status_code == 502


@pytest.mark.parametrize("exc", [ValueError("unrelated"), KeyError("k"), RuntimeError("x")])
def test_unrelated_exceptions_are_not_translated(exc: Exception) -> None:
    assert translate(exc) is None


def test_upstream_error_defaults() -> None:
    err = UpstreamError("m")
    assert err.status_code == 502
    assert err.upstream_status is None
    assert str(err) == "m"
