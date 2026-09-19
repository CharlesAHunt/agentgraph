"""Console entry point: run the FastAPI app under uvicorn.

Usable either as ``lgraph`` (the installed script) or ``python -m lgraph``.
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from .config import ConfigError, get_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    try:
        settings = get_settings()
    except ConfigError as exc:
        # A missing API key is an operator error, not a crash worth a traceback.
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    # Passing an import string (rather than the app object) is what lets
    # uvicorn support --reload/workers-style respawning later on.
    uvicorn.run(
        "lgraph.api:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
