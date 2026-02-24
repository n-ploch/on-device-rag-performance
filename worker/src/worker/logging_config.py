"""Logging configuration for the worker service."""

from __future__ import annotations

import logging
import os
import sys


def setup_logging() -> None:
    """Configure structured logging for the worker service."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Quiet noisy libraries
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
