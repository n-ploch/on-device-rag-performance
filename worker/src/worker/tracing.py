"""Langfuse SDK tracing setup for the worker.

Provides a Langfuse client and helper to extract trace context from
incoming request headers for distributed tracing.
"""

from __future__ import annotations

import logging
import os

from langfuse import Langfuse

logger = logging.getLogger(__name__)

# Module-level client, initialized once at startup
_client: Langfuse | None = None


def setup_tracing() -> Langfuse:
    """Initialize the Langfuse client from environment variables.

    Environment variables:
        LANGFUSE_PUBLIC_KEY: API public key
        LANGFUSE_SECRET_KEY: API secret key
        LANGFUSE_BASE_URL: Optional custom host (default: https://cloud.langfuse.com)

    Returns:
        Configured Langfuse client instance.
    """
    global _client
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY/SECRET_KEY not set, worker tracing will not export spans"
        )

    _client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    logger.info("Worker Langfuse client initialized with host: %s", host)
    return _client


def get_client() -> Langfuse:
    """Get the initialized Langfuse client, initializing if needed.

    Returns:
        The Langfuse client instance.
    """
    global _client
    if _client is None:
        _client = setup_tracing()
    return _client


def extract_trace_context(headers: dict[str, str]) -> tuple[str | None, str | None]:
    """Extract trace_id and parent_span_id from custom Langfuse headers.

    The orchestrator injects these headers when making requests to the worker
    so that child spans can be linked to the orchestrator's root trace.

    Args:
        headers: Request headers dict (values are case-insensitive).

    Returns:
        Tuple of (trace_id, parent_span_id), either may be None if not present.
    """
    # FastAPI normalizes header keys to lowercase
    trace_id = headers.get("x-langfuse-trace-id")
    parent_span_id = headers.get("x-langfuse-parent-span-id")
    return trace_id, parent_span_id


def shutdown_tracing() -> None:
    """Flush pending spans and shut down the Langfuse client."""
    global _client
    if _client is not None:
        _client.flush()
        logger.info("Worker Langfuse tracing flushed and shut down")
