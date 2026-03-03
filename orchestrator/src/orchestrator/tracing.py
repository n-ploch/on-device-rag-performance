"""Langfuse SDK tracing setup for the orchestrator.

Provides a Langfuse client configured from environment variables.
Uses Langfuse's native SDK instead of the OTEL abstraction layer,
giving direct access to scores, usage tracking, and trace metadata.
"""

from __future__ import annotations

import logging
import os

from langfuse import Langfuse

logger = logging.getLogger(__name__)


def get_langfuse_client() -> Langfuse:
    """Create a Langfuse client from environment variables.

    Environment variables:
        LANGFUSE_PUBLIC_KEY: API public key
        LANGFUSE_SECRET_KEY: API secret key
        LANGFUSE_BASE_URL: Optional custom host (default: https://cloud.langfuse.com)

    Returns:
        Configured Langfuse client instance.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY/SECRET_KEY not set, tracing will not export spans"
        )

    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )
    logger.info("Langfuse client initialized with host: %s", host)
    return client


def shutdown_tracing(client: Langfuse) -> None:
    """Flush pending spans and shut down the Langfuse client.

    Args:
        client: The Langfuse client to shut down.
    """
    client.flush()
    logger.info("Langfuse tracing flushed and shut down")
