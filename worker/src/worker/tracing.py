"""Active OTEL tracing setup for the worker.

Provides TracerProvider configuration with OTLP export to Langfuse.
Extracts trace context from incoming requests for distributed tracing.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

logger = logging.getLogger(__name__)

# Module-level tracer, initialized on first call to get_tracer()
_tracer: Tracer | None = None


def setup_tracing(service_name: str = "rag-worker") -> Tracer:
    """Initialize OTEL tracing with Langfuse OTLP exporter.

    Configures TracerProvider with BatchSpanProcessor and OTLP export.

    Environment variables:
        LANGFUSE_PUBLIC_KEY: API public key for Langfuse auth
        LANGFUSE_SECRET_KEY: API secret key for Langfuse auth
        LANGFUSE_BASE_URL: Optional custom host (default: https://cloud.langfuse.com)

    Args:
        service_name: Service name for resource identification.

    Returns:
        Configured Tracer instance.
    """
    global _tracer

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Configure OTLP exporter with Langfuse auth
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if public_key and secret_key:
        auth_string = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        endpoint = f"{host.rstrip('/')}/api/public/otel/v1/traces"

        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers={"Authorization": f"Basic {auth_string}"},
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info("Worker OTEL tracing configured with Langfuse endpoint: %s", endpoint)
    else:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY/SECRET_KEY not set, worker tracing will not export spans"
        )

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(__name__)

    return _tracer


def get_tracer() -> Tracer:
    """Get the configured tracer, initializing if needed.

    Returns:
        The configured Tracer instance.
    """
    global _tracer
    if _tracer is None:
        _tracer = setup_tracing()
    return _tracer


def extract_context(headers: dict[str, str]) -> Context:
    """Extract trace context from incoming request headers.

    Uses W3C trace context propagation to extract trace_id and span_id
    from the traceparent header.

    Args:
        headers: Request headers dict (case-insensitive keys).

    Returns:
        OpenTelemetry Context with extracted trace information.
    """
    # Convert headers to lowercase keys for consistent extraction
    normalized = {k.lower(): v for k, v in headers.items()}
    return extract(normalized)


def shutdown_tracing() -> None:
    """Shutdown the tracer provider and flush pending spans."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
        logger.info("Worker OTEL tracing shut down")
