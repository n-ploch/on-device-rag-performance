"""Active OTEL tracing setup for the orchestrator.

Provides TracerProvider configuration with OTLP export to Langfuse.
Uses W3C trace context propagation for distributed tracing.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

logger = logging.getLogger(__name__)


def setup_tracing(service_name: str = "rag-orchestrator") -> Tracer:
    """Initialize OTEL tracing with Langfuse OTLP exporter.

    Configures TracerProvider with BatchSpanProcessor and OTLP export.
    Sets up W3C trace context propagation for distributed tracing.

    Environment variables:
        LANGFUSE_PUBLIC_KEY: API public key for Langfuse auth
        LANGFUSE_SECRET_KEY: API secret key for Langfuse auth
        LANGFUSE_BASE_URL: Optional custom host (default: https://cloud.langfuse.com)

    Args:
        service_name: Service name for resource identification.

    Returns:
        Configured Tracer instance.
    """
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
        logger.info("OTEL tracing configured with Langfuse endpoint: %s", endpoint)
    else:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY/SECRET_KEY not set, tracing will not export spans"
        )

    trace.set_tracer_provider(provider)

    # Set up W3C trace context propagation
    set_global_textmap(CompositePropagator([TraceContextTextMapPropagator()]))

    return trace.get_tracer(__name__)


def shutdown_tracing() -> None:
    """Shutdown the tracer provider and flush pending spans."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
        logger.info("OTEL tracing shut down")
