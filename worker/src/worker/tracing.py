"""Active OTEL tracing setup for the worker.

Provides TracerProvider configuration with multi-backend OTLP export.
Auto-detects active backends from environment variables.
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

# Auto-detection order for backends
_AUTO_BACKENDS = ["langfuse", "weave", "generic"]


def _build_exporter(backend_type: str) -> OTLPSpanExporter | None:
    """Build an OTLP exporter for the given backend type.

    Reads credentials from environment variables. Returns None if required
    variables are absent so the caller can skip registering the processor.

    Required env vars per backend:
        langfuse: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
        weave:    WANDB_API_KEY, WANDB_BASE_URL
        generic:  OTEL_EXPORTER_OTLP_ENDPOINT

    Args:
        backend_type: One of "langfuse", "weave", or "generic".

    Returns:
        Configured OTLPSpanExporter or None if credentials are missing.
    """
    if backend_type == "langfuse":
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
        host = os.environ.get("LANGFUSE_BASE_URL", "")
        if not (public_key and secret_key and host):
            return None
        auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        return OTLPSpanExporter(
            endpoint=f"{host.rstrip('/')}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}"},
        )

    if backend_type == "weave":
        api_key = os.environ.get("WANDB_API_KEY", "")
        base_url = os.environ.get("WANDB_BASE_URL", "")
        if not (api_key and base_url):
            return None
        return OTLPSpanExporter(
            endpoint=f"{base_url.rstrip('/')}/otel/v1/traces",
            headers={"wandb-api-key": api_key},
        )

    if backend_type == "generic":
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if not endpoint:
            return None
        headers: dict[str, str] = {}
        raw_headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
        if raw_headers:
            for pair in raw_headers.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    headers[k.strip()] = v.strip()
        return OTLPSpanExporter(endpoint=endpoint, headers=headers)

    logger.warning("Unknown OTEL backend type %r — skipping", backend_type)
    return None


def setup_tracing(service_name: str = "rag-worker") -> Tracer:
    """Initialize OTEL tracing with auto-detected OTLP exporters.

    Auto-detects active backends by attempting to build an exporter for each
    known backend type. Any backend whose required env vars are present will
    be registered. Multiple backends can be active simultaneously.

    Args:
        service_name: Service name for resource identification.

    Returns:
        Configured Tracer instance.
    """
    global _tracer

    # Collect Weave resource attributes if Weave credentials are present
    weave_attrs: dict[str, str] = {}
    if os.environ.get("WANDB_API_KEY"):
        entity = os.environ.get("WANDB_ENTITY", "")
        project = os.environ.get("WANDB_PROJECT", "")
        if entity:
            weave_attrs["wandb.entity"] = entity
        if project:
            weave_attrs["wandb.project"] = project

    resource = Resource.create({"service.name": service_name, **weave_attrs})
    provider = TracerProvider(resource=resource)

    registered = 0
    for backend_type in _AUTO_BACKENDS:
        exporter = _build_exporter(backend_type)
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("Worker OTEL backend registered: %s", backend_type)
            registered += 1

    if registered == 0:
        logger.warning("No OTEL backends registered — worker spans will not be exported")

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
