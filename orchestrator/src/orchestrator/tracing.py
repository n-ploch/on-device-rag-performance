"""Active OTEL tracing setup for the orchestrator.

Provides TracerProvider configuration with multi-backend OTLP export.
Supports Langfuse, W&B Weave, and generic OTLP endpoints simultaneously.
Uses W3C trace context propagation for distributed tracing.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING

import httpx

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

    from orchestrator.config import ObservabilityConfig

logger = logging.getLogger(__name__)


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
    url, headers = _endpoint_and_headers(backend_type)
    if url is None:
        if backend_type not in ("langfuse", "weave", "generic"):
            logger.warning("Unknown OTEL backend type %r — skipping", backend_type)
        return None
    return OTLPSpanExporter(endpoint=url, headers=headers)


def _endpoint_and_headers(backend_type: str) -> tuple[str | None, dict[str, str]]:
    """Return the OTLP endpoint URL and auth headers for a backend type.

    Returns (None, {}) if required credentials are missing.
    """
    if backend_type == "langfuse":
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
        host = os.environ.get("LANGFUSE_BASE_URL", "")
        if not (public_key and secret_key and host):
            return None, {}
        auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        return f"{host.rstrip('/')}/api/public/otel/v1/traces", {
            "Authorization": f"Basic {auth}"
        }

    if backend_type == "weave":
        api_key = os.environ.get("WANDB_API_KEY", "")
        base_url = os.environ.get("WANDB_BASE_URL", "")
        if not (api_key and base_url):
            return None, {}
        return f"{base_url.rstrip('/')}/otel/v1/traces", {"wandb-api-key": api_key}

    if backend_type == "generic":
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if not endpoint:
            return None, {}
        headers: dict[str, str] = {}
        raw_headers = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
        if raw_headers:
            for pair in raw_headers.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    headers[k.strip()] = v.strip()
        return endpoint, headers

    return None, {}


async def verify_tracing_connection(observability: "ObservabilityConfig | None") -> None:
    """Verify each enabled tracing backend is reachable before evaluation starts.

    Makes a lightweight HTTP GET to each backend's OTLP endpoint.
    Raises RuntimeError if any enabled backend is unreachable or returns
    an authentication error (401/403).

    Connection errors and auth failures abort the experiment; other HTTP
    error codes (e.g. 405 Method Not Allowed for a GET probe) are treated
    as "reachable" — the server is up, the endpoint just doesn't accept GET.
    """
    from orchestrator.config import OTLPBackendConfig

    if observability is not None and observability.backends:
        active_backends = [b for b in observability.backends if b.enabled]
    elif observability is None or observability.langfuse:
        active_backends = [OTLPBackendConfig(type="langfuse")]
    else:
        return  # tracing disabled — nothing to check

    for backend in active_backends:
        url, headers = _endpoint_and_headers(backend.type)
        if url is None:
            continue  # missing credentials — _check_run_preconditions already handles this

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
                resp = await client.get(url, headers=headers)
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    f"Tracing backend '{backend.type}' rejected the request "
                    f"(HTTP {resp.status_code}). Check your credentials."
                )
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to tracing backend '{backend.type}' at {url}. "
                "Check the URL and ensure the backend is running."
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Tracing backend '{backend.type}' at {url} timed out. "
                "Check the URL and ensure the backend is running."
            )


def setup_tracing(
    service_name: str = "rag-orchestrator",
    observability: ObservabilityConfig | None = None,
) -> Tracer:
    """Initialize OTEL tracing with one or more OTLP exporters.

    Configures TracerProvider with BatchSpanProcessor for each active backend.
    Sets up W3C trace context propagation for distributed tracing.

    Backend selection:
      - When observability.backends is non-empty, those backends are used.
      - When observability.langfuse is True (legacy flag), a Langfuse backend is used.
      - When observability is None, falls back to the Langfuse legacy path.

    Args:
        service_name: Service name for resource identification.
        observability: Observability config; when None uses Langfuse legacy path.

    Returns:
        Configured Tracer instance.
    """
    from orchestrator.config import OTLPBackendConfig

    # Determine active backend list
    if observability is not None and observability.backends:
        active_backends = [b for b in observability.backends if b.enabled]
    elif observability is None or observability.langfuse:
        active_backends = [OTLPBackendConfig(type="langfuse")]
    else:
        active_backends = []

    # Collect Weave resource attributes (must be on Resource, not span attrs)
    weave_attrs: dict[str, str] = {}
    if any(b.type == "weave" for b in active_backends):
        entity = os.environ.get("WANDB_ENTITY", "")
        project = os.environ.get("WANDB_PROJECT", "")
        if entity:
            weave_attrs["wandb.entity"] = entity
        if project:
            weave_attrs["wandb.project"] = project

    resource = Resource.create({"service.name": service_name, **weave_attrs})
    provider = TracerProvider(resource=resource)

    registered = 0
    for backend in active_backends:
        exporter = _build_exporter(backend.type)
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("OTEL backend registered: %s", backend.type)
            registered += 1
        else:
            logger.warning(
                "OTEL backend %r skipped — required credentials not set", backend.type
            )

    if registered == 0:
        logger.warning("No OTEL backends registered — spans will not be exported")

    # Always write spans locally when output_jsonl is configured
    if observability is not None and observability.output_jsonl:
        from orchestrator.exporters.jsonl import JSONLSpanExporter

        local_exporter = JSONLSpanExporter(observability.output_jsonl)
        provider.add_span_processor(SimpleSpanProcessor(local_exporter))
        logger.info("Local JSONL export enabled: %s", observability.output_jsonl)

    trace.set_tracer_provider(provider)
    set_global_textmap(CompositePropagator([TraceContextTextMapPropagator()]))

    return trace.get_tracer(__name__)


def shutdown_tracing() -> None:
    """Shutdown the tracer provider and flush pending spans."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
        logger.info("OTEL tracing shut down")
