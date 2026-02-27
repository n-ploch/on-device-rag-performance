"""Langfuse exporter using native OpenTelemetry OTLP.

Exports EvaluationSpan data to Langfuse via their OTEL endpoint.
Uses standard OTLP HTTP exporter with Langfuse-specific attributes
for proper mapping to their data model.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING, Any

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, TraceFlags
from opentelemetry.trace.status import Status, StatusCode

if TYPE_CHECKING:
    from orchestrator.exporters.span_converter import EvaluationSpan

logger = logging.getLogger(__name__)


class LangfuseExporter:
    """Export evaluation spans to Langfuse via OTLP HTTP.

    Uses OpenTelemetry's standard OTLP exporter to send spans to
    Langfuse's /api/public/otel endpoint. Configuration is read
    from environment variables:

    - LANGFUSE_PUBLIC_KEY: API public key
    - LANGFUSE_SECRET_KEY: API secret key
    - LANGFUSE_BASE_URL: Optional custom host (default: https://cloud.langfuse.com)
    """

    def __init__(self):
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
        host = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set"
            )

        auth_string = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        endpoint = f"{host.rstrip('/')}/api/public/otel/v1/traces"

        self._exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers={"Authorization": f"Basic {auth_string}"},
        )
        self._resource = Resource.create({"service.name": "rag-orchestrator"})

        logger.info("LangfuseExporter initialized with endpoint %s", endpoint)

    def export(self, spans: list[EvaluationSpan]) -> None:
        """Export spans to Langfuse via OTLP.

        Args:
            spans: List of evaluation spans to export (typically root + children).
        """
        readable_spans = [self._to_readable_span(s) for s in spans]
        result = self._exporter.export(readable_spans)
        logger.debug("Exported %d spans, result: %s", len(spans), result)

    def _to_readable_span(self, span: EvaluationSpan) -> ReadableSpan:
        """Convert EvaluationSpan to OTEL ReadableSpan with Langfuse attributes."""
        attrs = self._build_langfuse_attributes(span)

        context = SpanContext(
            trace_id=span.context.trace_id,
            span_id=span.context.span_id,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )

        # Build parent context if span has a parent
        parent_context = None
        if span.parent is not None:
            parent_context = SpanContext(
                trace_id=span.parent.trace_id,
                span_id=span.parent.span_id,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )

        status = Status(StatusCode.OK)
        if span.status.status_code.name == "ERROR":
            status = Status(StatusCode.ERROR)

        return ReadableSpan(
            name=span.name,
            context=context,
            parent=parent_context,
            resource=self._resource,
            attributes=attrs,
            start_time=span.start_time,
            end_time=span.end_time,
            status=status,
        )

    def _build_langfuse_attributes(self, span: EvaluationSpan) -> dict[str, Any]:
        """Build attributes dict with Langfuse-specific namespaced attributes.

        Langfuse maps certain attribute prefixes to their data model:
        - langfuse.user.id -> user identification
        - langfuse.session.id -> session grouping
        - langfuse.observation.type -> span type (generation, span, event)
        - langfuse.observation.usage.* -> token usage
        - langfuse.score.* -> evaluation scores
        - langfuse.trace.metadata.* -> filterable metadata
        """
        attrs = span.attributes
        result = dict(attrs)

        # Add user/session IDs for trace grouping
        result["langfuse.user.id"] = attrs.get("run_id", "unknown")
        result["langfuse.session.id"] = attrs.get("run_id", "unknown")

        # Set observation type based on span name
        if span.name == "rag.generation":
            result["langfuse.observation.type"] = "generation"
        elif span.name == "rag.retrieval":
            result["langfuse.observation.type"] = "span"
        # Root span (rag.evaluation) doesn't need observation.type

        # Add scores only for root span (they're trace-level metrics)
        if span.parent is None:
            score_mappings = [
                ("custom.metrics.recall_at_k", "recall_at_k"),
                ("custom.metrics.precision_at_k", "precision_at_k"),
                ("custom.metrics.mrr", "mrr"),
            ]
            for attr_key, score_name in score_mappings:
                value = attrs.get(attr_key)
                if value is not None:
                    result[f"langfuse.score.{score_name}"] = value

            if attrs.get("custom.metrics.abstention") is True:
                result["langfuse.score.abstention"] = 1.0

        # Add filterable metadata
        result["langfuse.trace.metadata.run_id"] = attrs.get("run_id")
        result["langfuse.trace.metadata.claim_id"] = attrs.get("claim_id")

        return result

    def shutdown(self) -> None:
        """Shutdown the OTLP exporter."""
        self._exporter.shutdown()
        logger.info("LangfuseExporter shut down")
