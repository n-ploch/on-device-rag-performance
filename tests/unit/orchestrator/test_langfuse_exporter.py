"""Tests for Langfuse OTEL exporter."""

import os
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.exporters.langfuse_exporter import LangfuseExporter
from orchestrator.exporters.span_converter import (
    EvaluationSpan,
    SpanContext,
    SpanStatus,
    StatusCode,
)


def make_evaluation_span(
    trace_id: int = 0x12345678901234567890123456789012,
    span_id: int = 0x1234567890123456,
    start_time_ns: int = 1700000000000000000,
    end_time_ns: int = 1700000001500000000,
    name: str = "rag.evaluation",
    parent: SpanContext | None = None,
    attributes: dict | None = None,
) -> EvaluationSpan:
    """Create a test EvaluationSpan."""
    default_attrs = {
        "run_id": "test_run_001",
        "claim_id": "claim_001",
        "gen_ai.prompt": "What is the effect of aspirin?",
        "gen_ai.completion": "Aspirin reduces pain and inflammation.",
        "custom.metrics.recall_at_k": 1.0,
        "custom.metrics.precision_at_k": 0.33,
        "custom.metrics.mrr": 1.0,
        "custom.metrics.abstention": False,
        "custom.latency.e2e_latency_ms": 1500.0,
        "custom.latency.retrieval_latency_ms": 150.0,
        "custom.latency.ttft_ms": 300.0,
        "custom.latency.llm_generation_latency_ms": 1000.0,
        "custom.generation.prompt_tokens": 256,
        "custom.generation.completion_tokens": 128,
        "custom.generation.tokens_per_second": 128.0,
        "custom.hardware.max_ram_usage_mb": 4500.0,
        "custom.hardware.avg_cpu_utilization_pct": 85.0,
        "custom.hardware.peak_cpu_temp_c": 72.0,
        "custom.hardware.swap_in_bytes": 1024,
        "custom.hardware.swap_out_bytes": 512,
        "custom.retrieval.k": 3,
        "custom.retrieval.cited_doc_ids": "doc1,doc2,doc3",
    }
    if attributes:
        default_attrs.update(attributes)

    return EvaluationSpan(
        context=SpanContext(trace_id=trace_id, span_id=span_id),
        name=name,
        start_time=start_time_ns,
        end_time=end_time_ns,
        status=SpanStatus(status_code=StatusCode(name="OK")),
        attributes=default_attrs,
        parent=parent,
    )


@pytest.fixture
def mock_env(monkeypatch):
    """Set required environment variables."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-123")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-456")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://test.langfuse.com")


class TestLangfuseExporterInit:
    """Tests for LangfuseExporter initialization."""

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_creates_otlp_exporter_with_correct_endpoint(self, mock_otlp, mock_env):
        """Creates OTLP exporter with Langfuse endpoint."""
        exporter = LangfuseExporter()

        mock_otlp.assert_called_once()
        call_kwargs = mock_otlp.call_args.kwargs
        assert call_kwargs["endpoint"] == "https://test.langfuse.com/api/public/otel/v1/traces"

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_creates_otlp_exporter_with_auth_header(self, mock_otlp, mock_env):
        """Creates OTLP exporter with Basic auth header."""
        exporter = LangfuseExporter()

        call_kwargs = mock_otlp.call_args.kwargs
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"].startswith("Basic ")

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_uses_default_host_when_not_set(self, mock_otlp, monkeypatch):
        """Uses cloud.langfuse.com when LANGFUSE_BASE_URL not set."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)

        exporter = LangfuseExporter()

        call_kwargs = mock_otlp.call_args.kwargs
        assert "cloud.langfuse.com" in call_kwargs["endpoint"]

    def test_raises_when_keys_missing(self, monkeypatch):
        """Raises ValueError when API keys not set."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
            LangfuseExporter()


class TestLangfuseExporterExport:
    """Tests for export functionality."""

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_export_calls_otlp_exporter(self, mock_otlp_class, mock_env):
        """Export calls the underlying OTLP exporter."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        span = make_evaluation_span()
        exporter.export([span])

        mock_exporter.export.assert_called_once()

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_export_converts_to_readable_span(self, mock_otlp_class, mock_env):
        """Export converts EvaluationSpan to ReadableSpan."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        span = make_evaluation_span()
        exporter.export([span])

        exported_spans = mock_exporter.export.call_args[0][0]
        assert len(exported_spans) == 1

        readable_span = exported_spans[0]
        assert readable_span.name == "rag.evaluation"

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_export_preserves_trace_context(self, mock_otlp_class, mock_env):
        """Export preserves trace_id and span_id."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        span = make_evaluation_span(
            trace_id=0x12345678901234567890123456789012,
            span_id=0x1234567890123456,
        )
        exporter.export([span])

        exported_spans = mock_exporter.export.call_args[0][0]
        readable_span = exported_spans[0]
        assert readable_span.context.trace_id == 0x12345678901234567890123456789012
        assert readable_span.context.span_id == 0x1234567890123456

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_export_preserves_timing(self, mock_otlp_class, mock_env):
        """Export preserves start and end times."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        span = make_evaluation_span(
            start_time_ns=1700000000000000000,
            end_time_ns=1700000001500000000,
        )
        exporter.export([span])

        exported_spans = mock_exporter.export.call_args[0][0]
        readable_span = exported_spans[0]
        assert readable_span.start_time == 1700000000000000000
        assert readable_span.end_time == 1700000001500000000

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_export_multiple_spans(self, mock_otlp_class, mock_env):
        """Export handles multiple spans."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        spans = [
            make_evaluation_span(trace_id=0x1111, span_id=0x1111),
            make_evaluation_span(trace_id=0x2222, span_id=0x2222),
            make_evaluation_span(trace_id=0x3333, span_id=0x3333),
        ]
        exporter.export(spans)

        exported_spans = mock_exporter.export.call_args[0][0]
        assert len(exported_spans) == 3


class TestLangfuseAttributes:
    """Tests for Langfuse-specific attribute mapping."""

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_adds_langfuse_user_id(self, mock_otlp_class, mock_env):
        """Adds langfuse.user.id attribute from run_id."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        span = make_evaluation_span(attributes={"run_id": "my_run_123"})
        exporter.export([span])

        exported_spans = mock_exporter.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        assert attrs["langfuse.user.id"] == "my_run_123"

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_adds_langfuse_session_id(self, mock_otlp_class, mock_env):
        """Adds langfuse.session.id attribute from run_id."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        span = make_evaluation_span(attributes={"run_id": "my_run_123"})
        exporter.export([span])

        exported_spans = mock_exporter.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        assert attrs["langfuse.session.id"] == "my_run_123"

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_marks_as_generation_type(self, mock_otlp_class, mock_env):
        """Sets langfuse.observation.type to generation for generation spans."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        # Create a generation span (child of root)
        parent_ctx = SpanContext(trace_id=0x1234, span_id=0x5678)
        span = make_evaluation_span(name="rag.generation", parent=parent_ctx)
        exporter.export([span])

        exported_spans = mock_exporter.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        assert attrs["langfuse.observation.type"] == "generation"

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_marks_retrieval_as_span_type(self, mock_otlp_class, mock_env):
        """Sets langfuse.observation.type to span for retrieval spans."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        parent_ctx = SpanContext(trace_id=0x1234, span_id=0x5678)
        span = make_evaluation_span(name="rag.retrieval", parent=parent_ctx)
        exporter.export([span])

        exported_spans = mock_exporter.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        assert attrs["langfuse.observation.type"] == "span"

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_root_span_has_no_observation_type(self, mock_otlp_class, mock_env):
        """Root span (rag.evaluation) has no observation.type."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        span = make_evaluation_span()  # Root span, no parent
        exporter.export([span])

        exported_spans = mock_exporter.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        assert "langfuse.observation.type" not in attrs

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_preserves_usage_attributes_from_span(self, mock_otlp_class, mock_env):
        """Preserves langfuse.observation.usage.* attributes set by span_converter."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        parent_ctx = SpanContext(trace_id=0x1234, span_id=0x5678)
        # Generation span with usage attributes (as set by span_converter)
        span = make_evaluation_span(
            name="rag.generation",
            parent=parent_ctx,
            attributes={
                "langfuse.observation.usage.input": 100,
                "langfuse.observation.usage.output": 50,
            }
        )
        exporter.export([span])

        exported_spans = mock_exporter.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        assert attrs["langfuse.observation.usage.input"] == 100
        assert attrs["langfuse.observation.usage.output"] == 50

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_adds_score_attributes(self, mock_otlp_class, mock_env):
        """Adds langfuse.score.* attributes for metrics."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        span = make_evaluation_span(attributes={
            "custom.metrics.recall_at_k": 0.9,
            "custom.metrics.precision_at_k": 0.8,
            "custom.metrics.mrr": 0.95,
        })
        exporter.export([span])

        exported_spans = mock_exporter.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        assert attrs["langfuse.score.recall_at_k"] == 0.9
        assert attrs["langfuse.score.precision_at_k"] == 0.8
        assert attrs["langfuse.score.mrr"] == 0.95

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_adds_abstention_score(self, mock_otlp_class, mock_env):
        """Adds langfuse.score.abstention when model abstains."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        span = make_evaluation_span(attributes={"custom.metrics.abstention": True})
        exporter.export([span])

        exported_spans = mock_exporter.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        assert attrs["langfuse.score.abstention"] == 1.0

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_adds_trace_metadata(self, mock_otlp_class, mock_env):
        """Adds langfuse.trace.metadata.* for filterable fields."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        span = make_evaluation_span(attributes={
            "run_id": "run_abc",
            "claim_id": "claim_xyz",
        })
        exporter.export([span])

        exported_spans = mock_exporter.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        assert attrs["langfuse.trace.metadata.run_id"] == "run_abc"
        assert attrs["langfuse.trace.metadata.claim_id"] == "claim_xyz"

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_preserves_original_attributes(self, mock_otlp_class, mock_env):
        """Preserves original OTEL attributes."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        span = make_evaluation_span()
        exporter.export([span])

        exported_spans = mock_exporter.export.call_args[0][0]
        attrs = exported_spans[0].attributes
        assert attrs["gen_ai.prompt"] == "What is the effect of aspirin?"
        assert attrs["gen_ai.completion"] == "Aspirin reduces pain and inflammation."


class TestLangfuseExporterShutdown:
    """Tests for shutdown functionality."""

    @patch("orchestrator.exporters.langfuse_exporter.OTLPSpanExporter")
    def test_shutdown_calls_otlp_shutdown(self, mock_otlp_class, mock_env):
        """Shutdown calls underlying OTLP exporter shutdown."""
        mock_exporter = MagicMock()
        mock_otlp_class.return_value = mock_exporter

        exporter = LangfuseExporter()
        exporter.shutdown()

        mock_exporter.shutdown.assert_called_once()
