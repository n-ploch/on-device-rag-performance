"""Tests defining exporter APIs (JSONL and Langfuse)."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock

from orchestrator.exporters.jsonl import JSONLSpanExporter
from shared_types.schemas import InferenceMeasurement, HardwareMeasurement


def make_otel_span(
    name: str,
    trace_id: str = "0" * 32,
    span_id: str = "0" * 16,
    parent_span_id: str | None = None,
    start_time_unix_nano: int = 1700000000000000000,
    end_time_unix_nano: int = 1700000001000000000,
    attributes: dict | None = None,
    status_code: str = "OK",
) -> Mock:
    """Create a mock OTEL ReadableSpan for testing."""
    span = Mock()
    span.name = name
    span.context.trace_id = int(trace_id, 16)
    span.context.span_id = int(span_id, 16)
    span.parent.span_id = int(parent_span_id, 16) if parent_span_id else None
    span.start_time = start_time_unix_nano
    span.end_time = end_time_unix_nano
    span.attributes = attributes or {}
    span.status.status_code.name = status_code
    return span


def make_inference_measurement() -> InferenceMeasurement:
    """Create a sample InferenceMeasurement for testing."""
    return InferenceMeasurement(
        e2e_latency_ms=1450.5,
        retrieval_latency_ms=150.0,
        ttft_ms=300.0,
        llm_generation_latency_ms=1000.5,
        prompt_tokens=256,
        completion_tokens=128,
        tokens_per_second=133.3,
    )


def make_hardware_measurement() -> HardwareMeasurement:
    """Create a sample HardwareMeasurement for testing."""
    return HardwareMeasurement(
        max_ram_usage_mb=4500.2,
        avg_cpu_utilization_pct=85.5,
        peak_cpu_temp_c=72.0,
        swap_in_bytes=1024,
        swap_out_bytes=512,
    )


class TestJSONLExporter:
    """Tests for JSONL span exporter with OTEL format."""

    def test_exports_otel_required_fields(self, tmp_path):
        """Exported JSON contains required OTEL span fields."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        span = make_otel_span(name="rag.generate")
        exporter.export([span])
        exporter.shutdown()

        with open(output) as f:
            data = json.loads(f.readline())

        # Required OTEL fields
        assert "trace_id" in data
        assert "span_id" in data
        assert "name" in data
        assert "start_time_unix_nano" in data
        assert "end_time_unix_nano" in data
        assert "status" in data

    def test_trace_id_is_32_hex_chars(self, tmp_path):
        """trace_id is 32 hex characters (128-bit)."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        span = make_otel_span(name="test", trace_id="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4")
        exporter.export([span])
        exporter.shutdown()

        with open(output) as f:
            data = json.loads(f.readline())

        assert len(data["trace_id"]) == 32
        assert all(c in "0123456789abcdef" for c in data["trace_id"])

    def test_span_id_is_16_hex_chars(self, tmp_path):
        """span_id is 16 hex characters (64-bit)."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        span = make_otel_span(name="test", span_id="a1b2c3d4e5f6a1b2")
        exporter.export([span])
        exporter.shutdown()

        with open(output) as f:
            data = json.loads(f.readline())

        assert len(data["span_id"]) == 16
        assert all(c in "0123456789abcdef" for c in data["span_id"])

    def test_parent_span_id_when_present(self, tmp_path):
        """parent_span_id included when span has parent."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        span = make_otel_span(name="child", parent_span_id="1234567890abcdef")
        exporter.export([span])
        exporter.shutdown()

        with open(output) as f:
            data = json.loads(f.readline())

        assert "parent_span_id" in data
        assert len(data["parent_span_id"]) == 16

    def test_timestamps_are_nanoseconds(self, tmp_path):
        """Timestamps are Unix nanoseconds (19 digits)."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        span = make_otel_span(
            name="test",
            start_time_unix_nano=1700000000000000000,
            end_time_unix_nano=1700000001000000000,
        )
        exporter.export([span])
        exporter.shutdown()

        with open(output) as f:
            data = json.loads(f.readline())

        assert isinstance(data["start_time_unix_nano"], int)
        assert isinstance(data["end_time_unix_nano"], int)
        assert data["end_time_unix_nano"] > data["start_time_unix_nano"]

    def test_attributes_preserved(self, tmp_path):
        """Span attributes are preserved in export."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        span = make_otel_span(
            name="rag.generate",
            attributes={
                "claim_id": "claim_001",
                "run_id": "mistral_q4_baseline_001",
                "retrieval.k": 3,
                "retrieval.latency_ms": 150.5,
            },
        )
        exporter.export([span])
        exporter.shutdown()

        with open(output) as f:
            data = json.loads(f.readline())

        attrs = {a["key"]: a["value"] for a in data["attributes"]}
        assert attrs["claim_id"]["stringValue"] == "claim_001"
        assert attrs["run_id"]["stringValue"] == "mistral_q4_baseline_001"
        assert attrs["retrieval.k"]["intValue"] == 3

    def test_status_code_exported(self, tmp_path):
        """Status code (OK, ERROR) is exported."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        span = make_otel_span(name="test", status_code="ERROR")
        exporter.export([span])
        exporter.shutdown()

        with open(output) as f:
            data = json.loads(f.readline())

        assert data["status"]["code"] == "ERROR"

    def test_appends_multiple_spans(self, tmp_path):
        """Multiple exports append, don't overwrite."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        exporter.export([make_otel_span(name="span1")])
        exporter.export([make_otel_span(name="span2")])
        exporter.shutdown()

        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_batch_export(self, tmp_path):
        """Can export multiple spans in single batch."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        spans = [
            make_otel_span(name="span1", span_id="1" * 16),
            make_otel_span(name="span2", span_id="2" * 16),
            make_otel_span(name="span3", span_id="3" * 16),
        ]
        exporter.export(spans)
        exporter.shutdown()

        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_creates_parent_dirs(self, tmp_path):
        """Creates parent directories if they don't exist."""
        output = tmp_path / "nested" / "dir" / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))
        exporter.export([make_otel_span(name="test")])
        exporter.shutdown()
        assert output.exists()

    def test_shutdown_flushes(self, tmp_path):
        """shutdown() ensures all spans are written."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))
        exporter.export([make_otel_span(name="test")])
        exporter.shutdown()
        assert output.exists()
        with open(output) as f:
            assert len(f.readlines()) == 1


class TestOTELAttributeFormat:
    """Tests for OTEL typed attribute value format."""

    def test_attributes_use_typed_values(self, tmp_path):
        """Attributes use OTEL typed value format (stringValue, doubleValue, etc)."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        span = make_otel_span(
            name="rag.generate",
            attributes={
                "claim_id": "claim_001",
                "retrieval.k": 3,
                "latency_ms": 150.5,
                "abstention": False,
            },
        )
        exporter.export([span])
        exporter.shutdown()

        with open(output) as f:
            data = json.loads(f.readline())

        # Attributes should be a list of key-value pairs with typed values
        attrs = {a["key"]: a["value"] for a in data["attributes"]}

        assert "stringValue" in attrs["claim_id"]
        assert attrs["claim_id"]["stringValue"] == "claim_001"

        assert "intValue" in attrs["retrieval.k"]
        assert attrs["retrieval.k"]["intValue"] == 3

        assert "doubleValue" in attrs["latency_ms"]
        assert attrs["latency_ms"]["doubleValue"] == 150.5

        assert "boolValue" in attrs["abstention"]
        assert attrs["abstention"]["boolValue"] is False

    def test_inference_measurement_attributes(self, tmp_path):
        """InferenceMeasurement fields exported with correct types and namespaces."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        measurement = make_inference_measurement()
        # Convert measurement to span attributes with proper namespacing
        span = make_otel_span(
            name="rag.generate",
            attributes={
                f"custom.latency.{field}": getattr(measurement, field)
                for field in ["e2e_latency_ms", "retrieval_latency_ms", "ttft_ms", "llm_generation_latency_ms"]
            } | {
                f"custom.generation.{field}": getattr(measurement, field)
                for field in ["prompt_tokens", "completion_tokens", "tokens_per_second"]
            },
        )
        exporter.export([span])
        exporter.shutdown()

        with open(output) as f:
            data = json.loads(f.readline())

        attrs = {a["key"]: a["value"] for a in data["attributes"]}

        # Latency metrics should be doubleValue
        assert attrs["custom.latency.e2e_latency_ms"]["doubleValue"] == 1450.5
        assert attrs["custom.latency.retrieval_latency_ms"]["doubleValue"] == 150.0
        assert attrs["custom.latency.ttft_ms"]["doubleValue"] == 300.0
        assert attrs["custom.latency.llm_generation_latency_ms"]["doubleValue"] == 1000.5

        # Token counts should be intValue
        assert attrs["custom.generation.prompt_tokens"]["intValue"] == 256
        assert attrs["custom.generation.completion_tokens"]["intValue"] == 128

        # Tokens per second should be doubleValue
        assert attrs["custom.generation.tokens_per_second"]["doubleValue"] == 133.3

    def test_hardware_measurement_attributes(self, tmp_path):
        """HardwareMeasurement fields exported with correct types and namespaces."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        measurement = make_hardware_measurement()
        span = make_otel_span(
            name="rag.generate",
            attributes={
                f"custom.hardware.{field}": getattr(measurement, field)
                for field in measurement.model_fields.keys()
            },
        )
        exporter.export([span])
        exporter.shutdown()

        with open(output) as f:
            data = json.loads(f.readline())

        attrs = {a["key"]: a["value"] for a in data["attributes"]}

        # RAM and CPU metrics should be doubleValue
        assert attrs["custom.hardware.max_ram_usage_mb"]["doubleValue"] == 4500.2
        assert attrs["custom.hardware.avg_cpu_utilization_pct"]["doubleValue"] == 85.5
        assert attrs["custom.hardware.peak_cpu_temp_c"]["doubleValue"] == 72.0

        # Swap bytes should be intValue
        assert attrs["custom.hardware.swap_in_bytes"]["intValue"] == 1024
        assert attrs["custom.hardware.swap_out_bytes"]["intValue"] == 512

    def test_deterministic_metrics_attributes(self, tmp_path):
        """Deterministic metrics (recall, precision, mrr, abstention) exported correctly."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        span = make_otel_span(
            name="rag.generate",
            attributes={
                "custom.metrics.recall_at_k": 1.0,
                "custom.metrics.precision_at_k": 0.33,
                "custom.metrics.mrr": 1.0,
                "custom.metrics.abstention": False,
            },
        )
        exporter.export([span])
        exporter.shutdown()

        with open(output) as f:
            data = json.loads(f.readline())

        attrs = {a["key"]: a["value"] for a in data["attributes"]}

        # Retrieval metrics should be doubleValue
        assert attrs["custom.metrics.recall_at_k"]["doubleValue"] == 1.0
        assert attrs["custom.metrics.precision_at_k"]["doubleValue"] == 0.33
        assert attrs["custom.metrics.mrr"]["doubleValue"] == 1.0

        # Abstention should be boolValue
        assert attrs["custom.metrics.abstention"]["boolValue"] is False

    def test_gen_ai_semantic_convention_attributes(self, tmp_path):
        """gen_ai.* attributes follow OTEL semantic conventions."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        span = make_otel_span(
            name="rag.generate",
            attributes={
                "gen_ai.prompt": "What is the effect of aspirin?",
                "gen_ai.completion": "Based on the retrieved context, aspirin reduces pain.",
                "gen_ai.system": "llama.cpp",
                "gen_ai.request.model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            },
        )
        exporter.export([span])
        exporter.shutdown()

        with open(output) as f:
            data = json.loads(f.readline())

        attrs = {a["key"]: a["value"] for a in data["attributes"]}

        assert attrs["gen_ai.prompt"]["stringValue"] == "What is the effect of aspirin?"
        assert attrs["gen_ai.completion"]["stringValue"] == "Based on the retrieved context, aspirin reduces pain."
        assert attrs["gen_ai.system"]["stringValue"] == "llama.cpp"
        assert attrs["gen_ai.request.model"]["stringValue"] == "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"

    def test_null_attribute_value_handled(self, tmp_path):
        """Null attribute values (e.g., peak_cpu_temp_c=None) handled gracefully."""
        output = tmp_path / "traces.jsonl"
        exporter = JSONLSpanExporter(str(output))

        hw = HardwareMeasurement(
            max_ram_usage_mb=512.0,
            avg_cpu_utilization_pct=50.0,
            peak_cpu_temp_c=None,  # Sensor unavailable
            swap_in_bytes=0,
            swap_out_bytes=0,
        )

        span = make_otel_span(
            name="rag.generate",
            attributes={
                "custom.hardware.peak_cpu_temp_c": hw.peak_cpu_temp_c,
            },
        )
        exporter.export([span])
        exporter.shutdown()

        with open(output) as f:
            data = json.loads(f.readline())

        # Null values should either be omitted or have explicit null representation
        attrs = {a["key"]: a["value"] for a in data["attributes"]}
        # Either not present, or has explicit null
        if "custom.hardware.peak_cpu_temp_c" in attrs:
            assert attrs["custom.hardware.peak_cpu_temp_c"].get("doubleValue") is None or \
                   "nullValue" in attrs["custom.hardware.peak_cpu_temp_c"]
