"""Convert evaluation results to OTEL span format."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from orchestrator.datasets.schemas import GroundTruthEntry
from shared_types.schemas import GenerateResponse


@dataclass
class SpanContext:
    """OTEL span context with trace and span identifiers."""

    trace_id: int
    span_id: int


@dataclass
class StatusCode:
    """OTEL status code."""

    name: str


@dataclass
class SpanStatus:
    """OTEL span status."""

    status_code: StatusCode


@dataclass
class EvaluationSpan:
    """OTEL-compatible span representation of an evaluation result."""

    context: SpanContext
    name: str
    start_time: int  # nanoseconds
    end_time: int  # nanoseconds
    status: SpanStatus
    attributes: dict[str, Any]
    parent: None = None


def result_to_span(
    run_id: str,
    claim_id: str,
    ground_truth: GroundTruthEntry,
    response: GenerateResponse,
    recall_at_k: float | None,
    precision_at_k: float | None,
    mrr: float | None,
    is_abstention: bool,
) -> EvaluationSpan:
    """Convert evaluation result fields to an OTEL span.

    Args:
        run_id: Unique identifier for this evaluation run.
        claim_id: Unique identifier for the claim being evaluated.
        ground_truth: Ground truth entry with input and expected output.
        response: Model response with output and measurements.
        recall_at_k: Recall@k metric value.
        precision_at_k: Precision@k metric value.
        mrr: Mean reciprocal rank metric value.
        is_abstention: Whether the model abstained from answering.

    Returns:
        OTEL-compatible span with all evaluation data as attributes.
    """
    # Generate trace/span IDs from run_id and claim_id
    trace_id = int(hashlib.sha256(run_id.encode()).hexdigest()[:32], 16)
    span_id = int(hashlib.sha256(f"{run_id}:{claim_id}".encode()).hexdigest()[:16], 16)

    # Use inference timing if available
    inf = response.inference_measurement
    end_time = time.time_ns()
    start_time = end_time - int(inf.e2e_latency_ms * 1_000_000)

    # Build attributes following OTEL semantic conventions
    attributes: dict[str, Any] = {
        "run_id": run_id,
        "claim_id": claim_id,
        "gen_ai.prompt": ground_truth.input,
        "gen_ai.completion": response.output,
    }

    # Metrics
    if recall_at_k is not None:
        attributes["custom.metrics.recall_at_k"] = recall_at_k
    if precision_at_k is not None:
        attributes["custom.metrics.precision_at_k"] = precision_at_k
    if mrr is not None:
        attributes["custom.metrics.mrr"] = mrr
    attributes["custom.metrics.abstention"] = is_abstention

    # Inference measurements
    attributes["custom.latency.e2e_latency_ms"] = inf.e2e_latency_ms
    attributes["custom.latency.retrieval_latency_ms"] = inf.retrieval_latency_ms
    attributes["custom.latency.ttft_ms"] = inf.ttft_ms
    attributes["custom.latency.llm_generation_latency_ms"] = inf.llm_generation_latency_ms
    attributes["custom.generation.prompt_tokens"] = inf.prompt_tokens
    attributes["custom.generation.completion_tokens"] = inf.completion_tokens
    attributes["custom.generation.tokens_per_second"] = inf.tokens_per_second

    # Hardware measurements
    hw = response.hardware_measurement
    attributes["custom.hardware.max_ram_usage_mb"] = hw.max_ram_usage_mb
    attributes["custom.hardware.avg_cpu_utilization_pct"] = hw.avg_cpu_utilization_pct
    if hw.peak_cpu_temp_c is not None:
        attributes["custom.hardware.peak_cpu_temp_c"] = hw.peak_cpu_temp_c
    attributes["custom.hardware.swap_in_bytes"] = hw.swap_in_bytes
    attributes["custom.hardware.swap_out_bytes"] = hw.swap_out_bytes

    # Retrieval data
    ret = response.retrieval_data
    attributes["custom.retrieval.k"] = len(ret.cited_doc_ids)
    attributes["custom.retrieval.cited_doc_ids"] = ",".join(ret.cited_doc_ids)

    return EvaluationSpan(
        context=SpanContext(trace_id=trace_id, span_id=span_id),
        name="rag.evaluation",
        start_time=start_time,
        end_time=end_time,
        status=SpanStatus(status_code=StatusCode(name="OK")),
        attributes=attributes,
    )
