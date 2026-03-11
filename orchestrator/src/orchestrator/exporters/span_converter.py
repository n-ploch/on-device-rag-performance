"""Convert evaluation results to OTEL span format."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
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
    parent: SpanContext | None = None


def _generate_span_id(seed: str) -> int:
    """Generate a deterministic 64-bit span ID from a seed string."""
    return int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16)


def _generate_trace_id(seed: str) -> int:
    """Generate a unique 128-bit trace ID with UUID prefix for global uniqueness."""
    unique_prefix = uuid.uuid4().hex[:8]
    combined_seed = f"{unique_prefix}:{seed}"
    return int(hashlib.sha256(combined_seed.encode()).hexdigest()[:32], 16)


def _ms_to_ns(ms: float) -> int:
    """Convert milliseconds to nanoseconds."""
    return int(ms * 1_000_000)


def result_to_spans(
    run_id: str,
    claim_id: str,
    ground_truth: GroundTruthEntry,
    response: GenerateResponse,
    recall_at_k: float | None,
    precision_at_k: float | None,
    mrr: float | None,
    is_abstention: bool,
) -> list[EvaluationSpan]:
    """Convert evaluation result to OTEL spans with trace hierarchy.

    Creates a trace with the following structure:
    - rag.evaluation (root): Overall pipeline span with metrics
      - rag.retrieval: Document retrieval phase
      - rag.generation: LLM generation phase

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
        List of OTEL-compatible spans: [root, retrieval, generation].
    """
    # Generate IDs
    trace_id = _generate_trace_id(f"{run_id}:{claim_id}")
    root_span_id = _generate_span_id(f"{run_id}:{claim_id}:root")
    retrieval_span_id = _generate_span_id(f"{run_id}:{claim_id}:retrieval")
    generation_span_id = _generate_span_id(f"{run_id}:{claim_id}:generation")

    # Calculate timing
    inf = response.inference_measurement
    end_time = time.time_ns()
    start_time = end_time - _ms_to_ns(inf.e2e_latency_ms)

    # Retrieval timing: starts at beginning, ends after retrieval_latency_ms
    retrieval_end_time = start_time + _ms_to_ns(inf.retrieval_latency_ms)

    # Generation timing: starts at end - llm_generation_latency_ms, ends at end
    generation_start_time = end_time - _ms_to_ns(inf.llm_generation_latency_ms)

    root_context = SpanContext(trace_id=trace_id, span_id=root_span_id)
    ok_status = SpanStatus(status_code=StatusCode(name="OK"))

    # Hardware measurements (shared across spans)
    hw = response.hardware_measurement
    ret = response.retrieval_data

    # Build retrieval context string for evaluation
    retrieval_context = "\n---\n".join(ret.retrieved_chunks) if ret.retrieved_chunks else ""

    # === Root span: overall evaluation with metrics ===
    root_attrs: dict[str, Any] = {
        "run_id": run_id,
        "langfuse.session.id": run_id,
        "claim_id": claim_id,
        "gen_ai.prompt": ground_truth.input,
        "gen_ai.completion": response.output,
        # Evaluation attributes
        "custom.retrieval.context": retrieval_context,
        "custom.evaluation.expected_answer": ground_truth.expected_label,
        # Metrics
        "custom.metrics.abstention": is_abstention,
        # E2E latency
        "custom.latency.e2e_latency_ms": inf.e2e_latency_ms,
        # Hardware
        "custom.hardware.max_ram_usage_mb": hw.max_ram_usage_mb,
        "custom.hardware.avg_cpu_utilization_pct": hw.avg_cpu_utilization_pct,
        "custom.hardware.swap_in_bytes": hw.swap_in_bytes,
        "custom.hardware.swap_out_bytes": hw.swap_out_bytes,
    }
    # Optional evaluation attributes
    if recall_at_k is not None:
        root_attrs["custom.metrics.recall_at_k"] = recall_at_k
    if precision_at_k is not None:
        root_attrs["custom.metrics.precision_at_k"] = precision_at_k
    if mrr is not None:
        root_attrs["custom.metrics.mrr"] = mrr
    if hw.peak_cpu_temp_c is not None:
        root_attrs["custom.hardware.peak_cpu_temp_c"] = hw.peak_cpu_temp_c

    root_span = EvaluationSpan(
        context=root_context,
        name="rag.evaluation",
        start_time=start_time,
        end_time=end_time,
        status=ok_status,
        attributes=root_attrs,
        parent=None,
    )

    # === Retrieval span ===
    retrieval_attrs: dict[str, Any] = {
        "run_id": run_id,
        "claim_id": claim_id,
        "custom.latency.retrieval_latency_ms": inf.retrieval_latency_ms,
        "custom.retrieval.k": len(ret.cited_doc_ids),
        "custom.retrieval.cited_doc_ids": ",".join(ret.cited_doc_ids),
        "custom.retrieval.context": retrieval_context,
    }

    retrieval_span = EvaluationSpan(
        context=SpanContext(trace_id=trace_id, span_id=retrieval_span_id),
        name="rag.retrieval",
        start_time=start_time,
        end_time=retrieval_end_time,
        status=ok_status,
        attributes=retrieval_attrs,
        parent=root_context,
    )

    # === Generation span ===
    generation_attrs: dict[str, Any] = {
        "run_id": run_id,
        "claim_id": claim_id,
        "gen_ai.prompt": ground_truth.input,
        "gen_ai.completion": response.output,
        "custom.latency.ttft_ms": inf.ttft_ms,
        "custom.latency.llm_generation_latency_ms": inf.llm_generation_latency_ms,
        "custom.generation.prompt_tokens": inf.prompt_tokens,
        "custom.generation.completion_tokens": inf.completion_tokens,
        "custom.generation.tokens_per_second": inf.tokens_per_second,
        "custom.retrieval.context": retrieval_context,
        # Mark as generation type for Langfuse
        "langfuse.observation.type": "generation",
        "langfuse.observation.usage.input": inf.prompt_tokens,
        "langfuse.observation.usage.output": inf.completion_tokens,
    }

    if ground_truth.expected_label is not None:
        generation_attrs["custom.evaluation.expected_answer"] = ground_truth.expected_label

    generation_span = EvaluationSpan(
        context=SpanContext(trace_id=trace_id, span_id=generation_span_id),
        name="rag.generation",
        start_time=generation_start_time,
        end_time=end_time,
        status=ok_status,
        attributes=generation_attrs,
        parent=root_context,
    )

    return [root_span, retrieval_span, generation_span]
