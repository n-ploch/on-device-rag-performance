"""Orchestrator runner for RAG evaluation benchmarks.

This module provides the main entry point for running RAG evaluations:
- Loads configuration from YAML
- Ensures datasets, models, and collections are available
- Iterates over ground truth entries calling the worker's /generate endpoint
- Computes retrieval metrics and exports results to JSONL
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file before any other imports that might use env vars
load_dotenv()

import httpx
import pyarrow.parquet as pq
from opentelemetry import trace
from opentelemetry.propagate import inject
from pydantic import BaseModel

from orchestrator.config import EvalConfig
from orchestrator.datasets import HuggingFaceRAGBench, HuggingFaceSciFact
from orchestrator.datasets.schemas import GroundTruthEntry
from orchestrator.metrics import detect_abstention, mrr, precision_at_k, recall_at_k
from orchestrator.orchestrator import DatasetNotFoundError, Orchestrator
from orchestrator.tracing import setup_tracing, shutdown_tracing
from shared_types.schemas import GenerateRequest, GenerateResponse, RunConfig

logger = logging.getLogger(__name__)


def configure_logging(
    level: int,
    print_logs: bool = True,
    sys_logs_path: str | None = None,
) -> None:
    """Configure logging with optional file and console handlers.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        print_logs: Whether to print logs to terminal.
        sys_logs_path: Optional path to write logs to a file.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if print_logs:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    if sys_logs_path:
        # Ensure parent directory exists
        log_path = Path(sys_logs_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(sys_logs_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # If no handlers were added, add a NullHandler to prevent "No handlers" warning
    if not root_logger.handlers:
        root_logger.addHandler(logging.NullHandler())


class EvaluationResult(BaseModel):
    """Combined result for a single evaluation sample."""

    run_id: str
    claim_id: str
    ground_truth: GroundTruthEntry
    response: GenerateResponse
    recall_at_k: float | None = None
    precision_at_k: float | None = None
    mrr: float | None = None
    is_abstention: bool = False


def load_ground_truth(ground_truth_path: Path) -> list[GroundTruthEntry]:
    """Load ground truth entries from parquet file.

    Args:
        ground_truth_path: Path to the ground_truth.parquet file.

    Returns:
        List of GroundTruthEntry objects.
    """
    table = pq.read_table(ground_truth_path)
    return [GroundTruthEntry.model_validate(row) for row in table.to_pylist()]


def ensure_dataset(orchestrator: Orchestrator) -> None:
    """Ensure dataset is available, loading if necessary.

    Args:
        orchestrator: The orchestrator instance.

    Environment Variables:
        HF_TOKEN: Optional HuggingFace token for authenticated access.
    """
    try:
        orchestrator.validate_dataset()
        logger.info("Dataset already available")
    except DatasetNotFoundError:
        logger.info("Dataset not found, loading from HuggingFace...")
        token = os.environ.get("HF_TOKEN")
        name = orchestrator.config.dataset.name

        # Select loader based on dataset name
        if name == "ragbench":
            loader = HuggingFaceRAGBench(subset="emanual", token=token)
        elif name == "scifact":
            loader = HuggingFaceSciFact(token=token)
        else:
            raise ValueError(f"Unknown dataset name: {name}")

        orchestrator.load_dataset(loader)
        logger.info("Dataset loaded successfully")


async def evaluate_single(
    client: httpx.AsyncClient,
    entry: GroundTruthEntry,
    run_config: RunConfig,
    tracer: trace.Tracer,
) -> EvaluationResult:
    """Evaluate a single ground truth entry with active tracing.

    Creates a root span for the evaluation and injects trace context
    into the HTTP request for distributed tracing to the worker.

    Args:
        client: HTTP client for worker communication.
        entry: Ground truth entry to evaluate.
        run_config: Run configuration for this evaluation.
        tracer: OTEL tracer for span creation.

    Returns:
        EvaluationResult with response and computed metrics.
    """
    with tracer.start_as_current_span(
        "rag.evaluation",
        attributes={
            "run_id": run_config.run_id,
            "claim_id": entry.id,
            "gen_ai.prompt": entry.input,
            "ground_truth": entry.expected_response or "",
        },
    ) as span:
        # Prepare request with expected_response for worker tracing
        request = GenerateRequest(
            claim_id=entry.id,
            input_prompt=entry.input,
            run_config=run_config,
            expected_response=entry.expected_response,
        )

        # Inject trace context into headers for distributed tracing
        headers: dict[str, str] = {}
        inject(headers)

        response = await client.post(
            "/generate",
            json=request.model_dump(),
            headers=headers,
            timeout=httpx.Timeout(300.0),
        )
        response.raise_for_status()

        gen_response = GenerateResponse.model_validate(response.json())

        # Compute retrieval metrics
        relevant_docs = set(entry.supporting_documents)
        retrieved_docs = gen_response.retrieval_data.cited_doc_ids
        k = run_config.retrieval.k

        result_recall = None
        result_precision = None
        result_mrr = None

        if relevant_docs:
            result_recall = recall_at_k(retrieved_docs, relevant_docs, k)
            result_precision = precision_at_k(retrieved_docs, relevant_docs, k)
            result_mrr = mrr(retrieved_docs, relevant_docs)

        is_abstention = detect_abstention(gen_response.output)

        # Enrich span with response data and metrics
        span.set_attribute("gen_ai.completion", gen_response.output)

        # Build retrieval context string
        ret = gen_response.retrieval_data
        retrieval_context = "\n---\n".join(ret.retrieved_chunks) if ret.retrieved_chunks else ""
        span.set_attribute("retrieval_context", retrieval_context)

        # Metrics
        span.set_attribute("custom.metrics.abstention", is_abstention)
        if result_recall is not None:
            span.set_attribute("custom.metrics.recall_at_k", result_recall)
        if result_precision is not None:
            span.set_attribute("custom.metrics.precision_at_k", result_precision)
        if result_mrr is not None:
            span.set_attribute("custom.metrics.mrr", result_mrr)

        # Latency and hardware from inference measurement
        inf = gen_response.inference_measurement
        span.set_attribute("custom.latency.e2e_latency_ms", inf.e2e_latency_ms)

        hw = gen_response.hardware_measurement
        span.set_attribute("custom.hardware.max_ram_usage_mb", hw.max_ram_usage_mb)
        span.set_attribute("custom.hardware.avg_cpu_utilization_pct", hw.avg_cpu_utilization_pct)
        span.set_attribute("custom.hardware.swap_in_bytes", hw.swap_in_bytes)
        span.set_attribute("custom.hardware.swap_out_bytes", hw.swap_out_bytes)
        if hw.peak_cpu_temp_c is not None:
            span.set_attribute("custom.hardware.peak_cpu_temp_c", hw.peak_cpu_temp_c)

        return EvaluationResult(
            run_id=run_config.run_id,
            claim_id=entry.id,
            ground_truth=entry,
            response=gen_response,
            recall_at_k=result_recall,
            precision_at_k=result_precision,
            mrr=result_mrr,
            is_abstention=is_abstention,
        )


async def run_evaluation(
    orchestrator: Orchestrator,
    run_config: RunConfig,
    entries: list[GroundTruthEntry],
    tracer: trace.Tracer,
    show_progress: bool = True,
) -> list[EvaluationResult]:
    """Run evaluation for a single run config across all entries.

    Uses active OTEL tracing - spans are created during evaluation and
    exported automatically via the configured BatchSpanProcessor.

    Args:
        orchestrator: The orchestrator instance with HTTP client.
        run_config: Run configuration to evaluate.
        entries: List of ground truth entries.
        tracer: OTEL tracer for span creation.
        show_progress: Whether to show progress output.

    Returns:
        List of evaluation results.
    """
    if not orchestrator._client:
        raise RuntimeError("Orchestrator must be used as async context manager")

    results: list[EvaluationResult] = []
    total = len(entries)

    for i, entry in enumerate(entries, start=1):
        if show_progress:
            print(f"\r[{run_config.run_id}] {i}/{total}", end="", flush=True)

        try:
            result = await evaluate_single(
                orchestrator._client,
                entry,
                run_config,
                tracer,
            )
            results.append(result)
            # Spans are exported automatically by BatchSpanProcessor
        except httpx.HTTPStatusError as e:
            logger.error(
                "HTTP error for claim %s: %s %s",
                entry.id,
                e.response.status_code,
                e.response.text[:200],
            )
            raise
        except httpx.RequestError as e:
            logger.error("Request error for claim %s: %s", entry.id, e)
            raise

    if show_progress:
        print()

    return results


async def _run(
    config_path: Path,
    dry_run: bool = False,
    quiet: bool = False,
    run_id_filter: str | None = None,
    log_level: int = logging.INFO,
) -> int:
    """Main async entry point.

    Args:
        config_path: Path to evaluation config YAML.
        dry_run: If True, validate config and exit without evaluation.
        quiet: If True, suppress progress output.
        run_id_filter: If set, only run the specified run_id.
        log_level: Logging level to use.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    # Load configuration
    config = EvalConfig.from_yaml(config_path)

    # Reconfigure logging with settings from config
    configure_logging(
        level=log_level,
        print_logs=config.observability.print_logs,
        sys_logs_path=config.observability.sys_logs_path,
    )

    logger.info("Loaded config with %d run_configs", len(config.run_configs))

    # Filter run configs if specified
    run_configs = config.run_configs
    if run_id_filter:
        run_configs = [rc for rc in run_configs if rc.run_id == run_id_filter]
        if not run_configs:
            logger.error("No run config found with run_id=%s", run_id_filter)
            return 1

    async with Orchestrator(config) as orchestrator:
        # Ensure dataset is available
        ensure_dataset(orchestrator)

        if dry_run:
            # Just validate prerequisites
            validation = orchestrator.validate_dataset()
            entries = load_ground_truth(validation.ground_truth_path)
            logger.info(
                "Dry run complete. Would evaluate %d entries x %d configs",
                len(entries),
                len(run_configs),
            )
            return 0

        # Validate all prerequisites (models, worker, collections)
        await orchestrator.validate_prerequisites()

        # Load ground truth
        validation = orchestrator.validate_dataset()
        entries = load_ground_truth(validation.ground_truth_path)
        logger.info("Loaded %d ground truth entries", len(entries))

        # Set up active tracing (spans exported via BatchSpanProcessor)
        tracer = setup_tracing()

        try:
            for run_config in run_configs:
                logger.info("Starting evaluation: %s", run_config.run_id)

                results = await run_evaluation(
                    orchestrator,
                    run_config,
                    entries,
                    tracer,
                    show_progress=not quiet,
                )

                # Compute and print summary
                results_with_recall = [r for r in results if r.recall_at_k is not None]
                if results_with_recall:
                    avg_recall = sum(r.recall_at_k for r in results_with_recall) / len(
                        results_with_recall
                    )
                    avg_precision = sum(
                        r.precision_at_k for r in results_with_recall
                    ) / len(results_with_recall)
                    avg_mrr = sum(r.mrr for r in results_with_recall) / len(
                        results_with_recall
                    )
                else:
                    avg_recall = avg_precision = avg_mrr = 0.0

                abstention_count = sum(1 for r in results if r.is_abstention)

                logger.info(
                    "Completed %s: recall@k=%.3f, precision@k=%.3f, mrr=%.3f, "
                    "abstentions=%d/%d",
                    run_config.run_id,
                    avg_recall,
                    avg_precision,
                    avg_mrr,
                    abstention_count,
                    len(results),
                )
        finally:
            shutdown_tracing()

    return 0


def main() -> int:
    """CLI entry point for the orchestrator runner.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without running evaluation",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run only the specified run_id from config",
    )

    args = parser.parse_args()

    # Configure initial logging (will be reconfigured after loading config)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    configure_logging(level=log_level, print_logs=True, sys_logs_path=None)

    # Validate config path
    if not args.config.exists():
        logging.error("Config file not found: %s", args.config)
        return 1

    # Run async main
    return asyncio.run(
        _run(
            config_path=args.config,
            dry_run=args.dry_run,
            quiet=args.quiet,
            run_id_filter=args.run_id,
            log_level=log_level,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
