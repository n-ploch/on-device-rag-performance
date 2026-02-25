"""Factory for creating exporters based on configuration."""

from __future__ import annotations

from orchestrator.config import ObservabilityConfig
from orchestrator.exporters.jsonl import JSONLSpanExporter


def create_exporter(config: ObservabilityConfig) -> JSONLSpanExporter:
    """Create an exporter based on observability config.

    Args:
        config: Observability configuration.

    Returns:
        Configured exporter instance.
    """
    # TODO: Return LangfuseExporter when config.langfuse is True
    return JSONLSpanExporter(config.output_jsonl)
