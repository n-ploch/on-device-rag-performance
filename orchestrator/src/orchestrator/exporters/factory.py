"""Factory for creating exporters based on configuration."""

from __future__ import annotations

from typing import Union

from orchestrator.config import ObservabilityConfig
from orchestrator.exporters.jsonl import JSONLSpanExporter
from orchestrator.exporters.langfuse_exporter import LangfuseExporter


def create_exporter(config: ObservabilityConfig) -> Union[JSONLSpanExporter, LangfuseExporter]:
    """Create an exporter based on observability config.

    Args:
        config: Observability configuration.

    Returns:
        Configured exporter instance.
    """
    if config.langfuse:
        return LangfuseExporter()
    return JSONLSpanExporter(config.output_jsonl)
