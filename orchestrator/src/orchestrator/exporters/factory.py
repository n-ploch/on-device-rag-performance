"""Factory for creating exporters based on configuration."""

from __future__ import annotations

from typing import Union

from orchestrator.config import ObservabilityConfig
from orchestrator.exporters.jsonl import JSONLSpanExporter


def create_exporter(config: ObservabilityConfig) -> Union[JSONLSpanExporter]:
    """Create an exporter based on observability config.

    Args:
        config: Observability configuration.

    Returns:
        Configured exporter instance.
    """
    # TODO: logic needs to be corrected, no single exporter, but active tracing. Relevant when doing local exporting.
    if config.langfuse:
        return True
    return JSONLSpanExporter(config.output_jsonl)
