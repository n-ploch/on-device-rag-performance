"""Configuration models for the RAG evaluation orchestrator.

Loads and validates the evaluation config from YAML, providing typed access
to dataset settings, observability options, and run configurations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from shared_types.schemas import RunConfig

logger = logging.getLogger(__name__)


class DatasetLimits(BaseModel):
    """Limits for corpus and ground truth export."""

    corpus: int | None = None
    ground_truth: int | None = None


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    id: str
    name: str
    limits: DatasetLimits = Field(default_factory=DatasetLimits)


class OTLPBackendConfig(BaseModel):
    """Configuration for a single OTEL export backend.

    Credentials are always read from environment variables at runtime.
    Supported types and their required env vars:
      langfuse: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
      weave:    WANDB_API_KEY, WANDB_BASE_URL (+ optional WANDB_ENTITY, WANDB_PROJECT)
      generic:  OTEL_EXPORTER_OTLP_ENDPOINT (+ optional OTEL_EXPORTER_OTLP_HEADERS)
    """

    type: Literal["langfuse", "weave", "generic"]
    enabled: bool = True


class ObservabilityConfig(BaseModel):
    """Observability and tracing configuration."""

    langfuse: bool = False  # Legacy single-backend flag; ignored when backends is non-empty
    backends: list[OTLPBackendConfig] = []  # Multi-backend list; takes precedence when non-empty
    output_jsonl: str = "./logs/traces.jsonl"
    sys_logs_path: str | None = None  # Path to write Python logs (None = no file)
    print_logs: bool = True  # Whether to print logs to terminal


class EvalConfig(BaseModel):
    """Complete evaluation configuration.

    This is the root config model that corresponds to config.yaml.
    """

    dataset: DatasetConfig
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    run_configs: list[RunConfig]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "EvalConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated EvalConfig instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If the YAML is malformed.
            pydantic.ValidationError: If the config doesn't match the schema.
        """
        path = Path(path)
        logger.info("Loading config from %s", path)

        with path.open() as f:
            data = yaml.safe_load(f)

        config = cls.model_validate(data)
        logger.info(
            "Loaded config: dataset=%s, %d run_configs",
            config.dataset.id,
            len(config.run_configs),
        )
        return config
