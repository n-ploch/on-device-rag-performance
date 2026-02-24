"""Configuration models for the RAG evaluation orchestrator.

Loads and validates the evaluation config from YAML, providing typed access
to dataset settings, observability options, and run configurations.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
    source: str
    limits: DatasetLimits = Field(default_factory=DatasetLimits)


class ObservabilityConfig(BaseModel):
    """Observability and tracing configuration."""

    langfuse: bool = False
    output_jsonl: str = "./logs/traces.jsonl"


class PathsConfig(BaseModel):
    """Path configuration for logs and artifacts."""

    log_dir: str = "./logs/"


class EvalConfig(BaseModel):
    """Complete evaluation configuration.

    This is the root config model that corresponds to config.yaml.
    """

    dataset: DatasetConfig
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
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
