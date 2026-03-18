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

    backends: list[OTLPBackendConfig] = []
    output_jsonl: str | None = None
    sys_logs_path: str | None = None  # Path to write Python logs (None = no file)
    print_logs: bool = True  # Whether to print logs to terminal


class ServerSettings(BaseModel):
    """llama-server settings exposed in config.yaml.

    Shared hardware/performance flags are applied to both the embedding and
    generation servers. Context sizes are configured per server type because
    they have meaningfully different optimal values.
    """

    embedding_n_ctx: int = 512    # Context window for the embedding server (-c)
    generation_n_ctx: int = 2048  # Context window for the generation server (-c)
    n_gpu_layers: int = -1        # GPU layers to offload (-ngl); -1 = all, 0 = CPU-only
    n_threads: int | None = None  # CPU threads (-t); None = llama-server default
    n_batch: int | None = None    # Logical batch size (-b); None = llama-server default (512)
    flash_attn: bool = False      # Flash attention (-fa); requires compatible GPU/Metal
    tensor_split: str | None = None   # Multi-GPU split (-ts); comma-separated fractions, e.g. "3,1"
    no_kv_offload: bool = False   # Disable KV cache GPU offload (-nkvo); helps Metal with low RAM


class EvalConfig(BaseModel):
    """Complete evaluation configuration.

    This is the root config model that corresponds to config.yaml.
    """

    dataset: DatasetConfig
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    run_configs: list[RunConfig]
    server: ServerSettings = Field(default_factory=ServerSettings)  # llama-server hardware/perf settings shared across all runs

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
