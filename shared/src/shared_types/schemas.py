"""Pydantic schemas for RAG evaluation system.

This module defines the data contracts shared between:
- Worker (edge device FastAPI service)
- Orchestrator (host machine test runner)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ChunkingConfig(BaseModel):
    """Chunking strategy and strategy-specific parameters."""

    strategy: Literal["fixed", "char_split"]
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int = Field(default=0, ge=0)
    split_sequence: str | None = None

    @model_validator(mode="after")
    def _validate_strategy_fields(self) -> "ChunkingConfig":
        if self.strategy == "fixed" and self.chunk_size is None:
            raise ValueError("chunk_size is required when strategy='fixed'")

        if self.strategy == "char_split" and not self.split_sequence:
            raise ValueError("split_sequence is required when strategy='char_split'")

        return self


class RetrievalConfig(BaseModel):
    """Retrieval model and ANN/query parameters."""

    dataset_id: str  # Dataset identifier (e.g., "scifact", "ragbench_emanual")
    model: str
    quantization: str
    dimensions: int = Field(gt=0)
    chunking: ChunkingConfig | None = None
    k: int = Field(default=3, gt=0)


class GenerationConfig(BaseModel):
    """Generation model and quantization."""

    model: str
    quantization: str = "q4_k_m"


class RunConfig(BaseModel):
    """Single benchmark run configuration."""

    run_id: str
    retrieval: RetrievalConfig
    generation: GenerationConfig


class GenerateRequest(BaseModel):
    """Worker request payload."""

    claim_id: str
    input_prompt: str
    run_config: RunConfig


class RetrievalData(BaseModel):
    """Retrieved context returned by worker."""

    cited_doc_ids: list[str]
    retrieved_chunks: list[str]

    @model_validator(mode="after")
    def _validate_lengths(self) -> "RetrievalData":
        if len(self.cited_doc_ids) != len(self.retrieved_chunks):
            raise ValueError("cited_doc_ids and retrieved_chunks must have equal length")
        return self


class InferenceMeasurement(BaseModel):
    """Inference latency and token statistics."""

    e2e_latency_ms: float = Field(ge=0)
    retrieval_latency_ms: float = Field(ge=0)
    ttft_ms: float = Field(ge=0)
    llm_generation_latency_ms: float = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    tokens_per_second: float = Field(ge=0)


class HardwareMeasurement(BaseModel):
    """Hardware usage metrics observed during inference."""

    max_ram_usage_mb: float = Field(ge=0)
    avg_cpu_utilization_pct: float = Field(ge=0)
    peak_cpu_temp_c: float | None = None
    swap_in_bytes: int = Field(ge=0)
    swap_out_bytes: int = Field(ge=0)


class GenerateResponse(BaseModel):
    """Worker response payload."""

    output: str
    retrieval_data: RetrievalData
    inference_measurement: InferenceMeasurement
    hardware_measurement: HardwareMeasurement


class CorpusDocument(BaseModel):
    """Generic corpus document for embedding.

    This is the dataset-agnostic interface used by the worker's EmbeddingService.
    Any dataset loader must transform its documents into this format.
    """

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# Collection management schemas (orchestrator <-> worker)


class CollectionStatusRequest(BaseModel):
    """Request to check collection status.

    The dataset_id is included in retrieval_config.dataset_id.
    """

    retrieval_config: RetrievalConfig


class CollectionStatusResponse(BaseModel):
    """Response with collection status."""

    exists: bool
    populated: bool
    chunk_count: int | None = None
    collection_name: str | None = None


class CollectionBuildRequest(BaseModel):
    """Request to build a collection.

    The dataset_id is included in retrieval_config.dataset_id.
    """

    retrieval_config: RetrievalConfig


class CollectionBuildResponse(BaseModel):
    """Response after building a collection."""

    collection_name: str
    chunks_embedded: int
    already_existed: bool


# Model management schemas (orchestrator <-> worker)


class ServerConfig(BaseModel):
    """Optional server configuration for llama-server instances."""

    n_ctx: int | None = None  # Context size (embedding: 512, generation: 2048)
    n_gpu_layers: int = -1  # GPU layers to offload (-1 = all)
    parallel_slots: int = 4  # Parallel request slots (generation only)
    pooling: str = "mean"  # Pooling strategy (embedding only): mean, cls, last


class LoadModelsRequest(BaseModel):
    """Request to load models on worker.

    Models are loaded by starting llama-server processes. The server_config
    allows customizing server parameters if needed.
    """

    embedder_repo: str
    embedder_quantization: str
    generator_repo: str
    generator_quantization: str
    embedder_config: ServerConfig | None = None  # Optional embedding server config
    generator_config: ServerConfig | None = None  # Optional generation server config


class LoadModelsResponse(BaseModel):
    """Response after loading models."""

    embedder: str
    generator: str
    message: str


class ServerMetrics(BaseModel):
    """Metrics from a llama-server instance."""

    prompt_tokens_total: int = 0
    tokens_predicted_total: int = 0
    prompt_seconds_total: float = 0.0
    tokens_predicted_seconds_total: float = 0.0
    n_decode_total: int = 0
    requests_processing: int = 0
