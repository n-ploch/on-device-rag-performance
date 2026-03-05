"""RAG evaluation orchestrator.

The orchestrator is the main controller for running RAG evaluation benchmarks.
It handles:
- Loading and validating configuration
- Checking that datasets and models exist
- Ensuring collections are built (via worker API)
- Running evaluation loops and collecting metrics
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

from orchestrator.config import EvalConfig
from orchestrator.models.loader import ModelLoaderError, ensure_models
from shared_types.dataset_loader import DatasetLoader, get_dataset_dir
from shared_types.schemas import LoadModelsRequest, RunConfig, ServerConfig

logger = logging.getLogger(__name__)


class DatasetNotFoundError(Exception):
    """Raised when required dataset files are missing."""

    pass


class WorkerNotReadyError(Exception):
    """Raised when the worker is not ready to accept requests."""

    pass


class ModelPreparationError(Exception):
    """Raised when model preparation fails."""

    pass


@dataclass
class DatasetValidation:
    """Result of dataset validation."""

    dataset_id: str
    corpus_path: Path
    ground_truth_path: Path
    corpus_exists: bool
    ground_truth_exists: bool

    @property
    def is_valid(self) -> bool:
        """Check if all required files exist."""
        return self.corpus_exists and self.ground_truth_exists


@dataclass
class WorkerHealth:
    """Worker health status."""

    status: str
    backend: str
    models_loaded: bool

    @property
    def is_ready(self) -> bool:
        """Check if worker is ready to accept requests."""
        return self.status == "healthy" and self.models_loaded


@dataclass
class CollectionStatus:
    """Status of a collection for a run config."""

    run_id: str
    exists: bool
    populated: bool
    chunk_count: int | None = None


class Orchestrator:
    """Main orchestrator for RAG evaluation.

    The orchestrator coordinates the evaluation process:
    1. Validates that required datasets exist locally
    2. Checks that the worker is healthy and models are loaded
    3. Ensures collections are built for each run config
    4. Runs evaluation and collects metrics (future)
    """

    def __init__(
        self,
        config: EvalConfig,
        worker_url: str | None = None,
        datasets_dir: Path | None = None,
        models_dir: Path | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            config: Evaluation configuration.
            worker_url: URL of the worker service. Falls back to WORKER_URL
                environment variable or http://localhost:8000.
            datasets_dir: Base directory for datasets. Falls back to
                LOCAL_DATASETS_DIR environment variable.
            models_dir: Base directory for models. Falls back to
                LOCAL_MODELS_DIR environment variable.
        """
        self.config = config
        self.worker_url = worker_url or os.environ.get(
            "WORKER_URL", "http://localhost:8000"
        )
        self.datasets_dir = (
            Path(datasets_dir)
            if datasets_dir
            else Path(os.environ.get("LOCAL_DATASETS_DIR", "./local/datasets"))
        )
        self.models_dir = (
            Path(models_dir)
            if models_dir
            else Path(os.environ.get("LOCAL_MODELS_DIR", "./local/models"))
        )
        self._client: httpx.AsyncClient | None = None

        logger.info("Orchestrator initialized: worker_url=%s", self.worker_url)

    async def __aenter__(self) -> "Orchestrator":
        """Enter async context and create HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.worker_url,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context and close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def validate_dataset(self) -> DatasetValidation:
        """Check that required dataset files exist locally.

        Returns:
            DatasetValidation with paths and existence status.

        Raises:
            DatasetNotFoundError: If required files are missing.
        """
        dataset_id = self.config.dataset.id
        dataset_dir = get_dataset_dir(dataset_id, self.datasets_dir)

        corpus_path = dataset_dir / "corpus.parquet"
        ground_truth_path = dataset_dir / "ground_truth.parquet"

        validation = DatasetValidation(
            dataset_id=dataset_id,
            corpus_path=corpus_path,
            ground_truth_path=ground_truth_path,
            corpus_exists=corpus_path.exists(),
            ground_truth_exists=ground_truth_path.exists(),
        )

        if not validation.is_valid:
            missing = []
            if not validation.corpus_exists:
                missing.append(f"corpus.parquet (expected at {corpus_path})")
            if not validation.ground_truth_exists:
                missing.append(f"ground_truth.parquet (expected at {ground_truth_path})")

            raise DatasetNotFoundError(
                f"Dataset '{dataset_id}' is missing required files: {', '.join(missing)}. "
                f"Run the dataset export first to create these files."
            )

        logger.info(
            "Dataset '%s' validated: corpus=%s, ground_truth=%s",
            dataset_id,
            corpus_path,
            ground_truth_path,
        )
        return validation

    def load_dataset(self, loader: DatasetLoader) -> dict[str, Path]:
        """Load a dataset using the provided loader.

        Exports corpus, ground truth, and metadata to the local dataset folder
        using the limits specified in the configuration.

        Args:
            loader: A DatasetLoader instance to use for loading the dataset.

        Returns:
            Dictionary mapping artifact names to their file paths:
            - "corpus": Path to corpus.parquet
            - "ground_truth": Path to ground_truth.parquet
            - "metadata": Path to metadata.json
        """
        dataset_id = self.config.dataset.id
        output_dir = self.datasets_dir / dataset_id

        corpus_limit = self.config.dataset.limits.corpus
        ground_truth_limit = self.config.dataset.limits.ground_truth

        logger.info(
            "Loading dataset '%s' to %s (corpus_limit=%s, ground_truth_limit=%s)",
            dataset_id,
            output_dir,
            corpus_limit,
            ground_truth_limit,
        )

        paths = loader.export_all(
            output_dir=output_dir,
            corpus_limit=corpus_limit,
            ground_truth_limit=ground_truth_limit,
        )

        logger.info(
            "Dataset '%s' loaded successfully: corpus=%s, ground_truth=%s, metadata=%s",
            dataset_id,
            paths["corpus"],
            paths["ground_truth"],
            paths["metadata"],
        )

        return paths

    async def check_worker_health(self, require_models: bool = False) -> WorkerHealth:
        """Check that the worker is healthy.

        Args:
            require_models: If True, also check that models are loaded.

        Returns:
            WorkerHealth with status information.

        Raises:
            WorkerNotReadyError: If worker is not ready (or models not loaded when required).
            httpx.HTTPError: If connection fails.
        """
        if not self._client:
            raise RuntimeError("Orchestrator must be used as async context manager")

        response = await self._client.get("/health")
        response.raise_for_status()

        data = response.json()
        health = WorkerHealth(
            status=data.get("status", "unknown"),
            backend=data.get("backend", "unknown"),
            models_loaded=data.get("models_loaded", False),
        )

        if health.status != "healthy":
            raise WorkerNotReadyError(f"Worker is not healthy: status={health.status}")

        if require_models and not health.models_loaded:
            raise WorkerNotReadyError("Worker models not loaded. Call load_worker_models first.")

        logger.info(
            "Worker health check passed: status=%s, backend=%s, models_loaded=%s",
            health.status,
            health.backend,
            health.models_loaded,
        )
        return health

    async def load_worker_models(self, run_config: RunConfig) -> None:
        """Load models on the worker based on RunConfig.

        Calls the worker's /load_models endpoint to load the embedder and
        generator models specified in the run configuration.

        Args:
            run_config: The run configuration specifying which models to load.

        Raises:
            RuntimeError: If not in async context.
            httpx.HTTPError: If worker communication fails.
        """
        if not self._client:
            raise RuntimeError("Orchestrator must be used as async context manager")

        request = LoadModelsRequest(
            embedder_repo=run_config.retrieval.model,
            embedder_quantization=run_config.retrieval.quantization,
            generator_repo=run_config.generation.model,
            generator_quantization=run_config.generation.quantization,
            embedder_config=ServerConfig(
                n_ctx=512,
                parallel_slots=1
            ),
            generator_config=ServerConfig(
                n_ctx=2048,
                parallel_slots=1
            ),
        )

        logger.info(
            "Loading models on worker: embedder=%s/%s, generator=%s/%s",
            request.embedder_repo,
            request.embedder_quantization,
            request.generator_repo,
            request.generator_quantization,
        )

        response = await self._client.post(
            "/load_models",
            json=request.model_dump(),
            timeout=httpx.Timeout(300.0),  # Model loading can take time
        )
        response.raise_for_status()

        data = response.json()
        logger.info("Worker models loaded: %s", data.get("message", "success"))

    async def check_collection_status(
        self,
        run_config: RunConfig,
    ) -> CollectionStatus:
        """Check if a collection exists for a run config.

        Args:
            run_config: The run configuration to check.

        Returns:
            CollectionStatus with existence and population info.
        """
        if not self._client:
            raise RuntimeError("Orchestrator must be used as async context manager")

        response = await self._client.post(
            "/collection/status",
            json={
                "dataset_id": self.config.dataset.id,
                "retrieval_config": run_config.retrieval.model_dump(),
            },
        )
        response.raise_for_status()

        data = response.json()
        status = CollectionStatus(
            run_id=run_config.run_id,
            exists=data.get("exists", False),
            populated=data.get("populated", False),
            chunk_count=data.get("chunk_count"),
        )

        logger.debug(
            "Collection status for %s: exists=%s, populated=%s, chunks=%s",
            run_config.run_id,
            status.exists,
            status.populated,
            status.chunk_count,
        )
        return status

    async def build_collection(self, run_config: RunConfig) -> CollectionStatus:
        """Build a collection for a run config via worker API.

        This triggers the worker to embed the corpus and create a
        ChromaDB collection for the given retrieval configuration.

        Args:
            run_config: The run configuration to build collection for.

        Returns:
            CollectionStatus after building.
        """
        if not self._client:
            raise RuntimeError("Orchestrator must be used as async context manager")

        logger.info("Building collection for run_id=%s", run_config.run_id)

        response = await self._client.post(
            "/collection/build",
            json={
                "dataset_id": self.config.dataset.id,
                "retrieval_config": run_config.retrieval.model_dump(),
            },
            timeout=httpx.Timeout(600.0),  # Long timeout for embedding
        )
        response.raise_for_status()

        data = response.json()
        logger.info(
            "Collection built for %s: chunks=%d, already_existed=%s",
            run_config.run_id,
            data.get("chunks_embedded", 0),
            data.get("already_existed", False),
        )

        return CollectionStatus(
            run_id=run_config.run_id,
            exists=True,
            populated=True,
            chunk_count=data.get("chunks_embedded"),
        )

    async def ensure_collections(self) -> list[CollectionStatus]:
        """Ensure all collections exist for configured run configs.

        Checks each run config and builds missing collections.

        Returns:
            List of CollectionStatus for all run configs.
        """
        statuses: list[CollectionStatus] = []

        for run_config in self.config.run_configs:
            status = await self.check_collection_status(run_config)

            if not status.populated:
                status = await self.build_collection(run_config)

            statuses.append(status)

        return statuses

    def _ensure_models(self) -> None:
        """Ensure all required models are available locally.

        Downloads missing models from HuggingFace Hub.

        Raises:
            ModelPreparationError: If model preparation fails.
        """
        try:
            statuses = ensure_models(self.config, self.models_dir)
            for status in statuses:
                if status.downloaded:
                    logger.info(
                        "Downloaded model: %s (%s)", status.repo, status.quantization
                    )
                else:
                    logger.debug(
                        "Model already available: %s (%s)",
                        status.repo,
                        status.quantization,
                    )
        except ModelLoaderError as e:
            raise ModelPreparationError(f"Model preparation failed: {e}") from e

    async def validate_global_prerequisites(self) -> None:
        """Validate one-time prerequisites shared across all run configs.

        Checks that models are available locally, the dataset exists, and the
        worker process is reachable. Does NOT load worker models or build
        collections — those are run-config-specific and handled by
        prepare_run_config().

        Raises:
            ModelPreparationError: If model preparation fails.
            DatasetNotFoundError: If dataset files are missing.
            WorkerNotReadyError: If worker is not reachable.
            httpx.HTTPError: If worker communication fails.
        """
        logger.info("Validating global prerequisites...")
        self._ensure_models()
        self.validate_dataset()
        await self.check_worker_health(require_models=False)
        logger.info("Global prerequisites validated")

    async def _ensure_collection(self, run_config: RunConfig) -> CollectionStatus:
        """Ensure the collection for a single run config exists, building it if needed."""
        status = await self.check_collection_status(run_config)
        if not status.populated:
            status = await self.build_collection(run_config)
        return status

    async def prepare_run_config(self, run_config: RunConfig) -> None:
        """Prepare the worker for a specific run config.

        Loads the correct models on the worker, verifies they are ready, and
        ensures the ChromaDB collection for this config exists. Must be called
        before running evaluation for each run config.

        Args:
            run_config: The run configuration to prepare.

        Raises:
            WorkerNotReadyError: If worker is not ready after model load.
            httpx.HTTPError: If worker communication fails.
        """
        logger.info("Preparing run config: %s", run_config.run_id)
        await self.load_worker_models(run_config)
        await self.check_worker_health(require_models=True)
        await self._ensure_collection(run_config)
        logger.info("Run config ready: %s", run_config.run_id)
