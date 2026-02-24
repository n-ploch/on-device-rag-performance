"""Model loader for downloading missing models from HuggingFace Hub.

This module provides functionality to:
- Extract required models from evaluation configuration
- Check if models exist locally via the registry and filesystem
- Download missing models from HuggingFace Hub
- Update the model registry after successful downloads
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

from orchestrator.config import EvalConfig
from shared_types.naming import model_storage_key

logger = logging.getLogger(__name__)


class ModelLoaderError(Exception):
    """Base exception for model loading errors."""

    pass


class ModelNotFoundError(ModelLoaderError):
    """Model or quantization not found in HuggingFace repo."""

    pass


class DownloadError(ModelLoaderError):
    """Failed to download model file."""

    pass


@dataclass
class ModelSpec:
    """Specification for a model to load."""

    repo: str
    quantization: str
    model_type: Literal["generation", "retrieval"]


@dataclass
class ModelStatus:
    """Status after ensuring a model is available."""

    repo: str
    quantization: str
    path: Path
    downloaded: bool
    filename: str


def extract_required_models(config: EvalConfig) -> list[ModelSpec]:
    """Extract unique model specifications from config.

    Deduplicates models that appear in multiple run configs.

    Args:
        config: Evaluation configuration containing run_configs.

    Returns:
        List of unique ModelSpec instances.
    """
    seen: set[tuple[str, str]] = set()
    models: list[ModelSpec] = []

    for run_config in config.run_configs:
        # Retrieval model
        retrieval_key = (run_config.retrieval.model, run_config.retrieval.quantization)
        if retrieval_key not in seen:
            seen.add(retrieval_key)
            models.append(
                ModelSpec(
                    repo=run_config.retrieval.model,
                    quantization=run_config.retrieval.quantization,
                    model_type="retrieval",
                )
            )

        # Generation model
        generation_key = (
            run_config.generation.model,
            run_config.generation.quantization,
        )
        if generation_key not in seen:
            seen.add(generation_key)
            models.append(
                ModelSpec(
                    repo=run_config.generation.model,
                    quantization=run_config.generation.quantization,
                    model_type="generation",
                )
            )

    return models


def _load_registry(models_dir: Path) -> dict:
    """Load model registry, returning empty structure if not exists."""
    registry_path = models_dir / "model_registry.json"
    if registry_path.exists():
        return json.loads(registry_path.read_text())
    return {"models": {}}


def _save_registry(models_dir: Path, registry: dict) -> None:
    """Save registry atomically via temp file + rename."""
    registry_path = models_dir / "model_registry.json"
    tmp_path = registry_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(registry, indent=2, sort_keys=True))
    tmp_path.rename(registry_path)


def is_model_available(
    repo: str, quantization: str, models_dir: Path
) -> tuple[bool, str | None]:
    """Check if model exists in registry and filesystem.

    Args:
        repo: HuggingFace repository identifier.
        quantization: Quantization level.
        models_dir: Directory containing models and registry.

    Returns:
        Tuple of (is_available, filename_if_available).
    """
    registry = _load_registry(models_dir)

    # Check registry
    if repo not in registry.get("models", {}):
        return False, None
    if quantization not in registry["models"][repo]:
        return False, None

    filename = registry["models"][repo][quantization].get("filename")
    if not filename:
        return False, None

    # Check filesystem
    storage_dir = models_dir / model_storage_key(repo, quantization)
    model_path = storage_dir / filename

    if not model_path.exists():
        return False, None

    return True, filename


def find_gguf_file(repo: str, quantization: str) -> str:
    """Find GGUF filename matching quantization in HuggingFace repo.

    Args:
        repo: HuggingFace repository identifier.
        quantization: Quantization level (e.g., "q4_k_m").

    Returns:
        Filename of the matching GGUF file.

    Raises:
        ModelNotFoundError: If no matching GGUF file is found.
    """
    try:
        files = list_repo_files(repo)
    except Exception as e:
        raise ModelNotFoundError(f"Cannot list files in repo '{repo}': {e}") from e

    gguf_files = [f for f in files if f.endswith(".gguf")]

    if not gguf_files:
        raise ModelNotFoundError(f"No GGUF files found in repo '{repo}'")

    # Match quantization (case-insensitive)
    quant_pattern = quantization.upper()
    matches = [f for f in gguf_files if quant_pattern in f.upper()]

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Prefer shorter filenames (more likely to be canonical)
        return min(matches, key=len)
    else:
        # Extract available quantizations from filenames for error message
        available = []
        for f in gguf_files:
            parts = f.rsplit(".", 2)
            if len(parts) >= 2:
                available.append(parts[-2])
        raise ModelNotFoundError(
            f"No GGUF file matching quantization '{quantization}' in '{repo}'. "
            f"Available: {available}"
        )


def download_gguf_model(spec: ModelSpec, models_dir: Path) -> ModelStatus:
    """Download a GGUF model file from HuggingFace Hub.

    Args:
        spec: Model specification with repo and quantization.
        models_dir: Directory to store the model.

    Returns:
        ModelStatus with download result.

    Raises:
        DownloadError: If download fails.
    """
    logger.info("Downloading GGUF model: %s (%s)", spec.repo, spec.quantization)

    # Find the correct file
    filename = find_gguf_file(spec.repo, spec.quantization)

    # Create storage directory
    storage_dir = models_dir / model_storage_key(spec.repo, spec.quantization)
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Download file
    try:
        downloaded_path = hf_hub_download(
            repo_id=spec.repo,
            filename=filename,
            local_dir=storage_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        raise DownloadError(
            f"Failed to download '{filename}' from '{spec.repo}': {e}"
        ) from e

    # Update registry
    registry = _load_registry(models_dir)
    if spec.repo not in registry["models"]:
        registry["models"][spec.repo] = {}
    registry["models"][spec.repo][spec.quantization] = {"filename": filename}
    _save_registry(models_dir, registry)

    logger.info("Downloaded: %s -> %s", filename, downloaded_path)

    return ModelStatus(
        repo=spec.repo,
        quantization=spec.quantization,
        path=Path(downloaded_path),
        downloaded=True,
        filename=filename,
    )


def download_retrieval_model(spec: ModelSpec, models_dir: Path) -> ModelStatus:
    """Download a retrieval model (sentence transformer) from HuggingFace Hub.

    Downloads the full model repository using snapshot_download.

    Args:
        spec: Model specification with repo and quantization.
        models_dir: Directory to store the model.

    Returns:
        ModelStatus with download result.

    Raises:
        DownloadError: If download fails.
    """
    logger.info("Downloading retrieval model: %s (%s)", spec.repo, spec.quantization)

    # Create storage directory
    storage_dir = models_dir / model_storage_key(spec.repo, spec.quantization)
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Download full repository
    try:
        snapshot_download(
            repo_id=spec.repo,
            local_dir=storage_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        raise DownloadError(
            f"Failed to download retrieval model '{spec.repo}': {e}"
        ) from e

    # Use config.json as sentinel file for registry
    sentinel_file = "config.json"

    # Update registry
    registry = _load_registry(models_dir)
    if spec.repo not in registry["models"]:
        registry["models"][spec.repo] = {}
    registry["models"][spec.repo][spec.quantization] = {
        "filename": sentinel_file,
        "type": "retrieval",
    }
    _save_registry(models_dir, registry)

    logger.info("Downloaded retrieval model to: %s", storage_dir)

    return ModelStatus(
        repo=spec.repo,
        quantization=spec.quantization,
        path=storage_dir / sentinel_file,
        downloaded=True,
        filename=sentinel_file,
    )


def download_model(spec: ModelSpec, models_dir: Path) -> ModelStatus:
    """Download a model from HuggingFace Hub.

    Routes to the appropriate downloader based on model type.

    Args:
        spec: Model specification with repo, quantization, and type.
        models_dir: Directory to store the model.

    Returns:
        ModelStatus with download result.
    """
    if spec.model_type == "generation":
        return download_gguf_model(spec, models_dir)
    else:
        return download_retrieval_model(spec, models_dir)


def ensure_models(
    config: EvalConfig,
    models_dir: Path | None = None,
) -> list[ModelStatus]:
    """Ensure all models from config are available locally.

    Downloads any missing models from HuggingFace Hub.

    Args:
        config: Evaluation configuration containing run_configs.
        models_dir: Directory for model storage. Defaults to LOCAL_MODELS_DIR env var.

    Returns:
        List of ModelStatus for each unique model in config.

    Raises:
        ModelLoaderError: If any model cannot be loaded.
    """
    if models_dir is None:
        env_dir = os.environ.get("LOCAL_MODELS_DIR")
        if env_dir is None:
            raise ModelLoaderError("LOCAL_MODELS_DIR environment variable not set")
        models_dir = Path(env_dir)

    models_dir.mkdir(parents=True, exist_ok=True)

    specs = extract_required_models(config)
    statuses: list[ModelStatus] = []

    for spec in specs:
        available, filename = is_model_available(
            spec.repo, spec.quantization, models_dir
        )

        if available:
            storage_dir = models_dir / model_storage_key(spec.repo, spec.quantization)
            statuses.append(
                ModelStatus(
                    repo=spec.repo,
                    quantization=spec.quantization,
                    path=storage_dir / filename,
                    downloaded=False,
                    filename=filename,
                )
            )
            logger.debug(
                "Model already available: %s (%s)", spec.repo, spec.quantization
            )
        else:
            status = download_model(spec, models_dir)
            statuses.append(status)

    return statuses
