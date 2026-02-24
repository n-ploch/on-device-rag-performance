"""Tests for the model loader module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.config import EvalConfig
from orchestrator.models.loader import (
    DownloadError,
    ModelNotFoundError,
    ModelSpec,
    ModelStatus,
    _load_registry,
    _save_registry,
    download_gguf_model,
    download_retrieval_model,
    ensure_models,
    extract_required_models,
    find_gguf_file,
    is_model_available,
)


@pytest.fixture
def sample_config_data() -> dict:
    """Sample config data for testing."""
    return {
        "dataset": {"id": "scifact", "source": "allenai/scifact"},
        "run_configs": [
            {
                "run_id": "test_run_1",
                "retrieval": {
                    "model": "intfloat/multilingual-e5-small",
                    "quantization": "fp16",
                    "dimensions": 384,
                },
                "generation": {
                    "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                    "quantization": "q4_k_m",
                },
            },
            {
                "run_id": "test_run_2",
                "retrieval": {
                    "model": "intfloat/multilingual-e5-small",
                    "quantization": "fp16",
                    "dimensions": 384,
                },
                "generation": {
                    "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                    "quantization": "q8_0",
                },
            },
        ],
    }


@pytest.fixture
def sample_config(sample_config_data) -> EvalConfig:
    """Sample EvalConfig for testing."""
    return EvalConfig.model_validate(sample_config_data)


class TestExtractRequiredModels:
    """Tests for extract_required_models function."""

    def test_extracts_unique_models(self, sample_config):
        """Should extract unique (repo, quantization) pairs."""
        models = extract_required_models(sample_config)

        assert len(models) == 3
        repos = {m.repo for m in models}
        assert "intfloat/multilingual-e5-small" in repos
        assert "TheBloke/Mistral-7B-Instruct-v0.1-GGUF" in repos

    def test_deduplicates_same_model(self, sample_config):
        """Should not duplicate models used in multiple runs."""
        models = extract_required_models(sample_config)

        retrieval_models = [m for m in models if m.model_type == "retrieval"]
        assert len(retrieval_models) == 1

    def test_assigns_correct_model_type(self, sample_config):
        """Should assign retrieval and generation types correctly."""
        models = extract_required_models(sample_config)

        for model in models:
            if "e5-small" in model.repo:
                assert model.model_type == "retrieval"
            else:
                assert model.model_type == "generation"


class TestRegistryIO:
    """Tests for registry load/save functions."""

    def test_load_registry_creates_empty_if_missing(self, tmp_path):
        """Should return empty structure if registry doesn't exist."""
        registry = _load_registry(tmp_path)
        assert registry == {"models": {}}

    def test_load_registry_reads_existing(self, tmp_path):
        """Should load existing registry file."""
        expected = {"models": {"test/repo": {"q4": {"filename": "test.gguf"}}}}
        (tmp_path / "model_registry.json").write_text(json.dumps(expected))

        registry = _load_registry(tmp_path)
        assert registry == expected

    def test_save_registry_atomic_write(self, tmp_path):
        """Should write registry atomically via temp file."""
        registry = {"models": {"test/repo": {"q4": {"filename": "test.gguf"}}}}

        _save_registry(tmp_path, registry)

        registry_path = tmp_path / "model_registry.json"
        assert registry_path.exists()
        assert json.loads(registry_path.read_text()) == registry
        # Temp file should be cleaned up
        assert not (tmp_path / "model_registry.tmp").exists()


class TestIsModelAvailable:
    """Tests for is_model_available function."""

    def test_returns_false_if_not_in_registry(self, tmp_path):
        """Should return False if model not in registry."""
        (tmp_path / "model_registry.json").write_text('{"models": {}}')

        available, filename = is_model_available("test/repo", "q4", tmp_path)

        assert available is False
        assert filename is None

    def test_returns_false_if_quantization_missing(self, tmp_path):
        """Should return False if quantization not in registry."""
        registry = {"models": {"test/repo": {"q8": {"filename": "test.gguf"}}}}
        (tmp_path / "model_registry.json").write_text(json.dumps(registry))

        available, filename = is_model_available("test/repo", "q4", tmp_path)

        assert available is False
        assert filename is None

    def test_returns_false_if_file_missing(self, tmp_path):
        """Should return False if file doesn't exist on filesystem."""
        registry = {"models": {"test/repo": {"q4": {"filename": "test.gguf"}}}}
        (tmp_path / "model_registry.json").write_text(json.dumps(registry))

        available, filename = is_model_available("test/repo", "q4", tmp_path)

        assert available is False
        assert filename is None

    def test_returns_true_if_available(self, tmp_path):
        """Should return True if model exists in registry and filesystem."""
        registry = {"models": {"test/repo": {"q4": {"filename": "test.gguf"}}}}
        (tmp_path / "model_registry.json").write_text(json.dumps(registry))

        # Create the model file
        storage_dir = tmp_path / "repo__q4"
        storage_dir.mkdir()
        (storage_dir / "test.gguf").write_text("dummy")

        available, filename = is_model_available("test/repo", "q4", tmp_path)

        assert available is True
        assert filename == "test.gguf"


class TestFindGgufFile:
    """Tests for find_gguf_file function."""

    def test_finds_matching_gguf_file(self):
        """Should find GGUF file matching quantization."""
        mock_files = [
            "README.md",
            "model.Q4_K_M.gguf",
            "model.Q8_0.gguf",
            "config.json",
        ]

        with patch(
            "orchestrator.models.loader.list_repo_files", return_value=mock_files
        ):
            result = find_gguf_file("test/repo", "q4_k_m")

        assert result == "model.Q4_K_M.gguf"

    def test_case_insensitive_match(self):
        """Should match quantization case-insensitively."""
        mock_files = ["model.q4_k_m.gguf"]

        with patch(
            "orchestrator.models.loader.list_repo_files", return_value=mock_files
        ):
            result = find_gguf_file("test/repo", "Q4_K_M")

        assert result == "model.q4_k_m.gguf"

    def test_prefers_shorter_filename_on_multiple_matches(self):
        """Should prefer shorter filename when multiple match."""
        mock_files = [
            "model.Q4_K_M.gguf",
            "model-instruct.Q4_K_M.gguf",
        ]

        with patch(
            "orchestrator.models.loader.list_repo_files", return_value=mock_files
        ):
            result = find_gguf_file("test/repo", "q4_k_m")

        assert result == "model.Q4_K_M.gguf"

    def test_raises_if_no_gguf_files(self):
        """Should raise ModelNotFoundError if no GGUF files in repo."""
        mock_files = ["README.md", "config.json"]

        with patch(
            "orchestrator.models.loader.list_repo_files", return_value=mock_files
        ):
            with pytest.raises(ModelNotFoundError, match="No GGUF files found"):
                find_gguf_file("test/repo", "q4_k_m")

    def test_raises_if_no_matching_quantization(self):
        """Should raise ModelNotFoundError if no matching quantization."""
        mock_files = ["model.Q8_0.gguf"]

        with patch(
            "orchestrator.models.loader.list_repo_files", return_value=mock_files
        ):
            with pytest.raises(ModelNotFoundError, match="No GGUF file matching"):
                find_gguf_file("test/repo", "q4_k_m")


class TestDownloadGgufModel:
    """Tests for download_gguf_model function."""

    def test_downloads_and_updates_registry(self, tmp_path):
        """Should download file and update registry."""
        spec = ModelSpec(
            repo="test/repo", quantization="q4_k_m", model_type="generation"
        )

        mock_files = ["model.Q4_K_M.gguf"]
        download_path = tmp_path / "repo__q4_k_m" / "model.Q4_K_M.gguf"

        with (
            patch(
                "orchestrator.models.loader.list_repo_files", return_value=mock_files
            ),
            patch(
                "orchestrator.models.loader.hf_hub_download",
                return_value=str(download_path),
            ),
        ):
            status = download_gguf_model(spec, tmp_path)

        assert status.downloaded is True
        assert status.filename == "model.Q4_K_M.gguf"
        assert status.repo == "test/repo"

        # Check registry was updated
        registry = json.loads((tmp_path / "model_registry.json").read_text())
        assert registry["models"]["test/repo"]["q4_k_m"]["filename"] == "model.Q4_K_M.gguf"

    def test_raises_download_error_on_failure(self, tmp_path):
        """Should raise DownloadError if download fails."""
        spec = ModelSpec(
            repo="test/repo", quantization="q4_k_m", model_type="generation"
        )

        mock_files = ["model.Q4_K_M.gguf"]

        with (
            patch(
                "orchestrator.models.loader.list_repo_files", return_value=mock_files
            ),
            patch(
                "orchestrator.models.loader.hf_hub_download",
                side_effect=Exception("Network error"),
            ),
        ):
            with pytest.raises(DownloadError, match="Failed to download"):
                download_gguf_model(spec, tmp_path)


class TestDownloadRetrievalModel:
    """Tests for download_retrieval_model function."""

    def test_downloads_gguf_and_updates_registry(self, tmp_path):
        """Should download GGUF file and update registry."""
        spec = ModelSpec(
            repo="test/embed", quantization="fp16", model_type="retrieval"
        )

        mock_files = ["embed-model.FP16.gguf", "embed-model.Q8_0.gguf"]
        download_path = tmp_path / "embed__fp16" / "embed-model.FP16.gguf"

        with (
            patch(
                "orchestrator.models.loader.list_repo_files",
                return_value=mock_files,
            ),
            patch(
                "orchestrator.models.loader.hf_hub_download",
                return_value=str(download_path),
            ) as mock_download,
        ):
            status = download_retrieval_model(spec, tmp_path)

        assert status.downloaded is True
        assert status.filename == "embed-model.FP16.gguf"
        assert status.repo == "test/embed"

        mock_download.assert_called_once()

        # Check registry was updated
        registry = json.loads((tmp_path / "model_registry.json").read_text())
        assert (
            registry["models"]["test/embed"]["fp16"]["filename"]
            == "embed-model.FP16.gguf"
        )


class TestEnsureModels:
    """Tests for ensure_models function."""

    def test_skips_existing_models(self, tmp_path, sample_config):
        """Should not download models that already exist."""
        # Set up existing model
        registry = {
            "models": {
                "intfloat/multilingual-e5-small": {
                    "fp16": {"filename": "multilingual-e5-small-fp16.gguf"}
                },
                "TheBloke/Mistral-7B-Instruct-v0.1-GGUF": {
                    "q4_k_m": {"filename": "model.Q4_K_M.gguf"},
                    "q8_0": {"filename": "model.Q8_0.gguf"},
                },
            }
        }
        (tmp_path / "model_registry.json").write_text(json.dumps(registry))

        # Create model files
        for dirname, filename in [
            ("multilingual-e5-small__fp16", "multilingual-e5-small-fp16.gguf"),
            ("Mistral-7B-Instruct-v0.1-GGUF__q4_k_m", "model.Q4_K_M.gguf"),
            ("Mistral-7B-Instruct-v0.1-GGUF__q8_0", "model.Q8_0.gguf"),
        ]:
            (tmp_path / dirname).mkdir()
            (tmp_path / dirname / filename).write_text("dummy")

        statuses = ensure_models(sample_config, tmp_path)

        assert len(statuses) == 3
        assert all(not s.downloaded for s in statuses)

    def test_downloads_missing_models(self, tmp_path, sample_config):
        """Should download models that don't exist."""
        # Empty registry - no models available
        (tmp_path / "model_registry.json").write_text('{"models": {}}')

        mock_gguf_files = [
            "multilingual-e5-small-fp16.gguf",
            "model.Q4_K_M.gguf",
            "model.Q8_0.gguf",
        ]

        with (
            patch(
                "orchestrator.models.loader.list_repo_files",
                return_value=mock_gguf_files,
            ),
            patch("orchestrator.models.loader.hf_hub_download") as mock_hf_download,
        ):
            mock_hf_download.return_value = str(tmp_path / "dummy.gguf")
            statuses = ensure_models(sample_config, tmp_path)

        assert len(statuses) == 3
        assert all(s.downloaded for s in statuses)

    def test_uses_env_var_for_models_dir(self, tmp_path, sample_config):
        """Should use LOCAL_MODELS_DIR env var if models_dir not provided."""
        (tmp_path / "model_registry.json").write_text('{"models": {}}')

        mock_gguf_files = [
            "multilingual-e5-small-fp16.gguf",
            "model.Q4_K_M.gguf",
            "model.Q8_0.gguf",
        ]

        with (
            patch.dict("os.environ", {"LOCAL_MODELS_DIR": str(tmp_path)}),
            patch(
                "orchestrator.models.loader.list_repo_files",
                return_value=mock_gguf_files,
            ),
            patch("orchestrator.models.loader.hf_hub_download") as mock_hf_download,
        ):
            mock_hf_download.return_value = str(tmp_path / "dummy.gguf")
            statuses = ensure_models(sample_config)

        assert len(statuses) == 3
