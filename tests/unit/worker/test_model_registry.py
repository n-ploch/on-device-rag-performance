"""Tests defining model registry resolution."""

import json
from unittest.mock import patch

import pytest


class TestModelRegistry:
    """Tests for get_model_path() function."""

    def test_resolve_model_path(self, tmp_path):
        """Resolves HF repo + quantization to local path."""
        # Create mock registry
        registry = {
            "models": {
                "TheBloke/Mistral-7B-Instruct-v0.1-GGUF": {
                    "q4_k_m": {
                        "filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                        "size_bytes": 4368439296,
                    }
                }
            }
        }
        (tmp_path / "model_registry.json").write_text(json.dumps(registry))

        with patch.dict("os.environ", {"LOCAL_MODELS_DIR": str(tmp_path)}):
            from worker.models.registry import get_model_path

            path = get_model_path(
                repo="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                quantization="q4_k_m",
            )
            assert path == tmp_path / "Mistral-7B-Instruct-v0.1-GGUF__q4_k_m" / "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

    def test_missing_model_raises(self, tmp_path):
        """Unknown model/quantization raises KeyError."""
        registry = {"models": {}}
        (tmp_path / "model_registry.json").write_text(json.dumps(registry))

        with patch.dict("os.environ", {"LOCAL_MODELS_DIR": str(tmp_path)}):
            from worker.models.registry import get_model_path

            with pytest.raises(KeyError):
                get_model_path("unknown/model", "q4_k_m")

    def test_missing_quantization_raises(self, tmp_path):
        """Known model but unknown quantization raises KeyError."""
        registry = {
            "models": {
                "TheBloke/Mistral-7B-Instruct-v0.1-GGUF": {
                    "q4_k_m": {"filename": "test.gguf"}
                }
            }
        }
        (tmp_path / "model_registry.json").write_text(json.dumps(registry))

        with patch.dict("os.environ", {"LOCAL_MODELS_DIR": str(tmp_path)}):
            from worker.models.registry import get_model_path

            with pytest.raises(KeyError):
                get_model_path("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", "q8_0")

    def test_missing_env_var_raises(self):
        """Missing LOCAL_MODELS_DIR raises KeyError."""
        with patch.dict("os.environ", {}, clear=True):
            from worker.models.registry import get_model_path

            with pytest.raises(KeyError):
                get_model_path("any/model", "q4_k_m")

    def test_multiple_models_in_registry(self, tmp_path):
        """Can resolve multiple models from same registry."""
        registry = {
            "models": {
                "model-a": {"q4": {"filename": "a.gguf"}},
                "model-b": {"q4": {"filename": "b.gguf"}},
            }
        }
        (tmp_path / "model_registry.json").write_text(json.dumps(registry))

        with patch.dict("os.environ", {"LOCAL_MODELS_DIR": str(tmp_path)}):
            from worker.models.registry import get_model_path

            path_a = get_model_path("model-a", "q4")
            path_b = get_model_path("model-b", "q4")

            assert path_a == tmp_path / "model-a__q4" / "a.gguf"
            assert path_b == tmp_path / "model-b__q4" / "b.gguf"
