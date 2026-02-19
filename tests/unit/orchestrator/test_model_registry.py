"""Tests defining orchestrator and worker model naming parity."""

from __future__ import annotations

import json
from unittest.mock import patch


class TestModelRegistryParity:
    """Orchestrator and worker should resolve identical model keys/paths."""

    def test_worker_and_orchestrator_use_same_model_storage_key(self, tmp_path):
        registry = {
            "models": {
                "TheBloke/Mistral-7B-Instruct-v0.1-GGUF": {
                    "q4_k_m": {"filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf"}
                }
            }
        }
        (tmp_path / "model_registry.json").write_text(json.dumps(registry))

        with patch.dict("os.environ", {"LOCAL_MODELS_DIR": str(tmp_path)}):
            from orchestrator.models.registry import get_model_path as orchestrator_get_model_path
            from worker.models.registry import get_model_path as worker_get_model_path

            worker_path = worker_get_model_path("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", "q4_k_m")
            orchestrator_path = orchestrator_get_model_path("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", "q4_k_m")

            assert worker_path == orchestrator_path
            assert worker_path == (
                tmp_path / "Mistral-7B-Instruct-v0.1-GGUF__q4_k_m" / "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
            )
