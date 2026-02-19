"""Tests defining Worker FastAPI endpoints."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_services():
    """Mock all worker services for testing."""
    with patch("worker.main.RetrievalService") as mock_retrieval, \
         patch("worker.main.GenerationService") as mock_generation, \
         patch("worker.main.HardwareMonitor") as mock_monitor:

        # Configure mock retrieval
        mock_retrieval.return_value.retrieve.return_value = [
            {"id": "doc1", "text": "chunk1", "score": 0.9}
        ]

        # Configure mock generation
        mock_generation.return_value.generate.return_value = "Test answer"

        # Configure mock hardware monitor
        mock_monitor_instance = AsyncMock()
        mock_monitor_instance.__aenter__.return_value = mock_monitor_instance
        mock_monitor_instance.__aexit__.return_value = None
        mock_monitor_instance.metrics = Mock(
            max_ram_usage_mb=512.0,
            avg_cpu_utilization_pct=50.0,
            peak_cpu_temp_c=65.0,
            swap_in_bytes=0,
            swap_out_bytes=0,
        )
        mock_monitor.return_value = mock_monitor_instance

        yield {
            "retrieval": mock_retrieval,
            "generation": mock_generation,
            "monitor": mock_monitor,
        }


@pytest.fixture
def client(mock_services):
    """FastAPI test client with mocked services."""
    from worker.main import create_app

    app = create_app()
    return TestClient(app)


class TestGenerateEndpoint:
    """Tests for POST /generate endpoint."""

    def test_success_response_structure(self, client):
        """Response has all required fields per spec."""
        response = client.post(
            "/generate",
            json={
                "claim_id": "c1",
                "input_prompt": "Is aspirin effective?",
                "run_config": {
                    "run_id": "test_run",
                    "retrieval": {"model": "embed", "quantization": "fp16", "dimensions": 384, "k": 3},
                    "generation": {"model": "llm"},
                },
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Verify required fields exist
        assert "output" in data
        assert "retrieval_data" in data
        assert "inference_measurement" in data
        assert "hardware_measurement" in data

        # Verify retrieval_data structure
        rd = data["retrieval_data"]
        assert "cited_doc_ids" in rd
        assert "retrieved_chunks" in rd

        # Verify inference_measurement structure
        im = data["inference_measurement"]
        assert "e2e_latency_ms" in im
        assert "retrieval_latency_ms" in im
        assert "ttft_ms" in im
        assert "llm_generation_latency_ms" in im
        assert "prompt_tokens" in im
        assert "completion_tokens" in im
        assert "tokens_per_second" in im

        # Verify hardware_measurement structure
        hm = data["hardware_measurement"]
        assert "max_ram_usage_mb" in hm
        assert "avg_cpu_utilization_pct" in hm
        assert "peak_cpu_temp_c" in hm
        assert "swap_in_bytes" in hm
        assert "swap_out_bytes" in hm

    def test_validation_error(self, client):
        """Invalid request returns 422."""
        response = client.post("/generate", json={})
        assert response.status_code == 422

    def test_missing_claim_id(self, client):
        """Missing claim_id returns 422."""
        response = client.post(
            "/generate",
            json={
                "input_prompt": "test",
                "run_config": {
                    "run_id": "r1",
                    "retrieval": {"model": "m", "quantization": "fp16", "dimensions": 384},
                    "generation": {"model": "m"},
                },
            },
        )
        assert response.status_code == 422

    def test_missing_input_prompt(self, client):
        """Missing input_prompt returns 422."""
        response = client.post(
            "/generate",
            json={
                "claim_id": "c1",
                "run_config": {
                    "run_id": "r1",
                    "retrieval": {"model": "m", "quantization": "fp16", "dimensions": 384},
                    "generation": {"model": "m"},
                },
            },
        )
        assert response.status_code == 422

    def test_request_with_fixed_chunking(self, client):
        """Request with fixed chunking strategy succeeds."""
        response = client.post(
            "/generate",
            json={
                "claim_id": "c1",
                "input_prompt": "Is aspirin effective?",
                "run_config": {
                    "run_id": "mistral_q4_baseline_001",
                    "retrieval": {
                        "model": "intfloat/multilingual-e5-small",
                        "quantization": "fp16",
                        "dimensions": 384,
                        "chunking": {
                            "strategy": "fixed",
                            "chunk_size": 500,
                            "chunk_overlap": 64,
                        },
                        "k": 3,
                    },
                    "generation": {
                        "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                        "quantization": "q4_k_m",
                    },
                },
            },
        )
        assert response.status_code == 200

    def test_request_with_char_split_chunking(self, client):
        """Request with char_split chunking strategy succeeds."""
        response = client.post(
            "/generate",
            json={
                "claim_id": "c2",
                "input_prompt": "Does exercise help?",
                "run_config": {
                    "run_id": "mistral_q4_k5_001",
                    "retrieval": {
                        "model": "intfloat/multilingual-e5-small",
                        "quantization": "fp16",
                        "dimensions": 384,
                        "chunking": {
                            "strategy": "char_split",
                            "split_sequence": ". ",
                        },
                        "k": 5,
                    },
                    "generation": {
                        "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                        "quantization": "q4_k_m",
                    },
                },
            },
        )
        assert response.status_code == 200


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_returns_healthy(self, client):
        """Health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_includes_backend_info(self, client):
        """Health check includes compute backend info."""
        response = client.get("/health")
        data = response.json()
        assert "backend" in data
        assert data["backend"] in ["mps", "cuda", "cpu"]

    def test_includes_model_info(self, client):
        """Health check includes loaded model info."""
        response = client.get("/health")
        data = response.json()
        assert "models" in data
