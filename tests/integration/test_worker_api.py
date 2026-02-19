"""Tests defining Worker FastAPI endpoints."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from shared_types.schemas import (
    GenerateRequest,
    GenerateResponse,
    HardwareMeasurement,
    InferenceMeasurement,
    RetrievalData,
)
from worker.models.generator import GenerationResult


@pytest.fixture
def mock_app():
    """Create a FastAPI app with mocked services for testing."""
    app = FastAPI(title="RAG Worker Test")

    # Mock services
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = [
        {"id": "doc1", "text": "chunk1", "metadata": {}},
    ]

    mock_generation = MagicMock()
    mock_generation.generate.return_value = GenerationResult(
        text="Test answer",
        prompt_tokens=10,
        completion_tokens=5,
    )

    app.state.retrieval_service = mock_retrieval
    app.state.generation_service = mock_generation
    app.state.embedder = MagicMock()
    app.state.generator = MagicMock()

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(gen_request: GenerateRequest, request: Request) -> GenerateResponse:
        retrieval_service = request.app.state.retrieval_service
        generation_service = request.app.state.generation_service

        e2e_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        retrieved = retrieval_service.retrieve(
            gen_request.input_prompt,
            gen_request.run_config.retrieval,
        )
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        generation_start = time.perf_counter()
        gen_result = generation_service.generate(
            prompt=gen_request.input_prompt,
            retrieval_chunks=retrieved,
        )
        llm_generation_latency_ms = (time.perf_counter() - generation_start) * 1000

        e2e_latency_ms = (time.perf_counter() - e2e_start) * 1000
        duration_seconds = max(llm_generation_latency_ms / 1000.0, 1e-6)

        return GenerateResponse(
            output=gen_result.text,
            retrieval_data=RetrievalData(
                cited_doc_ids=[item["id"] for item in retrieved],
                retrieved_chunks=[item["text"] for item in retrieved],
            ),
            inference_measurement=InferenceMeasurement(
                e2e_latency_ms=e2e_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                ttft_ms=retrieval_latency_ms,
                llm_generation_latency_ms=llm_generation_latency_ms,
                prompt_tokens=gen_result.prompt_tokens,
                completion_tokens=gen_result.completion_tokens,
                tokens_per_second=gen_result.completion_tokens / duration_seconds,
            ),
            hardware_measurement=HardwareMeasurement(
                max_ram_usage_mb=512.0,
                avg_cpu_utilization_pct=50.0,
                peak_cpu_temp_c=65.0,
                swap_in_bytes=0,
                swap_out_bytes=0,
            ),
        )

    @app.get("/health")
    async def health(request: Request) -> dict:
        from worker.models.llm import detect_backend

        models_loaded = hasattr(request.app.state, "embedder") and hasattr(request.app.state, "generator")
        return {
            "status": "healthy" if models_loaded else "initializing",
            "backend": detect_backend(),
            "models_loaded": models_loaded,
        }

    return app


@pytest.fixture
def client(mock_app):
    """FastAPI test client with mocked services."""
    return TestClient(mock_app)


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

    def test_includes_models_loaded_info(self, client):
        """Health check includes models_loaded info."""
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data
        assert data["models_loaded"] is True
