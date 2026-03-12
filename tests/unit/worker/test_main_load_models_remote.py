"""Tests for /load_models and /generate endpoints with remote generation hosting."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute

from shared_types.schemas import (
    ChunkingConfig,
    GenerationConfig,
    HardwareMeasurement,
    LoadModelsRequest,
    RemoteGenerationConfig,
    RetrievalConfig,
    RunConfig,
    GenerateRequest,
)
from worker.main import create_app
from worker.models.generator_remote import RemoteGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_handler(path: str):
    """Extract a route handler from the worker app by path."""
    app = create_app()
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == path:
            return route.endpoint
    raise AssertionError(f"Route {path!r} not found")


def _remote_load_request(
    api_key_env: str = "TEST_API_KEY",
    base_url: str = "https://api.example.com/v1",
    model: str = "mistral-large-latest",
    rate_limit_rps: float | None = None,
    extra_headers: dict | None = None,
) -> LoadModelsRequest:
    return LoadModelsRequest(
        embedder_repo="ChristianAzinn/e5-large-v2-gguf",
        embedder_quantization="q4_k_m",
        generator_repo=model,  # ignored for remote but kept for compat
        generator_quantization="q4_k_m",  # ignored for remote
        generation_config=GenerationConfig(
            model=model,
            hosting="remote",
            remote=RemoteGenerationConfig(
                base_url=base_url,
                api_key_env=api_key_env,
                extra_headers=extra_headers or {},
                rate_limit_rps=rate_limit_rps,
            ),
        ),
    )


def _make_req_state(**overrides) -> SimpleNamespace:
    """Create a mock FastAPI request with app.state pre-populated."""
    state = SimpleNamespace(
        server_manager=MagicMock(),
        embedder=None,
        generator=None,
        loaded_models=None,
        generator_hosting=None,
        collections_dir=MagicMock(),
        retrieval_service=None,
        embedding_service=None,
        generation_service=None,
    )
    for k, v in overrides.items():
        setattr(state, k, v)
    return SimpleNamespace(app=SimpleNamespace(state=state))


# ---------------------------------------------------------------------------
# /load_models — remote branch
# ---------------------------------------------------------------------------


class TestLoadModelsRemote:
    @pytest.mark.asyncio
    async def test_creates_remote_generator(self):
        """RemoteGenerator must be placed in app.state.generator when hosting=remote."""
        handler = _get_handler("/load_models")
        request = _remote_load_request()
        req = _make_req_state()

        server_manager = req.app.state.server_manager
        server_manager.start_embedding_server = AsyncMock(return_value=True)
        server_manager.embedding_url = "http://localhost:8001"

        embedder_path = MagicMock()

        with (
            patch.dict("os.environ", {"TEST_API_KEY": "sk-test"}),
            patch("worker.main.get_model_path", return_value=embedder_path),
            patch("worker.main.LlamaServerEmbedder"),
            patch("worker.main.RetrievalService"),
            patch("worker.main.EmbeddingService"),
            patch("worker.main.GenerationService"),
        ):
            await handler(request, req)

        assert isinstance(req.app.state.generator, RemoteGenerator)
        assert req.app.state.generator_hosting == "remote"

    @pytest.mark.asyncio
    async def test_generation_server_not_started_for_remote(self):
        """start_generation_server must NOT be called when hosting=remote."""
        handler = _get_handler("/load_models")
        request = _remote_load_request()
        req = _make_req_state()

        server_manager = req.app.state.server_manager
        server_manager.start_embedding_server = AsyncMock(return_value=True)
        server_manager.start_generation_server = AsyncMock(return_value=True)
        server_manager.embedding_url = "http://localhost:8001"

        with (
            patch.dict("os.environ", {"TEST_API_KEY": "sk-test"}),
            patch("worker.main.get_model_path", return_value=MagicMock()),
            patch("worker.main.LlamaServerEmbedder"),
            patch("worker.main.RetrievalService"),
            patch("worker.main.EmbeddingService"),
            patch("worker.main.GenerationService"),
        ):
            await handler(request, req)

        server_manager.start_generation_server.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_500(self):
        """Missing API key env var must raise HTTPException 500."""
        handler = _get_handler("/load_models")
        request = _remote_load_request(api_key_env="NONEXISTENT_KEY_XYZ")
        req = _make_req_state()

        server_manager = req.app.state.server_manager
        server_manager.start_embedding_server = AsyncMock(return_value=True)
        server_manager.stop_embedding_server = AsyncMock()
        server_manager.embedding_url = "http://localhost:8001"

        with (
            patch.dict("os.environ", {}, clear=True),  # no env vars
            patch("worker.main.get_model_path", return_value=MagicMock()),
            patch("worker.main.LlamaServerEmbedder"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await handler(request, req)

        assert exc_info.value.status_code == 500
        assert "NONEXISTENT_KEY_XYZ" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_embedding_server_stopped_on_missing_api_key(self):
        """Embedding server must be stopped when API key check fails."""
        handler = _get_handler("/load_models")
        request = _remote_load_request(api_key_env="MISSING_KEY")
        req = _make_req_state()

        server_manager = req.app.state.server_manager
        server_manager.start_embedding_server = AsyncMock(return_value=True)
        server_manager.stop_embedding_server = AsyncMock()
        server_manager.embedding_url = "http://localhost:8001"

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("worker.main.get_model_path", return_value=MagicMock()),
            patch("worker.main.LlamaServerEmbedder"),
        ):
            with pytest.raises(HTTPException):
                await handler(request, req)

        server_manager.stop_embedding_server.assert_called_once()

    @pytest.mark.asyncio
    async def test_loaded_models_tracks_remote_label(self):
        """loaded_models["generator"] should use (model, "remote") tuple."""
        handler = _get_handler("/load_models")
        request = _remote_load_request(model="mistral-large-latest")
        req = _make_req_state()

        server_manager = req.app.state.server_manager
        server_manager.start_embedding_server = AsyncMock(return_value=True)
        server_manager.embedding_url = "http://localhost:8001"

        with (
            patch.dict("os.environ", {"TEST_API_KEY": "sk-test"}),
            patch("worker.main.get_model_path", return_value=MagicMock()),
            patch("worker.main.LlamaServerEmbedder"),
            patch("worker.main.RetrievalService"),
            patch("worker.main.EmbeddingService"),
            patch("worker.main.GenerationService"),
        ):
            await handler(request, req)

        assert req.app.state.loaded_models["generator"] == ("mistral-large-latest", "remote")

    @pytest.mark.asyncio
    async def test_rate_limiter_passed_to_remote_generator(self):
        """rate_limit_rps from config is forwarded to the RemoteGenerator."""
        handler = _get_handler("/load_models")
        request = _remote_load_request(rate_limit_rps=2.0)
        req = _make_req_state()

        server_manager = req.app.state.server_manager
        server_manager.start_embedding_server = AsyncMock(return_value=True)
        server_manager.embedding_url = "http://localhost:8001"

        with (
            patch.dict("os.environ", {"TEST_API_KEY": "sk-test"}),
            patch("worker.main.get_model_path", return_value=MagicMock()),
            patch("worker.main.LlamaServerEmbedder"),
            patch("worker.main.RetrievalService"),
            patch("worker.main.EmbeddingService"),
            patch("worker.main.GenerationService"),
        ):
            await handler(request, req)

        generator: RemoteGenerator = req.app.state.generator
        assert generator._rate_limiter is not None
        assert generator._rate_limiter.rps == 2.0

    @pytest.mark.asyncio
    async def test_no_rate_limiter_when_rps_is_none(self):
        """No rate limiter created when rate_limit_rps is None."""
        handler = _get_handler("/load_models")
        request = _remote_load_request(rate_limit_rps=None)
        req = _make_req_state()

        server_manager = req.app.state.server_manager
        server_manager.start_embedding_server = AsyncMock(return_value=True)
        server_manager.embedding_url = "http://localhost:8001"

        with (
            patch.dict("os.environ", {"TEST_API_KEY": "sk-test"}),
            patch("worker.main.get_model_path", return_value=MagicMock()),
            patch("worker.main.LlamaServerEmbedder"),
            patch("worker.main.RetrievalService"),
            patch("worker.main.EmbeddingService"),
            patch("worker.main.GenerationService"),
        ):
            await handler(request, req)

        generator: RemoteGenerator = req.app.state.generator
        assert generator._rate_limiter is None


# ---------------------------------------------------------------------------
# /generate — model validation skipped for remote
# ---------------------------------------------------------------------------


def _sample_run_config(hosting: str = "remote") -> RunConfig:
    gen_cfg = (
        GenerationConfig(
            model="mistral-large-latest",
            hosting="remote",
            remote=RemoteGenerationConfig(
                base_url="https://api.example.com/v1",
                api_key_env="TEST_API_KEY",
            ),
        )
        if hosting == "remote"
        else GenerationConfig(model="local/model", quantization="q4_k_m", hosting="local")
    )
    return RunConfig(
        run_id="test_run",
        retrieval=RetrievalConfig(
            dataset_id="scifact",
            model="intfloat/multilingual-e5-small",
            quantization="fp16",
            dimensions=384,
            chunking=ChunkingConfig(strategy="fixed", chunk_size=500),
            k=3,
        ),
        generation=gen_cfg,
    )


class TestGenerateRemoteValidation:
    @pytest.mark.asyncio
    async def test_generator_model_check_skipped_for_remote(self):
        """Mismatched quantization on remote config must not raise 400."""
        handler = _get_handler("/generate")

        # loaded_models says ("mistral-large-latest", "remote") but run_config
        # has quantization "q4_k_m" — this mismatch is acceptable for remote
        req = _make_req_state(
            embedder=MagicMock(),
            generator=MagicMock(),
            generator_hosting="remote",
            loaded_models={
                "embedder": ("intfloat/multilingual-e5-small", "fp16"),
                "generator": ("mistral-large-latest", "remote"),
            },
        )

        from worker.models.generator_http import GenerationResult as HttpResult
        mock_gen_result = HttpResult(
            text="answer",
            prompt_tokens=5,
            completion_tokens=3,
            prompt_ms=100.0,
            predicted_ms=500.0,
            predicted_per_token_ms=0.0,
            predicted_per_second=6.0,
        )
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve.return_value = [{"id": "d1", "text": "chunk", "metadata": {}}]
        req.app.state.retrieval_service = mock_retrieval

        mock_generation = MagicMock()
        mock_generation.generate.return_value = mock_gen_result
        req.app.state.generation_service = mock_generation

        req.headers = {}  # FastAPI Request.headers used by extract_context

        generate_request = GenerateRequest(
            claim_id="c1",
            input_prompt="test question",
            run_config=_sample_run_config(hosting="remote"),
        )

        with patch("worker.main.HardwareMonitor") as mock_hw:
            mock_hw_instance = AsyncMock()
            mock_hw_instance.metrics = HardwareMeasurement(
                max_ram_usage_mb=100.0,
                avg_cpu_utilization_pct=10.0,
                peak_cpu_temp_c=None,
                swap_in_bytes=0,
                swap_out_bytes=0,
            )
            mock_hw.return_value.__aenter__ = AsyncMock(return_value=mock_hw_instance)
            mock_hw.return_value.__aexit__ = AsyncMock(return_value=False)
            with patch("worker.main.get_tracer") as mock_tracer:
                mock_span = MagicMock()
                mock_span.__enter__ = MagicMock(return_value=mock_span)
                mock_span.__exit__ = MagicMock(return_value=False)
                mock_tracer.return_value.start_span.return_value = mock_span
                with patch("worker.main.extract_context", return_value=MagicMock()):
                    response = await handler(generate_request, req)

        assert response.output == "answer"
        assert response.inference_measurement.prompt_tokens == 5

    @pytest.mark.asyncio
    async def test_embedder_mismatch_still_raises_for_remote(self):
        """Embedder mismatch must still raise 400 even for remote generation."""
        handler = _get_handler("/generate")

        req = _make_req_state(
            embedder=MagicMock(),
            generator=MagicMock(),
            generator_hosting="remote",
            loaded_models={
                "embedder": ("wrong/embedder", "fp16"),  # mismatched
                "generator": ("mistral-large-latest", "remote"),
            },
        )

        req.headers = {}

        generate_request = GenerateRequest(
            claim_id="c1",
            input_prompt="test",
            run_config=_sample_run_config(hosting="remote"),
        )

        with pytest.raises(HTTPException) as exc_info:
            with patch("worker.main.extract_context", return_value=MagicMock()):
                await handler(generate_request, req)

        assert exc_info.value.status_code == 400
