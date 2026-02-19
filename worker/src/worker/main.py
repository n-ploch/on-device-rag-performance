"""Worker FastAPI application."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from shared_types.schemas import (
    GenerateRequest,
    GenerateResponse,
    InferenceMeasurement,
    RetrievalData,
)
from worker.models.embedder import Embedder
from worker.models.generator import Generator
from worker.models.llm import detect_backend
from worker.models.registry import get_model_path
from worker.services.generation import GenerationService
from worker.services.hardware_monitor import HardwareMonitor
from worker.services.retrieval import RetrievalService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, release on shutdown.

    llama.cpp locks memory upon instantiation, so we create both the
    embedder and generator wrappers here and store them in app.state.
    """
    embedder_repo = os.environ.get("EMBEDDER_MODEL_REPO", "mistral-embed")
    embedder_quant = os.environ.get("EMBEDDER_QUANTIZATION", "q4_k_m")
    generator_repo = os.environ.get("GENERATOR_MODEL_REPO", "mistral-7b-instruct")
    generator_quant = os.environ.get("GENERATOR_QUANTIZATION", "q4_k_m")

    embedder_path = get_model_path(embedder_repo, embedder_quant)
    generator_path = get_model_path(generator_repo, generator_quant)

    app.state.embedder = Embedder(embedder_path, n_ctx=512)
    app.state.generator = Generator(generator_path, n_ctx=2048)

    collections_dir = Path(os.environ.get("LOCAL_COLLECTIONS_DIR", "./collections"))
    app.state.retrieval_service = RetrievalService(
        embedder=app.state.embedder,
        collections_dir=collections_dir,
    )
    app.state.generation_service = GenerationService(generator=app.state.generator)

    yield

    del app.state.generator
    del app.state.embedder


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Worker", lifespan=lifespan)

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest, req: Request) -> GenerateResponse:
        retrieval_service: RetrievalService = req.app.state.retrieval_service
        generation_service: GenerationService = req.app.state.generation_service

        e2e_start = time.perf_counter()
        async with HardwareMonitor() as monitor:
            retrieval_start = time.perf_counter()
            retrieved = retrieval_service.retrieve(
                request.input_prompt,
                request.run_config.retrieval,
            )
            retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

            generation_start = time.perf_counter()
            gen_result = generation_service.generate(
                prompt=request.input_prompt,
                retrieval_chunks=retrieved,
            )
            llm_generation_latency_ms = (time.perf_counter() - generation_start) * 1000

        e2e_latency_ms = (time.perf_counter() - e2e_start) * 1000
        duration_seconds = max(llm_generation_latency_ms / 1000.0, 1e-6)

        retrieval_data = RetrievalData(
            cited_doc_ids=[item["id"] for item in retrieved],
            retrieved_chunks=[item["text"] for item in retrieved],
        )
        inference_measurement = InferenceMeasurement(
            e2e_latency_ms=e2e_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            ttft_ms=retrieval_latency_ms,
            llm_generation_latency_ms=llm_generation_latency_ms,
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
            tokens_per_second=gen_result.completion_tokens / duration_seconds,
        )

        return GenerateResponse(
            output=gen_result.text,
            retrieval_data=retrieval_data,
            inference_measurement=inference_measurement,
            hardware_measurement=monitor.metrics,
        )

    @app.get("/health")
    async def health(req: Request) -> dict:
        models_loaded = hasattr(req.app.state, "embedder") and hasattr(req.app.state, "generator")
        return {
            "status": "healthy" if models_loaded else "initializing",
            "backend": detect_backend(),
            "models_loaded": models_loaded,
        }

    return app
