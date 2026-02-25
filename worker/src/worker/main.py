"""Worker FastAPI application."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from fastapi import FastAPI, Request

from worker.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

from shared_types.schemas import (
    CollectionBuildRequest,
    CollectionBuildResponse,
    CollectionStatusRequest,
    CollectionStatusResponse,
    GenerateRequest,
    GenerateResponse,
    InferenceMeasurement,
    RetrievalData,
)
from worker.models.embedder import Embedder
from worker.models.generator import Generator
from worker.models.llm import detect_backend
from worker.models.registry import get_model_path
from worker.datasets.corpus_reader import CorpusReader
from worker.services.embedding import EmbeddingService
from worker.services.generation import GenerationService
from worker.services.hardware_monitor import HardwareMonitor
from worker.services.retrieval import RetrievalService


def _is_missing_collection_error(exc: Exception) -> bool:
    """Return True if exception indicates a missing Chroma collection."""
    return isinstance(exc, ValueError) or exc.__class__.__name__ == "NotFoundError"


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

    logger.info("Loading embedder: %s (%s)", embedder_repo, embedder_quant)
    embedder_path = get_model_path(embedder_repo, embedder_quant)
    app.state.embedder = Embedder(embedder_path, n_ctx=512)

    logger.info("Loading generator: %s (%s)", generator_repo, generator_quant)
    generator_path = get_model_path(generator_repo, generator_quant)
    app.state.generator = Generator(generator_path, n_ctx=2048)

    collections_dir = Path(os.environ.get("LOCAL_COLLECTIONS_DIR", "./collections"))
    app.state.collections_dir = collections_dir
    app.state.retrieval_service = RetrievalService(
        embedder=app.state.embedder,
        collections_dir=collections_dir,
    )
    app.state.embedding_service = EmbeddingService(
        embedder=app.state.embedder,
        collections_dir=collections_dir,
    )
    app.state.generation_service = GenerationService(generator=app.state.generator)

    logger.info("Models loaded, worker ready")
    yield

    del app.state.generator
    del app.state.embedder


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Worker", lifespan=lifespan)

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest, req: Request) -> GenerateResponse:
        logger.info("POST /generate run_id=%s", request.run_config.run_id)
        logger.debug(
            "Retrieval config: k=%d, model=%s",
            request.run_config.retrieval.k,
            request.run_config.retrieval.model,
        )
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

        logger.info(
            "Request complete: retrieval=%.1fms, generation=%.1fms, total=%.1fms",
            retrieval_latency_ms,
            llm_generation_latency_ms,
            e2e_latency_ms,
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

    @app.post("/collection/status", response_model=CollectionStatusResponse)
    async def collection_status(
        request: CollectionStatusRequest,
        req: Request,
    ) -> CollectionStatusResponse:
        """Check if a collection exists and is populated."""
        logger.info(
            "POST /collection/status dataset=%s, model=%s",
            request.dataset_id,
            request.retrieval_config.model,
        )
        embedding_service: EmbeddingService = req.app.state.embedding_service

        exists = embedding_service.collection_exists(
            dataset_id=request.dataset_id,
            retrieval_config=request.retrieval_config,
        )

        # Get chunk count if collection exists
        chunk_count = None
        resolved_path = embedding_service.resolve_collection_path(
            request.dataset_id,
            request.retrieval_config,
        )
        collection_name = resolved_path.name if resolved_path is not None else None
        if exists and resolved_path is not None:
            client = chromadb.PersistentClient(path=str(resolved_path))
            try:
                collection = client.get_collection(name=EmbeddingService.CHROMA_COLLECTION_NAME)
                chunk_count = collection.count()
            except Exception as e:
                if _is_missing_collection_error(e):
                    pass
                else:
                    raise

        return CollectionStatusResponse(
            exists=exists,
            populated=exists and (chunk_count is not None and chunk_count > 0),
            chunk_count=chunk_count,
            collection_name=collection_name,
        )

    @app.post("/collection/build", response_model=CollectionBuildResponse)
    async def collection_build(
        request: CollectionBuildRequest,
        req: Request,
    ) -> CollectionBuildResponse:
        """Build a collection by embedding the corpus."""
        logger.info(
            "POST /collection/build dataset=%s, model=%s",
            request.dataset_id,
            request.retrieval_config.model,
        )
        embedding_service: EmbeddingService = req.app.state.embedding_service

        # Load corpus from dataset
        corpus_reader = CorpusReader.from_dataset_id(request.dataset_id)
        corpus = corpus_reader.read_all()

        logger.info("Loaded %d documents from corpus", len(corpus))

        # Build the collection
        result = embedding_service.embed_corpus(
            corpus=corpus,
            dataset_id=request.dataset_id,
            retrieval_config=request.retrieval_config,
        )

        logger.info(
            "Collection built: %s, chunks=%d, already_existed=%s",
            result.collection_name,
            result.total_chunks,
            result.already_existed,
        )

        return CollectionBuildResponse(
            collection_name=result.collection_name,
            chunks_embedded=result.total_chunks,
            already_existed=result.already_existed,
        )

    return app
