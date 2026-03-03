"""Worker FastAPI application using llama-server for inference."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from fastapi import FastAPI, HTTPException, Request

from worker.logging_config import setup_logging
from worker.tracing import extract_context, get_tracer, setup_tracing, shutdown_tracing

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
    LoadModelsRequest,
    LoadModelsResponse,
    RetrievalData,
)
from worker.datasets.corpus_reader import CorpusReader
from worker.models.embedder_http import LlamaServerEmbedder
from worker.models.generator_http import LlamaServerGenerator
from worker.models.registry import get_model_path
from worker.services.embedding import EmbeddingService
from worker.services.generation import GenerationService
from worker.services.hardware_monitor import HardwareMonitor
from worker.services.retrieval import RetrievalService
from worker.services.server_manager import (
    DEFAULT_EMBEDDING_PORT,
    DEFAULT_GENERATION_PORT,
    LlamaServerManager,
)


def _is_missing_collection_error(exc: Exception) -> bool:
    """Return True if exception indicates a missing Chroma collection."""
    return isinstance(exc, ValueError) or exc.__class__.__name__ == "NotFoundError"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize worker state. Models are loaded via /load_models endpoint.

    The worker manages llama-server processes for both embedding and generation.
    Servers are started/stopped via the /load_models endpoint.
    """
    # Initialize server manager
    llama_server_path = os.environ.get("LLAMA_SERVER_PATH")
    embedding_port = int(os.environ.get("EMBEDDING_PORT", str(DEFAULT_EMBEDDING_PORT)))
    generation_port = int(os.environ.get("GENERATION_PORT", str(DEFAULT_GENERATION_PORT)))

    app.state.server_manager = LlamaServerManager(
        llama_server_path=llama_server_path,
        embedding_port=embedding_port,
        generation_port=generation_port,
    )

    # Initialize state - models will be loaded via /load_models
    app.state.embedder = None
    app.state.generator = None
    app.state.loaded_models = None

    # Initialize collections directory
    collections_dir = Path(os.environ.get("LOCAL_COLLECTIONS_DIR", "./collections"))
    app.state.collections_dir = collections_dir

    # Services will be initialized when models are loaded
    app.state.retrieval_service = None
    app.state.embedding_service = None
    app.state.generation_service = None

    # Initialize OTEL tracing
    setup_tracing()

    logger.info(
        "Worker initialized with llama-server backend, waiting for /load_models call "
        "(embedding port: %d, generation port: %d)",
        embedding_port,
        generation_port,
    )
    yield

    # Cleanup: stop all servers and shutdown tracing
    logger.info("Shutting down, stopping llama-server instances")
    await app.state.server_manager.close()
    shutdown_tracing()


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Worker (llama-server)", lifespan=lifespan)

    @app.post("/load_models", response_model=LoadModelsResponse)
    async def load_models(request: LoadModelsRequest, req: Request) -> LoadModelsResponse:
        """Load embedder and generator models by starting llama-server processes.

        If servers are already running, they are stopped first before starting
        new ones with the requested models.
        """
        logger.info(
            "POST /load_models embedder=%s/%s, generator=%s/%s",
            request.embedder_repo,
            request.embedder_quantization,
            request.generator_repo,
            request.generator_quantization,
        )

        server_manager: LlamaServerManager = req.app.state.server_manager

        # Stop existing servers and cleanup clients
        if req.app.state.embedder is not None:
            logger.info("Cleaning up previous embedder client")
            req.app.state.embedder.close()
            req.app.state.embedder = None
        if req.app.state.generator is not None:
            logger.info("Cleaning up previous generator client")
            req.app.state.generator.close()
            req.app.state.generator = None

        # Get model paths
        embedder_path = get_model_path(request.embedder_repo, request.embedder_quantization)
        generator_path = get_model_path(request.generator_repo, request.generator_quantization)

        # Start embedding server with optional custom config
        embedder_cfg = request.embedder_config
        logger.info("Starting embedding server: %s (%s)", request.embedder_repo, request.embedder_quantization)
        if not await server_manager.start_embedding_server(
            model_path=embedder_path,
            n_ctx=embedder_cfg.n_ctx if embedder_cfg and embedder_cfg.n_ctx else 512,
            n_gpu_layers=embedder_cfg.n_gpu_layers if embedder_cfg else -1,
            pooling=embedder_cfg.pooling if embedder_cfg else "mean",
        ):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start embedding server for {request.embedder_repo}",
            )

        # Start generation server with optional custom config
        generator_cfg = request.generator_config
        logger.info("Starting generation server: %s (%s)", request.generator_repo, request.generator_quantization)
        if not await server_manager.start_generation_server(
            model_path=generator_path,
            n_ctx=generator_cfg.n_ctx if generator_cfg and generator_cfg.n_ctx else 2048,
            n_gpu_layers=generator_cfg.n_gpu_layers if generator_cfg else -1,
            parallel_slots=generator_cfg.parallel_slots if generator_cfg else 4,
        ):
            # Stop embedding server if generation server fails
            await server_manager.stop_embedding_server()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start generation server for {request.generator_repo}",
            )

        # Create HTTP clients
        req.app.state.embedder = LlamaServerEmbedder(
            server_url=server_manager.embedding_url,
        )
        req.app.state.generator = LlamaServerGenerator(
            server_url=server_manager.generation_url,
        )

        # Track what's loaded for validation
        req.app.state.loaded_models = {
            "embedder": (request.embedder_repo, request.embedder_quantization),
            "generator": (request.generator_repo, request.generator_quantization),
        }

        # Initialize services with new clients
        collections_dir = req.app.state.collections_dir
        req.app.state.retrieval_service = RetrievalService(
            embedder=req.app.state.embedder,
            collections_dir=collections_dir,
        )
        req.app.state.embedding_service = EmbeddingService(
            embedder=req.app.state.embedder,
            collections_dir=collections_dir,
        )
        req.app.state.generation_service = GenerationService(
            generator=req.app.state.generator,
        )

        logger.info("Models loaded successfully via llama-server")
        return LoadModelsResponse(
            embedder=f"{request.embedder_repo}/{request.embedder_quantization}",
            generator=f"{request.generator_repo}/{request.generator_quantization}",
            message="Models loaded successfully via llama-server",
        )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest, req: Request) -> GenerateResponse:
        logger.info("POST /generate run_id=%s", request.run_config.run_id)

        # Validate models are loaded
        if req.app.state.embedder is None or req.app.state.generator is None:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Call /load_models first.",
            )

        # Validate loaded models match request
        loaded = req.app.state.loaded_models
        expected_embedder = (request.run_config.retrieval.model, request.run_config.retrieval.quantization)
        if expected_embedder != loaded["embedder"]:
            raise HTTPException(
                status_code=400,
                detail=f"RunConfig expects embedder {expected_embedder}, but {loaded['embedder']} is loaded",
            )
        expected_generator = (request.run_config.generation.model, request.run_config.generation.quantization)
        if expected_generator != loaded["generator"]:
            raise HTTPException(
                status_code=400,
                detail=f"RunConfig expects generator {expected_generator}, but {loaded['generator']} is loaded",
            )

        logger.debug(
            "Retrieval config: k=%d, model=%s",
            request.run_config.retrieval.k,
            request.run_config.retrieval.model,
        )
        retrieval_service: RetrievalService = req.app.state.retrieval_service
        generation_service: GenerationService = req.app.state.generation_service

        # Extract trace context from incoming request headers
        ctx = extract_context(dict(req.headers))
        tracer = get_tracer()

        e2e_start = time.perf_counter()
        async with HardwareMonitor() as monitor:
            # Create retrieval span as child of orchestrator's root span
            with tracer.start_span(
                "rag.retrieval",
                context=ctx,
                attributes={
                    "run_id": request.run_config.run_id,
                    "claim_id": request.claim_id,
                },
            ) as retrieval_span:
                retrieval_start = time.perf_counter()
                retrieved = retrieval_service.retrieve(
                    request.input_prompt,
                    request.run_config.retrieval,
                )
                retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

                # Enrich retrieval span
                doc_ids = [item["metadata"].get("doc_id", item["id"]) for item in retrieved]
                retrieval_span.set_attribute("custom.latency.retrieval_latency_ms", retrieval_latency_ms)
                retrieval_span.set_attribute("custom.retrieval.k", len(retrieved))
                retrieval_span.set_attribute("custom.retrieval.cited_doc_ids", ",".join(doc_ids))

            # Create generation span as child of orchestrator's root span
            with tracer.start_span(
                "rag.generation",
                context=ctx,
                attributes={
                    "run_id": request.run_config.run_id,
                    "claim_id": request.claim_id,
                    "langfuse.observation.type": "generation",
                },
            ) as generation_span:
                gen_result = generation_service.generate(
                    prompt=request.input_prompt,
                    retrieval_chunks=retrieved,
                )

                # Build retrieval context for the span
                retrieval_context = "\n---\n".join(item["text"] for item in retrieved)

                # Enrich generation span with llama measurements
                generation_span.set_attribute("gen_ai.prompt", request.input_prompt)
                generation_span.set_attribute("gen_ai.completion", gen_result.text)
                generation_span.set_attribute("custom.generation.retrieval_context", retrieval_context)

                # Include expected response (ground truth) if provided
                if request.expected_response:
                    generation_span.set_attribute("custom.generation.ground_truth", request.expected_response)

                # Latency measurements from llama.cpp
                ttft_ms = gen_result.prompt_ms + gen_result.predicted_per_token_ms
                generation_span.set_attribute("custom.latency.ttft_ms", ttft_ms)
                generation_span.set_attribute("custom.latency.llm_generation_latency_ms", gen_result.predicted_ms)
                generation_span.set_attribute("custom.latency.prompt_ms", gen_result.prompt_ms)
                generation_span.set_attribute("custom.latency.predicted_ms", gen_result.predicted_ms)
                generation_span.set_attribute("custom.latency.predicted_per_token_ms", gen_result.predicted_per_token_ms)

                # Token metrics
                generation_span.set_attribute("custom.generation.prompt_tokens", gen_result.prompt_tokens)
                generation_span.set_attribute("custom.generation.completion_tokens", gen_result.completion_tokens)
                generation_span.set_attribute("custom.generation.tokens_per_second", gen_result.predicted_per_second)

                # Langfuse-specific usage attributes
                generation_span.set_attribute("langfuse.observation.usage.input", gen_result.prompt_tokens)
                generation_span.set_attribute("langfuse.observation.usage.output", gen_result.completion_tokens)

        e2e_latency_ms = (time.perf_counter() - e2e_start) * 1000

        retrieval_data = RetrievalData(
            cited_doc_ids=[item["metadata"].get("doc_id", item["id"]) for item in retrieved],
            retrieved_chunks=[item["text"] for item in retrieved],
        )
        inference_measurement = InferenceMeasurement(
            e2e_latency_ms=e2e_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            ttft_ms=gen_result.prompt_ms + gen_result.predicted_per_token_ms,
            llm_generation_latency_ms=gen_result.predicted_ms,
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
            tokens_per_second=gen_result.predicted_per_second,
        )

        logger.info(
            "Request complete: retrieval=%.1fms, generation=%.1fms, total=%.1fms",
            retrieval_latency_ms,
            gen_result.predicted_ms,
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
        server_manager: LlamaServerManager = req.app.state.server_manager

        # Check server health
        server_health = await server_manager.health_check()

        models_loaded = (
            req.app.state.embedder is not None
            and req.app.state.generator is not None
        )

        return {
            "status": "healthy",
            "backend": "llama-server",
            "models_loaded": models_loaded,
            "loaded_models": req.app.state.loaded_models,
            "servers": {
                "embedding": {
                    "running": server_manager.is_embedding_running,
                    "healthy": server_health.get("embedding", False),
                    "url": server_manager.embedding_url,
                },
                "generation": {
                    "running": server_manager.is_generation_running,
                    "healthy": server_health.get("generation", False),
                    "url": server_manager.generation_url,
                },
            },
        }

    @app.get("/metrics")
    async def metrics(req: Request) -> dict:
        """Scrape and return metrics from llama-server instances."""
        server_manager: LlamaServerManager = req.app.state.server_manager
        server_metrics = await server_manager.scrape_metrics()

        return {
            "embedding": (
                {
                    "prompt_tokens_total": server_metrics["embedding"].prompt_tokens_total,
                    "tokens_predicted_total": server_metrics["embedding"].tokens_predicted_total,
                    "prompt_seconds_total": server_metrics["embedding"].prompt_seconds_total,
                    "tokens_predicted_seconds_total": server_metrics["embedding"].tokens_predicted_seconds_total,
                    "n_decode_total": server_metrics["embedding"].n_decode_total,
                    "requests_processing": server_metrics["embedding"].requests_processing,
                }
                if "embedding" in server_metrics
                else None
            ),
            "generation": (
                {
                    "prompt_tokens_total": server_metrics["generation"].prompt_tokens_total,
                    "tokens_predicted_total": server_metrics["generation"].tokens_predicted_total,
                    "prompt_seconds_total": server_metrics["generation"].prompt_seconds_total,
                    "tokens_predicted_seconds_total": server_metrics["generation"].tokens_predicted_seconds_total,
                    "n_decode_total": server_metrics["generation"].n_decode_total,
                    "requests_processing": server_metrics["generation"].requests_processing,
                }
                if "generation" in server_metrics
                else None
            ),
        }

    @app.post("/collection/status", response_model=CollectionStatusResponse)
    async def collection_status(
        request: CollectionStatusRequest,
        req: Request,
    ) -> CollectionStatusResponse:
        """Check if a collection exists and is populated."""
        retrieval_config = request.retrieval_config
        logger.info(
            "POST /collection/status dataset=%s, model=%s",
            retrieval_config.dataset_id,
            retrieval_config.model,
        )

        if req.app.state.embedding_service is None:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Call /load_models first.",
            )

        embedding_service: EmbeddingService = req.app.state.embedding_service

        exists = embedding_service.collection_exists(retrieval_config)

        # Get chunk count if collection exists
        chunk_count = None
        resolved_path = embedding_service.resolve_collection_path(retrieval_config)
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
        retrieval_config = request.retrieval_config
        logger.info(
            "POST /collection/build dataset=%s, model=%s",
            retrieval_config.dataset_id,
            retrieval_config.model,
        )

        if req.app.state.embedding_service is None:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Call /load_models first.",
            )

        embedding_service: EmbeddingService = req.app.state.embedding_service

        # Load corpus from dataset
        corpus_reader = CorpusReader.from_dataset_id(retrieval_config.dataset_id)
        corpus = corpus_reader.read_all()

        logger.info("Loaded %d documents from corpus", len(corpus))

        # Build the collection
        result = embedding_service.embed_corpus(
            corpus=corpus,
            retrieval_config=retrieval_config,
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
