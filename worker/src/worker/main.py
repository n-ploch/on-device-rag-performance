"""Worker FastAPI application."""

from __future__ import annotations

import time

from fastapi import FastAPI

from shared_types.schemas import (
    GenerateRequest,
    GenerateResponse,
    HardwareMeasurement,
    InferenceMeasurement,
    RetrievalData,
)
from worker.models.llm import detect_backend
from worker.services.generation import GenerationService
from worker.services.hardware_monitor import HardwareMonitor
from worker.services.retrieval import RetrievalService


def _coerce_hardware_measurement(raw) -> HardwareMeasurement:
    if isinstance(raw, HardwareMeasurement):
        return raw
    return HardwareMeasurement(
        max_ram_usage_mb=float(getattr(raw, "max_ram_usage_mb", 0.0)),
        avg_cpu_utilization_pct=float(getattr(raw, "avg_cpu_utilization_pct", 0.0)),
        peak_cpu_temp_c=getattr(raw, "peak_cpu_temp_c", None),
        swap_in_bytes=int(getattr(raw, "swap_in_bytes", 0)),
        swap_out_bytes=int(getattr(raw, "swap_out_bytes", 0)),
    )


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Worker")
    retrieval_service = RetrievalService()
    generation_service = GenerationService()

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        e2e_start = time.perf_counter()
        async with HardwareMonitor() as monitor:
            retrieval_start = time.perf_counter()
            retrieved = retrieval_service.retrieve(request.input_prompt, request.run_config.retrieval)
            retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

            generation_start = time.perf_counter()
            output = generation_service.generate(
                prompt=request.input_prompt,
                run_config=request.run_config,
                retrieval_chunks=retrieved,
            )
            llm_generation_latency_ms = (time.perf_counter() - generation_start) * 1000

        e2e_latency_ms = (time.perf_counter() - e2e_start) * 1000
        prompt_tokens = len(request.input_prompt.split())
        completion_tokens = len(output.split())
        duration_seconds = max(llm_generation_latency_ms / 1000.0, 1e-6)

        retrieval_data = RetrievalData(
            cited_doc_ids=[str(item.get("id", "")) for item in retrieved],
            retrieved_chunks=[str(item.get("text", "")) for item in retrieved],
        )
        inference_measurement = InferenceMeasurement(
            e2e_latency_ms=e2e_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            ttft_ms=retrieval_latency_ms,
            llm_generation_latency_ms=llm_generation_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=completion_tokens / duration_seconds,
        )

        return GenerateResponse(
            output=output,
            retrieval_data=retrieval_data,
            inference_measurement=inference_measurement,
            hardware_measurement=_coerce_hardware_measurement(monitor.metrics),
        )

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "healthy",
            "backend": detect_backend(),
            "models": {},
        }

    return app
