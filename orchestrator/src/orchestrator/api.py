"""FastAPI server for the RAG evaluation orchestrator.

Wraps the existing orchestrator/runner logic to expose a local web API,
enabling the browser-based frontend to load configs and stream structured
evaluation progress. Environment variables are configured via .env / docker-compose.

Usage:
    uvicorn orchestrator.api:app --port 8080
    # or via entry point:
    rag-api
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from sse_starlette.sse import EventSourceResponse

# Load .env before importing orchestrator modules that read env vars
load_dotenv()

from orchestrator.config import EvalConfig  # noqa: E402
from orchestrator.orchestrator import Orchestrator  # noqa: E402
from orchestrator.runner import (  # noqa: E402
    EvaluationResult,
    configure_logging,
    ensure_dataset,
    evaluate_single,
    load_ground_truth,
)
from orchestrator.tracing import setup_tracing, shutdown_tracing  # noqa: E402
from shared_types.schemas import RunConfig  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE event helpers
# ---------------------------------------------------------------------------

# The queue carries JSON strings; None is the sentinel that signals completion.
_Queue = asyncio.Queue  # type: ignore[type-arg]


async def _emit(queue: _Queue, event: dict) -> None:  # type: ignore[type-arg]
    await queue.put(json.dumps(event))


async def _sse_generator(queue: _Queue, task: asyncio.Task):  # type: ignore[type-arg]
    """Forward structured JSON events from the queue as SSE data frames."""
    while True:
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=1.0)
            if msg is None:
                # Sentinel: evaluation task exited cleanly
                break
            yield {"data": msg}
        except asyncio.TimeoutError:
            if task.done():
                exc = task.exception()
                if exc:
                    yield {"data": json.dumps({"type": "error", "message": str(exc)})}
                break
            yield {"comment": "keep-alive"}

    yield {"data": json.dumps({"type": "done"})}



# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


@dataclass
class _AppState:
    config: EvalConfig | None = None
    config_yaml_text: str = ""
    run_task: asyncio.Task | None = None  # type: ignore[type-arg]
    stop_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Orchestrator API")
app.state.data = _AppState()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ConfigLoadRequest(BaseModel):
    path: str | None = None
    content: str | None = None


class ConfigLoadResponse(BaseModel):
    ok: bool
    config: dict | None = None  # type: ignore[type-arg]
    yaml_text: str = ""
    error: str | None = None


class WorkerUrlResponse(BaseModel):
    url: str


class WorkerUrlUpdateRequest(BaseModel):
    url: str


class WorkerCheckResponse(BaseModel):
    ok: bool
    status: str | None = None
    backend: str | None = None
    models_loaded: bool | None = None
    error: str | None = None


class RunRequest(BaseModel):
    pass  # no parameters needed — progress events carry all info


class StatusResponse(BaseModel):
    config_loaded: bool
    run_id: str | None = None
    is_running: bool


# ---------------------------------------------------------------------------
# Progress-aware evaluation loop (does not modify runner.py)
# ---------------------------------------------------------------------------


def _summary_metrics(results: list[EvaluationResult]) -> dict:  # type: ignore[type-arg]
    with_metrics = [r for r in results if r.recall_at_k is not None]
    if with_metrics:
        avg_recall = sum(r.recall_at_k for r in with_metrics) / len(with_metrics)  # type: ignore[misc]
        avg_precision = sum(r.precision_at_k for r in with_metrics) / len(with_metrics)  # type: ignore[misc]
        avg_mrr = sum(r.mrr for r in with_metrics) / len(with_metrics)  # type: ignore[misc]
    else:
        avg_recall = avg_precision = avg_mrr = 0.0
    return {
        "avg_recall": round(avg_recall, 4),
        "avg_precision": round(avg_precision, 4),
        "avg_mrr": round(avg_mrr, 4),
        "abstentions": sum(1 for r in results if r.is_abstention),
        "total": len(results),
    }


async def _run_configs_loop_with_progress(
    run_configs: list[RunConfig],
    orchestrator: Orchestrator,
    entries: list,
    tracer,
    queue: _Queue,
    stop_event: asyncio.Event,
) -> None:
    """Evaluation loop that emits structured progress events to *queue*.

    After each entry completes (never mid-request), checks *stop_event*.
    If set, emits a ``stopped`` event with a partial summary and returns.
    """
    total_configs = len(run_configs)

    for config_idx, run_config in enumerate(run_configs):
        await orchestrator.prepare_run_config(run_config)

        repeat_count = run_config.repeat if run_config.repeat is not None else 1

        for rep in range(repeat_count):
            session_id = f"{run_config.run_id}_{uuid4().hex[:8]}"
            run_entries = (
                entries[: run_config.limit]
                if run_config.limit is not None
                else entries
            )
            total_entries = len(run_entries)

            gen_cfg = run_config.generation
            gen_model = (
                gen_cfg.remote.base_url
                if gen_cfg.hosting == "remote" and gen_cfg.remote
                else gen_cfg.model
            )

            await _emit(
                queue,
                {
                    "type": "run_start",
                    "run_id": run_config.run_id,
                    "config_index": config_idx + 1,
                    "total_configs": total_configs,
                    "rep": rep + 1,
                    "total_reps": repeat_count,
                    "session_id": session_id,
                    "total_entries": total_entries,
                    "retrieval_model": run_config.retrieval.model,
                    "retrieval_quantization": run_config.retrieval.quantization,
                    "generation_model": gen_model,
                    "generation_quantization": run_config.generation.quantization,
                    "k": run_config.retrieval.k,
                },
            )

            results: list[EvaluationResult] = []

            for entry_idx, entry in enumerate(run_entries):
                try:
                    result = await evaluate_single(
                        orchestrator._client,  # type: ignore[arg-type]
                        entry,
                        run_config,
                        tracer,
                        session_id,
                    )
                    results.append(result)

                    inf = result.response.inference_measurement
                    hw = result.response.hardware_measurement

                    await _emit(
                        queue,
                        {
                            "type": "entry_result",
                            "entry_index": entry_idx + 1,
                            "total_entries": total_entries,
                            "run_id": run_config.run_id,
                            "request": {
                                "claim_id": entry.id,
                                "input": entry.input,
                            },
                            "response": {
                                "output": result.response.output,
                                "recall_at_k": result.recall_at_k,
                                "precision_at_k": result.precision_at_k,
                                "mrr": result.mrr,
                                "is_abstention": result.is_abstention,
                                "latency_ms": inf.e2e_latency_ms,
                                "tokens_per_second": inf.tokens_per_second,
                                "ram_mb": hw.max_ram_usage_mb,
                            },
                        },
                    )

                except Exception as exc:
                    await _emit(
                        queue,
                        {
                            "type": "entry_error",
                            "entry_index": entry_idx + 1,
                            "claim_id": entry.id,
                            "message": str(exc),
                        },
                    )

                # Check stop signal after each entry (never mid-request)
                if stop_event.is_set():
                    await _emit(
                        queue,
                        {
                            "type": "stopped",
                            "run_id": run_config.run_id,
                            "completed_entries": entry_idx + 1,
                            "total_entries": total_entries,
                            **_summary_metrics(results),
                        },
                    )
                    return

            await _emit(
                queue,
                {
                    "type": "run_complete",
                    "run_id": run_config.run_id,
                    **_summary_metrics(results),
                },
            )


async def _run_with_progress(
    config: EvalConfig,
    queue: _Queue,
    stop_event: asyncio.Event,
    dry_run: bool = False,
    run_id_filter: str | None = None,
) -> None:
    """Main evaluation coroutine for the API.

    Emits structured events; places a None sentinel in the queue when done.
    """
    # Suppress chatty library logs — errors are caught and emitted as events
    configure_logging(level=logging.WARNING, print_logs=False, sys_logs_path=None)

    try:
        run_configs = config.run_configs
        if run_id_filter:
            run_configs = [rc for rc in run_configs if rc.run_id == run_id_filter]
            if not run_configs:
                await _emit(
                    queue,
                    {
                        "type": "error",
                        "message": f"No run config found with run_id='{run_id_filter}'.",
                    },
                )
                return

        async with Orchestrator(config) as orchestrator:
            ensure_dataset(orchestrator)

            if dry_run:
                validation = orchestrator.validate_dataset()
                entries = load_ground_truth(validation.ground_truth_path)
                await _emit(
                    queue,
                    {
                        "type": "dry_run_result",
                        "total_entries": len(entries),
                        "total_configs": len(run_configs),
                        "run_ids": [rc.run_id for rc in run_configs],
                    },
                )
                return

            await orchestrator.validate_global_prerequisites()
            validation = orchestrator.validate_dataset()
            entries = load_ground_truth(validation.ground_truth_path)

            tracer = setup_tracing(observability=config.observability)
            try:
                await _run_configs_loop_with_progress(
                    run_configs=run_configs,
                    orchestrator=orchestrator,
                    entries=entries,
                    tracer=tracer,
                    queue=queue,
                    stop_event=stop_event,
                )
            finally:
                shutdown_tracing()

    except Exception as exc:
        await _emit(queue, {"type": "error", "message": str(exc)})
        raise
    finally:
        await queue.put(None)  # sentinel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_run_preconditions(state: _AppState, dry_run: bool = False) -> None:
    if state.config is None:
        raise HTTPException(
            status_code=422,
            detail="Please load a configuration file in the Setup tab first.",
        )
    if state.run_task is not None and not state.run_task.done():
        raise HTTPException(status_code=409, detail="An evaluation is already running.")
    if dry_run:
        return

    obs = state.config.observability

    _BACKEND_REQUIRED_VARS: dict[str, list[str]] = {
        "langfuse": ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"],
        "weave": ["WANDB_API_KEY", "WANDB_BASE_URL"],
        "generic": ["OTEL_EXPORTER_OTLP_ENDPOINT"],
    }

    if obs.backends:
        # Multi-backend path: validate each enabled backend
        for backend in obs.backends:
            if not backend.enabled:
                continue
            required = _BACKEND_REQUIRED_VARS.get(backend.type, [])
            missing = [k for k in required if not os.environ.get(k)]
            if missing:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Backend '{backend.type}' is enabled but "
                        f"{' and '.join(missing)} are not set. "
                        "Configure them in the Environment tab."
                    ),
                )
    elif obs.langfuse:
        # Legacy single-backend path
        missing = [
            k
            for k in _BACKEND_REQUIRED_VARS["langfuse"]
            if not os.environ.get(k)
        ]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Langfuse is enabled in the config but {' and '.join(missing)} "
                    "are not set. Configure them in the Environment tab or set "
                    "'langfuse: false' in your config."
                ),
            )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/config/load", response_model=ConfigLoadResponse)
async def load_config(req: ConfigLoadRequest) -> ConfigLoadResponse:
    state: _AppState = app.state.data
    if not req.path and not req.content:
        raise HTTPException(status_code=422, detail="Provide either 'path' or 'content'.")
    try:
        if req.content:
            yaml_text = req.content
            raw = yaml.safe_load(yaml_text)
            config = EvalConfig.model_validate(raw)
        else:
            config_path = Path(req.path)  # type: ignore[arg-type]
            if not config_path.exists():
                return ConfigLoadResponse(
                    ok=False, error=f"File not found: {config_path.resolve()}"
                )
            yaml_text = config_path.read_text()
            config = EvalConfig.from_yaml(config_path)
    except yaml.YAMLError as exc:
        return ConfigLoadResponse(ok=False, error=f"YAML parse error: {exc}")
    except ValidationError as exc:
        return ConfigLoadResponse(ok=False, error=f"Config validation error: {exc}")
    except Exception as exc:
        return ConfigLoadResponse(ok=False, error=str(exc))

    state.config = config
    state.config_yaml_text = yaml_text
    return ConfigLoadResponse(
        ok=True, config=config.model_dump(mode="json"), yaml_text=yaml_text
    )


@app.get("/api/worker/url", response_model=WorkerUrlResponse)
async def get_worker_url() -> WorkerUrlResponse:
    return WorkerUrlResponse(url=os.environ.get("WORKER_URL", "http://localhost:8000"))


@app.post("/api/worker/url", response_model=WorkerUrlResponse)
async def set_worker_url(req: WorkerUrlUpdateRequest) -> WorkerUrlResponse:
    os.environ["WORKER_URL"] = req.url.rstrip("/")
    return WorkerUrlResponse(url=os.environ["WORKER_URL"])


@app.get("/api/worker/check", response_model=WorkerCheckResponse)
async def check_worker() -> WorkerCheckResponse:
    url = os.environ.get("WORKER_URL", "http://localhost:8000")
    try:
        async with httpx.AsyncClient(
            base_url=url, timeout=httpx.Timeout(5.0, connect=3.0)
        ) as client:
            response = await client.get("/health")
            response.raise_for_status()
            data = response.json()
            return WorkerCheckResponse(
                ok=True,
                status=data.get("status"),
                backend=data.get("backend"),
                models_loaded=data.get("models_loaded"),
            )
    except httpx.ConnectError:
        return WorkerCheckResponse(ok=False, error=f"Could not connect to worker at {url}")
    except httpx.TimeoutException:
        return WorkerCheckResponse(ok=False, error=f"Connection to worker at {url} timed out")
    except httpx.HTTPStatusError as exc:
        return WorkerCheckResponse(ok=False, error=f"Worker returned HTTP {exc.response.status_code}")
    except Exception as exc:  # noqa: BLE001
        return WorkerCheckResponse(ok=False, error=str(exc))


@app.post("/api/run")
async def run_evaluation(req: RunRequest):
    state: _AppState = app.state.data
    _check_run_preconditions(state, dry_run=False)
    state.stop_event.clear()
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(
        _run_with_progress(  # type: ignore[arg-type]
            config=state.config,
            queue=queue,
            stop_event=state.stop_event,
            dry_run=False,
        )
    )
    state.run_task = task
    return EventSourceResponse(_sse_generator(queue, task))


@app.post("/api/dry-run")
async def dry_run_evaluation(req: RunRequest):
    state: _AppState = app.state.data
    _check_run_preconditions(state, dry_run=True)
    state.stop_event.clear()
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(
        _run_with_progress(  # type: ignore[arg-type]
            config=state.config,
            queue=queue,
            stop_event=state.stop_event,
            dry_run=True,
        )
    )
    state.run_task = task
    return EventSourceResponse(_sse_generator(queue, task))


@app.post("/api/stop")
async def stop_evaluation():
    """Signal the running evaluation to stop after the current entry finishes."""
    state: _AppState = app.state.data
    if state.run_task is None or state.run_task.done():
        raise HTTPException(status_code=409, detail="No evaluation is currently running.")
    state.stop_event.set()
    return {"ok": True}


@app.get("/api/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    state: _AppState = app.state.data
    is_running = state.run_task is not None and not state.run_task.done()
    run_id = (
        state.config.run_configs[0].run_id
        if state.config and state.config.run_configs
        else None
    )
    return StatusResponse(
        config_loaded=state.config is not None,
        run_id=run_id,
        is_running=is_running,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import uvicorn

    uvicorn.run("orchestrator.api:app", host="127.0.0.1", port=8080, reload=False)


if __name__ == "__main__":
    main()
