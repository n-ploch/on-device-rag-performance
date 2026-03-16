"""FastAPI server for the RAG evaluation orchestrator.

Wraps the existing orchestrator/runner logic to expose a local web API,
enabling the browser-based frontend to load configs, set env vars, and
stream evaluation output.

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
from dataclasses import dataclass, field
from pathlib import Path

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
    _run_configs_loop,
    configure_logging,
    ensure_dataset,
    load_ground_truth,
)
from orchestrator.tracing import setup_tracing, shutdown_tracing  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known env keys exposed to the frontend
# ---------------------------------------------------------------------------

ENV_KEYS: list[str] = [
    "WORKER_URL",
    "LOCAL_MODELS_DIR",
    "LOCAL_DATASETS_DIR",
    "HF_TOKEN",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LLM_API_KEY",
    "LLAMA_SERVER_PATH",
    "EMBEDDING_PORT",
    "GENERATION_PORT",
]


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


@dataclass
class _AppState:
    config: EvalConfig | None = None
    config_yaml_text: str = ""
    run_task: asyncio.Task | None = None  # type: ignore[type-arg]


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


class EnvResponse(BaseModel):
    values: dict[str, str]


class EnvUpdateRequest(BaseModel):
    values: dict[str, str]


class EnvUpdateResponse(BaseModel):
    ok: bool


class RunRequest(BaseModel):
    verbose: bool = False


class StatusResponse(BaseModel):
    config_loaded: bool
    run_id: str | None = None
    is_running: bool


# ---------------------------------------------------------------------------
# SSE log capture
# ---------------------------------------------------------------------------


class _QueueLogHandler(logging.Handler):
    """Routes log records into an asyncio.Queue for SSE streaming."""

    def __init__(self, queue: asyncio.Queue[str | None]) -> None:
        super().__init__()
        self._queue = queue
        self._loop = asyncio.get_event_loop()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # call_soon_threadsafe because OTEL's BatchSpanProcessor uses threads
            self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)
        except Exception:
            self.handleError(record)


async def _sse_generator(
    queue: asyncio.Queue[str | None],
    task: asyncio.Task,  # type: ignore[type-arg]
):
    """Yield SSE-formatted events from the log queue until the task finishes."""
    while True:
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=1.0)
            if msg is None:
                # Sentinel: evaluation task finished cleanly
                break
            yield {"data": json.dumps({"type": "log", "text": msg})}
        except asyncio.TimeoutError:
            if task.done():
                exc = task.exception()
                if exc:
                    yield {"data": json.dumps({"type": "error", "text": str(exc)})}
                break
            # Keep-alive comment to prevent proxy timeouts
            yield {"comment": "keep-alive"}

    yield {"data": json.dumps({"type": "done"})}


# ---------------------------------------------------------------------------
# Core evaluation runner (accepts pre-parsed EvalConfig, no file I/O)
# ---------------------------------------------------------------------------


async def _run_from_config(
    config: EvalConfig,
    queue: asyncio.Queue[str | None],
    dry_run: bool = False,
    run_id_filter: str | None = None,
    log_level: int = logging.INFO,
) -> None:
    """Mirror of runner._run() that accepts a pre-parsed EvalConfig.

    All log output is routed to *queue* via _QueueLogHandler.
    A ``None`` sentinel is placed in the queue when this coroutine exits.
    """
    handler = _QueueLogHandler(queue)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    # Replace root handlers so all log output goes to the SSE queue
    configure_logging(level=log_level, print_logs=False, sys_logs_path=None)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)

    try:
        run_configs = config.run_configs
        if run_id_filter:
            run_configs = [rc for rc in run_configs if rc.run_id == run_id_filter]
            if not run_configs:
                logger.error("No run config found with run_id=%s", run_id_filter)
                return

        async with Orchestrator(config) as orchestrator:
            ensure_dataset(orchestrator)

            if dry_run:
                validation = orchestrator.validate_dataset()
                entries = load_ground_truth(validation.ground_truth_path)
                logger.info(
                    "Dry run complete. Would evaluate %d entries x %d configs.",
                    len(entries),
                    len(run_configs),
                )
                return

            await orchestrator.validate_global_prerequisites()

            validation = orchestrator.validate_dataset()
            entries = load_ground_truth(validation.ground_truth_path)
            logger.info("Loaded %d ground truth entries", len(entries))

            tracer = setup_tracing()
            try:
                await _run_configs_loop(
                    run_configs=run_configs,
                    orchestrator=orchestrator,
                    entries=entries,
                    tracer=tracer,
                    quiet=True,  # progress via logging, not print()
                )
            finally:
                shutdown_tracing()

    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        raise
    finally:
        await queue.put(None)  # sentinel: tells the SSE generator to close


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_run_preconditions(state: _AppState, dry_run: bool = False) -> None:
    """Raise HTTP 422/409 for the most common misconfiguration cases."""
    if state.config is None:
        raise HTTPException(
            status_code=422,
            detail="Please load a configuration file in the Setup tab first.",
        )

    if state.run_task is not None and not state.run_task.done():
        raise HTTPException(
            status_code=409,
            detail="An evaluation is already running.",
        )

    if not dry_run and state.config.observability.langfuse:
        missing = [
            k
            for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
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
    """Load and validate a YAML evaluation config.

    Accepts either a server-side file path or raw YAML content (from the
    browser file picker).
    """
    state: _AppState = app.state.data

    if not req.path and not req.content:
        raise HTTPException(
            status_code=422, detail="Provide either 'path' or 'content'."
        )

    try:
        if req.content:
            yaml_text = req.content
            raw = yaml.safe_load(yaml_text)
            config = EvalConfig.model_validate(raw)
        else:
            config_path = Path(req.path)  # type: ignore[arg-type]
            if not config_path.exists():
                return ConfigLoadResponse(
                    ok=False,
                    error=f"File not found: {config_path.resolve()}",
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
        ok=True,
        config=config.model_dump(mode="json"),
        yaml_text=yaml_text,
    )


@app.get("/api/env", response_model=EnvResponse)
async def get_env() -> EnvResponse:
    """Return current values of all known orchestrator env keys."""
    return EnvResponse(
        values={k: os.environ.get(k, "") for k in ENV_KEYS}
    )


@app.post("/api/env", response_model=EnvUpdateResponse)
async def update_env(req: EnvUpdateRequest) -> EnvUpdateResponse:
    """Write env var values to os.environ for this session."""
    for k, v in req.values.items():
        if k in ENV_KEYS:
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
    return EnvUpdateResponse(ok=True)


@app.post("/api/run")
async def run_evaluation(req: RunRequest):
    """Start a full evaluation run; stream log output via SSE."""
    state: _AppState = app.state.data
    _check_run_preconditions(state, dry_run=False)

    log_level = logging.DEBUG if req.verbose else logging.INFO
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    task = asyncio.create_task(
        _run_from_config(
            config=state.config,  # type: ignore[arg-type]
            queue=queue,
            dry_run=False,
            log_level=log_level,
        )
    )
    state.run_task = task

    return EventSourceResponse(_sse_generator(queue, task))


@app.post("/api/dry-run")
async def dry_run_evaluation(req: RunRequest):
    """Validate config and dataset without running evaluation; stream via SSE."""
    state: _AppState = app.state.data
    _check_run_preconditions(state, dry_run=True)

    log_level = logging.DEBUG if req.verbose else logging.INFO
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    task = asyncio.create_task(
        _run_from_config(
            config=state.config,  # type: ignore[arg-type]
            queue=queue,
            dry_run=True,
            log_level=log_level,
        )
    )
    state.run_task = task

    return EventSourceResponse(_sse_generator(queue, task))


@app.get("/api/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Return current app state for the frontend to check."""
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
    """Start the orchestrator API server."""
    import uvicorn

    uvicorn.run("orchestrator.api:app", host="127.0.0.1", port=8080, reload=False)


if __name__ == "__main__":
    main()
