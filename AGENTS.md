# AGENTS.md — RAGrig

> Machine-readable context for AI coding agents (Codex, Cursor, Gemini CLI, Copilot, etc.).
> Agent skills (setup, evaluate, analyze, troubleshoot) are in `skills/` — see [`skills/README.md`](skills/README.md).
> Claude Code users: invoke `/ragrig`, `/ragrig-setup`, `/ragrig-evaluate`, `/ragrig-analyze`, or `/ragrig-troubleshoot`.

---

## Project overview

**RAGrig** is a distributed RAG evaluation system for benchmarking quantized LLMs on edge devices.

Two components communicate over HTTP:

| Component | Default port | Role |
|-----------|-------------|------|
| **Worker** | 8000 | Edge device — embedding, ChromaDB retrieval, GGUF generation, hardware monitoring |
| **Orchestrator** | 8080 | Host — evaluation loop, quality metrics, OTEL tracing, optional frontend |

All shared Pydantic schemas live in `shared/src/shared_types/schemas.py`.

---

## Repository layout

```
.
├── orchestrator/          # Host-side evaluation driver
│   └── src/orchestrator/
│       ├── runner.py      # CLI entry point (rag-orchestrator)
│       ├── api.py         # FastAPI server (rag-api, port 8080)
│       ├── orchestrator.py
│       ├── metrics.py
│       └── tracing.py
├── worker/                # Edge-device FastAPI inference service (port 8000)
│   └── src/worker/
│       ├── main.py
│       ├── models/
│       └── services/
├── shared/                # Pydantic schemas shared by orchestrator + worker
│   └── src/shared_types/schemas.py
├── analysis/              # Langfuse → Parquet export, Jupyter notebooks
├── config/                # YAML evaluation configs (sample_config.yaml)
├── docs/                  # cli.md, orchestrator-api.md, worker-api.md
├── scripts/               # setup.sh, start-worker.sh, start-docker.sh, start-ui.sh
├── tests/
│   ├── unit/
│   └── integration/
├── docker-compose.yml     # orchestrator-api (8080) + frontend (3003)
└── Makefile
```

`local/` is git-ignored and holds runtime data (models, ChromaDB collections, datasets, exports).

---

## Setup

### First-time bootstrap

```bash
bash scripts/setup.sh        # creates .rag/ venv, editable installs, copies .env.example → .env
```

Edit `.env` — at minimum set:

```bash
LOCAL_MODELS_DIR=./local/models
LOCAL_COLLECTIONS_DIR=./local/collections
LOCAL_DATASETS_DIR=./local/datasets
```

### Manual install (without the script)

```bash
python3 -m venv .rag && source .rag/bin/activate
pip install -e shared/ -e orchestrator/ -e worker/ -e analysis/
cp .env.example .env
```

---

## Running tests

```bash
source .rag/bin/activate
make test                    # unit tests (pytest tests/unit -v)
pytest tests/unit -v         # equivalent
pytest tests/integration -v  # requires worker running on localhost:8000
```

---

## Key commands

```bash
# Start worker (bare-metal, edge device)
./scripts/start-worker.sh
# or: uvicorn worker.main:create_app --factory --host 0.0.0.0 --port 8000

# Start orchestrator + frontend via Docker
./scripts/start-docker.sh    # ports 8080, 3003

# Development mode (no Docker)
./scripts/start-ui.sh        # rag-api on 8080 + Vite on 5173

# Run evaluation (CLI)
source .rag/bin/activate
make eval CONFIG=config/sample_config.yaml
# or: rag-orchestrator --config config/sample_config.yaml

# Validate config without running
rag-orchestrator --config config/sample_config.yaml --dry-run

# Worker health check
curl http://localhost:8000/health
```

---

## CLI reference (`rag-orchestrator`)

Entry point defined in `orchestrator/pyproject.toml`.

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` / `-c` | `config/config.yaml` | Path to evaluation YAML |
| `--dry-run` | off | Validate config and count entries, no inference |
| `--run-id ID` | all | Run only one `run_id` from the config |
| `--log-level LEVEL` / `-l` | `info` | Logging level (`debug`, `info`, `warning`, `error`, `critical`) |
| `--quiet` / `-q` | off | Suppress per-entry output |

Exit 0 = success. Non-zero = error (check stderr).

---

## API surface

### Orchestrator API — `http://localhost:8080`

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/config/load` | Load config from `{"path": "..."}` or `{"content": "..."}` |
| `POST` | `/api/config/reset` | Clear loaded config |
| `GET` | `/api/worker/url` | Get current worker URL |
| `POST` | `/api/worker/url` | Set worker URL |
| `GET` | `/api/worker/check` | Test worker connectivity |
| `POST` | `/api/run` | Start evaluation — returns SSE stream |
| `POST` | `/api/dry-run` | Validate config — returns SSE stream |
| `POST` | `/api/stop` | Stop after current entry |
| `GET` | `/api/status` | Server state JSON |
| `GET` | `/docs` | Interactive OpenAPI UI |

### Worker API — `http://localhost:8000`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Worker status |
| `POST` | `/load_models` | Load embedder + generator GGUF models |
| `POST` | `/collection/status` | Check ChromaDB collection |
| `POST` | `/collection/build` | Build/embed collection from corpus |
| `POST` | `/generate` | Run RAG inference |
| `GET` | `/metrics` | llama-server telemetry |

Full schema in `shared/src/shared_types/schemas.py`. Human-readable docs in `docs/`.

---

## Config file structure

See `config/sample_config.yaml` for a full working example.

```yaml
dataset:
  name: ragbench          # or: scifact

observability:
  backends:
    - type: langfuse
      enabled: true
  output_jsonl: ./logs/traces.jsonl

run_configs:
  - run_id: my_run
    limit: 50             # null = all entries
    retrieval:
      dataset_id: emanual
      model: intfloat/multilingual-e5-small
      quantization: fp16
      dimensions: 384
      chunking:
        strategy: fixed
        chunk_size: 500
        chunk_overlap: 64
      k: 5
    generation:
      model: TheBloke/Mistral-7B-Instruct-v0.1-GGUF
      quantization: q4_k_m
      hosting: local       # or: remote
```

---

## Architecture constraints (read before editing)

- **Langfuse v1 only.** Local Langfuse deployments do not expose the v2 API. Always use the v1 SDK (`fern_langfuse`) and `/api/public/scores` directly via httpx for score reads.
- **ChromaDB collection dimensions are immutable.** A collection created with 384-dim vectors cannot accept 1024-dim vectors. A dimension mismatch requires a new collection name.
- **Worker runs 100% offline after setup.** All models and datasets must be pre-provisioned under `LOCAL_MODELS_DIR` and `LOCAL_DATASETS_DIR` before evaluation starts. No runtime downloads.
- **Editable installs required.** `shared/` must be installed editable (`pip install -e shared/`) so orchestrator and worker share the same Pydantic schema objects.
- **OTEL trace propagation.** The orchestrator passes W3C `traceparent` headers to the worker on every `/generate` call. Do not strip these headers when modifying HTTP clients.
- **Collection naming convention.** Collections are named `{model-slug}__{quantization}__{dimensions}_{index}` (e.g. `multilingual-e5-small__fp16__384_0`). See `shared/src/shared_types/naming.py`.
- **`LOCAL_*_DIR` env vars.** All paths that cross the Docker boundary use these variables. The Docker Compose mounts `./local` → `/app/local`; the worker uses host paths via `host.docker.internal`.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LOCAL_MODELS_DIR` | yes | Directory containing GGUF model files |
| `LOCAL_COLLECTIONS_DIR` | yes | Directory for ChromaDB collections |
| `LOCAL_DATASETS_DIR` | yes | Directory for downloaded datasets |
| `WORKER_URL` | no (default `http://localhost:8000`) | Worker base URL seen from orchestrator |
| `LANGFUSE_BASE_URL` | if Langfuse enabled | Local Langfuse instance URL |
| `LANGFUSE_PUBLIC_KEY` | if Langfuse enabled | |
| `LANGFUSE_SECRET_KEY` | if Langfuse enabled | |
| `WANDB_API_KEY` | if W&B Weave enabled | |
| `HF_TOKEN` | if gated HuggingFace assets | |
| `LLM_API_KEY` | if remote generation | |
