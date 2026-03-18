# RAGrig — On-Device RAG Performance Evaluation

A benchmarking system for evaluating hardware performance and generation quality
of quantized LLMs for on-device Retrieval-Augmented Generation (RAG).

## What it measures

- **Hardware metrics**: RAM usage, CPU utilization, swap activity, thermal behavior
- **Inference metrics**: End-to-end latency, time-to-first-token, tokens/second
- **Quality metrics**: Retrieval recall/precision, MRR, abstention rates

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      HOST MACHINE                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │                   ORCHESTRATOR                      │  │
│  │  • Drives evaluation loop                          │  │
│  │  • Computes quality metrics (recall@k, MRR, …)    │  │
│  │  • Exports traces via OpenTelemetry → Langfuse     │  │
│  └────────────────────────────────────────────────────┘  │
│                          │                                │
│                          │ HTTP POST /generate            │
│                          ▼                                │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                      EDGE DEVICE                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │                    WORKER                           │  │
│  │  • FastAPI (100% offline operation)                │  │
│  │  • llama-server for embeddings & generation        │  │
│  │  • ChromaDB for vector retrieval                   │  │
│  │  • psutil for hardware monitoring                  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

| Component | Responsibility | Does NOT do |
|-----------|---------------|-------------|
| **Orchestrator** | Evaluation loop, quality metrics, telemetry | Model inference, hardware monitoring |
| **Worker** | Model loading, inference, retrieval, hardware metrics | Quality scoring, trace aggregation |
| **Shared** | Pydantic schemas for API contracts | Business logic |

The edge worker runs **100% offline**. Models, datasets, and ChromaDB
collections must be pre-provisioned to local paths before starting, which is normally performed by the orchestrator.

---

## Prerequisites: llama.cpp

The worker relies on `llama-server` (part of llama.cpp) for model inference and embeddings. Install it on the **edge device** before proceeding. Follow [here](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#quick-start) for the official installation guide.

### Option A — Package manager (simplest)

```bash
# macOS / Linux
brew install llama.cpp

# Windows
winget install llama.cpp
```

> **GPU backend check**: Package manager builds may not enable your GPU backend by default. After installing, run `llama-server --version` and check that the correct backend appears (e.g., `Metal`, `CUDA`, `Vulkan`). If it shows `CPU only`, build from source instead.

### Option B — Pre-built binaries

Download the latest release for your platform from [github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases). Each release includes backend-specific builds:

| Build suffix | Hardware |
|---|---|
| `*-metal` | Apple Silicon (M-series) |
| `*-cuda-cu12*` | NVIDIA GPU (CUDA 12.x) |
| `*-vulkan` | AMD / Intel GPU (cross-platform) |
| `*-hip` | AMD GPU (ROCm) |

Pick the build matching your device and add the binary directory to `PATH`.

### Option C — Build from source (recommended for hardware-specific tuning)

Building from source lets the compiler optimize for your exact CPU/GPU and is the best way to unlock peak performance. See the [llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for full instructions. Quick reference:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build \
  -DGGML_METAL=ON       # Apple Silicon — replace with the flag for your backend
  # -DGGML_CUDA=ON      # NVIDIA
  # -DGGML_VULKAN=ON    # AMD / Intel via Vulkan
  # -DGGML_HIP=ON       # AMD via ROCm
cmake --build build --config Release -j $(nproc)
```

After building, add `build/bin/` to your `PATH` or copy `llama-server` to a location already on it.

### Verify the installation

```bash
llama-server --version
```

You should see a version string and the active backend listed (e.g., `ggml_metal`, `ggml_cuda`). If `llama-server` is not found, check your `PATH`.

---

## Quick start

### 1. Bootstrap

```bash
./scripts/setup.sh
```

This creates a `.rag/` venv, installs all packages in editable mode, and copies
`.env.example → .env`. Edit `.env` with your local paths and backend tracing variables (i.e., Langfuse).

### 2. Run tests

```bash
make test
```

### 3a. Local development (separate terminals)

**Terminal 1 — Worker (edge device)**

```bash
./scripts/start-worker.sh
```

**Terminal 2 — Orchestrator + Frontend (host)**

```bash
make ui       # orchestrator API on :8080, Vite frontend on :5173
# or CLI:
make eval CONFIG=config/sample_config.yaml
```

### 3b. Docker (orchestrator + frontend only)

```bash
./scripts/start-docker.sh
```

Starts the orchestrator API on `localhost:8080` and the frontend on
`localhost:3003`. The worker still runs bare-metal on the edge device.

---

## Repository structure

```
on-device-rag-performance/
├── orchestrator/           # Host-side evaluation driver
│   └── src/orchestrator/
│       ├── api.py          # FastAPI server (rag-api entry point)
│       ├── runner.py       # CLI entry point (rag-orchestrator)
│       ├── exporters/      # JSONL and Langfuse span exporters
│       └── metrics.py      # recall@k, precision@k, MRR, abstention
│
├── worker/                 # Edge device FastAPI service
│   └── src/worker/
│       ├── main.py         # Factory app + endpoints
│       ├── models/         # Embedder, generator, registry
│       └── services/       # Retrieval, generation, hardware monitor
│
├── shared/                 # Shared Pydantic schemas
│   └── src/shared_types/schemas.py
│
├── analysis/               # Langfuse metrics export + notebooks
├── frontend/               # Browser UI (Vite + TypeScript)
├── tests/                  # Unit and integration tests
├── config/                 # YAML evaluation configs
├── scripts/                # Setup and startup scripts
├── docs/                   # API and CLI reference
└── .claude/commands/       # Claude Code skills (ragrig.md)
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/cli.md](docs/cli.md) | `rag-orchestrator` and `rag-api` CLI reference |
| [docs/worker-api.md](docs/worker-api.md) | Worker FastAPI endpoint reference |
| [docs/orchestrator-api.md](docs/orchestrator-api.md) | Orchestrator FastAPI endpoint reference |
| [config/sample_config.yaml](config/sample_config.yaml) | Annotated example evaluation config |

---

## Make targets

| Target | Description |
|--------|-------------|
| `make setup` | Bootstrap venv and install all packages |
| `make test` | Run unit tests |
| `make worker` | Start worker (bare-metal) |
| `make orchestrator` | Start orchestrator API only |
| `make ui` | Start orchestrator API + Vite frontend |
| `make eval CONFIG=...` | Run evaluation CLI with a config file |
| `make clean` | Remove build artifacts |

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LOCAL_MODELS_DIR` | Yes | Path to pre-downloaded GGUF models |
| `LOCAL_COLLECTIONS_DIR` | Yes | Path to ChromaDB collections |
| `LOCAL_DATASETS_DIR` | Yes | Path to downloaded datasets |
| `WORKER_URL` | No | Worker URL (default: `http://localhost:8000`) |
| `LANGFUSE_BASE_URL` | If using Langfuse | Langfuse instance URL |
| `LANGFUSE_PUBLIC_KEY` | If using Langfuse | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | If using Langfuse | Langfuse secret key |
| `HF_TOKEN` | If gated dataset | HuggingFace access token |
| `LLM_API_KEY` | If remote generation | Remote LLM API key |
| `LOG_LEVEL` | No | Logging level (default: `INFO`) |

See [.env.example](.env.example) for all variables with comments.

---

## Key constraints

1. **`embedding=True`**: The embedder must be started with embedding mode or
   llama.cpp will crash.
2. **Vector dimensions**: ChromaDB locks dimensions per collection — ensure
   `retrieval.dimensions` in the config matches the model's actual output.
3. **Editable installs**: `pip install -e ./shared` shares Pydantic models
   without duplication. `./scripts/setup.sh` handles this automatically.
4. **No runtime downloads**: All models and collections must exist locally
   before starting the worker.

---

## Agentic usage

A Claude Code skill is available at `.claude/commands/ragrig.md`. Invoke it in
Claude Code with `/ragrig` to get a full machine-oriented reference for driving
the tool end-to-end, including pre-flight checks, worker lifecycle, evaluation
commands, and failure-mode guidance.
