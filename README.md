# WIP - Implementation ongoing
# On-Device RAG Performance Evaluation

A benchmarking system for evaluating hardware performance and generation quality of quantized Mistral models for on-device Retrieval-Augmented Generation (RAG).

## Project Goal

This project measures how different quantization levels of LLMs affect:
- **Hardware metrics**: RAM usage, CPU utilization, swap activity, thermal behavior
- **Inference metrics**: End-to-end latency, time-to-first-token, tokens/second
- **Quality metrics**: Retrieval recall/precision, MRR, abstention rates

The system is designed for edge device evaluation where models run entirely offline.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           HOST MACHINE                                   │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        ORCHESTRATOR                                │  │
│  │  • Drives evaluation loop                                         │  │
│  │  • Calculates quality metrics (recall@k, precision@k, MRR)        │  │
│  │  • Exports traces via OpenTelemetry (JSONL / Langfuse)            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                  │                                       │
│                                  │ HTTP (POST /generate)                 │
│                                  ▼                                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           EDGE DEVICE                                    │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                          WORKER                                    │  │
│  │  • FastAPI server (100% offline operation)                        │  │
│  │  • llama-cpp-python for embeddings & generation                   │  │
│  │  • ChromaDB for vector retrieval                                  │  │
│  │  • psutil for hardware monitoring                                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Strict Decoupling

The system enforces a clear separation of concerns:

| Component | Responsibility | Does NOT do |
|-----------|---------------|-------------|
| **Orchestrator** | Evaluation loop, quality metrics, telemetry export | Model inference, hardware monitoring |
| **Worker** | Model loading, inference, retrieval, hardware metrics | Quality scoring, trace aggregation |
| **Shared** | Pydantic schemas for API contracts | Business logic |

### Network Isolation

The Edge Worker operates **100% offline**. No models or datasets are downloaded at runtime—all artifacts must be pre-provisioned to local paths.

## Repository Structure

```
on-device-rag-performance/
├── orchestrator/           # Host-side evaluation driver
│   └── src/orchestrator/
│       ├── exporters/      # JSONL and Langfuse span exporters
│       ├── models/         # Model registry (path resolution)
│       └── metrics.py      # recall@k, precision@k, MRR, abstention
│
├── worker/                 # Edge device FastAPI service
│   └── src/worker/
│       ├── models/
│       │   ├── embedder.py    # Llama wrapper with embedding=True
│       │   ├── generator.py   # Llama wrapper for text generation
│       │   └── registry.py    # GGUF path resolution
│       ├── services/
│       │   ├── chromadb_bridge.py    # EmbeddingFunction adapter
│       │   ├── collection_registry.py # Collection metadata management
│       │   ├── generation.py         # RAG prompt building + inference
│       │   ├── hardware_monitor.py   # Async psutil sampling
│       │   └── retrieval.py          # ChromaDB query execution
│       └── main.py         # FastAPI app with lifespan model loading
│
├── shared/                 # Shared Pydantic schemas
│   └── src/shared_types/
│       ├── schemas.py      # GenerateRequest, GenerateResponse, etc.
│       └── naming.py       # Collection/model naming conventions
│
├── tests/
│   ├── unit/               # Unit tests per component
│   └── integration/        # API contract tests
│
└── config/                 # YAML configuration files
```

## Mental Models

### 1. Model Loading & Lifespan

llama.cpp locks memory upon model instantiation. The Worker creates **two separate wrappers** during FastAPI's lifespan:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Embedder: small context, embedding mode
    app.state.embedder = Embedder(path, n_ctx=512, embedding=True)

    # Generator: large context for RAG
    app.state.generator = Generator(path, n_ctx=2048)

    yield  # App runs here

    # Cleanup on shutdown
```

### 2. ChromaDB Integration

ChromaDB requires an `EmbeddingFunction` to generate vectors. We bridge our local Embedder:

```python
class LlamaEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embedder.embed(doc) for doc in input]
```

### 3. Collection Storage Convention

Collection folders are named dynamically to prevent model/dimension mismatches:

```
{model}__{quantization}__{dimensions}_{index}

Example: mistral-embed__q4_k_m__1024_0
```

Each collection is fully self-contained in its own folder:

- `<collections_root>/metadata.json`: top-level tree index
- `<collections_root>/<collection_folder>/metadata.json`: leaf metadata
- `<collections_root>/<collection_folder>/chroma.sqlite3`: Chroma sqlite
- `<collections_root>/<collection_folder>/<segment-uuid>/...bin`: Chroma segment files

Inside each collection folder, the internal Chroma collection name is fixed to `chunks`.

### 4. Hardware Monitoring

An async context manager samples hardware metrics during inference:

```python
async with HardwareMonitor() as monitor:
    result = generation_service.generate(prompt, chunks)

# monitor.metrics contains:
#   max_ram_usage_mb, avg_cpu_utilization_pct,
#   peak_cpu_temp_c, swap_in_bytes, swap_out_bytes
```

## Interface Definitions

### API Contract: `POST /generate`

**Request** (`GenerateRequest`):
```json
{
  "claim_id": "c1",
  "input_prompt": "Is aspirin effective for pain relief?",
  "run_config": {
    "run_id": "mistral_q4_k5_001",
    "retrieval": {
      "model": "intfloat/multilingual-e5-small",
      "quantization": "fp16",
      "dimensions": 384,
      "chunking": {"strategy": "fixed", "chunk_size": 500, "chunk_overlap": 64},
      "k": 5
    },
    "generation": {
      "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
      "quantization": "q4_k_m"
    }
  }
}
```

**Response** (`GenerateResponse`):
```json
{
  "output": "Based on the provided context, aspirin is effective...",
  "retrieval_data": {
    "cited_doc_ids": ["doc_123", "doc_456"],
    "retrieved_chunks": ["Aspirin reduces inflammation...", "Clinical trials show..."]
  },
  "inference_measurement": {
    "e2e_latency_ms": 1250.5,
    "retrieval_latency_ms": 45.2,
    "ttft_ms": 120.0,
    "llm_generation_latency_ms": 1100.3,
    "prompt_tokens": 512,
    "completion_tokens": 128,
    "tokens_per_second": 11.6
  },
  "hardware_measurement": {
    "max_ram_usage_mb": 4500.2,
    "avg_cpu_utilization_pct": 85.5,
    "peak_cpu_temp_c": 72.0,
    "swap_in_bytes": 0,
    "swap_out_bytes": 0
  }
}
```

### Shared Types

All API contracts are defined as Pydantic models in `shared/src/shared_types/schemas.py`. Both orchestrator and worker import these via editable install:

```bash
pip install -e ./shared
```

This ensures type safety and prevents schema drift between components.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LOCAL_MODELS_DIR` | Yes | Path to pre-downloaded GGUF models |
| `LOCAL_COLLECTIONS_DIR` | Yes | Path to pre-built ChromaDB collections |
| `EMBEDDER_MODEL_REPO` | No | Embedding model name (default: `mistral-embed`) |
| `EMBEDDER_QUANTIZATION` | No | Embedding quantization (default: `q4_k_m`) |
| `GENERATOR_MODEL_REPO` | No | Generation model name (default: `mistral-7b-instruct`) |
| `GENERATOR_QUANTIZATION` | No | Generation quantization (default: `q4_k_m`) |
| `WORKER_URL` | No | Worker API URL for orchestrator (default: `http://localhost:8000`) |

## Getting Started

```bash
# Create virtual environment
python -m venv .rag
source .rag/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install packages in editable mode
pip install -e ./shared -e ./worker -e ./orchestrator

# Run tests
pytest tests/ -v

# Start worker (on edge device)
uvicorn worker.main:create_app --factory --host 0.0.0.0 --port 8000
```

## Key Constraints

1. **`embedding=True` flag**: The Embedder MUST be instantiated with `embedding=True` or llama.cpp will crash
2. **Vector dimensions**: ChromaDB locks dimensions per collection—ensure collection names include the dimension value
3. **Editable installs**: Use `pip install -e ./shared` to share Pydantic models without duplication
4. **No runtime downloads**: All models and collections must exist locally before starting the worker
