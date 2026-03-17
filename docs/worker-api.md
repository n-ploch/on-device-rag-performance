# Worker API Reference

The Worker is a FastAPI service that runs on the **edge device**. It handles
model loading, retrieval, inference, and hardware monitoring. The orchestrator
communicates with it over HTTP.

---

## Starting the worker

### Bare-metal (recommended)

```bash
./scripts/start-worker.sh
```

The script activates the venv, validates required environment variables, and
launches uvicorn. Requires `.rag/` venv (run `./scripts/setup.sh` first).

### Manual uvicorn command

```bash
source .rag/bin/activate
uvicorn worker.main:create_app --factory --host 0.0.0.0 --port 8000
```

### Required environment variables

| Variable | Description |
|----------|-------------|
| `LOCAL_MODELS_DIR` | Path to pre-downloaded GGUF model files |
| `LOCAL_COLLECTIONS_DIR` | Path to ChromaDB collection folders |
| `LOCAL_DATASETS_DIR` | Path to downloaded dataset files |

---

## Endpoints

### `GET /health`

Returns the worker's current status.

**Response**

```json
{
  "status": "ok",
  "backend": "llama-server",
  "models_loaded": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | Always `"ok"` if the server is running |
| `backend` | `string` | Active inference backend identifier |
| `models_loaded` | `boolean` | Whether models are currently loaded via `/load_models` |

---

### `POST /load_models`

Starts the embedding and generation llama-server processes and initialises all
services. Must be called before `/generate`.

**Request body** (`LoadModelsRequest`)

```json
{
  "embedder_repo": "intfloat/multilingual-e5-small",
  "embedder_quantization": "fp16",
  "generator_repo": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
  "generator_quantization": "q4_k_m",
  "embedder_config": null,
  "generator_config": null,
  "generation_config": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `embedder_repo` | `string` | HuggingFace repo name for the embedding model |
| `embedder_quantization` | `string` | Quantization tag (e.g. `fp16`, `q4_k_m`) |
| `generator_repo` | `string` | HuggingFace repo name for the generation model |
| `generator_quantization` | `string` | Quantization tag |
| `embedder_config` | `ServerConfig \| null` | Optional llama-server overrides for embedder |
| `generator_config` | `ServerConfig \| null` | Optional llama-server overrides for generator |
| `generation_config` | `GenerationConfig \| null` | Optional generation config (e.g. remote hosting) |

**Response**

```json
{
  "ok": true,
  "message": "Models loaded successfully"
}
```

---

### `POST /generate`

Run a RAG inference request. Returns the generated response along with
retrieval data and hardware/latency measurements.

**Request body** (`GenerateRequest`)

```json
{
  "claim_id": "c1",
  "input_prompt": "Is aspirin effective for pain relief?",
  "run_config": {
    "run_id": "mistral_q4_run1",
    "retrieval": {
      "dataset_id": "emanual",
      "model": "intfloat/multilingual-e5-small",
      "quantization": "fp16",
      "dimensions": 384,
      "chunking": {
        "strategy": "fixed",
        "chunk_size": 500,
        "chunk_overlap": 64
      },
      "k": 5
    },
    "generation": {
      "model": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
      "quantization": "q4_k_m",
      "hosting": "local"
    }
  },
  "expected_response": "Aspirin is effective for pain relief."
}
```

**OTEL headers** — Inject distributed trace context by including the standard
W3C `traceparent` / `tracestate` headers. The worker creates child spans
(`rag.retrieval`, `rag.generation`) under the parent span.

**Response body** (`GenerateResponse`)

```json
{
  "output": "Based on the provided context, aspirin is effective...",
  "retrieval_data": {
    "cited_doc_ids": ["doc_123", "doc_456"],
    "retrieved_chunks": [
      "Aspirin reduces inflammation...",
      "Clinical trials show..."
    ]
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

**Inference measurement fields**

| Field | Unit | Description |
|-------|------|-------------|
| `e2e_latency_ms` | ms | Total wall time from request received to response sent |
| `retrieval_latency_ms` | ms | ChromaDB query time |
| `ttft_ms` | ms | Time to first token |
| `llm_generation_latency_ms` | ms | Total LLM generation time |
| `prompt_tokens` | count | Input token count |
| `completion_tokens` | count | Output token count |
| `tokens_per_second` | tokens/s | Generation throughput |

**Hardware measurement fields**

| Field | Unit | Description |
|-------|------|-------------|
| `max_ram_usage_mb` | MB | Peak resident memory during inference |
| `avg_cpu_utilization_pct` | % | Average CPU utilization sampled during inference |
| `peak_cpu_temp_c` | °C | Peak CPU temperature (null if unavailable) |
| `swap_in_bytes` | bytes | Swap-in during inference |
| `swap_out_bytes` | bytes | Swap-out during inference |

---

### `POST /collection/status`

Check whether a ChromaDB collection for a given retrieval config exists and
has been populated.

**Request body** (`CollectionStatusRequest`)

```json
{
  "retrieval_config": {
    "dataset_id": "emanual",
    "model": "intfloat/multilingual-e5-small",
    "quantization": "fp16",
    "dimensions": 384,
    "chunking": { "strategy": "fixed", "chunk_size": 500, "chunk_overlap": 64 },
    "k": 5
  }
}
```

**Response** (`CollectionStatusResponse`)

```json
{
  "exists": true,
  "populated": true,
  "chunk_count": 14832,
  "collection_name": "multilingual-e5-small__fp16__384_0"
}
```

---

### `POST /collection/build`

Build a ChromaDB collection from the corpus for a given retrieval config.
This embeds all corpus chunks and stores them — can take several minutes.

**Request body** (`CollectionBuildRequest`)

```json
{
  "retrieval_config": {
    "dataset_id": "emanual",
    "model": "intfloat/multilingual-e5-small",
    "quantization": "fp16",
    "dimensions": 384,
    "chunking": { "strategy": "fixed", "chunk_size": 500, "chunk_overlap": 64 },
    "k": 5
  }
}
```

**Response** (`CollectionBuildResponse`)

```json
{
  "ok": true,
  "collection_name": "multilingual-e5-small__fp16__384_0",
  "chunk_count": 14832
}
```

---

## Collection naming convention

Collections are stored at `LOCAL_COLLECTIONS_DIR/<collection_name>/`:

```
{model-slug}__{quantization}__{dimensions}_{index}

Example: multilingual-e5-small__fp16__384_0
```

Each collection folder contains `chroma.sqlite3` and segment binary files.
The `metadata.json` at the collections root tracks all known collections.

---

## Key constraints

- **`embedding=True`**: The embedder llama-server is started with embedding
  mode. Do not use the same server instance for generation.
- **Vector dimensions**: ChromaDB locks dimensions per collection. Ensure
  `retrieval.dimensions` in the config matches the model's actual output size.
- **No runtime downloads**: All GGUF model files must exist under
  `LOCAL_MODELS_DIR` before calling `/load_models`.
- **Offline operation**: The worker makes no outbound network requests during
  inference. Dataset downloads happen via the orchestrator.
