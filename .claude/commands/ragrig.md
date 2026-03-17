You are operating RAGrig, a distributed RAG evaluation system.

## Architecture

Two components communicate over HTTP:

- **Worker** (edge device, port 8000): FastAPI service running quantized GGUF
  models via llama-server. Handles embedding, retrieval (ChromaDB), generation,
  and hardware monitoring. Runs **100% offline** after setup.
- **Orchestrator** (host): Drives the evaluation loop, computes quality metrics,
  exports OTEL spans to Langfuse (or other backends).

All shared schemas are Pydantic models in `shared/src/shared_types/schemas.py`.

---

## STOP and ask the user when

- Any required environment variable is missing or has a placeholder value
  (`LOCAL_MODELS_DIR`, `LOCAL_COLLECTIONS_DIR`, `LOCAL_DATASETS_DIR`,
  `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`)
- `HF_TOKEN` is needed (gated HuggingFace dataset/model) but not set
- `LLM_API_KEY` is needed (remote generation hosting) but not set
- `POST /load_models` returns an error (missing GGUF file, wrong path, etc.)
- `POST /collection/build` has made no progress for more than 10 minutes
- The worker health check (`GET /health`) fails to connect
- You are unsure which `run_id` from the config the user wants to execute
- The user has not told you which config file to use

---

## Pre-flight checklist

Run these checks before starting any evaluation:

1. **Verify `.env`** — read `.env` (or check env vars). Confirm that
   `LOCAL_MODELS_DIR`, `LOCAL_COLLECTIONS_DIR`, `LOCAL_DATASETS_DIR` are
   set and point to directories that exist. Confirm Langfuse keys are present
   if the config enables the Langfuse backend.

2. **Worker health** — `GET http://localhost:8000/health` (or `$WORKER_URL/health`).
   Expect `{"status":"ok"}`. If connection is refused, the worker is not running.

3. **Config dry-run** — `rag-orchestrator --config <path> --dry-run` to
   validate the YAML and count entries without touching the worker.

---

## Worker lifecycle

### 1. Start the worker (on the edge device)

```bash
./scripts/start-worker.sh
```

Or manually:

```bash
source .rag/bin/activate
uvicorn worker.main:create_app --factory --host 0.0.0.0 --port 8000
```

### 2. Load models

```bash
curl -s -X POST http://localhost:8000/load_models \
  -H "Content-Type: application/json" \
  -d '{
    "embedder_repo": "<embedder-repo>",
    "embedder_quantization": "<quant>",
    "generator_repo": "<generator-repo>",
    "generator_quantization": "<quant>",
    "embedder_config": null,
    "generator_config": null,
    "generation_config": null
  }'
```

Read the `run_config` in the YAML to determine the correct repo and quantization
values. GGUF files must exist under `LOCAL_MODELS_DIR`.

### 3. Check collection status

```bash
curl -s -X POST http://localhost:8000/collection/status \
  -H "Content-Type: application/json" \
  -d '{
    "retrieval_config": {
      "dataset_id": "<dataset_id>",
      "model": "<embedder-repo>",
      "quantization": "<quant>",
      "dimensions": <int>,
      "chunking": { "strategy": "fixed", "chunk_size": 500, "chunk_overlap": 64 },
      "k": 5
    }
  }'
```

If `"populated": false`, build the collection (step 4). Otherwise skip.

### 4. Build collection (if needed)

```bash
curl -s -X POST http://localhost:8000/collection/build \
  -H "Content-Type: application/json" \
  -d '{ "retrieval_config": { ... } }'
```

This can take several minutes. Poll `/collection/status` to track progress.
**Stop and notify the user if there is no progress after 10 minutes.**

---

## Running an evaluation

### CLI (recommended for scripting)

```bash
source .rag/bin/activate

# All run configs in file
rag-orchestrator --config config/my_experiment.yaml

# Dry-run validation only
rag-orchestrator --config config/my_experiment.yaml --dry-run

# Single run config
rag-orchestrator --config config/my_experiment.yaml --run-id <run_id>

# Verbose logging
rag-orchestrator --config config/my_experiment.yaml -v
```

Exit code 0 = success. Non-zero = error (check stderr).

### Make shortcut

```bash
make eval CONFIG=config/my_experiment.yaml
```

### Via HTTP API (programmatic)

```bash
# 1. Load config
curl -s -X POST http://localhost:8080/api/config/load \
  -H "Content-Type: application/json" \
  -d '{"path": "config/my_experiment.yaml"}'

# 2. Stream evaluation (SSE)
curl -s -N http://localhost:8080/api/run \
  -X POST -H "Content-Type: application/json" -d '{}'

# 3. Stop early if needed
curl -s -X POST http://localhost:8080/api/stop
```

---

## Interpreting results

### Output files

| File | Location | Description |
|------|----------|-------------|
| JSONL spans | `local/` | Raw OTEL spans exported during evaluation |
| Langfuse | `$LANGFUSE_BASE_URL` | Interactive trace viewer — session ID logged to stdout |
| Parquet export | `local/metric-export/` | Run `python analysis/langfuse_export.py --session-id <id>` |

### Key metrics

| Metric | Good value | Description |
|--------|-----------|-------------|
| `recall_at_k` | > 0.7 | Fraction of relevant docs retrieved in top-k |
| `precision_at_k` | > 0.5 | Fraction of retrieved docs that are relevant |
| `mrr` | > 0.6 | Mean reciprocal rank of first relevant doc |
| `abstention` | low | Model declined to answer (indicates retrieval failure) |
| `ttft_ms` | < 500ms | Time to first token |
| `tokens_per_second` | device-dependent | Generation throughput |

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Connection refused` on `/health` | Worker not running | Run `./scripts/start-worker.sh` |
| `/load_models` returns 500 | GGUF file not found | Check `LOCAL_MODELS_DIR`; verify file names match registry |
| `dimensions` mismatch error | Wrong `dimensions` in config | Match `retrieval.dimensions` to model output size |
| High abstention rate | Collection not populated or wrong collection | Check `/collection/status`; rebuild if needed |
| `LANGFUSE_PUBLIC_KEY not set` | Missing env var | Edit `.env`; ask user for key |
| TTFT timeout (>300s) | Model too large for device RAM | Try a smaller quantization level |

---

## Config file structure (quick reference)

See `config/sample_config.yaml` for a full example.

```yaml
dataset:
  name: ragbench          # or: scifact
  subset: emanual

observability:
  backends:
    - type: langfuse
      enabled: true

run_configs:
  - run_id: run1
    limit: 50             # max entries (null = all)
    repeat: 1
    retrieval:
      dataset_id: emanual
      model: intfloat/multilingual-e5-small
      quantization: fp16
      dimensions: 384
      chunking: { strategy: fixed, chunk_size: 500, chunk_overlap: 64 }
      k: 5
    generation:
      model: TheBloke/Mistral-7B-Instruct-v0.1-GGUF
      quantization: q4_k_m
      hosting: local      # or: remote
```

---

## Useful commands at a glance

```bash
./scripts/setup.sh                          # first-time bootstrap
./scripts/start-worker.sh                   # start worker
./scripts/start-docker.sh                   # start orchestrator + frontend via Docker
make test                                   # run unit tests
make eval CONFIG=config/sample_config.yaml  # run evaluation
rag-orchestrator --dry-run -c config/...    # validate without running
curl localhost:8000/health                  # worker health
```
