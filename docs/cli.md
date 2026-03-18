# CLI Reference

RAGrig ships two CLI entry points, both installed into the active venv via
`pip install -e ./orchestrator`.

---

## `rag-orchestrator` — run an evaluation

Drives a full evaluation loop: loads ground truth, calls the worker for each
entry, computes retrieval metrics, and exports results.

### Usage

```
rag-orchestrator [OPTIONS]
```

### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--config PATH` | `-c` | `config/config.yaml` | Path to evaluation config YAML |
| `--dry-run` | | off | Validate config and count entries without running evaluation |
| `--quiet` | `-q` | off | Suppress per-entry progress output |
| `--log-level LEVEL` | `-l` | `info` | Logging level (`debug`, `info`, `warning`, `error`, `critical`) |
| `--run-id ID` | | all | Run only the specified `run_id` from the config |

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Evaluation completed successfully |
| `1` | Config file not found, unknown dataset, or evaluation error |

### Examples

```bash
# Run all configs in the default config file
rag-orchestrator

# Point at a specific config
rag-orchestrator --config config/my_experiment.yaml

# Dry-run to validate config and count entries
rag-orchestrator --config config/my_experiment.yaml --dry-run

# Run only one specific run config
rag-orchestrator --config config/my_experiment.yaml --run-id mistral_q4_run1

# Enable debug logging
rag-orchestrator --log-level debug
```

### Output files

| File | Location | Description |
|------|----------|-------------|
| JSONL spans | `local/` (configurable) | Raw OTEL spans, one JSON object per line |
| Parquet export | `local/metric-export/` | Per-evaluation metric rows (via `analysis/langfuse_export.py`) |
| Logs | `logs/` (configurable) | Application logs if `sys_logs_path` is set in config |

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LOCAL_MODELS_DIR` | Yes | Path to pre-downloaded GGUF models |
| `LOCAL_COLLECTIONS_DIR` | Yes | Path to ChromaDB collections |
| `LOCAL_DATASETS_DIR` | Yes | Path to downloaded datasets |
| `WORKER_URL` | No | Worker base URL (default: `http://localhost:8000`) |
| `LANGFUSE_BASE_URL` | If using Langfuse | Langfuse instance URL |
| `LANGFUSE_PUBLIC_KEY` | If using Langfuse | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | If using Langfuse | Langfuse secret key |
| `HF_TOKEN` | If gated dataset | HuggingFace access token |
| `LLM_API_KEY` | If remote generation | API key for remote LLM |

---

## Config YAML format

See `config/sample_config.yaml` for a full working example.

```yaml
dataset:
  name: ragbench          # or: scifact
  subset: emanual         # dataset subset (ragbench only)

observability:
  print_logs: true
  sys_logs_path: null     # set to a file path to write logs to disk
  backends:
    - type: langfuse
      enabled: true

run_configs:
  - run_id: mistral_q4_run1
    limit: 50             # max ground truth entries (null = all)
    repeat: 1             # how many times to repeat this run
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
      hosting: local      # or: remote
```

---

## `rag-api` — start the orchestrator HTTP server

Wraps the evaluation logic in a FastAPI server for the browser frontend and
programmatic clients. See [orchestrator-api.md](orchestrator-api.md) for the
full HTTP API reference.

### Usage

```bash
rag-api [--log-level LEVEL]
```

The server listens on `http://127.0.0.1:8080` by default.
Interactive API docs are available at `http://127.0.0.1:8080/docs`.

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--log-level LEVEL` | `-l` | `info` | Logging level (`debug`, `info`, `warning`, `error`, `critical`) |

For the full UI (API + frontend), use `./scripts/start-ui.sh` instead.

---

## `rag-worker` — start the worker HTTP server

Starts the edge-device FastAPI worker. Installed into the venv via
`pip install -e ./worker`.

### Usage

```bash
rag-worker [--host HOST] [--port PORT] [--log-level LEVEL]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--host HOST` | | `0.0.0.0` | Bind host |
| `--port PORT` | | `8000` | Bind port |
| `--log-level LEVEL` | `-l` | `info` | Logging level (`debug`, `info`, `warning`, `error`, `critical`) |

For the standard startup (env validation + logging banner), use `./scripts/start-worker.sh` instead.
