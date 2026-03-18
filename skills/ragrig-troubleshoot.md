---
name: ragrig-troubleshoot
version: "1.0"
description: >
  Diagnose RAGrig failures: worker connectivity, model loading errors,
  collection issues, and observability misconfigurations.
scope: project
apply: on-demand
triggers:
  - "troubleshoot"
  - "debug ragrig"
  - "something is wrong"
  - "error"
  - "not working"
  - "connection refused"
  - "fix issue"
prerequisites: []
tools_required: [bash, http]
compatible_with: [claude-code, cursor, copilot, codex]
---

## Diagnostic commands

```bash
# Worker health
curl -s $WORKER_URL$/health

# Orchestrator status
curl -s http://localhost:8080/api/status

# Worker connectivity check (from orchestrator)
curl -s http://localhost:8080/api/worker/check

# Validate config without running
source .rag/bin/activate
rag-orchestrator --config config/my_experiment.yaml --dry-run -v

# Collection status
curl -s -X POST $WORKER_URL$/collection/status \
  -H "Content-Type: application/json" \
  -d '{"retrieval_config": { ... }}'
```

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Connection refused` on `/health` | Worker not running | Run `./scripts/start-worker.sh` |
| `/load_models` returns 500 | GGUF file not found | Check `LOCAL_MODELS_DIR`; verify file names match registry |
| `dimensions` mismatch error | Wrong `dimensions` in config | Match `retrieval.dimensions` to model output size; rebuild collection with correct name |
| High abstention rate | Collection not populated or wrong collection | Check `/collection/status`; rebuild if `populated: false` |
| `LANGFUSE_PUBLIC_KEY not set` | Missing env var | Edit `.env`; ask user for key |
| TTFT timeout (> 300 s) | Model too large for device RAM | Try a smaller quantization level (e.g. q4_k_m → q2_k) |
| `422 Unprocessable Entity` on `/generate` | Schema mismatch (stale shared package) | `pip install -e shared/` to refresh the editable install |
| Collection build stuck > 10 min | Dataset not downloaded or wrong path | Check `LOCAL_DATASETS_DIR`; verify the dataset corpus exists |
| `httpx.ConnectError` during export | Langfuse unreachable | Check `LANGFUSE_BASE_URL`; confirm Langfuse container is running |
| Empty Parquet export | Wrong session ID or v2 API called | Use v1 SDK only; confirm session ID from stdout log |

---

## Environment variable checklist

| Variable | Required for | Check |
|----------|-------------|-------|
| `LOCAL_MODELS_DIR` | Worker — model loading | Directory exists, contains GGUF files |
| `LOCAL_COLLECTIONS_DIR` | Worker — ChromaDB | Directory exists (created automatically if empty) |
| `LOCAL_DATASETS_DIR` | Worker — corpus reading | Directory exists, contains dataset files |
| `LANGFUSE_BASE_URL` | Observability (Langfuse) | Reachable; no trailing slash |
| `LANGFUSE_PUBLIC_KEY` | Observability (Langfuse) | Non-empty |
| `LANGFUSE_SECRET_KEY` | Observability (Langfuse) | Non-empty |
| `HF_TOKEN` | Gated HuggingFace assets | Set if dataset or model is gated |
| `LLM_API_KEY` | Remote generation hosting | Set if `generation.hosting: remote` |
| `WORKER_URL` | Orchestrator → worker | Defaults to `http://localhost:8000`; override if worker is on a remote device |
