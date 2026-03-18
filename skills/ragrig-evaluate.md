---
name: ragrig-evaluate
version: "1.0"
description: >
  Run a RAGrig evaluation: pre-flight checks, execute the evaluation loop
  via CLI or HTTP API, and interpret the results.
scope: project
apply: on-demand
triggers:
  - "run evaluation"
  - "start eval"
  - "execute benchmark"
  - "run benchmark"
  - "assess performance"
  - "evaluate rag"
arguments:
  - name: config
    description: Path to evaluation YAML config file
    required: false
    default: "config/sample_config.yaml"
  - name: run_id
    description: Single run_id from the config to execute (omit to run all)
    required: false
  - name: dry_run
    description: Validate config and count entries without running inference
    required: false
    default: "false"
prerequisites:
  - "Worker running and healthy (GET /health returns 200)"
  - ".env configured with observability backend credentials if enabled"
tools_required: [bash, http]
compatible_with: [claude-code, cursor, copilot, codex]
---

## Config file first — mandatory

**Before doing anything else, verify that a config file exists and was created
for this experiment.**

1. If a `--config` argument was passed and the file exists → proceed to
   [Pre-flight checklist](#pre-flight-checklist).
2. If no argument was passed, check whether `config/config.yaml` or
   `config/my_experiment.yaml` exists. If a plausible config file is present,
   confirm with the user before using it.
3. If **no config file exists** → stop and run the config creation flow:
   - Tell the user: *"No config file found. Creating a config file is the
     required first step. Let me guide you through it."*
   - Follow **`skills/ragrig-config.md`** (or invoke `/ragrig-config`) to
     create one interactively.
   - Do **not** fall back to `config/sample_config.yaml` for real runs — the
     sample is for reference only.

The config file controls the dataset, models, quantization, chunking, and
observability backends. All options are documented in
[`docs/config.md`](../docs/config.md).

---

## STOP and ask the user when

- `GET /health` on the worker fails (connection refused or non-200)
- Required observability env vars are missing:
  - Langfuse: `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`
  - W&B Weave: `WANDB_API_KEY`
- `LLM_API_KEY` is needed (remote generation hosting) but not set
- You are unsure which config file to use
- The config has multiple `run_configs` and the user has not specified which to run

---

## Pre-flight checklist

Run these checks before starting any evaluation:

**1. Worker health**

```bash
curl -s $WORKER_URL$/health
# expect: {"status":"ok"}
```

If this fails: run `./scripts/start-worker.sh` and retry, or ask the user.

**2. Config dry-run**

```bash
source .rag/bin/activate
rag-orchestrator --config <path> --dry-run
```

Validates the YAML and counts entries without touching the worker. Confirm the
entry count looks correct before proceeding.

---

## Running an evaluation

### CLI (recommended)

```bash
source .rag/bin/activate

# All run configs in file
rag-orchestrator --config config/my_experiment.yaml

# Single run config
rag-orchestrator --config config/my_experiment.yaml --run-id <run_id>

# Verbose logging
rag-orchestrator --config config/my_experiment.yaml --log-level debug

# Quiet (suppress per-entry output)
rag-orchestrator --config config/my_experiment.yaml -q
```

Exit code 0 = success. Non-zero = error (check stderr).

### Via HTTP API (programmatic / streaming)

```bash
# 1. Load config
curl -s -X POST http://localhost:8080/api/config/load \
  -H "Content-Type: application/json" \
  -d '{"path": "config/my_experiment.yaml"}'

# 2. Stream evaluation (SSE — keep connection open)
curl -s -N -X POST http://localhost:8080/api/run \
  -H "Content-Type: application/json" -d '{}'

# 3. Stop early if needed
curl -s -X POST http://localhost:8080/api/stop
```

SSE events: `run_start` → `entry_result` (one per entry) → `run_complete` → `done`.

---

## Interpreting results

### Output files

| File | Location | Description |
|------|----------|-------------|
| JSONL spans | `logs/traces.jsonl` (configurable) | Raw OTEL spans per entry |
| Langfuse traces | `$LANGFUSE_BASE_URL` | Interactive trace viewer — session ID printed to stdout |
| Parquet export | `local/metric-export/` | Run `python analysis/langfuse_export.py --session-id <id>` |

### Key metrics

| Metric | Good value | Description |
|--------|-----------|-------------|
| `recall_at_k` | > 0.7 | Fraction of relevant docs retrieved in top-k |
| `precision_at_k` | > 0.5 | Fraction of retrieved docs that are relevant |
| `mrr` | > 0.6 | Mean reciprocal rank of first relevant doc |
| `ttft_ms` | < 500 ms | Time to first token |
| `tokens_per_second` | device-dependent | Generation throughput |
