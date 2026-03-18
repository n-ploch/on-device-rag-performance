---
name: ragrig-setup
version: "1.0"
description: >
  Bootstrap RAGrig end-to-end: create venv, start the worker,
  load GGUF models, and build the ChromaDB collection.
scope: project
apply: on-demand
triggers:
  - "set up ragrig"
  - "bootstrap worker"
  - "load models"
  - "build collection"
  - "start worker"
  - "initialize ragrig"
  - "prepare for evaluation"
arguments:
  - name: config
    description: Path to evaluation YAML config file (used to read model and dataset IDs)
    required: false
    default: "config/sample_config.yaml"
prerequisites:
  - ".env exists with LOCAL_MODELS_DIR, LOCAL_COLLECTIONS_DIR, LOCAL_DATASETS_DIR set"
  - "GGUF model files present under LOCAL_MODELS_DIR"
tools_required: [bash, read_file, http]
compatible_with: [claude-code, cursor, copilot, codex]
---

## Config file first — mandatory

**Setup reads model and dataset IDs from a config file.** Before starting,
ensure a config file exists for this experiment.

1. If a `--config` argument was passed and the file exists → proceed to
   [Step 1](#step-1--verify-environment).
2. If no config file is found → stop and create one first:
   - Tell the user: *"No config file found. Let me guide you through creating
     one before we set up the worker."*
   - Follow **`skills/ragrig-config.md`** (or invoke `/ragrig-config`).
3. Do **not** use `config/sample_config.yaml` for real experiments — it is a
   reference file only. All options are documented in
   [`docs/config.md`](../docs/config.md).

---

## STOP and ask the user when

- `LOCAL_MODELS_DIR`, `LOCAL_COLLECTIONS_DIR`, or `LOCAL_DATASETS_DIR` is missing from `.env` or has a placeholder value
- `HF_TOKEN` is needed (gated HuggingFace dataset or model) but is not set
- `POST /load_models` returns a non-200 response (GGUF file not found, wrong path)
- `POST /collection/build` has made no progress for more than 10 minutes
- You are unsure which `run_id` from the config the user wants to set up for

---

## Step 1 — Verify environment

Read `.env` (or check env vars). Confirm:

- `LOCAL_MODELS_DIR` → exists on disk
- `LOCAL_COLLECTIONS_DIR` → exists on disk
- `LOCAL_DATASETS_DIR` → exists on disk
- `WORKER_URL` → exists on disk
- `LLAMA_SERVER_PATH` → confirm you can run llama-server command with this path or, if missing, with default 'llama-server'

If any path is missing or the directory doesn't exist, stop and ask the user.

---

## Step 2 — Start the worker

```bash
./scripts/start-worker.sh
```

Or manually:

```bash
source .rag/bin/activate
uvicorn worker.main:create_app --factory --host 0.0.0.0 --port 8000
```

Confirm it is up:

```bash
curl -s $WORKER_URL$/health
# expect: {"status":"ok"}
```

If connection is refused, the worker is not running. Do not proceed and report to the user with error description.

---

## Step 3 — Load models

Read the `run_config` in the YAML (default: `config/sample_config.yaml`) to get
`retrieval.model`, `retrieval.quantization`, `generation.model`, and
`generation.quantization`.

```bash
curl -s -X POST http://localhost:8000/load_models \
  -H "Content-Type: application/json" \
  -d '{
    "embedder_repo": "<retrieval.model>",
    "embedder_quantization": "<retrieval.quantization>",
    "generator_repo": "<generation.model>",
    "generator_quantization": "<generation.quantization>",
    "embedder_config": null,
    "generator_config": null,
    "generation_config": null
  }'
```

GGUF files must exist under `LOCAL_MODELS_DIR`. If the response is not 200, stop and
report the error to the user.

---

## Step 4 — Check collection status

```bash
curl -s -X POST http://localhost:8000/collection/status \
  -H "Content-Type: application/json" \
  -d '{
    "retrieval_config": {
      "dataset_id": "<dataset.id from YAML>",
      "model": "<retrieval.model>",
      "quantization": "<retrieval.quantization>",
      "dimensions": <retrieval.dimensions>,
      "chunking": {
        "strategy": "fixed",
        "chunk_size": 500,
        "chunk_overlap": 64
      },
      "k": 5
    }
  }'
```

If `"populated": true` → skip step 5. If `"populated": false` → continue.

---

## Step 5 — Build collection (if needed)

```bash
curl -s -X POST http://localhost:8000/collection/build \
  -H "Content-Type: application/json" \
  -d '{ "retrieval_config": { ... } }'
```

This can take several minutes. Poll `/collection/status` to track progress.
**Stop and notify the user if there is no progress after 10 minutes.**

When `"populated": true` is returned, setup is complete.
