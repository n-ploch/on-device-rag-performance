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
tools_required: [bash, read_file, http]
compatible_with: [claude-code, cursor, copilot, codex]

---

## STOP and ask the user when

- `LOCAL_MODELS_DIR`, `LOCAL_COLLECTIONS_DIR`, or `LOCAL_DATASETS_DIR` is missing from `.env` or has a placeholder value
- `HF_TOKEN` is needed (gated HuggingFace dataset or model) but is not set
- `POST /load_models` still returns non-200 **after** attempting an orchestrator dry-run (model truly unresolvable)
- `POST /collection/build` has made no progress for more than 10 minutes
- You are unsure which `run_id` from the config the user wants to set up for

---

## Step 0 — Bootstrap the environment (if needed)

Check whether the Python venv exists:

```bash
ls .rag/bin/activate 2>/dev/null && echo "venv ok" || echo "venv missing"
```

If missing, run the setup script first:

```bash
./scripts/setup.sh
```

This creates the venv, installs all packages in editable mode, and copies
`.env.example → .env` if `.env` does not yet exist. You do **not** need to
write any installation code — the script handles everything.

After setup, verify `llama-server` is available:

```bash
which llama-server || echo "NOT FOUND"
```

If `llama-server` is not found, inform the user to install
[llama.cpp](https://github.com/ggerganov/llama.cpp) and re-run setup before
proceeding. If `LLAMA_SERVER_PATH` is set in `.env`, use that path instead.

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

Remember the PID for the worker process to grep logs and information on the worker later. Important for troubleshooting.

Confirm it is up:

```bash
curl -s $WORKER_URL/health
# expect: {"status":"ok"}
```

If connection is refused, the worker is not running. Do not proceed and report to the user with error description.

---
