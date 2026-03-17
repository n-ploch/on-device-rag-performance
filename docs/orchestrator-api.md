# Orchestrator API Reference

The Orchestrator API is a FastAPI server (`rag-api`) that wraps the evaluation
logic and exposes it over HTTP. It is used by the browser frontend and can be
used by any programmatic client.

**Base URL:** `http://localhost:8080` (default)
**Interactive docs:** `http://localhost:8080/docs`

---

## Starting the server

```bash
rag-api
# or:
uvicorn orchestrator.api:app --host 127.0.0.1 --port 8080
# or via Docker:
./scripts/start-docker.sh
```

---

## Endpoints

### Config

#### `POST /api/config/load`

Load an evaluation config from a file path or inline YAML content.

**Request body**

```json
{ "path": "config/my_experiment.yaml" }
// or
{ "content": "dataset:\n  name: ragbench\n..." }
```

One of `path` or `content` must be provided.

**Response**

```json
{
  "ok": true,
  "config": { "dataset": { "name": "ragbench" }, "run_configs": [...] },
  "yaml_text": "dataset:\n  name: ragbench\n...",
  "error": null
}
```

On failure, `ok` is `false` and `error` contains the reason.

---

#### `POST /api/config/reset`

Clear the currently loaded config.

**Response**

```json
{ "ok": true }
```

---

### Worker

#### `GET /api/worker/url`

Return the current worker URL.

**Response**

```json
{ "url": "http://localhost:8000" }
```

---

#### `POST /api/worker/url`

Update the worker URL for this session.

**Request body**

```json
{ "url": "http://192.168.1.100:8000" }
```

**Response**

```json
{ "url": "http://192.168.1.100:8000" }
```

---

#### `GET /api/worker/check`

Test connectivity to the current worker URL by calling its `/health` endpoint.

**Response (success)**

```json
{
  "ok": true,
  "status": "ok",
  "backend": "llama-server",
  "models_loaded": false,
  "error": null
}
```

**Response (failure)**

```json
{
  "ok": false,
  "error": "Could not connect to worker at http://localhost:8000"
}
```

---

### Evaluation

#### `POST /api/run`

Start an evaluation and stream progress as Server-Sent Events. A config must
be loaded first via `/api/config/load`.

**Response:** `text/event-stream` — each `data:` line is a JSON object.

**SSE event types**

| `type` | Emitted when | Key fields |
|--------|-------------|-----------|
| `run_start` | A run config begins | `run_id`, `session_id`, `total_entries`, `config_index`, `total_configs`, `retrieval_model`, `generation_model` |
| `entry_result` | Each entry completes | `entry_index`, `total_entries`, `run_id`, `request.claim_id`, `response.output`, `metrics.*`, `inference.*`, `hardware.*` |
| `run_summary` | A run config finishes | `run_id`, `session_id`, `avg_recall`, `avg_precision`, `avg_mrr`, `abstentions`, `total` |
| `stopped` | User stopped the run | partial summary |
| `error` | Unhandled exception | `message` |
| `done` | All configs complete | — |

**Example stream**

```
data: {"type":"run_start","run_id":"mistral_q4","session_id":"mistral_q4_a1b2c3d4","total_entries":50,...}
data: {"type":"entry_result","entry_index":1,"total_entries":50,...}
...
data: {"type":"run_summary","run_id":"mistral_q4","avg_recall":0.82,...}
data: {"type":"done"}
```

---

#### `POST /api/dry-run`

Validate the loaded config and count entries without running inference.

**Response:** `text/event-stream` — emits a single `dry_run_result` event then `done`.

```json
{
  "type": "dry_run_result",
  "total_entries": 50,
  "total_configs": 3,
  "run_ids": ["run1", "run2", "run3"]
}
```

---

#### `POST /api/stop`

Signal the running evaluation to stop after the current entry completes.
Does nothing if no evaluation is running.

**Response**

```json
{ "ok": true }
```

---

### Status

#### `GET /api/status`

Return the server's current state.

**Response**

```json
{
  "config_loaded": true,
  "run_id": null,
  "is_running": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `config_loaded` | `boolean` | Whether a config is loaded |
| `run_id` | `string \| null` | Active run ID if running |
| `is_running` | `boolean` | Whether an evaluation is in progress |

---

## Required environment variables

The orchestrator API reads these from the environment (or `.env`):

| Variable | Required | Description |
|----------|----------|-------------|
| `WORKER_URL` | No | Worker base URL (default: `http://localhost:8000`) |
| `LANGFUSE_BASE_URL` | If using Langfuse | Langfuse instance URL |
| `LANGFUSE_PUBLIC_KEY` | If using Langfuse | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | If using Langfuse | Langfuse secret key |
| `HF_TOKEN` | If gated dataset | HuggingFace access token |
| `LLM_API_KEY` | If remote generation | API key for remote LLM |

If a required variable is missing when `/api/run` is called, the server
returns HTTP 422 with a descriptive error message.

---

## CORS

The server allows requests from `http://localhost:5173` (the Vite dev server).
Adjust `allow_origins` in `orchestrator/src/orchestrator/api.py` if you deploy
the frontend elsewhere.
