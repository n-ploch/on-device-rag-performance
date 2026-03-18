# Configuration Reference

RAGrig is driven by a single YAML file passed to `rag-orchestrator --config <path>`.
The default path is `config/config.yaml`.

A minimal working example is at [`config/sample_config_quick.yaml`](../config/sample_config_quick.yaml).
An annotated example of every parameter is at [`config/sample_config_full.yaml`](../config/sample_config_full.yaml).

---

## Top-level structure

```yaml
dataset:      { ... }   # required — which dataset to benchmark against
observability: { ... }  # optional — where to send traces and logs
server:        { ... }  # optional — llama-server hardware/performance settings
run_configs:   [ ... ]  # required — one or more benchmark runs
```

---

## `dataset`

Specifies the evaluation dataset. Downloaded from HuggingFace on first use and
cached in `local/`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | **required** | Unique dataset identifier used in traces (e.g. `"ragbench_emanual"`). Defining this as a YAML anchor (`&dataset_id`) lets `run_configs` reference it. |
| `name` | string | **required** | Loader name. Supported values: `"ragbench"`. Each dataset needs a loader and to comply with minimal interface requirements |
| `limits.corpus` | int \| null | `null` | Cap on corpus documents to load. `null` = load all. |
| `limits.ground_truth` | int \| null | `null` | Cap on ground truth entries to load. `null` = load all. |

---

## `observability`

Controls tracing backends and local log output. All fields are optional.

### `backends`

A list of OTEL export targets. Multiple backends can be enabled simultaneously.
Credentials are always read from environment variables at runtime — never put
secrets in the config file.

| `type` | Required env vars | Optional env vars |
|--------|-------------------|-------------------|
| `langfuse` | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | — |
| `weave` | `WANDB_API_KEY`, `WANDB_BASE_URL` `WANDB_ENTITY` `WANDB_PROJECT` | — |
| `generic` | `OTEL_EXPORTER_OTLP_ENDPOINT` | `OTEL_EXPORTER_OTLP_HEADERS` |

The `generic` backend accepts any OTLP/HTTP endpoint. Set the endpoint URL via
`OTEL_EXPORTER_OTLP_ENDPOINT` and pass auth headers via
`OTEL_EXPORTER_OTLP_HEADERS` as a comma-separated list of `key=value` pairs:

```
OTEL_EXPORTER_OTLP_ENDPOINT=https://your-backend.example.com/v1/traces
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer your-api-key
```

Multiple headers:

```
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer token123,x-custom-header=somevalue
```
If the OTLP Backend requires its specific attributes in the TracerObject like Weave, this is not supported with the generic type as of now.

Each entry in the list:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | **required** | One of `"langfuse"`, `"weave"`, `"generic"`. |
| `enabled` | bool | `true` | Set to `false` to disable without removing the entry. |

### Other observability fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_jsonl` | string \| null | `null` | File path for a local JSONL export of every span. Useful for offline analysis. |
| `sys_logs_path` | string \| null | `null` | File path for Python log output. `null` = log to stdout only. |
| `print_logs` | bool | `true` | Whether to echo log lines to stdout. |

---

## `server`

Hardware and performance settings for the two llama-server processes (embedding
and generation). All fields are optional — omitting the entire `server:` block
uses the defaults shown below, which match the previous hardcoded behaviour.

Context sizes are specified per server type because their optimal values differ:
the embedding server typically needs 512 tokens (one chunk), while the
generation server needs enough room for the system prompt, all retrieved chunks,
and the model's answer.

| Field | Type | Default | llama-server flag | Description |
|-------|------|---------|-------------------|-------------|
| `embedding_n_ctx` | int | `512` | `-c` | Context window for the embedding server. Should be ≥ `chunk_size`. |
| `generation_n_ctx` | int | `2048` | `-c` | Context window for the generation server. Must fit: system prompt + `k` chunks + answer. |
| `n_gpu_layers` | int | `-1` | `-ngl` | Layers to offload to GPU. `-1` = all layers (full GPU), `0` = CPU-only. |
| `n_threads` | int \| null | `null` | `-t` | CPU threads to use. `null` = llama-server default (all physical cores). |
| `n_batch` | int \| null | `null` | `-b` | Logical prompt batch size. `null` = llama-server default (512). |
| `flash_attn` | bool | `false` | `-fa` | Enable flash attention. Speeds up generation on compatible GPUs and Apple Metal. **Do not enable for encoder-only embedding models** (e.g. E5, BERT) — it may cause a startup failure. Applied to the generation server only. |
| `tensor_split` | string \| null | `null` | `-ts` | Comma-separated GPU layer fractions for multi-GPU setups (e.g. `"3,1"`). Only applied when the string contains a comma. |
| `no_kv_offload` | bool | `false` | `-nkvo` | Keep the KV cache in system RAM instead of GPU VRAM. Useful on Apple Silicon when unified memory is under pressure. Applied to the generation server only. |

---

## `run_configs`

A list of benchmark runs executed sequentially. Each run loads a fresh pair of
models onto the worker, then evaluates every ground truth entry.

### Top-level run fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `run_id` | string | **required** | Unique identifier for this run. Appears in every trace and the exported Parquet file. |
| `limit` | int \| null | `null` | Evaluate only the first N ground truth entries. `null` = all entries. |
| `repeat` | int \| null | `null` | Repeat this run N times. Each repetition gets a fresh session ID, enabling variance analysis. `null` = run once. |
| `retrieval` | object | **required** | Retrieval model and chunking settings — see below. |
| `generation` | object | **required** | Generation model and hosting settings — see below. |

### `retrieval`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_id` | string | **required** | Must match the top-level `dataset.id`. Use the YAML anchor (`*dataset_id`) to avoid repetition. |
| `model` | string | **required** | HuggingFace repo for the embedding model (e.g. `"ChristianAzinn/e5-large-v2-gguf"`). |
| `quantization` | string | **required** | GGUF quantization file to use from that repo (e.g. `"q4_k_m"`). |
| `dimensions` | int | **required** | Embedding vector dimensions. Must match the model (e.g. `1024` for E5-large). |
| `k` | int | `3` | Number of chunks to retrieve per query. |
| `chunking` | object \| null | `null` | Chunking strategy. If `null`, the corpus must already be indexed. |

#### `retrieval.chunking`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | string | **required** | `"fixed"` splits by character count; `"char_split"` splits on a delimiter. |
| `chunk_size` | int | — | Characters per chunk. **Required** when `strategy: fixed`. |
| `chunk_overlap` | int | `0` | Overlap in characters between consecutive chunks. |
| `split_sequence` | string | — | Delimiter to split on. **Required** when `strategy: char_split` (e.g. `"\n\n"`). |

### `generation`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | **required** | HuggingFace repo (local) or model name sent to the API (remote). |
| `quantization` | string | `"q4_k_m"` | GGUF quantization file to use. Ignored when `hosting: remote`. |
| `hosting` | string | `"local"` | `"local"` starts a llama-server process on the worker; `"remote"` calls an OpenAI-compatible API. |
| `remote` | object | — | **Required** when `hosting: remote`. See below. |

#### `generation.remote`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_url` | string | **required** | OpenAI-compatible API base URL (e.g. `"https://api.mistral.ai/v1"`). |
| `api_key_env` | string | `"OPENAI_API_KEY"` | Name of the environment variable that holds the API key. |
| `extra_headers` | map | `{}` | Additional HTTP headers to include in every request. |
| `rate_limit_rps` | float \| null | `null` | Maximum requests per second. `null` = unlimited. |

---

## Common patterns

### Sweep over quantizations

Define multiple `run_configs` with the same retrieval settings but different
`generation.quantization` values and unique `run_id`s:

```yaml
run_configs:
  - run_id: "llama3_q4"
    generation: { model: "unsloth/Llama-3.2-3B-Instruct-GGUF", quantization: "q4_k_m", hosting: "local" }
    retrieval: &retrieval_base
      dataset_id: *dataset_id
      model: "ChristianAzinn/e5-large-v2-gguf"
      quantization: "q4_k_m"
      dimensions: 1024
      k: 6
      chunking: { strategy: "fixed", chunk_size: 500, chunk_overlap: 64 }

  - run_id: "llama3_q8"
    generation: { model: "unsloth/Llama-3.2-3B-Instruct-GGUF", quantization: "q8_0", hosting: "local" }
    retrieval: *retrieval_base
```

### Measure variance with `repeat`

Set `repeat: N` to run the same configuration N times. Each repetition is
exported as a separate session in Langfuse, making it easy to compute
confidence intervals over latency and quality metrics.

### CPU-only edge device

```yaml
server:
  n_gpu_layers: 0     # All layers on CPU
  n_threads: 4        # Match physical core count
  n_batch: 256        # Smaller batch reduces peak RAM
```

### Apple Silicon with limited unified memory

```yaml
server:
  n_gpu_layers: -1    # Still offload to Metal GPU
  no_kv_offload: true # Keep KV cache in system RAM to avoid VRAM pressure
  flash_attn: false   # Leave off for embedding models; safe to enable for generation
```
