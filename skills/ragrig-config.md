---
name: ragrig-config
version: "1.0"
description: >
  Interactively guide the user through creating a RAGrig evaluation config file
  (YAML) step by step, then write it to config/. Full reference: docs/config.md.
scope: project
apply: on-demand
triggers:
  - "create config"
  - "new config"
  - "configure evaluation"
  - "set up config"
  - "make a config"
  - "write config"
  - "config file"
arguments:
  - name: output
    description: Output path for the generated config file
    required: false
    default: "config/my_experiment.yaml"
prerequisites:
  - ".env exists with LOCAL_MODELS_DIR set (needed to verify model files exist)"
tools_required: [bash, read_file, write_file]
compatible_with: [claude-code, cursor, copilot, codex]
---

## Purpose

**Creating a config file is the required first step before any evaluation.**
This skill walks the user through the YAML interactively, section by section,
offering sensible defaults so they can move quickly or configure every detail.

Full config reference: [`docs/config.md`](../docs/config.md)
Sample files: [`config/sample_config_quick.yaml`](../config/sample_config_quick.yaml),
[`config/sample_config_full.yaml`](../config/sample_config_full.yaml)

---

## STOP and ask the user when

- You do not know what output filename to use
- The user's device type is unclear (needed to suggest server presets)
- The user wants a remote generation backend but has not provided a `base_url`
- The user skips a **required** field (marked below)
- You are about to write the config file (always ask for confirmation first)

---

## Step 0 — Paste or interactive?

**Always start here.** Ask:

> "You can paste your key settings (models, observability backend, entry limit,
> etc.) and I'll build the config for you — or I can walk you through it
> interactively. Which do you prefer?"

- **Paste** → the user provides a free-form summary (e.g. "use Llama-3.2-1B q4,
  local, langfuse, 10 entries"). Extract all recognisable fields, fill gaps with
  defaults, show the assembled YAML for review, then ask: *"Does this look right?
  I'll write it to `<output_path>`."* Do **not** write without confirmation.
- **Interactive / quick** → follow the [Quick path](#quick-path) below
- **Interactive / full** → follow the [Full path](#full-path) below

If the user doesn't answer or says "quick / just go / defaults / yes", take the
quick path.

---

## Quick path

Collect only what is strictly necessary, then write the file.

### Q1 — Run ID

Ask: *"What should this run be called? (e.g. `llama3_q4_test`)"*
Default if skipped: `"my_run"`.

### Q2 — Retrieval model and chunking

Ask: *"Which embedding model should be used for retrieval?
- `ChristianAzinn/e5-large-v2-gguf` (recommended, 1024-dim)
- Or paste any HuggingFace GGUF embedding repo"*

Default: `"ChristianAzinn/e5-large-v2-gguf"`.

Follow-up: *"How many chunks (k) should be retrieved per query?"*
Default: `5`.

Follow-up: *"Chunk size in characters? (chunk overlap defaults to 64)"*
Default: `500`.

Note: if the user picks a different embedding model, remind them to verify
`dimensions` matches that model's output size (default 1024 is for E5-large).

### Q3 — Local or remote generation?

Ask: *"Should generation run locally on the worker, or via a remote
OpenAI-compatible API?"*
Default: `"local"`.

**If local**, ask:
- *"Which generation model?"*
  Common choices:
  - `unsloth/Llama-3.2-3B-Instruct-GGUF` (small, fast, recommended for edge)
  - `unsloth/Llama-3.2-1B-Instruct-GGUF` (tiny, CPU-friendly)
  - `mistralai/Ministral-3B-Instruct-2410-GGUF` (Mistral 3B)
  - Or paste any HuggingFace GGUF repo
  Default: `"unsloth/Llama-3.2-3B-Instruct-GGUF"`.
- *"Quantization level? (`q4_k_m` is a good default, `q8_0` for higher quality,
  `q2_k` for minimum RAM)"*
  Default: `"q4_k_m"`.

**If remote**, ask:
- `base_url` (**required** — e.g. `https://api.mistral.ai/v1`)
- Model name as the API expects it (e.g. `mistral-small-latest`) (**required**)
- `api_key_env` — name of the env var that holds the API key
  Default: `"OPENAI_API_KEY"`.
- Then **check `.env`**: confirm the named env var is set and non-empty.
  If missing, warn the user and note it must be added before evaluation.

### Q4 — Observability backend

Ask: *"Which observability backend do you want?
- `langfuse` (local Langfuse — requires `LANGFUSE_*` env vars)
- `weave` (Weights & Biases Weave — requires `WANDB_API_KEY`)
- `generic` (any OTLP endpoint — requires `OTEL_EXPORTER_OTLP_ENDPOINT`)
- `none` (no remote traces, JSONL only)"*

Default: `"langfuse"`.

If the user picks `none`, set `backends: []` and enable `output_jsonl`.

For any backend other than `none`, check `.env` for the required vars (see
table in the Full path Block 2) and warn if any are missing.

### Q5 — Entry limit

Ask: *"How many ground truth entries should this run evaluate?
Enter a number for a quick test (e.g. `10`), or press Enter to evaluate all."*

Default: `null` (all entries).

### Write the file

Assemble the YAML using the template below, **show the user the full file
content**, and ask: *"Does this look right? I'll write it to `<output_path>`."*
Do **not** write the file until the user confirms.

```yaml
# Generated by ragrig-config
# Full reference: docs/config.md

dataset:
  id: &dataset_id "ragbench_emanual"
  name: "ragbench"
  limits:
    corpus: null
    ground_truth: <Q5_limit>

observability:
  backends:
    - type: "<Q4_backend>"      # omit this block entirely if none
      enabled: true
  output_jsonl: "./logs/traces.jsonl"
  sys_logs_path: "./logs/runner.log"
  print_logs: true

# server: block omitted — llama-server defaults apply
# Edit or add a server: block for GPU/CPU tuning (see docs/config.md)

run_configs:

  - run_id: "<Q1_run_id>"
    limit: <Q5_limit>
    retrieval:
      dataset_id: *dataset_id
      model: "<Q2_embedding_model>"
      quantization: "q4_k_m"
      dimensions: 1024
      k: <Q2_k>
      chunking:
        strategy: "fixed"
        chunk_size: <Q2_chunk_size>
        chunk_overlap: 64
    generation:
      model: "<Q3_model>"
      quantization: "<Q3_quantization>"   # omit when hosting: remote
      hosting: "<Q3_hosting>"
      # Uncomment and fill when hosting: remote:
      # remote:
      #   base_url: "<Q3_base_url>"
      #   api_key_env: "<Q3_api_key_env>"
          # extra_headers:                          # [optional: {}] Additional HTTP headers
          #   Content-Type: "application/json"
          # rate_limit_rps: 1.0                    # [optional: null] Max requests/second (null = unlimited)
```

After writing, tell the user:
> "Config written to `<output_path>`. Run `rag-orchestrator --config <output_path> --dry-run`
> to validate, then `/ragrig-evaluate` (or `/ragrig-setup` first if the worker
> isn't running yet)."

---

## Full path

Walk through each top-level YAML block in order. After each block, show a
running preview of what has been collected so far.

### Block 1 — `dataset`

Explain: *"The dataset block defines what corpus and ground truth to evaluate
against. RAGrig currently ships a loader for `ragbench`. The `id` is a free-form
label that appears in every trace."*

| Question | Default |
|----------|---------|
| Dataset ID (label used in traces) | `"ragbench_emanual"` |
| Loader name | `"ragbench"` |
| Corpus limit (null = all) | `null` |
| Ground truth limit (null = all) | `null` |

Recommend setting `ground_truth: 10` for a first test run to keep it fast.

### Block 2 — `observability`

Explain: *"Observability backends receive OTEL spans from every evaluation entry.
You can enable multiple at once. Credentials are always read from env vars — never
put secrets in the YAML."*

**Backend selection** (multi-select, can pick several or none):

| Backend | Required env vars | Notes |
|---------|-------------------|-------|
| `langfuse` | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | Local or cloud |
| `weave` | `WANDB_API_KEY`, `WANDB_BASE_URL`, `WANDB_ENTITY`, `WANDB_PROJECT` | W&B cloud |
| `generic` | `OTEL_EXPORTER_OTLP_ENDPOINT` | Any OTLP/HTTP endpoint |

For each selected backend, check that the required env vars are set in `.env`.
Warn (but do not block) if any are missing.

Additional observability fields:

| Question | Default |
|----------|---------|
| Local JSONL output path (null = disabled) | `"./logs/traces.jsonl"` |
| Python log file path (null = stdout only) | `"./logs/runner.log"` |
| Print logs to terminal? | `true` |

### Block 3 — `server`

Explain: *"The `server` block controls the two llama-server processes (embedding
and generation). Omitting it entirely uses safe defaults that work on most
hardware. I can suggest a preset based on your device."*

Ask: *"What type of device is the worker running on?"*
- **NVIDIA GPU** → suggest `n_gpu_layers: -1`, `flash_attn: true`
- **Apple Silicon** → suggest `n_gpu_layers: -1`, `flash_attn: false`, `no_kv_offload: true` if RAM is tight
- **CPU only** → suggest `n_gpu_layers: 0`, ask for core count → set `n_threads`
- **Skip / use defaults** → omit the `server:` block entirely

Show the preset and ask if the user wants to adjust any individual field before
continuing. Reference the full field table in [`docs/config.md § server`](../docs/config.md#server).

### Block 4 — `run_configs`

Explain: *"Each entry in `run_configs` is one benchmark run. Runs execute
sequentially; each loads a fresh pair of models. You can define multiple runs to
sweep quantizations or compare models."*

For each run (keep asking "add another run?" until the user says no):

**Run-level fields:**

| Question | Default |
|----------|---------|
| `run_id` (unique label) | `"run_1"` |
| Entry limit (null = all) | `null` |
| Repeat N times (null = once) | `null` |

**Retrieval fields:**

| Question | Default |
|----------|---------|
| Embedding model (HF repo) | `"ChristianAzinn/e5-large-v2-gguf"` |
| Quantization | `"q4_k_m"` |
| Dimensions | `1024` (for E5-large) |
| k (chunks to retrieve) | `5` |
| Chunking strategy (`fixed` / `char_split`) | `"fixed"` |
| Chunk size (chars, for `fixed`) | `500` |
| Chunk overlap (chars) | `64` |

Note: if the user picks a different embedding model, remind them to set
`dimensions` to match that model's output size.

**Generation — hosting first:**

Ask: *"Should generation run locally on the worker, or via a remote
OpenAI-compatible API?"*
Default: `"local"`.

If **local**, ask:

| Question | Default |
|----------|---------|
| Model (HF repo) | `"unsloth/Llama-3.2-3B-Instruct-GGUF"` |
| Quantization | `"q4_k_m"` |

If **remote**, ask:

| Question | Notes |
|----------|-------|
| `base_url` | **required** (e.g. `https://api.mistral.ai/v1`) |
| Model name (as the API expects it) | **required** (e.g. `mistral-small-latest`) |
| `api_key_env` | env var name holding the key (default: `"OPENAI_API_KEY"`) |
| `rate_limit_rps` | requests per second cap (default: `null`) |

Then check `.env` for the named `api_key_env` variable. Warn (but do not block)
if it is missing or empty.

### Write the file

After all blocks are collected, render the complete YAML. Show the user the
final file content for review and ask "Does this look right? I'll write it to
`<output_path>`."

After confirming, write the file. Then tell the user:
> "Config written to `<output_path>`. Next steps:
> 1. Validate: `rag-orchestrator --config <output_path> --dry-run`
> 2. If the worker is not running: `/ragrig-setup --config <output_path>`
> 3. Run evaluation: `/ragrig-evaluate --config <output_path>`"

---

## YAML anchors tip

Remind the user that YAML anchors avoid repeating the dataset ID across run
configs. The template already uses `&dataset_id` on the `dataset.id` line and
`*dataset_id` in each `retrieval.dataset_id` — keep this pattern when editing
manually.
