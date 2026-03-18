---
name: ragrig-analyze
version: "1.0"
description: >
  Export RAGrig evaluation traces from Langfuse to Parquet and interpret
  quality and hardware metrics.
scope: project
apply: on-demand
triggers:
  - "analyze results"
  - "export metrics"
  - "explore traces"
  - "export parquet"
  - "open notebook"
  - "download traces"
arguments:
  - name: session_id
    description: Langfuse session ID or glob pattern (e.g. "*_q4_*")
    required: true
  - name: from_date
    description: Filter traces from this date onward (YYYY-MM-DD)
    required: false
prerequisites:
  - "Langfuse instance reachable at LANGFUSE_BASE_URL"
  - "Evaluation has been run and traces are in Langfuse"
  - "analysis/ package installed: pip install -e analysis/ or pip install -e analysis/[notebooks]"
tools_required: [bash]
compatible_with: [claude-code, cursor, copilot, codex]
---

## STOP and ask the user when

- `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, or `LANGFUSE_SECRET_KEY` is not set
- The Langfuse instance is unreachable
- No session ID is provided and you cannot infer it from context (the session ID is printed to stdout at the end of each evaluation run)

---

## Export traces to Parquet

The session ID is printed to stdout at the end of `rag-orchestrator` runs.

```bash
source .rag/bin/activate
python analysis/langfuse_export.py --session-id <session_id>
```

Output: `local/metric-export/YYYY-MM-DD_HH-MM_langfuse_export_<session_id>.parquet`

**Important:** The local Langfuse deployment only supports the v1 API. The export
script uses `FernLangfuse` (v1 SDK) for observations and direct `httpx` calls to
`/api/public/scores`. Do not use SDK v3 `score_v_2` — it calls the v2 endpoint
which does not exist on local deployments.

---

## Explore in a notebook

```bash
source .rag/bin/activate
jupyter lab notebooks/01_explore_metrics.ipynb
```

Install notebook extras if needed:

```bash
pip install -e analysis/[notebooks]
```

---

## Parquet columns reference

After export, the Parquet file has one row per evaluation entry. Key columns:

| Column | Source span attribute | Description |
|--------|-----------------------|-------------|
| `metrics_recall_at_k` | `custom.metrics_recall_at_k` | Fraction of relevant docs in top-k |
| `metrics_precision_at_k` | `custom.metrics_precision_at_k` | Precision at k |
| `metrics_mrr` | `custom.metrics_mrr` | Mean reciprocal rank |
| `metrics_abstention` | `custom.metrics_abstention` | 1 if model abstained |
| `latency_e2e_latency_ms` | `custom.latency_e2e_latency_ms` | End-to-end request latency |
| `latency_ttft_ms` | `custom.latency_ttft_ms` | Time to first token |
| `latency_tokens_per_second` | `custom.generation_tokens_per_second` | Generation throughput |
| `hardware_max_ram_usage_mb` | `custom.hardware_max_ram_usage_mb` | Peak RAM during generation |
| `hardware_avg_cpu_utilization_pct` | `custom.hardware_avg_cpu_utilization_pct` | Mean CPU % |
| `hardware_peak_cpu_temp_c` | `custom.hardware_peak_cpu_temp_c` | Peak CPU temperature |
