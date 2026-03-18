<!-- RAGrig skill dispatcher for Claude Code -->
<!-- Canonical skills are in skills/ — do not edit workflow content here -->

You are operating RAGrig, a distributed RAG evaluation system (Worker on port 8000 + Orchestrator on port 8080).

## Standard workflow (always follow this order)

1. **`/ragrig-config`** — create a config YAML for your experiment (required first step)
2. **`/ragrig-setup`** — start the worker, load models, build ChromaDB collection
3. **`/ragrig-evaluate`** — run the evaluation, interpret results
4. **`/ragrig-analyze`** — export Langfuse traces to Parquet, explore metrics

**If the user asks to run an evaluation and no config file exists, always start
with `/ragrig-config` before anything else.** Do not use the sample configs for
real experiments. Config options are documented in `docs/config.md`.

## Other workflows

- `/ragrig-troubleshoot` — diagnose failures, consult failure mode table

Not sure which to use? Describe what you want to do and I'll pick the right workflow.

For project architecture and API reference: see `AGENTS.md`.
