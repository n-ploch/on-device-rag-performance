<!-- RAGrig skill dispatcher for Claude Code -->
<!-- Canonical skills are in skills/ — do not edit workflow content here -->

You are operating RAGrig, a distributed RAG evaluation system (Worker on port 8000 + Orchestrator on port 8080). Choose the right workflow:

- `/ragrig-setup` — bootstrap: venv, start worker, load GGUF models, build ChromaDB collection
- `/ragrig-evaluate` — pre-flight checks, run evaluation via CLI or API, interpret results
- `/ragrig-analyze` — export Langfuse traces to Parquet, explore metrics in notebooks
- `/ragrig-troubleshoot` — diagnose failures, consult failure mode table

Not sure which to use? Describe what you want to do and I'll pick the right workflow.

For project architecture and API reference: see `AGENTS.md`.
