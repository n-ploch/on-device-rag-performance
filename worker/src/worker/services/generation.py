"""Generation service wrapper around llama.cpp inference."""

from __future__ import annotations

from shared_types.schemas import RunConfig


class GenerationService:
    """Runs answer generation using the configured LLM."""

    def generate(self, prompt: str, run_config: RunConfig, retrieval_chunks: list[dict] | None = None) -> str:
        _ = run_config
        _ = retrieval_chunks
        return f"Generated answer: {prompt}"
