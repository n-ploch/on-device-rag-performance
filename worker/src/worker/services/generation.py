"""Generation service using llama.cpp for RAG answer generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worker.models.generator import Generator, GenerationResult


class GenerationService:
    """Runs RAG-style answer generation using the configured LLM."""

    def __init__(self, generator: Generator):
        self._generator = generator

    def generate(
        self,
        prompt: str,
        retrieval_chunks: list[dict] | None = None,
        max_tokens: int = 256,
    ) -> "GenerationResult":
        """Generate an answer using retrieved context.

        Args:
            prompt: The user's original query/claim.
            retrieval_chunks: List of retrieved chunks with 'text' keys.
            max_tokens: Maximum tokens to generate.

        Returns:
            GenerationResult with text and token counts.
        """
        rag_prompt = self._build_rag_prompt(prompt, retrieval_chunks or [])

        return self._generator.generate(
            prompt=rag_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["</s>", "\n\n---"],
        )

    def _build_rag_prompt(self, query: str, chunks: list[dict]) -> str:
        """Build a RAG prompt with retrieved context."""
        if not chunks:
            return self._format_prompt_no_context(query)

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            context_parts.append(f"[{i}] {text}")

        context_block = "\n\n".join(context_parts)

        return self._format_prompt_with_context(query, context_block)

    def _format_prompt_no_context(self, query: str) -> str:
        """Format prompt when no context is available."""
        return f"""<s>[INST] Answer the following question. If you cannot determine the answer, say "I don't know" or "Insufficient information".

Question: {query}

Answer: [/INST]"""

    def _format_prompt_with_context(self, query: str, context: str) -> str:
        """Format prompt with retrieved context."""
        return f"""<s>[INST] Use the following context to answer the question. If the context doesn't contain enough information, say "I don't know" or "Insufficient information".

Context:
{context}

Question: {query}

Answer: [/INST]"""
