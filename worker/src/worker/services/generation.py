"""Generation service for RAG answer generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from worker.models.generator_http import LlamaServerGenerator
    from worker.models.generator_remote import RemoteGenerator


@dataclass
class GenerationResult:
    """Result from text generation (matches both generator implementations)."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class GeneratorProtocol(Protocol):
    """Protocol defining the generator interface."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = ...,
        temperature: float = ...,
        top_p: float = ...,
        stop: list[str] | None = ...,
    ) -> GenerationResult: ...


# Type alias for generator implementations
GeneratorType = "LlamaServerGenerator | RemoteGenerator"

logger = logging.getLogger(__name__)


class GenerationService:
    """Runs RAG-style answer generation using the configured LLM.

    Supports local (llama-server) and remote generator backends through duck typing.
    """

    def __init__(self, generator: GeneratorType):
        self._generator = generator

    def generate(
        self,
        prompt: str,
        retrieval_chunks: list[dict] | None = None,
        max_tokens: int = 256,
    ) -> GenerationResult:
        """Generate an answer using retrieved context.

        Args:
            prompt: The user's original query/claim.
            retrieval_chunks: List of retrieved chunks with 'text' keys.
            max_tokens: Maximum tokens to generate.

        Returns:
            GenerationResult with text and token counts.
        """
        chunks = retrieval_chunks or []
        logger.debug("Building RAG prompt with %d chunks", len(chunks))
        rag_prompt = self._build_rag_prompt(prompt, chunks)

        result = self._generator.generate(
            prompt=rag_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["</s>", "\n\n---"],
        )
        logger.info(
            "Generated %d tokens (prompt: %d tokens)",
            result.completion_tokens,
            result.prompt_tokens,
        )
        return result

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
        return f"""<s>[INST] Use the following pieces of context to answer the question.
                Context:
                <No context available>

                Question: {query}
                
                Answer: [/INST]"""

    def _format_prompt_with_context(self, query: str, context: str) -> str:
        """Format prompt with retrieved context."""
        return f"""<s>[INST] Use the following pieces of context to answer the question.
                Context:
                {context}

                Question: {query}
                
                Answer: [/INST]"""
