"""Embedding model wrapper using llama-cpp-python."""

from __future__ import annotations

from pathlib import Path

from llama_cpp import Llama


class Embedder:
    """Wraps llama-cpp-python for embedding generation.

    IMPORTANT: Must be instantiated with embedding=True to enable the embedding
    endpoint in the llama.cpp backend. Using a small n_ctx (512) is sufficient
    for embedding tasks and reduces memory footprint.
    """

    def __init__(self, model_path: Path, n_ctx: int = 512, n_gpu_layers: int = -1):
        self._model = Llama(
            model_path=str(model_path),
            embedding=True,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text string."""
        result = self._model.embed(text)
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                return result[0]
            return result
        raise ValueError("Unexpected embedding result format")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts."""
        return [self.embed(text) for text in texts]

    @property
    def dimensions(self) -> int:
        """Return the embedding dimension size."""
        return self._model.n_embd()
