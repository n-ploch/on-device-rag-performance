"""ChromaDB embedding function bridge using local llama.cpp embedder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

if TYPE_CHECKING:
    from worker.models.embedder import Embedder


class LlamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB-compatible embedding function wrapping our Embedder.

    This class implements the chromadb.api.types.EmbeddingFunction protocol,
    allowing ChromaDB to use our local llama.cpp-based embedder for both
    indexing and querying.
    """

    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for a list of documents.

        Args:
            input: List of document strings to embed.

        Returns:
            List of embedding vectors, one per input document.
        """
        embeddings: Embeddings = []
        for doc in input:
            embedding = self._embedder.embed(doc)
            embeddings.append(embedding)
        return embeddings
