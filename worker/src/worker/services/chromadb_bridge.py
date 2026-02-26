"""ChromaDB embedding function bridge supporting local and HTTP embedders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

if TYPE_CHECKING:
    from worker.models.embedder import Embedder
    from worker.models.embedder_http import LlamaServerEmbedder


class EmbedderProtocol(Protocol):
    """Protocol defining the embedder interface."""

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# Type alias for embedder implementations
EmbedderType = Union["Embedder", "LlamaServerEmbedder"]


class LlamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB-compatible embedding function wrapping an embedder.

    This class implements the chromadb.api.types.EmbeddingFunction protocol,
    allowing ChromaDB to use our embedder (either local llama.cpp-based or
    HTTP-based llama-server) for both indexing and querying.

    Supports both the original in-process Embedder and the new HTTP-based
    LlamaServerEmbedder through duck typing.
    """

    def __init__(self, embedder: EmbedderType):
        """Initialize with any embedder implementing the EmbedderProtocol.

        Args:
            embedder: An embedder instance with embed() and embed_batch() methods.
                     Can be either Embedder (in-process) or LlamaServerEmbedder (HTTP).
        """
        self._embedder = embedder

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for a list of documents.

        Uses batch embedding when available for efficiency.

        Args:
            input: List of document strings to embed.

        Returns:
            List of embedding vectors, one per input document.
        """
        # Use batch embedding for efficiency
        return self._embedder.embed_batch(list(input))
