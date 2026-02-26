"""Embedding model client using llama-server HTTP API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LlamaServerEmbedder:
    """HTTP client for llama-server embedding endpoint.

    Provides the same interface as the original Embedder class but communicates
    with a running llama-server instance instead of loading the model in-process.
    """

    def __init__(
        self,
        server_url: str,
        timeout: float = 60.0,
        dimensions: int | None = None,
    ):
        """Initialize the embedder client.

        Args:
            server_url: Base URL of the llama-server (e.g., "http://localhost:8001").
            timeout: HTTP request timeout in seconds.
            dimensions: Expected embedding dimensions (for validation).
                       If None, determined from first embedding.
        """
        self._server_url = server_url.rstrip("/")
        self._client = httpx.Client(base_url=self._server_url, timeout=timeout)
        self._dimensions = dimensions
        logger.info("Initialized embedder client for %s", self._server_url)

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text string.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            RuntimeError: If the server request fails.
        """
        logger.debug("Embedding text of length %d", len(text))

        response = self._client.post(
            "/v1/embeddings",
            json={
                "input": text,
                "model": "embedding",
            },
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Embedding request failed: {response.status_code} - {response.text}"
            )

        data = response.json()
        embedding = data["data"][0]["embedding"]

        # Set dimensions on first call if not provided
        if self._dimensions is None:
            self._dimensions = len(embedding)
            logger.info("Detected embedding dimensions: %d", self._dimensions)

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts.

        Uses the batch embedding endpoint for efficiency.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: If the server request fails.
        """
        if not texts:
            return []

        logger.debug("Batch embedding %d texts", len(texts))

        response = self._client.post(
            "/v1/embeddings",
            json={
                "input": texts,
                "model": "embedding",
            },
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Batch embedding request failed: {response.status_code} - {response.text}"
            )

        data = response.json()

        # Sort by index to ensure correct order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in sorted_data]

        # Set dimensions on first call if not provided
        if self._dimensions is None and embeddings:
            self._dimensions = len(embeddings[0])
            logger.info("Detected embedding dimensions: %d", self._dimensions)

        return embeddings

    @property
    def dimensions(self) -> int:
        """Return the embedding dimension size.

        Raises:
            RuntimeError: If dimensions haven't been determined yet
                         (no embeddings generated).
        """
        if self._dimensions is None:
            raise RuntimeError(
                "Embedding dimensions not yet determined. "
                "Generate at least one embedding first."
            )
        return self._dimensions

    @property
    def server_url(self) -> str:
        """Return the server URL."""
        return self._server_url

    def health_check(self) -> bool:
        """Check if the server is healthy.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            response = self._client.get("/health")
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "ok"
        except httpx.RequestError:
            pass
        return False

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self._client.close()
        except Exception:
            pass
