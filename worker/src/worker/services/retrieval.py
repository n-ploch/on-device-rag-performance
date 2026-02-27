"""Retrieval service using ChromaDB with local embeddings."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb

logger = logging.getLogger(__name__)

from shared_types.schemas import RetrievalConfig
from worker.services.chromadb_bridge import LlamaEmbeddingFunction
from worker.services.collection_registry import CollectionRegistry

if TYPE_CHECKING:
    from worker.models.embedder import Embedder


class RetrievalService:
    """Handles retrieval using pre-built ChromaDB collections."""

    CHROMA_COLLECTION_NAME = "chunks"

    def __init__(
        self,
        embedder: Embedder,
        collections_dir: Path | None = None,
    ):
        self._embedder = embedder
        self._embedding_fn = LlamaEmbeddingFunction(embedder)

        if collections_dir is None:
            collections_dir = Path(os.environ["LOCAL_COLLECTIONS_DIR"])

        self._registry = CollectionRegistry(collections_dir)
        self._client_cache: dict[str, Any] = {}
        self._collection_cache: dict[str, chromadb.Collection] = {}

    def _get_collection(self, retrieval_config: RetrievalConfig) -> chromadb.Collection:
        """Get or cache a ChromaDB collection for the given config.

        The dataset_id is extracted from retrieval_config.dataset_id.
        """
        dataset_id = retrieval_config.dataset_id
        collection_path = self._registry.resolve_collection_path(dataset_id, retrieval_config)
        if collection_path is None:
            raise ValueError(
                "No collection found for dataset "
                f"'{dataset_id}' and retrieval config model='{retrieval_config.model}', "
                f"quantization='{retrieval_config.quantization}', dimensions={retrieval_config.dimensions}."
            )

        cache_key = str(collection_path)
        if cache_key not in self._collection_cache:
            logger.debug("Loading collection from path: %s", collection_path)
            if cache_key not in self._client_cache:
                self._client_cache[cache_key] = chromadb.PersistentClient(path=cache_key)
            client = self._client_cache[cache_key]
            self._collection_cache[cache_key] = client.get_collection(
                name=self.CHROMA_COLLECTION_NAME,
                embedding_function=self._embedding_fn,
            )

        return self._collection_cache[cache_key]

    def retrieve(self, query: str, retrieval_config: RetrievalConfig) -> list[dict]:
        """Query ChromaDB and return retrieved chunks with metadata.

        Args:
            query: The user's query text.
            retrieval_config: Retrieval parameters including model, quantization, k.

        Returns:
            List of dicts with 'id' and 'text' keys for each retrieved chunk.
        """
        collection = self._get_collection(retrieval_config)
        logger.debug("ChromaDB query: k=%d, query_len=%d", retrieval_config.k, len(query))

        results = collection.query(
            query_texts=[query],
            n_results=retrieval_config.k,
            include=["documents", "metadatas"],
        )

        retrieved: list[dict] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i, doc_id in enumerate(ids):
            text = documents[i] if i < len(documents) else ""
            metadata = metadatas[i] if i < len(metadatas) else {}
            retrieved.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata,
            })

        logger.info("Retrieved %d chunks", len(retrieved))
        return retrieved
