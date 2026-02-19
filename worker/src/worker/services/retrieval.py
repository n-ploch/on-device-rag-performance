"""Retrieval service using ChromaDB with local embeddings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb

from shared_types.naming import collection_base_key
from shared_types.schemas import RetrievalConfig
from worker.services.chromadb_bridge import LlamaEmbeddingFunction

if TYPE_CHECKING:
    from worker.models.embedder import Embedder


class RetrievalService:
    """Handles retrieval using pre-built ChromaDB collections."""

    def __init__(
        self,
        embedder: Embedder,
        collections_dir: Path | None = None,
        dataset_id: str = "scifact",
    ):
        self._embedder = embedder
        self._dataset_id = dataset_id
        self._embedding_fn = LlamaEmbeddingFunction(embedder)

        if collections_dir is None:
            collections_dir = Path(os.environ["LOCAL_COLLECTIONS_DIR"])

        self._client = chromadb.PersistentClient(path=str(collections_dir))
        self._collection_cache: dict[str, chromadb.Collection] = {}

    def _get_collection(self, retrieval_config: RetrievalConfig) -> chromadb.Collection:
        """Get or cache a ChromaDB collection for the given config."""
        collection_name = self._resolve_collection_name(retrieval_config)

        if collection_name not in self._collection_cache:
            self._collection_cache[collection_name] = self._client.get_collection(
                name=collection_name,
                embedding_function=self._embedding_fn,
            )

        return self._collection_cache[collection_name]

    def _resolve_collection_name(self, retrieval_config: RetrievalConfig) -> str:
        """Resolve collection name from config.

        Collections are named: {model}__{quantization}__{dimensions}_{index}
        For pre-built collections, we use index 0.
        """
        base = collection_base_key(
            retrieval_config.model,
            retrieval_config.quantization,
            retrieval_config.dimensions,
        )
        return f"{self._dataset_id}_{base}_0"

    def retrieve(self, query: str, retrieval_config: RetrievalConfig) -> list[dict]:
        """Query ChromaDB and return retrieved chunks with metadata.

        Args:
            query: The user's query text.
            retrieval_config: Retrieval parameters including model, quantization, k.

        Returns:
            List of dicts with 'id' and 'text' keys for each retrieved chunk.
        """
        collection = self._get_collection(retrieval_config)

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

        return retrieved
