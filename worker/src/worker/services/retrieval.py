"""Retrieval service wrapper around collection registry and vector DB."""

from __future__ import annotations

from shared_types.schemas import RetrievalConfig
from worker.services.collection_registry import CollectionRegistry


class RetrievalService:
    """Handles retrieval collection resolution and query execution."""

    def __init__(self, dataset_id: str = "scifact", collection_registry: CollectionRegistry | None = None):
        self.dataset_id = dataset_id
        self.collection_registry = collection_registry or CollectionRegistry()

    def retrieve(self, query: str, retrieval_config: RetrievalConfig) -> list[dict]:
        """Resolve a collection and return retrieved chunks.

        Retrieval execution against Chroma is intentionally stubbed here; tests focus on
        lifecycle and contracts, not ANN behavior.
        """

        _ = query
        self.collection_registry.get_or_create_collection(self.dataset_id, retrieval_config)
        return []
