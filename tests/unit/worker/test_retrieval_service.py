"""Tests for retrieval service collection lifecycle integration."""

from __future__ import annotations

from unittest.mock import Mock

from shared_types.schemas import ChunkingConfig, RetrievalConfig
from worker.services.retrieval import RetrievalService


def test_worker_retrieval_service_gets_or_creates_collection():
    registry = Mock()
    service = RetrievalService(dataset_id="scifact", collection_registry=registry)
    config = RetrievalConfig(
        model="intfloat/multilingual-e5-small",
        quantization="fp16",
        dimensions=384,
        chunking=ChunkingConfig(strategy="fixed", chunk_size=500, chunk_overlap=64),
        k=3,
    )

    result = service.retrieve("What reduces inflammation?", config)

    registry.get_or_create_collection.assert_called_once_with("scifact", config)
    assert result == []
