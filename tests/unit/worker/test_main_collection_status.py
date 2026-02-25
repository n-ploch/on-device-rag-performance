"""Tests for collection status endpoint behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.routing import APIRoute

from shared_types.schemas import ChunkingConfig, CollectionStatusRequest, RetrievalConfig
from worker.main import create_app
from worker.services.embedding import EmbeddingService


def _collection_status_handler():
    app = create_app()
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == "/collection/status":
            return route.endpoint
    raise AssertionError("Collection status route not found")


@pytest.mark.asyncio
async def test_collection_status_uses_resolved_folder_name(tmp_path):
    handler = _collection_status_handler()
    resolved_path = tmp_path / "multilingual-e5-small__fp16__384_0"

    embedding_service = MagicMock()
    embedding_service.collection_exists.return_value = True
    embedding_service.resolve_collection_path.return_value = resolved_path

    request = CollectionStatusRequest(
        dataset_id="scifact",
        retrieval_config=RetrievalConfig(
            model="intfloat/multilingual-e5-small",
            quantization="fp16",
            dimensions=384,
            chunking=ChunkingConfig(strategy="fixed", chunk_size=500, chunk_overlap=64),
            k=3,
        ),
    )
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(embedding_service=embedding_service)))

    with patch("worker.main.chromadb") as mock_chromadb:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 7
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        response = await handler(request, req)

    assert response.exists is True
    assert response.populated is True
    assert response.collection_name == resolved_path.name
    assert response.chunk_count == 7
    mock_chromadb.PersistentClient.assert_called_once_with(path=str(resolved_path))
    mock_client.get_collection.assert_called_once_with(name=EmbeddingService.CHROMA_COLLECTION_NAME)
