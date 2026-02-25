"""Tests for retrieval service ChromaDB integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shared_types.schemas import ChunkingConfig, RetrievalConfig


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 384
    return embedder


@pytest.fixture
def retrieval_config():
    """Standard retrieval config for tests."""
    return RetrievalConfig(
        model="intfloat/multilingual-e5-small",
        quantization="fp16",
        dimensions=384,
        chunking=ChunkingConfig(strategy="fixed", chunk_size=500, chunk_overlap=64),
        k=3,
    )


class TestRetrievalService:
    def test_retrieve_queries_chromadb_collection(self, tmp_path, mock_embedder, retrieval_config):
        """Retrieval service queries the correct ChromaDB collection."""
        with patch("worker.services.retrieval.chromadb") as mock_chromadb:
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["doc1", "doc2"]],
                "documents": [["chunk text 1", "chunk text 2"]],
                "metadatas": [[{"source": "paper1"}, {"source": "paper2"}]],
            }
            mock_client = MagicMock()
            mock_client.get_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            from worker.services.retrieval import RetrievalService

            service = RetrievalService(
                embedder=mock_embedder,
                collections_dir=tmp_path,
                dataset_id="scifact",
            )
            resolved_path = service._registry.get_or_create_collection("scifact", retrieval_config)

            results = service.retrieve("What reduces inflammation?", retrieval_config)

            mock_chromadb.PersistentClient.assert_called_once_with(path=str(resolved_path))
            mock_client.get_collection.assert_called_once()
            assert mock_client.get_collection.call_args[1]["name"] == service.CHROMA_COLLECTION_NAME
            mock_collection.query.assert_called_once()
            call_kwargs = mock_collection.query.call_args[1]
            assert call_kwargs["query_texts"] == ["What reduces inflammation?"]
            assert call_kwargs["n_results"] == 3

            assert len(results) == 2
            assert results[0]["id"] == "doc1"
            assert results[0]["text"] == "chunk text 1"

    def test_retrieve_respects_k_parameter(self, tmp_path, mock_embedder):
        """Retrieval uses the k parameter from config."""
        config = RetrievalConfig(
            model="intfloat/multilingual-e5-small",
            quantization="fp16",
            dimensions=384,
            k=10,
        )

        with patch("worker.services.retrieval.chromadb") as mock_chromadb:
            mock_collection = MagicMock()
            mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
            mock_client = MagicMock()
            mock_client.get_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            from worker.services.retrieval import RetrievalService

            service = RetrievalService(embedder=mock_embedder, collections_dir=tmp_path)
            resolved_path = service._registry.get_or_create_collection("scifact", config)
            service.retrieve("test query", config)

            mock_chromadb.PersistentClient.assert_called_once_with(path=str(resolved_path))
            call_kwargs = mock_collection.query.call_args[1]
            assert call_kwargs["n_results"] == 10

    def test_retrieve_uses_registry_resolved_collection_path(self, tmp_path, mock_embedder, retrieval_config):
        """Retrieval uses a per-collection folder and fixed Chroma collection name."""
        with patch("worker.services.retrieval.chromadb") as mock_chromadb:
            mock_collection = MagicMock()
            mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
            mock_client = MagicMock()
            mock_client.get_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            from worker.services.retrieval import RetrievalService

            service = RetrievalService(
                embedder=mock_embedder,
                collections_dir=tmp_path,
                dataset_id="scifact",
            )
            resolved_path = service._registry.get_or_create_collection("scifact", retrieval_config)
            service.retrieve("test", retrieval_config)

            mock_chromadb.PersistentClient.assert_called_once_with(path=str(resolved_path))
            call_kwargs = mock_client.get_collection.call_args[1]
            assert call_kwargs["name"] == service.CHROMA_COLLECTION_NAME
            assert "embedding_function" in call_kwargs

    def test_retrieve_raises_for_missing_registry_entry(self, tmp_path, mock_embedder, retrieval_config):
        """Retrieval raises a clear error when no registry entry exists."""
        with patch("worker.services.retrieval.chromadb") as mock_chromadb:
            from worker.services.retrieval import RetrievalService

            service = RetrievalService(
                embedder=mock_embedder,
                collections_dir=tmp_path,
                dataset_id="scifact",
            )

            with pytest.raises(ValueError, match="No collection found for dataset"):
                service.retrieve("test", retrieval_config)

            mock_chromadb.PersistentClient.assert_not_called()
