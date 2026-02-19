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

            results = service.retrieve("What reduces inflammation?", retrieval_config)

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
            service.retrieve("test query", config)

            call_kwargs = mock_collection.query.call_args[1]
            assert call_kwargs["n_results"] == 10

    def test_collection_name_includes_model_quant_dimensions(self, tmp_path, mock_embedder, retrieval_config):
        """Collection name follows naming convention."""
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
            service.retrieve("test", retrieval_config)

            collection_name = mock_client.get_collection.call_args[1]["name"]
            assert "multilingual-e5-small" in collection_name
            assert "fp16" in collection_name
            assert "384" in collection_name
