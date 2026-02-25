"""Tests for EmbeddingService."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shared_types import CorpusDocument
from shared_types.schemas import ChunkingConfig, RetrievalConfig
from worker.services.embedding import EmbeddingProgress, EmbeddingResult, EmbeddingService


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 384
    embedder.embed_batch.return_value = [[0.1] * 384, [0.2] * 384]
    return embedder


@pytest.fixture
def sample_corpus():
    """Sample corpus documents for testing."""
    return [
        CorpusDocument(id="doc1", text="First document with some text content.", metadata={}),
        CorpusDocument(id="doc2", text="Second document with more text.", metadata={}),
        CorpusDocument(id="doc3", text="Third document text.", metadata={}),
    ]


@pytest.fixture
def retrieval_config():
    """Standard retrieval config for tests."""
    return RetrievalConfig(
        model="intfloat/multilingual-e5-small",
        quantization="fp16",
        dimensions=384,
        chunking=ChunkingConfig(strategy="fixed", chunk_size=20, chunk_overlap=5),
        k=3,
    )


@pytest.fixture
def retrieval_config_no_chunking():
    """Retrieval config without chunking."""
    return RetrievalConfig(
        model="intfloat/multilingual-e5-small",
        quantization="fp16",
        dimensions=384,
        k=3,
    )


class TestEmbeddingService:
    def test_embed_corpus_creates_collection(self, tmp_path, mock_embedder, sample_corpus, retrieval_config):
        """embed_corpus creates a ChromaDB collection."""
        with patch("worker.services.embedding.chromadb") as mock_chromadb:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client = MagicMock()
            mock_client.get_collection.side_effect = ValueError("Not found")
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            service = EmbeddingService(embedder=mock_embedder, collections_dir=tmp_path)
            result = service.embed_corpus(
                corpus=sample_corpus,
                dataset_id="scifact",
                retrieval_config=retrieval_config,
            )

            assert isinstance(result, EmbeddingResult)
            assert result.total_documents == 3
            assert result.already_existed is False
            assert result.collection_path.parent == tmp_path
            assert result.collection_name == result.collection_path.name
            mock_chromadb.PersistentClient.assert_called_once_with(path=str(result.collection_path))
            mock_client.get_or_create_collection.assert_called_once()
            assert (
                mock_client.get_or_create_collection.call_args[1]["name"]
                == EmbeddingService.CHROMA_COLLECTION_NAME
            )
            mock_collection.add.assert_called()

    def test_embed_corpus_skips_existing_collection(
        self, tmp_path, mock_embedder, sample_corpus, retrieval_config
    ):
        """embed_corpus returns early if collection already exists."""
        with patch("worker.services.embedding.chromadb") as mock_chromadb:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 10  # Non-empty
            mock_client = MagicMock()
            mock_client.get_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            service = EmbeddingService(embedder=mock_embedder, collections_dir=tmp_path)
            result = service.embed_corpus(
                corpus=sample_corpus,
                dataset_id="scifact",
                retrieval_config=retrieval_config,
            )

            assert result.already_existed is True
            assert result.total_chunks == 10
            assert result.collection_path.parent == tmp_path
            assert result.collection_name == result.collection_path.name
            mock_chromadb.PersistentClient.assert_called_once_with(path=str(result.collection_path))
            mock_collection.add.assert_not_called()

    def test_embed_corpus_with_no_chunking(
        self, tmp_path, mock_embedder, sample_corpus, retrieval_config_no_chunking
    ):
        """embed_corpus uses full documents when no chunking specified."""
        with patch("worker.services.embedding.chromadb") as mock_chromadb:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client = MagicMock()
            mock_client.get_collection.side_effect = ValueError("Not found")
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            service = EmbeddingService(embedder=mock_embedder, collections_dir=tmp_path)
            result = service.embed_corpus(
                corpus=sample_corpus,
                dataset_id="scifact",
                retrieval_config=retrieval_config_no_chunking,
            )

            # Without chunking, chunks == documents
            assert result.total_chunks == 3

    def test_collection_exists_returns_true_for_populated(
        self, tmp_path, mock_embedder, retrieval_config
    ):
        """collection_exists returns True when collection has documents."""
        with patch("worker.services.embedding.chromadb") as mock_chromadb:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 10
            mock_client = MagicMock()
            mock_client.get_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            service = EmbeddingService(embedder=mock_embedder, collections_dir=tmp_path)
            resolved_path = service._registry.get_or_create_collection("scifact", retrieval_config)
            result = service.collection_exists("scifact", retrieval_config)

            assert result is True
            mock_chromadb.PersistentClient.assert_called_once_with(path=str(resolved_path))
            mock_client.get_collection.assert_called_once_with(name=EmbeddingService.CHROMA_COLLECTION_NAME)

    def test_collection_exists_returns_false_for_missing(
        self, tmp_path, mock_embedder, retrieval_config
    ):
        """collection_exists returns False when collection doesn't exist."""
        with patch("worker.services.embedding.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_client.get_collection.side_effect = ValueError("Not found")
            mock_chromadb.PersistentClient.return_value = mock_client

            service = EmbeddingService(embedder=mock_embedder, collections_dir=tmp_path)
            resolved_path = service._registry.get_or_create_collection("scifact", retrieval_config)
            result = service.collection_exists("scifact", retrieval_config)

            assert result is False
            mock_chromadb.PersistentClient.assert_called_once_with(path=str(resolved_path))

    def test_collection_exists_returns_false_for_chromadb_not_found(
        self, tmp_path, mock_embedder, retrieval_config
    ):
        """collection_exists returns False for Chroma NotFoundError."""
        with patch("worker.services.embedding.chromadb") as mock_chromadb:
            class NotFoundError(Exception):
                pass

            mock_client = MagicMock()
            mock_client.get_collection.side_effect = NotFoundError("Collection missing")
            mock_chromadb.PersistentClient.return_value = mock_client

            service = EmbeddingService(embedder=mock_embedder, collections_dir=tmp_path)
            resolved_path = service._registry.get_or_create_collection("scifact", retrieval_config)
            result = service.collection_exists("scifact", retrieval_config)

            assert result is False
            mock_chromadb.PersistentClient.assert_called_once_with(path=str(resolved_path))

    def test_collection_exists_returns_false_when_registry_has_no_match(
        self, tmp_path, mock_embedder, retrieval_config
    ):
        """collection_exists returns False when registry has no entry."""
        with patch("worker.services.embedding.chromadb") as mock_chromadb:
            service = EmbeddingService(embedder=mock_embedder, collections_dir=tmp_path)
            result = service.collection_exists("scifact", retrieval_config)

            assert result is False
            mock_chromadb.PersistentClient.assert_not_called()

    def test_progress_callback_invoked(
        self, tmp_path, mock_embedder, sample_corpus, retrieval_config_no_chunking
    ):
        """Progress callback is invoked during embedding."""
        progress_updates = []

        def track_progress(progress: EmbeddingProgress):
            progress_updates.append(progress)

        with patch("worker.services.embedding.chromadb") as mock_chromadb:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client = MagicMock()
            mock_client.get_collection.side_effect = ValueError("Not found")
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            service = EmbeddingService(embedder=mock_embedder, collections_dir=tmp_path)
            service.embed_corpus(
                corpus=sample_corpus,
                dataset_id="scifact",
                retrieval_config=retrieval_config_no_chunking,
                batch_size=2,
                progress_callback=track_progress,
            )

            assert len(progress_updates) > 0
            last_update = progress_updates[-1]
            assert last_update.chunks_embedded == 3
            assert last_update.total_chunks == 3

    def test_collection_name_matches_registry_folder_name(
        self, tmp_path, mock_embedder, retrieval_config
    ):
        """Embedding result collection_name is the allocated folder name."""
        with patch("worker.services.embedding.chromadb") as mock_chromadb:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client = MagicMock()
            mock_client.get_collection.side_effect = ValueError("Not found")
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            service = EmbeddingService(embedder=mock_embedder, collections_dir=tmp_path)
            result = service.embed_corpus(
                corpus=[CorpusDocument(id="doc1", text="test", metadata={})],
                dataset_id="scifact",
                retrieval_config=retrieval_config,
            )

            assert result.collection_name == result.collection_path.name
            assert "multilingual-e5-small" in result.collection_name
            assert "fp16" in result.collection_name
            assert "384" in result.collection_name


class TestChunking:
    def test_fixed_chunking_creates_overlapping_chunks(self, tmp_path, mock_embedder):
        """Fixed chunking creates chunks with overlap."""
        corpus = [
            CorpusDocument(
                id="doc1",
                text="0123456789" * 5,  # 50 chars
                metadata={},
            )
        ]
        config = RetrievalConfig(
            model="test",
            quantization="fp16",
            dimensions=384,
            chunking=ChunkingConfig(strategy="fixed", chunk_size=20, chunk_overlap=5),
        )

        with patch("worker.services.embedding.chromadb") as mock_chromadb:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client = MagicMock()
            mock_client.get_collection.side_effect = ValueError("Not found")
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            service = EmbeddingService(embedder=mock_embedder, collections_dir=tmp_path)
            result = service.embed_corpus(
                corpus=corpus,
                dataset_id="test",
                retrieval_config=config,
            )

            # With 50 chars, chunk_size=20, overlap=5:
            # Chunk 0: 0-20, Chunk 1: 15-35, Chunk 2: 30-50
            assert result.total_chunks == 3

    def test_char_split_chunking_splits_on_sequence(self, tmp_path, mock_embedder):
        """Char split chunking splits on the specified sequence."""
        corpus = [
            CorpusDocument(
                id="doc1",
                text="First sentence. Second sentence. Third sentence.",
                metadata={},
            )
        ]
        config = RetrievalConfig(
            model="test",
            quantization="fp16",
            dimensions=384,
            chunking=ChunkingConfig(strategy="char_split", split_sequence=". "),
        )

        with patch("worker.services.embedding.chromadb") as mock_chromadb:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client = MagicMock()
            mock_client.get_collection.side_effect = ValueError("Not found")
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            service = EmbeddingService(embedder=mock_embedder, collections_dir=tmp_path)
            result = service.embed_corpus(
                corpus=corpus,
                dataset_id="test",
                retrieval_config=config,
            )

            # Split on ". " creates 3 chunks
            assert result.total_chunks == 3

    def test_empty_chunks_are_filtered(self, tmp_path, mock_embedder):
        """Empty chunks from splitting are filtered out."""
        corpus = [
            CorpusDocument(
                id="doc1",
                text=". . . Something. . . ",  # Many empty splits
                metadata={},
            )
        ]
        config = RetrievalConfig(
            model="test",
            quantization="fp16",
            dimensions=384,
            chunking=ChunkingConfig(strategy="char_split", split_sequence=". "),
        )

        with patch("worker.services.embedding.chromadb") as mock_chromadb:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client = MagicMock()
            mock_client.get_collection.side_effect = ValueError("Not found")
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            service = EmbeddingService(embedder=mock_embedder, collections_dir=tmp_path)
            result = service.embed_corpus(
                corpus=corpus,
                dataset_id="test",
                retrieval_config=config,
            )

            # Only "Something" should remain as non-empty (empty strings are filtered)
            assert result.total_chunks == 1

    def test_chunk_metadata_includes_doc_id(self, tmp_path, mock_embedder):
        """Chunk metadata includes original doc_id."""
        corpus = [CorpusDocument(id="doc1", text="Test text", metadata={"title": "Test"})]
        config = RetrievalConfig(
            model="test",
            quantization="fp16",
            dimensions=384,
        )

        with patch("worker.services.embedding.chromadb") as mock_chromadb:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client = MagicMock()
            mock_client.get_collection.side_effect = ValueError("Not found")
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            service = EmbeddingService(embedder=mock_embedder, collections_dir=tmp_path)
            service.embed_corpus(
                corpus=corpus,
                dataset_id="test",
                retrieval_config=config,
            )

            # Check metadata passed to collection.add
            add_call = mock_collection.add.call_args
            metadatas = add_call[1]["metadatas"]
            assert metadatas[0]["doc_id"] == "doc1"
            assert metadatas[0]["title"] == "Test"
