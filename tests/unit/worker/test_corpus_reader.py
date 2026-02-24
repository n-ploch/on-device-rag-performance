"""Tests for CorpusReader."""

from __future__ import annotations

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from shared_types import CorpusDocument
from worker.datasets import CorpusReader


@pytest.fixture
def corpus_parquet(tmp_path):
    """Create a test corpus.parquet file."""
    records = [
        {"id": "doc1", "text": "First document text.", "metadata": {"title": "Doc 1"}},
        {"id": "doc2", "text": "Second document text.", "metadata": {"title": "Doc 2"}},
        {"id": "doc3", "text": "Third document text.", "metadata": {"title": "Doc 3"}},
        {"id": "doc4", "text": "Fourth document text.", "metadata": {}},
        {"id": "doc5", "text": "Fifth document text.", "metadata": {}},
    ]

    table = pa.Table.from_pylist(records)
    corpus_path = tmp_path / "corpus.parquet"
    pq.write_table(table, corpus_path)

    return corpus_path


class TestCorpusReader:
    def test_init_with_existing_file(self, corpus_parquet):
        """CorpusReader initializes with existing corpus file."""
        reader = CorpusReader(corpus_parquet)

        assert reader.path == corpus_parquet

    def test_init_raises_if_file_not_found(self, tmp_path):
        """CorpusReader raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            CorpusReader(tmp_path / "missing.parquet")

        assert "Corpus not found" in str(exc_info.value)

    def test_count_returns_document_count(self, corpus_parquet):
        """count() returns total number of documents."""
        reader = CorpusReader(corpus_parquet)

        assert reader.count() == 5

    def test_len_returns_document_count(self, corpus_parquet):
        """len(reader) returns document count."""
        reader = CorpusReader(corpus_parquet)

        assert len(reader) == 5

    def test_read_all_returns_all_documents(self, corpus_parquet):
        """read_all() returns all documents as CorpusDocument list."""
        reader = CorpusReader(corpus_parquet)

        documents = reader.read_all()

        assert len(documents) == 5
        assert all(isinstance(doc, CorpusDocument) for doc in documents)
        assert documents[0].id == "doc1"
        assert documents[0].text == "First document text."
        assert documents[0].metadata == {"title": "Doc 1"}

    def test_read_batched_yields_batches(self, corpus_parquet):
        """read_batched() yields documents in batches."""
        reader = CorpusReader(corpus_parquet)

        batches = list(reader.read_batched(batch_size=2))

        assert len(batches) == 3  # 5 docs / 2 per batch = 3 batches
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1  # Last batch has remainder

    def test_read_batched_returns_corpus_documents(self, corpus_parquet):
        """read_batched() yields CorpusDocument objects."""
        reader = CorpusReader(corpus_parquet)

        for batch in reader.read_batched(batch_size=2):
            assert all(isinstance(doc, CorpusDocument) for doc in batch)

    def test_read_by_ids_returns_matching_documents(self, corpus_parquet):
        """read_by_ids() returns documents matching given IDs."""
        reader = CorpusReader(corpus_parquet)

        documents = reader.read_by_ids(["doc2", "doc4"])

        assert len(documents) == 2
        ids = {doc.id for doc in documents}
        assert ids == {"doc2", "doc4"}

    def test_read_by_ids_returns_empty_for_no_matches(self, corpus_parquet):
        """read_by_ids() returns empty list for non-existent IDs."""
        reader = CorpusReader(corpus_parquet)

        documents = reader.read_by_ids(["doc999", "doc888"])

        assert documents == []

    def test_iter_yields_all_documents(self, corpus_parquet):
        """Iterating over reader yields all documents."""
        reader = CorpusReader(corpus_parquet)

        documents = list(reader)

        assert len(documents) == 5
        assert all(isinstance(doc, CorpusDocument) for doc in documents)

    def test_reader_is_dataset_agnostic(self, tmp_path):
        """Reader works with any corpus following the schema."""
        # Create a non-SciFact corpus
        records = [
            {"id": "custom1", "text": "Custom dataset text.", "metadata": {"source": "custom"}},
        ]
        table = pa.Table.from_pylist(records)
        corpus_path = tmp_path / "corpus.parquet"
        pq.write_table(table, corpus_path)

        reader = CorpusReader(corpus_path)
        documents = reader.read_all()

        assert len(documents) == 1
        assert documents[0].id == "custom1"
        assert documents[0].metadata["source"] == "custom"

    def test_from_dataset_id_resolves_path(self, tmp_path):
        """from_dataset_id resolves corpus path from dataset ID and base dir."""
        # Create dataset directory structure
        dataset_dir = tmp_path / "scifact"
        dataset_dir.mkdir()

        records = [{"id": "doc1", "text": "Test", "metadata": {"title": "Test Doc"}}]
        table = pa.Table.from_pylist(records)
        pq.write_table(table, dataset_dir / "corpus.parquet")

        reader = CorpusReader.from_dataset_id("scifact", datasets_dir=tmp_path)

        assert reader.count() == 1
        assert reader.path == dataset_dir / "corpus.parquet"

    def test_from_dataset_id_raises_if_not_found(self, tmp_path):
        """from_dataset_id raises FileNotFoundError if corpus doesn't exist."""
        with pytest.raises(FileNotFoundError):
            CorpusReader.from_dataset_id("nonexistent", datasets_dir=tmp_path)
