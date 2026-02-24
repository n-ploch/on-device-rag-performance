"""Tests for HuggingFaceSciFact dataset loader (BeIR format)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.datasets.huggingface_scifact import HuggingFaceSciFact


@pytest.fixture
def mock_corpus_data():
    """Mock BeIR corpus data."""
    return [
        {
            "_id": "1",
            "title": "Aspirin Study",
            "text": "Aspirin reduces inflammation. It is widely used.",
            "metadata": {},
        },
        {
            "_id": "2",
            "title": "Ibuprofen Research",
            "text": "Ibuprofen is an NSAID. It treats pain.",
            "metadata": {},
        },
    ]


@pytest.fixture
def mock_queries_data():
    """Mock BeIR queries data."""
    return [
        {
            "_id": "100",
            "text": "Aspirin reduces inflammation.",
            "metadata": {},
        },
        {
            "_id": "101",
            "text": "Ibuprofen causes drowsiness.",
            "metadata": {},
        },
    ]


@pytest.fixture
def mock_qrels_data():
    """Mock BeIR qrels data."""
    return [
        {"query-id": 100, "corpus-id": 1, "score": 1},
    ]


class TestHuggingFaceSciFact:
    def test_dataset_id(self):
        """dataset_id returns 'scifact'."""
        loader = HuggingFaceSciFact()
        assert loader.dataset_id == "scifact"

    def test_export_corpus_creates_parquet(self, tmp_path, mock_corpus_data):
        """export_corpus creates a corpus.parquet file."""
        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_corpus_data

            loader = HuggingFaceSciFact()
            result_path = loader.export_corpus(tmp_path, limit=2)

            assert result_path == tmp_path / "corpus.parquet"
            assert result_path.exists()

    def test_export_corpus_transforms_to_generic_format(self, tmp_path, mock_corpus_data):
        """Corpus is transformed to generic {id, text, metadata} format."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_corpus_data

            loader = HuggingFaceSciFact()
            result_path = loader.export_corpus(tmp_path, limit=2)

            table = pq.read_table(result_path)
            records = table.to_pylist()

            assert len(records) == 2
            assert records[0]["id"] == "1"
            assert "Aspirin reduces inflammation" in records[0]["text"]
            assert records[0]["metadata"]["title"] == "Aspirin Study"

    def test_export_corpus_preserves_text(self, tmp_path, mock_corpus_data):
        """Text content is preserved from BeIR format."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_corpus_data

            loader = HuggingFaceSciFact()
            loader.export_corpus(tmp_path)

            table = pq.read_table(tmp_path / "corpus.parquet")
            records = table.to_pylist()

            assert records[0]["text"] == "Aspirin reduces inflammation. It is widely used."

    def test_export_corpus_respects_limit(self, tmp_path, mock_corpus_data):
        """export_corpus respects the limit parameter."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_corpus_data

            loader = HuggingFaceSciFact()
            loader.export_corpus(tmp_path, limit=1)

            table = pq.read_table(tmp_path / "corpus.parquet")
            assert table.num_rows == 1

    def test_export_ground_truth_creates_parquet(
        self, tmp_path, mock_corpus_data, mock_queries_data, mock_qrels_data
    ):
        """export_ground_truth creates a ground_truth.parquet file."""
        with patch("datasets.load_dataset") as mock_load:

            def load_side_effect(fmt, data_files=None, delimiter=None, **kwargs):
                if data_files and "corpus" in data_files:
                    return mock_corpus_data
                elif data_files and "queries" in data_files:
                    return mock_queries_data
                elif delimiter == "\t":
                    return mock_qrels_data
                return mock_corpus_data

            mock_load.side_effect = load_side_effect

            loader = HuggingFaceSciFact()
            result_path = loader.export_ground_truth(tmp_path)

            assert result_path == tmp_path / "ground_truth.parquet"
            assert result_path.exists()

    def test_export_ground_truth_maps_qrels(
        self, tmp_path, mock_corpus_data, mock_queries_data, mock_qrels_data
    ):
        """Ground truth uses qrels for supporting documents."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:

            def load_side_effect(fmt, data_files=None, delimiter=None, **kwargs):
                if data_files and "corpus" in data_files:
                    return mock_corpus_data
                elif data_files and "queries" in data_files:
                    return mock_queries_data
                elif delimiter == "\t":
                    return mock_qrels_data
                return mock_corpus_data

            mock_load.side_effect = load_side_effect

            loader = HuggingFaceSciFact()
            loader.export_ground_truth(tmp_path)

            table = pq.read_table(tmp_path / "ground_truth.parquet")
            records = table.to_pylist()

            # First query has a qrel
            query1 = records[0]
            assert query1["id"] == "100"
            assert query1["input"] == "Aspirin reduces inflammation."
            assert query1["expected_label"] == "SUPPORT"
            assert "1" in query1["supporting_documents"]

    def test_export_ground_truth_handles_no_qrels(
        self, tmp_path, mock_corpus_data, mock_queries_data, mock_qrels_data
    ):
        """Queries without qrels get NOT_ENOUGH_INFO label."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:

            def load_side_effect(fmt, data_files=None, delimiter=None, **kwargs):
                if data_files and "corpus" in data_files:
                    return mock_corpus_data
                elif data_files and "queries" in data_files:
                    return mock_queries_data
                elif delimiter == "\t":
                    return mock_qrels_data
                return mock_corpus_data

            mock_load.side_effect = load_side_effect

            loader = HuggingFaceSciFact()
            loader.export_ground_truth(tmp_path)

            table = pq.read_table(tmp_path / "ground_truth.parquet")
            records = table.to_pylist()

            # Second query has no qrels
            query2 = records[1]
            assert query2["expected_label"] == "NOT_ENOUGH_INFO"
            assert query2["supporting_documents"] == []

    def test_export_ground_truth_includes_cited_doc_ids(
        self, tmp_path, mock_corpus_data, mock_queries_data, mock_qrels_data
    ):
        """cited_doc_ids is included as a direct field."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:

            def load_side_effect(fmt, data_files=None, delimiter=None, **kwargs):
                if data_files and "corpus" in data_files:
                    return mock_corpus_data
                elif data_files and "queries" in data_files:
                    return mock_queries_data
                elif delimiter == "\t":
                    return mock_qrels_data
                return mock_corpus_data

            mock_load.side_effect = load_side_effect

            loader = HuggingFaceSciFact()
            loader.export_ground_truth(tmp_path)

            table = pq.read_table(tmp_path / "ground_truth.parquet")
            records = table.to_pylist()

            assert "cited_doc_ids" in records[0]
            assert records[0]["cited_doc_ids"] == ["1"]

    def test_export_metadata_creates_json(
        self, tmp_path, mock_corpus_data, mock_queries_data, mock_qrels_data
    ):
        """export_metadata creates a metadata.json file."""
        with patch("datasets.load_dataset") as mock_load:

            def load_side_effect(fmt, data_files=None, delimiter=None, **kwargs):
                if data_files and "corpus" in data_files:
                    return mock_corpus_data
                elif data_files and "queries" in data_files:
                    return mock_queries_data
                elif delimiter == "\t":
                    return mock_qrels_data
                return mock_corpus_data

            mock_load.side_effect = load_side_effect

            loader = HuggingFaceSciFact()
            loader.export_corpus(tmp_path, limit=2)
            loader.export_ground_truth(tmp_path, limit=2)
            result_path = loader.export_metadata(tmp_path)

            assert result_path == tmp_path / "metadata.json"
            assert result_path.exists()

            metadata = json.loads(result_path.read_text())
            assert metadata["dataset_id"] == "scifact"
            assert metadata["source"] == "BeIR/scifact"
            assert metadata["corpus_count"] == 2
            assert metadata["ground_truth_count"] == 2

    def test_export_all_creates_all_files(
        self, tmp_path, mock_corpus_data, mock_queries_data, mock_qrels_data
    ):
        """export_all creates corpus, ground_truth, and metadata."""
        with patch("datasets.load_dataset") as mock_load:

            def load_side_effect(fmt, data_files=None, delimiter=None, **kwargs):
                if data_files and "corpus" in data_files:
                    return mock_corpus_data
                elif data_files and "queries" in data_files:
                    return mock_queries_data
                elif delimiter == "\t":
                    return mock_qrels_data
                return mock_corpus_data

            mock_load.side_effect = load_side_effect

            loader = HuggingFaceSciFact()
            paths = loader.export_all(tmp_path, corpus_limit=2, ground_truth_limit=2)

            assert "corpus" in paths
            assert "ground_truth" in paths
            assert "metadata" in paths

            assert paths["corpus"].exists()
            assert paths["ground_truth"].exists()
            assert paths["metadata"].exists()
