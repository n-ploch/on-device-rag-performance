"""Tests for HuggingFaceSciFact dataset loader (BigBio format)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.datasets.huggingface_scifact import HuggingFaceSciFact


@pytest.fixture
def mock_corpus_data():
    """Mock BigBio corpus data (scifact_corpus_source)."""
    return [
        {
            "doc_id": 1,
            "title": "Aspirin Study",
            "abstract": ["Aspirin reduces inflammation.", "It is widely used."],
            "structured": False,
        },
        {
            "doc_id": 2,
            "title": "Ibuprofen Research",
            "abstract": ["Ibuprofen is an NSAID.", "It treats pain."],
            "structured": False,
        },
    ]


@pytest.fixture
def mock_claims_data():
    """Mock BigBio claims data (scifact_claims_source)."""
    return [
        {
            "id": 100,
            "claim": "Aspirin reduces inflammation.",
            "evidences": [
                {
                    "doc_id": 1,
                    "sentence_ids": [0],
                    "label": "SUPPORT",
                }
            ],
            "cited_doc_ids": [1],
        },
        {
            "id": 101,
            "claim": "Ibuprofen causes drowsiness.",
            "evidences": [],
            "cited_doc_ids": [],
        },
        {
            "id": 102,
            "claim": "Aspirin is harmful.",
            "evidences": [
                {
                    "doc_id": 1,
                    "sentence_ids": [0, 1],
                    "label": "CONTRADICT",
                }
            ],
            "cited_doc_ids": [1],
        },
    ]


class TestHuggingFaceSciFact:
    def test_dataset_id(self):
        """dataset_id returns 'scifact_bigbio'."""
        loader = HuggingFaceSciFact()
        assert loader.dataset_id == "scifact_bigbio"

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

    def test_export_corpus_joins_abstract_sentences(self, tmp_path, mock_corpus_data):
        """Abstract sentences are joined into single text."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_corpus_data

            loader = HuggingFaceSciFact()
            loader.export_corpus(tmp_path)

            table = pq.read_table(tmp_path / "corpus.parquet")
            records = table.to_pylist()

            assert records[0]["text"] == "Aspirin reduces inflammation. It is widely used."
            assert records[0]["metadata"]["sentence_count"] == 2

    def test_export_corpus_respects_limit(self, tmp_path, mock_corpus_data):
        """export_corpus respects the limit parameter."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_corpus_data

            loader = HuggingFaceSciFact()
            loader.export_corpus(tmp_path, limit=1)

            table = pq.read_table(tmp_path / "corpus.parquet")
            assert table.num_rows == 1

    def test_export_ground_truth_creates_parquet(self, tmp_path, mock_claims_data):
        """export_ground_truth creates a ground_truth.parquet file."""
        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_claims_data

            loader = HuggingFaceSciFact()
            result_path = loader.export_ground_truth(tmp_path)

            assert result_path == tmp_path / "ground_truth.parquet"
            assert result_path.exists()

    def test_export_ground_truth_uses_embedded_evidence(self, tmp_path, mock_claims_data):
        """Ground truth uses embedded evidence for supporting documents."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_claims_data

            loader = HuggingFaceSciFact()
            loader.export_ground_truth(tmp_path)

            table = pq.read_table(tmp_path / "ground_truth.parquet")
            records = table.to_pylist()

            # First claim has SUPPORT evidence
            claim1 = records[0]
            assert claim1["id"] == "100"
            assert claim1["input"] == "Aspirin reduces inflammation."
            assert claim1["expected_label"] == "SUPPORT"
            assert "1" in claim1["supporting_documents"]

    def test_export_ground_truth_handles_no_evidence(self, tmp_path, mock_claims_data):
        """Claims without evidence get NOT_ENOUGH_INFO label."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_claims_data

            loader = HuggingFaceSciFact()
            loader.export_ground_truth(tmp_path)

            table = pq.read_table(tmp_path / "ground_truth.parquet")
            records = table.to_pylist()

            # Second claim has no evidence
            claim2 = records[1]
            assert claim2["expected_label"] == "NOT_ENOUGH_INFO"
            assert claim2["supporting_documents"] == []

    def test_export_ground_truth_handles_contradict_label(self, tmp_path, mock_claims_data):
        """Claims with CONTRADICT evidence get CONTRADICT label."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_claims_data

            loader = HuggingFaceSciFact()
            loader.export_ground_truth(tmp_path)

            table = pq.read_table(tmp_path / "ground_truth.parquet")
            records = table.to_pylist()

            # Third claim has CONTRADICT evidence
            claim3 = records[2]
            assert claim3["expected_label"] == "CONTRADICT"
            assert "1" in claim3["supporting_documents"]

    def test_export_ground_truth_includes_cited_doc_ids(self, tmp_path, mock_claims_data):
        """cited_doc_ids is included as a direct field."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_claims_data

            loader = HuggingFaceSciFact()
            loader.export_ground_truth(tmp_path)

            table = pq.read_table(tmp_path / "ground_truth.parquet")
            records = table.to_pylist()

            assert "cited_doc_ids" in records[0]
            assert records[0]["cited_doc_ids"] == ["1"]

    def test_export_ground_truth_includes_sentence_evidence(self, tmp_path, mock_claims_data):
        """Evidence includes sentence-level information."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_claims_data

            loader = HuggingFaceSciFact()
            loader.export_ground_truth(tmp_path)

            table = pq.read_table(tmp_path / "ground_truth.parquet")
            records = table.to_pylist()

            # First claim has sentence evidence
            evidence = records[0]["evidence"]
            assert len(evidence) == 1
            assert evidence[0]["doc_id"] == "1"
            assert len(evidence[0]["sentences"]) == 1
            assert evidence[0]["sentences"][0]["sentence_indices"] == [0]
            assert evidence[0]["sentences"][0]["label"] == "SUPPORT"

    def test_export_metadata_creates_json(self, tmp_path, mock_corpus_data, mock_claims_data):
        """export_metadata creates a metadata.json file."""
        loader = HuggingFaceSciFact()
        with patch.object(loader, "_load_corpus", return_value=mock_corpus_data):
            with patch.object(loader, "_load_claims", return_value=mock_claims_data):
                loader.export_corpus(tmp_path, limit=2)
                loader.export_ground_truth(tmp_path, limit=3)
                result_path = loader.export_metadata(tmp_path)

        assert result_path == tmp_path / "metadata.json"
        assert result_path.exists()

        metadata = json.loads(result_path.read_text())
        assert metadata["dataset_id"] == "scifact_bigbio"
        assert metadata["source"] == "bigbio/scifact"
        assert metadata["corpus_count"] == 2
        assert metadata["ground_truth_count"] == 3

    def test_export_all_creates_all_files(self, tmp_path, mock_corpus_data, mock_claims_data):
        """export_all creates corpus, ground_truth, and metadata."""
        loader = HuggingFaceSciFact()
        with patch.object(loader, "_load_corpus", return_value=mock_corpus_data):
            with patch.object(loader, "_load_claims", return_value=mock_claims_data):
                paths = loader.export_all(tmp_path, corpus_limit=2, ground_truth_limit=3)

        assert "corpus" in paths
        assert "ground_truth" in paths
        assert "metadata" in paths

        assert paths["corpus"].exists()
        assert paths["ground_truth"].exists()
        assert paths["metadata"].exists()
