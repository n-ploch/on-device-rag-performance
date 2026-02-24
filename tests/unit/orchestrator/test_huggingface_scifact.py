"""Tests for HuggingFaceSciFact dataset loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.datasets.huggingface_scifact import HuggingFaceSciFact


@pytest.fixture
def mock_corpus_data():
    """Mock SciFact corpus data."""
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
            "structured": True,
        },
    ]


@pytest.fixture
def mock_claims_data():
    """Mock SciFact claims data."""
    return [
        {
            "id": 1,
            "claim": "Aspirin reduces inflammation.",
            "evidence": {
                "1": [{"label": "SUPPORT", "sentences": [0]}],
            },
            "cited_doc_ids": [1],
        },
        {
            "id": 2,
            "claim": "Ibuprofen causes drowsiness.",
            "evidence": {},
            "cited_doc_ids": [2],
        },
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

    def test_export_corpus_joins_abstract_sentences(self, tmp_path, mock_corpus_data):
        """Abstract sentences are joined into single text."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:
            mock_load.return_value = mock_corpus_data

            loader = HuggingFaceSciFact()
            loader.export_corpus(tmp_path)

            table = pq.read_table(tmp_path / "corpus.parquet")
            records = table.to_pylist()

            # Abstract sentences should be joined with space
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
        self, tmp_path, mock_corpus_data, mock_claims_data
    ):
        """export_ground_truth creates a ground_truth.parquet file."""
        with patch("datasets.load_dataset") as mock_load:

            def load_side_effect(repo, split, **kwargs):
                if split == "corpus":
                    return mock_corpus_data
                return mock_claims_data

            mock_load.side_effect = load_side_effect

            loader = HuggingFaceSciFact()
            result_path = loader.export_ground_truth(tmp_path)

            assert result_path == tmp_path / "ground_truth.parquet"
            assert result_path.exists()

    def test_export_ground_truth_includes_evidence(
        self, tmp_path, mock_corpus_data, mock_claims_data
    ):
        """Ground truth includes sentence-level evidence."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:

            def load_side_effect(repo, split, **kwargs):
                if split == "corpus":
                    return mock_corpus_data
                return mock_claims_data

            mock_load.side_effect = load_side_effect

            loader = HuggingFaceSciFact()
            loader.export_ground_truth(tmp_path)

            table = pq.read_table(tmp_path / "ground_truth.parquet")
            records = table.to_pylist()

            # First claim has evidence
            claim1 = records[0]
            assert claim1["id"] == "1"
            assert claim1["input"] == "Aspirin reduces inflammation."
            assert claim1["expected_label"] == "SUPPORT"
            assert "1" in claim1["supporting_documents"]
            assert len(claim1["evidence"]) == 1

            # Evidence includes sentence texts
            doc_evidence = claim1["evidence"][0]
            assert doc_evidence["doc_id"] == "1"
            assert len(doc_evidence["sentences"]) == 1
            assert doc_evidence["sentences"][0]["sentence_texts"] == [
                "Aspirin reduces inflammation."
            ]

    def test_export_ground_truth_handles_no_evidence(
        self, tmp_path, mock_corpus_data, mock_claims_data
    ):
        """Claims without evidence get NOT_ENOUGH_INFO label."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:

            def load_side_effect(repo, split, **kwargs):
                if split == "corpus":
                    return mock_corpus_data
                return mock_claims_data

            mock_load.side_effect = load_side_effect

            loader = HuggingFaceSciFact()
            loader.export_ground_truth(tmp_path)

            table = pq.read_table(tmp_path / "ground_truth.parquet")
            records = table.to_pylist()

            # Second claim has no evidence
            claim2 = records[1]
            assert claim2["expected_label"] == "NOT_ENOUGH_INFO"
            assert claim2["supporting_documents"] == []

    def test_export_ground_truth_unpacks_cited_doc_ids(
        self, tmp_path, mock_corpus_data, mock_claims_data
    ):
        """cited_doc_ids is unpacked as a direct field."""
        import pyarrow.parquet as pq

        with patch("datasets.load_dataset") as mock_load:

            def load_side_effect(repo, split, **kwargs):
                if split == "corpus":
                    return mock_corpus_data
                return mock_claims_data

            mock_load.side_effect = load_side_effect

            loader = HuggingFaceSciFact()
            loader.export_ground_truth(tmp_path)

            table = pq.read_table(tmp_path / "ground_truth.parquet")
            records = table.to_pylist()

            # cited_doc_ids should be present as direct field
            assert "cited_doc_ids" in records[0]
            assert records[0]["cited_doc_ids"] == ["1"]

    def test_export_metadata_creates_json(self, tmp_path, mock_corpus_data, mock_claims_data):
        """export_metadata creates a metadata.json file."""
        import json

        with patch("datasets.load_dataset") as mock_load:

            def load_side_effect(repo, split, **kwargs):
                if split == "corpus":
                    return mock_corpus_data
                return mock_claims_data

            mock_load.side_effect = load_side_effect

            loader = HuggingFaceSciFact()
            loader.export_corpus(tmp_path, limit=2)
            loader.export_ground_truth(tmp_path, limit=2)
            result_path = loader.export_metadata(tmp_path)

            assert result_path == tmp_path / "metadata.json"
            assert result_path.exists()

            metadata = json.loads(result_path.read_text())
            assert metadata["dataset_id"] == "scifact"
            assert metadata["source"] == "allenai/scifact"
            assert metadata["corpus_count"] == 2
            assert metadata["ground_truth_count"] == 2

    def test_export_all_creates_all_files(self, tmp_path, mock_corpus_data, mock_claims_data):
        """export_all creates corpus, ground_truth, and metadata."""
        with patch("datasets.load_dataset") as mock_load:

            def load_side_effect(repo, split, **kwargs):
                if split == "corpus":
                    return mock_corpus_data
                return mock_claims_data

            mock_load.side_effect = load_side_effect

            loader = HuggingFaceSciFact()
            paths = loader.export_all(tmp_path, corpus_limit=2, ground_truth_limit=2)

            assert "corpus" in paths
            assert "ground_truth" in paths
            assert "metadata" in paths

            assert paths["corpus"].exists()
            assert paths["ground_truth"].exists()
            assert paths["metadata"].exists()
