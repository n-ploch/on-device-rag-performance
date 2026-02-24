"""Tests for ground truth schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestrator.datasets.schemas import (
    DocumentEvidence,
    GroundTruthEntry,
    SentenceEvidence,
)


class TestSentenceEvidence:
    def test_creates_with_required_fields(self):
        """SentenceEvidence can be created with required fields."""
        evidence = SentenceEvidence(
            sentence_indices=[0, 1],
            sentence_texts=["First sentence.", "Second sentence."],
            label="SUPPORT",
        )

        assert evidence.sentence_indices == [0, 1]
        assert evidence.sentence_texts == ["First sentence.", "Second sentence."]
        assert evidence.label == "SUPPORT"

    def test_serializes_to_dict(self):
        """SentenceEvidence serializes to dict correctly."""
        evidence = SentenceEvidence(
            sentence_indices=[0],
            sentence_texts=["Evidence sentence."],
            label="CONTRADICT",
        )

        data = evidence.model_dump()

        assert data["sentence_indices"] == [0]
        assert data["sentence_texts"] == ["Evidence sentence."]
        assert data["label"] == "CONTRADICT"


class TestDocumentEvidence:
    def test_creates_with_required_fields(self):
        """DocumentEvidence can be created with required fields."""
        evidence = DocumentEvidence(
            doc_id="doc1",
            doc_title="Test Document",
            sentences=[
                SentenceEvidence(
                    sentence_indices=[0],
                    sentence_texts=["Evidence."],
                    label="SUPPORT",
                )
            ],
        )

        assert evidence.doc_id == "doc1"
        assert evidence.doc_title == "Test Document"
        assert len(evidence.sentences) == 1

    def test_doc_title_is_optional(self):
        """doc_title can be None."""
        evidence = DocumentEvidence(doc_id="doc1")

        assert evidence.doc_title is None
        assert evidence.sentences == []


class TestGroundTruthEntry:
    def test_creates_with_required_fields(self):
        """GroundTruthEntry can be created with required fields."""
        entry = GroundTruthEntry(
            id="claim1",
            input="This is a test claim.",
        )

        assert entry.id == "claim1"
        assert entry.input == "This is a test claim."
        assert entry.expected_label is None
        assert entry.supporting_documents == []
        assert entry.evidence == []

    def test_creates_with_full_evidence(self):
        """GroundTruthEntry can include full evidence structure."""
        entry = GroundTruthEntry(
            id="claim1",
            input="Aspirin reduces inflammation.",
            expected_label="SUPPORT",
            supporting_documents=["doc1", "doc2"],
            evidence=[
                DocumentEvidence(
                    doc_id="doc1",
                    doc_title="Aspirin Study",
                    sentences=[
                        SentenceEvidence(
                            sentence_indices=[0, 1],
                            sentence_texts=[
                                "Aspirin is anti-inflammatory.",
                                "It reduces swelling.",
                            ],
                            label="SUPPORT",
                        )
                    ],
                )
            ],
        )

        assert entry.expected_label == "SUPPORT"
        assert len(entry.supporting_documents) == 2
        assert len(entry.evidence) == 1
        assert entry.evidence[0].doc_id == "doc1"

    def test_extra_fields_allowed(self):
        """GroundTruthEntry allows extra fields via extra='allow'."""
        entry = GroundTruthEntry(
            id="claim1",
            input="Test claim",
            cited_doc_ids=["doc1", "doc2"],  # SciFact-specific field
            custom_field="custom_value",
        )

        # Extra fields are accessible directly
        assert entry.cited_doc_ids == ["doc1", "doc2"]
        assert entry.custom_field == "custom_value"

    def test_get_all_evidence_sentences(self):
        """get_all_evidence_sentences extracts all sentence texts."""
        entry = GroundTruthEntry(
            id="claim1",
            input="Test",
            evidence=[
                DocumentEvidence(
                    doc_id="doc1",
                    sentences=[
                        SentenceEvidence(
                            sentence_indices=[0],
                            sentence_texts=["First."],
                            label="SUPPORT",
                        ),
                        SentenceEvidence(
                            sentence_indices=[1],
                            sentence_texts=["Second."],
                            label="SUPPORT",
                        ),
                    ],
                ),
                DocumentEvidence(
                    doc_id="doc2",
                    sentences=[
                        SentenceEvidence(
                            sentence_indices=[0],
                            sentence_texts=["Third."],
                            label="CONTRADICT",
                        ),
                    ],
                ),
            ],
        )

        sentences = entry.get_all_evidence_sentences()

        assert sentences == ["First.", "Second.", "Third."]

    def test_get_evidence_for_document(self):
        """get_evidence_for_document returns evidence for specific doc."""
        entry = GroundTruthEntry(
            id="claim1",
            input="Test",
            evidence=[
                DocumentEvidence(doc_id="doc1", doc_title="Doc 1"),
                DocumentEvidence(doc_id="doc2", doc_title="Doc 2"),
            ],
        )

        evidence = entry.get_evidence_for_document("doc2")

        assert evidence is not None
        assert evidence.doc_title == "Doc 2"

    def test_get_evidence_for_document_returns_none_if_not_found(self):
        """get_evidence_for_document returns None for missing doc."""
        entry = GroundTruthEntry(
            id="claim1",
            input="Test",
            evidence=[DocumentEvidence(doc_id="doc1")],
        )

        evidence = entry.get_evidence_for_document("doc999")

        assert evidence is None

    def test_serializes_to_dict(self):
        """GroundTruthEntry serializes to dict for Parquet export."""
        entry = GroundTruthEntry(
            id="claim1",
            input="Test claim",
            expected_label="SUPPORT",
            supporting_documents=["doc1"],
            evidence=[],
            cited_doc_ids=["doc1"],  # Extra field
        )

        data = entry.model_dump()

        assert data["id"] == "claim1"
        assert data["input"] == "Test claim"
        assert data["expected_label"] == "SUPPORT"
        assert data["cited_doc_ids"] == ["doc1"]
