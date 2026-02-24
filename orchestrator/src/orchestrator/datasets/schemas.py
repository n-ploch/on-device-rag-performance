"""Ground truth schemas for evaluation datasets.

These schemas define the structure of evaluation data with evidence
for computing retrieval and generation metrics. The schemas support
dataset-specific fields via Pydantic's extra="allow" configuration.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SentenceEvidence(BaseModel):
    """Evidence at sentence level within a document.

    Represents specific sentences that support or contradict a claim.
    """

    sentence_indices: list[int] = Field(
        description="0-based indices into the document's sentence array"
    )
    sentence_texts: list[str] = Field(
        description="Extracted text content of the evidence sentences"
    )
    label: str = Field(
        description="Evidence label (e.g., 'SUPPORT', 'CONTRADICT')"
    )


class DocumentEvidence(BaseModel):
    """All evidence from a single document.

    Groups sentence-level evidence by document, including document metadata.
    """

    doc_id: str = Field(description="Document identifier")
    doc_title: str | None = Field(default=None, description="Document title if available")
    sentences: list[SentenceEvidence] = Field(
        default_factory=list,
        description="Sentence-level evidence within this document",
    )


class GroundTruthEntry(BaseModel):
    """Complete ground truth for one evaluation sample.

    Contains all information needed to evaluate a single claim/query:
    - The input text
    - Expected label/output
    - Supporting documents and sentence-level evidence

    Dataset-specific fields are unpacked directly onto the model
    (e.g., cited_doc_ids for SciFact) via extra="allow", making
    them accessible as entry.field_name rather than entry.metadata["field_name"].
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="Unique identifier for this evaluation sample")
    input: str = Field(description="The claim or query text to evaluate")
    expected_label: str | None = Field(
        default=None,
        description="Ground truth label (e.g., 'SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO')",
    )
    supporting_documents: list[str] = Field(
        default_factory=list,
        description="Document IDs that contain evidence for this sample",
    )
    evidence: list[DocumentEvidence] = Field(
        default_factory=list,
        description="Detailed evidence structure with sentence-level annotations",
    )

    def get_all_evidence_sentences(self) -> list[str]:
        """Extract all evidence sentence texts across all documents.

        Returns:
            Flattened list of all evidence sentence texts.
        """
        sentences = []
        for doc_evidence in self.evidence:
            for sent_evidence in doc_evidence.sentences:
                sentences.extend(sent_evidence.sentence_texts)
        return sentences

    def get_evidence_for_document(self, doc_id: str) -> DocumentEvidence | None:
        """Get evidence for a specific document.

        Args:
            doc_id: The document identifier to look up.

        Returns:
            DocumentEvidence if found, None otherwise.
        """
        for doc_evidence in self.evidence:
            if doc_evidence.doc_id == doc_id:
                return doc_evidence
        return None
