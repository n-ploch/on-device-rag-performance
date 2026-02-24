"""HuggingFace SciFact dataset loader.

Loads the allenai/scifact dataset and transforms it to the standard
corpus and ground truth formats for the RAG evaluation system.

SciFact is a scientific claim verification dataset where:
- Corpus: Scientific paper abstracts (as sentences)
- Claims: Atomic claims that may be supported/contradicted by abstracts
- Evidence: Sentence-level annotations linking claims to abstract sentences
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from orchestrator.datasets.schemas import DocumentEvidence, GroundTruthEntry, SentenceEvidence
from shared_types import CorpusDocument, DatasetLoader

if TYPE_CHECKING:
    from datasets import Dataset


class HuggingFaceSciFact(DatasetLoader):
    """Dataset loader for SciFact from HuggingFace Hub.

    SciFact structure:
    - Corpus split: {doc_id, title, abstract (list of sentences), structured}
    - Claims split: {id, claim, evidence (dict), cited_doc_ids}

    Evidence structure: {doc_id: [{label, sentences: [indices]}]}
    """

    REPO_ID = "allenai/scifact"
    CORPUS_SPLIT = "corpus"
    CLAIMS_SPLIT = "claims"

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the SciFact loader.

        Args:
            cache_dir: Optional directory for HuggingFace cache.
        """
        self._cache_dir = cache_dir
        self._corpus_dataset: Dataset | None = None
        self._claims_dataset: Dataset | None = None
        self._corpus_count: int = 0
        self._ground_truth_count: int = 0

    @property
    def dataset_id(self) -> str:
        """Return the dataset identifier."""
        return "scifact"

    def _load_corpus(self) -> Dataset:
        """Lazy-load the corpus split."""
        if self._corpus_dataset is None:
            from datasets import load_dataset

            self._corpus_dataset = load_dataset(
                self.REPO_ID,
                split=self.CORPUS_SPLIT,
                cache_dir=str(self._cache_dir) if self._cache_dir else None,
                trust_remote_code=True,
            )
        return self._corpus_dataset

    def _load_claims(self) -> Dataset:
        """Lazy-load the claims split."""
        if self._claims_dataset is None:
            from datasets import load_dataset

            self._claims_dataset = load_dataset(
                self.REPO_ID,
                split=self.CLAIMS_SPLIT,
                cache_dir=str(self._cache_dir) if self._cache_dir else None,
                trust_remote_code=True,
            )
        return self._claims_dataset

    def _build_corpus_index(self) -> dict[str, dict[str, Any]]:
        """Build an index of corpus documents for evidence extraction.

        Returns:
            Dict mapping doc_id to {title, abstract_sentences}.
        """
        corpus = self._load_corpus()
        index = {}
        for row in corpus:
            doc_id = str(row["doc_id"])
            index[doc_id] = {
                "title": row.get("title"),
                "abstract": row.get("abstract", []),
            }
        return index

    def _transform_corpus_document(self, row: dict[str, Any]) -> CorpusDocument:
        """Transform a SciFact corpus row to generic CorpusDocument.

        SciFact stores abstracts as a list of sentences. We join them
        into a single text for embedding.
        """
        abstract_sentences = row.get("abstract", [])
        text = " ".join(abstract_sentences) if abstract_sentences else ""

        return CorpusDocument(
            id=str(row["doc_id"]),
            text=text,
            metadata={
                "title": row.get("title"),
                "structured": row.get("structured", False),
                "sentence_count": len(abstract_sentences),
            },
        )

    def _transform_ground_truth(
        self,
        row: dict[str, Any],
        corpus_index: dict[str, dict[str, Any]],
    ) -> GroundTruthEntry:
        """Transform a SciFact claim to GroundTruthEntry.

        Extracts sentence-level evidence and resolves sentence texts
        from the corpus index.
        """
        evidence_raw = row.get("evidence", {}) or {}
        cited_doc_ids = row.get("cited_doc_ids", []) or []

        # Determine overall label
        labels = set()
        for doc_evidence_list in evidence_raw.values():
            for ev in doc_evidence_list:
                labels.add(ev.get("label", ""))

        if "SUPPORT" in labels and "CONTRADICT" not in labels:
            expected_label = "SUPPORT"
        elif "CONTRADICT" in labels and "SUPPORT" not in labels:
            expected_label = "CONTRADICT"
        elif labels:
            expected_label = "MIXED"
        else:
            expected_label = "NOT_ENOUGH_INFO"

        # Build document evidence
        evidence_list: list[DocumentEvidence] = []
        supporting_documents: list[str] = []

        for doc_id_str, doc_evidence_raw in evidence_raw.items():
            doc_id = str(doc_id_str)
            supporting_documents.append(doc_id)

            # Get document info from corpus index
            doc_info = corpus_index.get(doc_id, {})
            abstract_sentences = doc_info.get("abstract", [])

            # Group evidence by label
            sentences_by_label: dict[str, list[tuple[list[int], list[str]]]] = {}
            for ev in doc_evidence_raw:
                label = ev.get("label", "UNKNOWN")
                sent_indices = ev.get("sentences", [])

                # Extract actual sentence texts
                sent_texts = []
                for idx in sent_indices:
                    if 0 <= idx < len(abstract_sentences):
                        sent_texts.append(abstract_sentences[idx])

                if label not in sentences_by_label:
                    sentences_by_label[label] = []
                sentences_by_label[label].append((sent_indices, sent_texts))

            # Create SentenceEvidence for each label group
            sentence_evidence_list: list[SentenceEvidence] = []
            for label, evidence_groups in sentences_by_label.items():
                for indices, texts in evidence_groups:
                    sentence_evidence_list.append(
                        SentenceEvidence(
                            sentence_indices=indices,
                            sentence_texts=texts,
                            label=label,
                        )
                    )

            evidence_list.append(
                DocumentEvidence(
                    doc_id=doc_id,
                    doc_title=doc_info.get("title"),
                    sentences=sentence_evidence_list,
                )
            )

        return GroundTruthEntry(
            id=str(row["id"]),
            input=row["claim"],
            expected_label=expected_label,
            supporting_documents=supporting_documents,
            evidence=evidence_list,
            # SciFact-specific fields unpacked directly
            cited_doc_ids=[str(d) for d in cited_doc_ids],
        )

    def _iter_corpus(self, limit: int | None = None) -> Iterator[CorpusDocument]:
        """Iterate over corpus documents."""
        corpus = self._load_corpus()
        for i, row in enumerate(corpus):
            if limit is not None and i >= limit:
                break
            yield self._transform_corpus_document(row)

    def _iter_ground_truth(
        self,
        corpus_index: dict[str, dict[str, Any]],
        limit: int | None = None,
    ) -> Iterator[GroundTruthEntry]:
        """Iterate over ground truth entries."""
        claims = self._load_claims()
        for i, row in enumerate(claims):
            if limit is not None and i >= limit:
                break
            yield self._transform_ground_truth(row, corpus_index)

    def export_corpus(self, output_dir: Path, limit: int | None = None) -> Path:
        """Export corpus to Parquet format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        records = [doc.model_dump() for doc in self._iter_corpus(limit=limit)]
        self._corpus_count = len(records)

        table = pa.Table.from_pylist(records)
        output_path = output_dir / "corpus.parquet"
        pq.write_table(table, output_path)

        return output_path

    def export_ground_truth(self, output_dir: Path, limit: int | None = None) -> Path:
        """Export ground truth to Parquet format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build corpus index for evidence text extraction
        corpus_index = self._build_corpus_index()

        records = [
            entry.model_dump() for entry in self._iter_ground_truth(corpus_index, limit=limit)
        ]
        self._ground_truth_count = len(records)

        table = pa.Table.from_pylist(records)
        output_path = output_dir / "ground_truth.parquet"
        pq.write_table(table, output_path)

        return output_path

    def export_metadata(self, output_dir: Path) -> Path:
        """Export dataset metadata to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "dataset_id": self.dataset_id,
            "source": self.REPO_ID,
            "corpus_split": self.CORPUS_SPLIT,
            "claims_split": self.CLAIMS_SPLIT,
            "corpus_count": self._corpus_count,
            "ground_truth_count": self._ground_truth_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        output_path = output_dir / "metadata.json"
        output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

        return output_path
