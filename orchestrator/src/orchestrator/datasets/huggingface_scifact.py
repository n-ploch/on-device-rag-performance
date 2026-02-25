"""HuggingFace BigBio SciFact dataset loader.

Loads the bigbio/scifact dataset and transforms it to the standard
corpus and ground truth formats for the RAG evaluation system.

SciFact is a scientific claim verification dataset where:
- Corpus: Scientific paper abstracts (from scifact_corpus_source)
- Claims: Scientific claims with evidence annotations (from scifact_claims_source)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from orchestrator.datasets.schemas import DocumentEvidence, GroundTruthEntry, SentenceEvidence
from shared_types import CorpusDocument, DatasetLoader

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


class HuggingFaceSciFact(DatasetLoader):
    """Dataset loader for SciFact from HuggingFace Hub (BigBio format).

    Uses the bigbio/scifact dataset which contains:
    - scifact_corpus_source: {doc_id, title, abstract, structured}
    - scifact_claims_source: {id, claim, evidences, cited_doc_ids}
    """

    DATASET_NAME = "bigbio/scifact"

    def __init__(self, cache_dir: Path | None = None, token: str | None = None):
        """Initialize the SciFact loader.

        Args:
            cache_dir: Optional directory for HuggingFace cache.
            token: Optional HuggingFace token for authenticated access.
        """
        self._cache_dir = cache_dir
        self._token = token
        self._corpus_dataset: Dataset | None = None
        self._claims_dataset: Dataset | None = None
        self._corpus_count: int = 0
        self._ground_truth_count: int = 0

    @property
    def dataset_id(self) -> str:
        """Return the dataset identifier."""
        return "scifact_bigbio"

    def _load_corpus(self) -> Dataset:
        """Lazy-load the corpus from scifact_corpus_source."""
        if self._corpus_dataset is None:
            from datasets import load_dataset

            logger.info("Loading SciFact corpus from bigbio/scifact")
            self._corpus_dataset = load_dataset(
                self.DATASET_NAME,
                "scifact_corpus_source",
                split="train",
                cache_dir=str(self._cache_dir) if self._cache_dir else None,
                token=self._token,
            )
        return self._corpus_dataset

    def _load_claims(self, split: str = "train") -> Dataset:
        """Lazy-load the claims from scifact_claims_source.

        Args:
            split: Dataset split to load ('train', 'validation', or 'test').
        """
        if self._claims_dataset is None:
            from datasets import load_dataset

            logger.info("Loading SciFact claims from bigbio/scifact (split=%s)", split)
            self._claims_dataset = load_dataset(
                self.DATASET_NAME,
                "scifact_claims_source",
                split=split,
                cache_dir=str(self._cache_dir) if self._cache_dir else None,
                token=self._token,
            )
        return self._claims_dataset

    def _transform_corpus_document(self, row: dict[str, Any]) -> CorpusDocument:
        """Transform a BigBio corpus row to generic CorpusDocument.

        BigBio format:
            - doc_id: int
            - title: str
            - abstract: list[str] (sentences)
            - structured: bool
        """
        # Join abstract sentences into single text
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

    def _transform_ground_truth(self, row: dict[str, Any]) -> GroundTruthEntry:
        """Transform a BigBio claim to GroundTruthEntry.

        BigBio format:
            - id: int
            - claim: str
            - evidences: list[{doc_id, sentence_ids, label}]
            - cited_doc_ids: list[int]
        """
        claim_id = str(row["id"])
        claim_text = row["claim"]
        evidences = row.get("evidences", [])
        cited_doc_ids = [str(doc_id) for doc_id in row.get("cited_doc_ids", [])]

        # Build evidence list from BigBio evidence annotations
        evidence_list: list[DocumentEvidence] = []
        labels_found: set[str] = set()

        for evidence in evidences:
            doc_id = str(evidence["doc_id"])
            sentence_ids = evidence.get("sentence_ids", [])
            label = evidence.get("label", "SUPPORT")
            labels_found.add(label)

            # Create sentence evidence (we don't have the actual text here)
            sentence_evidence = SentenceEvidence(
                sentence_indices=sentence_ids,
                sentence_texts=[],  # Texts would need to be looked up from corpus
                label=label,
            )

            evidence_list.append(
                DocumentEvidence(
                    doc_id=doc_id,
                    doc_title=None,
                    sentences=[sentence_evidence],
                )
            )

        # Determine expected label
        # Priority: CONTRADICT > SUPPORT > NOT_ENOUGH_INFO
        if "CONTRADICT" in labels_found:
            expected_label = "CONTRADICT"
        elif "SUPPORT" in labels_found:
            expected_label = "SUPPORT"
        elif cited_doc_ids:
            expected_label = "SUPPORT"
        else:
            expected_label = "NOT_ENOUGH_INFO"

        return GroundTruthEntry(
            id=claim_id,
            input=claim_text,
            expected_label=expected_label,
            supporting_documents=cited_doc_ids,
            evidence=evidence_list,
            cited_doc_ids=cited_doc_ids,
        )

    def _iter_corpus(self, limit: int | None = None) -> Iterator[CorpusDocument]:
        """Iterate over corpus documents."""
        corpus = self._load_corpus()
        for i, row in enumerate(corpus):
            if limit is not None and i >= limit:
                break
            yield self._transform_corpus_document(row)

    def _iter_ground_truth(self, limit: int | None = None) -> Iterator[GroundTruthEntry]:
        """Iterate over ground truth entries."""
        claims = self._load_claims()
        for i, row in enumerate(claims):
            if limit is not None and i >= limit:
                break
            yield self._transform_ground_truth(row)

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

        records = [entry.model_dump() for entry in self._iter_ground_truth(limit=limit)]
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
            "source": self.DATASET_NAME,
            "corpus_config": "scifact_corpus_source",
            "claims_config": "scifact_claims_source",
            "corpus_count": self._corpus_count,
            "ground_truth_count": self._ground_truth_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        output_path = output_dir / "metadata.json"
        output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

        return output_path
