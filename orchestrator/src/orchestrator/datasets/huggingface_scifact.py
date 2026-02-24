"""HuggingFace SciFact dataset loader.

Loads the BeIR/scifact dataset and transforms it to the standard
corpus and ground truth formats for the RAG evaluation system.

SciFact is a scientific claim verification dataset where:
- Corpus: Scientific paper abstracts
- Queries: Scientific claims to verify
- Qrels: Relevance judgments mapping claims to supporting documents
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
    """Dataset loader for SciFact from HuggingFace Hub (BeIR format).

    Uses the BeIR/scifact dataset which contains:
    - corpus.jsonl.gz: {_id, title, text, metadata}
    - queries.jsonl.gz: {_id, text, metadata}
    - qrels (from BeIR/scifact-qrels): query-id, corpus-id, score
    """

    CORPUS_URL = "hf://datasets/BeIR/scifact/corpus.jsonl.gz"
    QUERIES_URL = "hf://datasets/BeIR/scifact/queries.jsonl.gz"
    QRELS_URL = "hf://datasets/BeIR/scifact-qrels/test.tsv"

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the SciFact loader.

        Args:
            cache_dir: Optional directory for HuggingFace cache.
        """
        self._cache_dir = cache_dir
        self._corpus_dataset: Dataset | None = None
        self._queries_dataset: Dataset | None = None
        self._qrels_dataset: Dataset | None = None
        self._corpus_count: int = 0
        self._ground_truth_count: int = 0

    @property
    def dataset_id(self) -> str:
        """Return the dataset identifier."""
        return "scifact"

    def _load_corpus(self) -> Dataset:
        """Lazy-load the corpus."""
        if self._corpus_dataset is None:
            from datasets import load_dataset

            logger.info("Loading SciFact corpus from BeIR/scifact")
            self._corpus_dataset = load_dataset(
                "json",
                data_files=self.CORPUS_URL,
                split="train",
                cache_dir=str(self._cache_dir) if self._cache_dir else None,
            )
        return self._corpus_dataset

    def _load_queries(self) -> Dataset:
        """Lazy-load the queries."""
        if self._queries_dataset is None:
            from datasets import load_dataset

            logger.info("Loading SciFact queries from BeIR/scifact")
            self._queries_dataset = load_dataset(
                "json",
                data_files=self.QUERIES_URL,
                split="train",
                cache_dir=str(self._cache_dir) if self._cache_dir else None,
            )
        return self._queries_dataset

    def _load_qrels(self) -> Dataset:
        """Lazy-load the relevance judgments."""
        if self._qrels_dataset is None:
            from datasets import load_dataset

            logger.info("Loading SciFact qrels from BeIR/scifact-qrels")
            self._qrels_dataset = load_dataset(
                "csv",
                data_files=self.QRELS_URL,
                delimiter="\t",
                split="train",
                cache_dir=str(self._cache_dir) if self._cache_dir else None,
            )
        return self._qrels_dataset

    def _build_qrels_index(self) -> dict[str, list[str]]:
        """Build index mapping query_id to list of relevant corpus_ids.

        Returns:
            Dict mapping query_id (str) to list of corpus_ids (str).
        """
        qrels = self._load_qrels()
        index: dict[str, list[str]] = {}
        for row in qrels:
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])
            if query_id not in index:
                index[query_id] = []
            index[query_id].append(corpus_id)
        return index

    def _transform_corpus_document(self, row: dict[str, Any]) -> CorpusDocument:
        """Transform a BeIR corpus row to generic CorpusDocument."""
        return CorpusDocument(
            id=str(row["_id"]),
            text=row.get("text", ""),
            metadata={
                "title": row.get("title"),
            },
        )

    def _transform_ground_truth(
        self,
        row: dict[str, Any],
        qrels_index: dict[str, list[str]],
    ) -> GroundTruthEntry:
        """Transform a SciFact query to GroundTruthEntry."""
        query_id = str(row["_id"])
        supporting_docs = qrels_index.get(query_id, [])

        # BeIR format doesn't have sentence-level evidence, just doc-level
        evidence_list: list[DocumentEvidence] = []
        for doc_id in supporting_docs:
            evidence_list.append(
                DocumentEvidence(
                    doc_id=doc_id,
                    doc_title=None,
                    sentences=[],
                )
            )

        # Determine label based on whether there are supporting docs
        if supporting_docs:
            expected_label = "SUPPORT"
        else:
            expected_label = "NOT_ENOUGH_INFO"

        return GroundTruthEntry(
            id=query_id,
            input=row["text"],
            expected_label=expected_label,
            supporting_documents=supporting_docs,
            evidence=evidence_list,
            cited_doc_ids=supporting_docs,
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
        qrels_index: dict[str, list[str]],
        limit: int | None = None,
    ) -> Iterator[GroundTruthEntry]:
        """Iterate over ground truth entries."""
        queries = self._load_queries()
        for i, row in enumerate(queries):
            if limit is not None and i >= limit:
                break
            yield self._transform_ground_truth(row, qrels_index)

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

        # Build qrels index for relevance judgments
        qrels_index = self._build_qrels_index()

        records = [
            entry.model_dump() for entry in self._iter_ground_truth(qrels_index, limit=limit)
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
            "source": "BeIR/scifact",
            "corpus_url": self.CORPUS_URL,
            "queries_url": self.QUERIES_URL,
            "qrels_url": self.QRELS_URL,
            "corpus_count": self._corpus_count,
            "ground_truth_count": self._ground_truth_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        output_path = output_dir / "metadata.json"
        output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

        return output_path
