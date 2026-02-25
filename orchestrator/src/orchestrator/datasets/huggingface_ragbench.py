"""HuggingFace RAGBench dataset loader.

Loads the rungalileo/ragbench dataset and transforms it to the standard
corpus and ground truth formats for the RAG evaluation system.

RAGBench is a QA benchmark where each row contains:
- question: The query to answer
- documents: List of relevant document texts
- response: The expected answer
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from orchestrator.datasets.schemas import GroundTruthEntry
from shared_types import CorpusDocument, DatasetLoader

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


class HuggingFaceRAGBench(DatasetLoader):
    """Dataset loader for RAGBench from HuggingFace Hub.

    RAGBench contains multiple QA subsets. Each row has:
    - question: str
    - documents: list[str]
    - response: str
    - Various evaluation scores
    """

    DATASET_NAME = "rungalileo/ragbench"

    def __init__(
        self,
        subset: str = "emanual",
        split: str = "test",
        cache_dir: Path | None = None,
        token: str | None = None,
    ):
        """Initialize the RAGBench loader.

        Args:
            subset: RAGBench subset to load (e.g., "emanual", "hotpotqa").
            split: Dataset split to load ("train", "validation", or "test").
            cache_dir: Optional directory for HuggingFace cache.
            token: Optional HuggingFace token for authenticated access.
        """
        self._subset = subset
        self._split = split
        self._cache_dir = cache_dir
        self._token = token
        self._dataset: Dataset | None = None
        self._corpus_map: dict[str, str] | None = None
        self._corpus_count: int = 0
        self._ground_truth_count: int = 0

    @property
    def dataset_id(self) -> str:
        """Return the dataset identifier."""
        return f"ragbench_{self._subset}"

    def _load_dataset(self) -> Dataset:
        """Lazy-load the dataset."""
        if self._dataset is None:
            from datasets import load_dataset

            logger.info(
                "Loading RAGBench %s from %s (split=%s)",
                self._subset,
                self.DATASET_NAME,
                self._split,
            )
            self._dataset = load_dataset(
                self.DATASET_NAME,
                self._subset,
                split=self._split,
                cache_dir=str(self._cache_dir) if self._cache_dir else None,
                token=self._token,
            )
        return self._dataset

    def _generate_doc_id(self, text: str) -> str:
        """Generate a unique document ID by hashing content."""
        return f"doc_{hashlib.md5(text.encode()).hexdigest()[:12]}"

    def _build_corpus_map(self) -> dict[str, str]:
        """Build a mapping of document IDs to document texts.

        Extracts unique documents from all rows and assigns IDs.

        Returns:
            Dict mapping doc_id -> doc_text
        """
        if self._corpus_map is not None:
            return self._corpus_map

        dataset = self._load_dataset()
        corpus_map: dict[str, str] = {}

        for row in dataset:
            documents = row.get("documents", [])
            for doc_text in documents:
                if not doc_text:
                    continue
                doc_id = self._generate_doc_id(doc_text)
                if doc_id not in corpus_map:
                    corpus_map[doc_id] = doc_text

        self._corpus_map = corpus_map
        logger.info("Built corpus with %d unique documents", len(corpus_map))
        return corpus_map

    def _transform_corpus_document(self, doc_id: str, text: str) -> CorpusDocument:
        """Transform a document to CorpusDocument format."""
        return CorpusDocument(
            id=doc_id,
            text=text,
            metadata={
                "source": f"{self.DATASET_NAME}/{self._subset}",
            },
        )

    def _transform_ground_truth(self, row: dict[str, Any]) -> GroundTruthEntry:
        """Transform a RAGBench row to GroundTruthEntry.

        RAGBench format:
            - id: str
            - question: str
            - documents: list[str]
            - response: str
        """
        row_id = str(row["id"])
        question = row["question"]
        documents = row.get("documents", [])
        response = row.get("response", "")

        # Map documents to their IDs
        supporting_doc_ids = []
        for doc_text in documents:
            if doc_text:
                doc_id = self._generate_doc_id(doc_text)
                supporting_doc_ids.append(doc_id)

        return GroundTruthEntry(
            id=row_id,
            input=question,
            expected_label=None,  # RAGBench is QA, not classification
            supporting_documents=supporting_doc_ids,
            evidence=[],  # RAGBench has different evidence structure
            expected_response=response,
        )

    def _iter_corpus(self, limit: int | None = None) -> Iterator[CorpusDocument]:
        """Iterate over corpus documents."""
        corpus_map = self._build_corpus_map()
        for i, (doc_id, text) in enumerate(corpus_map.items()):
            if limit is not None and i >= limit:
                break
            yield self._transform_corpus_document(doc_id, text)

    def _iter_ground_truth(self, limit: int | None = None) -> Iterator[GroundTruthEntry]:
        """Iterate over ground truth entries."""
        # Ensure corpus map is built first so doc IDs are consistent
        self._build_corpus_map()

        dataset = self._load_dataset()
        for i, row in enumerate(dataset):
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
            "subset": self._subset,
            "split": self._split,
            "corpus_count": self._corpus_count,
            "ground_truth_count": self._ground_truth_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        output_path = output_dir / "metadata.json"
        output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

        return output_path
