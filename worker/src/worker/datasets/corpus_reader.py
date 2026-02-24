"""Generic corpus reader for pre-downloaded datasets.

This module provides a dataset-agnostic interface for reading corpus
documents from local Parquet files. The worker uses this to load
corpus documents for embedding without knowledge of the original
dataset structure.

Storage is organized as: $LOCAL_DATASETS_DIR/{dataset_id}/corpus.parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from shared_types import CorpusDocument, get_dataset_dir


class CorpusReader:
    """Generic corpus reader for local Parquet files.

    Reads corpus documents in the standardized format:
    {id: str, text: str, metadata: dict}

    The reader is completely dataset-agnostic - it works with any
    corpus that follows this schema, regardless of the original
    dataset source.
    """

    def __init__(self, corpus_path: Path):
        """Initialize the corpus reader.

        Args:
            corpus_path: Path to the corpus.parquet file.

        Raises:
            FileNotFoundError: If the corpus file doesn't exist.
        """
        self._corpus_path = Path(corpus_path)

        if not self._corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus not found at {self._corpus_path}. "
                f"Run dataset export first to create the corpus file."
            )

    @classmethod
    def from_dataset_id(
        cls,
        dataset_id: str,
        datasets_dir: Path | None = None,
    ) -> "CorpusReader":
        """Create a CorpusReader for a dataset by ID.

        Resolves the corpus path using LOCAL_DATASETS_DIR environment variable
        and the dataset_id: {datasets_dir}/{dataset_id}/corpus.parquet

        Args:
            dataset_id: Dataset identifier (e.g., 'scifact').
            datasets_dir: Base datasets directory. If None, uses LOCAL_DATASETS_DIR
                environment variable or falls back to ./local/datasets.

        Returns:
            CorpusReader instance for the dataset.

        Raises:
            FileNotFoundError: If the corpus file doesn't exist.
        """
        dataset_dir = get_dataset_dir(dataset_id, datasets_dir)
        corpus_path = dataset_dir / "corpus.parquet"
        return cls(corpus_path)

    @property
    def path(self) -> Path:
        """Return the path to the corpus file."""
        return self._corpus_path

    def count(self) -> int:
        """Return total document count without loading data.

        Returns:
            Number of documents in the corpus.
        """
        metadata = pq.read_metadata(self._corpus_path)
        return metadata.num_rows

    def read_all(self) -> list[CorpusDocument]:
        """Read entire corpus into memory.

        Returns:
            List of all corpus documents.
        """
        table = pq.read_table(self._corpus_path)
        return [CorpusDocument(**row) for row in table.to_pylist()]

    def read_batched(
        self,
        batch_size: int = 1000,
    ) -> Iterator[list[CorpusDocument]]:
        """Read corpus in memory-efficient batches.

        Args:
            batch_size: Number of documents per batch.

        Yields:
            Lists of CorpusDocument objects.
        """
        parquet_file = pq.ParquetFile(self._corpus_path)

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield [CorpusDocument(**row) for row in batch.to_pylist()]

    def read_by_ids(self, doc_ids: list[str]) -> list[CorpusDocument]:
        """Read specific documents by their IDs.

        Note: This loads the entire corpus and filters in memory.
        For large corpora with frequent lookups, consider building
        an index.

        Args:
            doc_ids: List of document IDs to retrieve.

        Returns:
            List of matching documents (order not guaranteed).
        """
        doc_id_set = set(doc_ids)
        table = pq.read_table(self._corpus_path)

        documents = []
        for row in table.to_pylist():
            if row["id"] in doc_id_set:
                documents.append(CorpusDocument(**row))

        return documents

    def __len__(self) -> int:
        """Return the number of documents in the corpus."""
        return self.count()

    def __iter__(self) -> Iterator[CorpusDocument]:
        """Iterate over all documents one at a time.

        Memory-efficient iteration for processing documents sequentially.

        Yields:
            Individual CorpusDocument objects.
        """
        for batch in self.read_batched(batch_size=100):
            yield from batch
