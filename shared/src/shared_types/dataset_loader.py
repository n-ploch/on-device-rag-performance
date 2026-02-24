"""Abstract base class for dataset loaders.

This module defines the interface for dataset loaders that transform
raw datasets into the standardized formats used by the RAG evaluation system:
- Corpus: Generic documents for embedding (worker)
- Ground Truth: Evaluation data with evidence (orchestrator)

Storage is organized as: $LOCAL_DATASETS_DIR/{dataset_id}/
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

DEFAULT_DATASETS_DIR = "./local/datasets"


def get_dataset_dir(dataset_id: str, datasets_dir: Path | None = None) -> Path:
    """Get the storage directory for a dataset.

    Args:
        dataset_id: Dataset identifier (e.g., 'scifact').
        datasets_dir: Base datasets directory. If None, uses LOCAL_DATASETS_DIR
            environment variable or falls back to ./local/datasets.

    Returns:
        Path to the dataset directory: {datasets_dir}/{dataset_id}/
    """
    if datasets_dir is None:
        datasets_dir = Path(os.environ.get("LOCAL_DATASETS_DIR", DEFAULT_DATASETS_DIR))

    return datasets_dir / dataset_id


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders.

    Dataset loaders are responsible for:
    1. Downloading/loading raw dataset from a source
    2. Transforming documents to generic corpus format
    3. Extracting ground truth with evidence for evaluation
    4. Exporting both artifacts to local Parquet files

    The loader is completely independent from the embedding logic.
    """

    @abstractmethod
    def export_corpus(
        self,
        output_dir: Path,
        limit: int | None = None,
    ) -> Path:
        """Export corpus documents to Parquet format.

        The corpus is exported in a generic format that the worker can
        consume without knowledge of the original dataset structure.

        Output schema: {id: str, text: str, metadata: dict}

        Args:
            output_dir: Directory to write corpus.parquet.
            limit: Maximum number of documents to export. None for all.

        Returns:
            Path to the created corpus.parquet file.
        """
        ...

    @abstractmethod
    def export_ground_truth(
        self,
        output_dir: Path,
        limit: int | None = None,
    ) -> Path:
        """Export ground truth data to Parquet format.

        The ground truth contains evaluation samples with evidence
        for computing retrieval and generation metrics.

        Output schema includes:
        - id: Sample identifier
        - input: Query/claim text
        - expected_label: Ground truth label (if applicable)
        - supporting_documents: List of relevant document IDs
        - evidence: Detailed evidence structure (dataset-specific)
        - Additional dataset-specific fields (unpacked, not in metadata)

        Args:
            output_dir: Directory to write ground_truth.parquet.
            limit: Maximum number of samples to export. None for all.

        Returns:
            Path to the created ground_truth.parquet file.
        """
        ...

    @abstractmethod
    def export_metadata(self, output_dir: Path) -> Path:
        """Export dataset metadata to JSON.

        Metadata includes:
        - dataset_id: Unique identifier
        - source: Original data source
        - corpus_count: Number of documents
        - ground_truth_count: Number of evaluation samples
        - created_at: Export timestamp

        Args:
            output_dir: Directory to write metadata.json.

        Returns:
            Path to the created metadata.json file.
        """
        ...

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """Unique identifier for this dataset.

        Returns:
            Short identifier string (e.g., 'scifact', 'msmarco').
        """
        ...

    def get_default_output_dir(self, datasets_dir: Path | None = None) -> Path:
        """Get the default output directory for this dataset.

        Uses LOCAL_DATASETS_DIR environment variable or falls back to ./local/datasets.
        The dataset files are stored under: {datasets_dir}/{dataset_id}/

        Args:
            datasets_dir: Base datasets directory. If None, uses env var.

        Returns:
            Path to this dataset's directory.
        """
        return get_dataset_dir(self.dataset_id, datasets_dir)

    def export_all(
        self,
        output_dir: Path | None = None,
        corpus_limit: int | None = None,
        ground_truth_limit: int | None = None,
    ) -> dict[str, Path]:
        """Export all dataset artifacts to a directory.

        Convenience method that exports corpus, ground truth, and metadata.

        Args:
            output_dir: Directory to write all files. If None, uses the default
                location: $LOCAL_DATASETS_DIR/{dataset_id}/
            corpus_limit: Maximum number of corpus documents.
            ground_truth_limit: Maximum number of ground truth samples.

        Returns:
            Dictionary mapping artifact names to their file paths.
        """
        if output_dir is None:
            output_dir = self.get_default_output_dir()
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        return {
            "corpus": self.export_corpus(output_dir, limit=corpus_limit),
            "ground_truth": self.export_ground_truth(output_dir, limit=ground_truth_limit),
            "metadata": self.export_metadata(output_dir),
        }
