"""Embedding service for creating and populating ChromaDB collections.

This service takes corpus documents in the generic format and creates
searchable vector collections. It is completely dataset-agnostic -
it only requires documents with {id, text} interface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

import chromadb

from shared_types import CorpusDocument
from shared_types.schemas import ChunkingConfig, RetrievalConfig
from worker.services.chromadb_bridge import LlamaEmbeddingFunction
from worker.services.collection_registry import CollectionRegistry

if TYPE_CHECKING:
    from worker.models.embedder import Embedder


def _is_missing_collection_error(exc: Exception) -> bool:
    """Return True if exception indicates a missing Chroma collection."""
    return isinstance(exc, ValueError) or exc.__class__.__name__ == "NotFoundError"


@dataclass
class EmbeddingProgress:
    """Progress information for embedding callback."""

    total_documents: int
    documents_processed: int
    total_chunks: int
    chunks_embedded: int
    current_batch: int
    total_batches: int


class ProgressCallback(Protocol):
    """Protocol for progress callbacks."""

    def __call__(self, progress: EmbeddingProgress) -> None:
        """Called with progress updates during embedding."""
        ...


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""

    collection_path: Path
    collection_name: str
    total_documents: int
    total_chunks: int
    already_existed: bool


class EmbeddingService:
    """Creates and populates ChromaDB collections from corpus documents.

    This service is completely dataset-agnostic - it only requires
    documents with {id, text} interface via CorpusDocument.
    """

    DEFAULT_BATCH_SIZE = 32
    CHROMA_COLLECTION_NAME = "chunks"

    def __init__(
        self,
        embedder: Embedder,
        collections_dir: Path | None = None,
    ):
        """Initialize the embedding service.

        Args:
            embedder: The Embedder model wrapper for generating embeddings.
            collections_dir: Directory for ChromaDB collections.
                Falls back to LOCAL_COLLECTIONS_DIR environment variable.
        """
        self._embedder = embedder
        self._embedding_fn = LlamaEmbeddingFunction(embedder)

        if collections_dir is None:
            collections_dir = Path(os.environ.get("LOCAL_COLLECTIONS_DIR", "./collections"))

        self._collections_dir = collections_dir
        self._registry = CollectionRegistry(collections_dir)

    def embed_corpus(
        self,
        corpus: list[CorpusDocument],
        dataset_id: str,
        retrieval_config: RetrievalConfig,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress_callback: Callable[[EmbeddingProgress], None] | None = None,
    ) -> EmbeddingResult:
        """Embed a corpus and populate a ChromaDB collection.

        Args:
            corpus: List of documents with id and text fields.
            dataset_id: Identifier for the dataset (e.g., "scifact").
            retrieval_config: Configuration including model, chunking, etc.
            batch_size: Number of chunks to embed per batch.
            progress_callback: Optional callback for progress updates.

        Returns:
            EmbeddingResult with collection info and statistics.
        """
        collection_path = self._registry.get_or_create_collection(dataset_id, retrieval_config)
        collection_name = collection_path.name

        # Check if collection already exists and is populated
        client = chromadb.PersistentClient(path=str(collection_path))

        try:
            existing = client.get_collection(name=self.CHROMA_COLLECTION_NAME)
            if existing.count() > 0:
                return EmbeddingResult(
                    collection_path=collection_path,
                    collection_name=collection_name,
                    total_documents=len(corpus),
                    total_chunks=existing.count(),
                    already_existed=True,
                )
        except Exception as e:
            if _is_missing_collection_error(e):
                pass  # Collection doesn't exist yet
            else:
                raise

        # Chunk documents
        chunks = self._chunk_documents(corpus, retrieval_config.chunking)

        # Create collection with embedding function
        collection = client.get_or_create_collection(
            name=self.CHROMA_COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={
                "dataset": dataset_id,
                "model": retrieval_config.model,
                "quantization": retrieval_config.quantization,
                "dimensions": retrieval_config.dimensions,
            },
        )

        # Embed and populate
        self._embed_and_populate(
            collection=collection,
            chunks=chunks,
            batch_size=batch_size,
            progress_callback=progress_callback,
            total_documents=len(corpus),
        )

        return EmbeddingResult(
            collection_path=collection_path,
            collection_name=collection_name,
            total_documents=len(corpus),
            total_chunks=len(chunks),
            already_existed=False,
        )

    def collection_exists(
        self,
        dataset_id: str,
        retrieval_config: RetrievalConfig,
    ) -> bool:
        """Check if a populated collection already exists for this config.

        Args:
            dataset_id: Dataset identifier.
            retrieval_config: Retrieval configuration.

        Returns:
            True if collection exists and has documents, False otherwise.
        """
        collection_path = self._registry.resolve_collection_path(dataset_id, retrieval_config)
        if collection_path is None:
            return False

        client = chromadb.PersistentClient(path=str(collection_path))

        try:
            collection = client.get_collection(name=self.CHROMA_COLLECTION_NAME)
            return collection.count() > 0
        except Exception as e:
            if _is_missing_collection_error(e):
                return False
            raise

    def resolve_collection_path(
        self,
        dataset_id: str,
        retrieval_config: RetrievalConfig,
    ) -> Path | None:
        """Resolve collection folder path for exact dataset/config match."""
        return self._registry.resolve_collection_path(dataset_id, retrieval_config)

    def _chunk_documents(
        self,
        corpus: list[CorpusDocument],
        chunking_config: ChunkingConfig | None,
    ) -> list[tuple[str, str, dict]]:
        """Chunk documents according to strategy.

        Returns:
            List of (chunk_id, chunk_text, metadata) tuples.
        """
        chunks: list[tuple[str, str, dict]] = []

        for doc in corpus:
            if chunking_config is None:
                # No chunking - use full document
                chunks.append((
                    doc.id,
                    doc.text,
                    {"doc_id": doc.id, "chunk_index": 0, **doc.metadata},
                ))
                continue

            if chunking_config.strategy == "fixed":
                doc_chunks = self._chunk_fixed(
                    doc_id=doc.id,
                    text=doc.text,
                    metadata=doc.metadata,
                    chunk_size=chunking_config.chunk_size or 500,
                    chunk_overlap=chunking_config.chunk_overlap,
                )
            else:  # char_split
                doc_chunks = self._chunk_char_split(
                    doc_id=doc.id,
                    text=doc.text,
                    metadata=doc.metadata,
                    split_sequence=chunking_config.split_sequence or ". ",
                )

            chunks.extend(doc_chunks)

        return chunks

    def _chunk_fixed(
        self,
        doc_id: str,
        text: str,
        metadata: dict,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[tuple[str, str, dict]]:
        """Fixed-size chunking with overlap.

        Args:
            doc_id: Document identifier.
            text: Document text to chunk.
            metadata: Document metadata to include in chunks.
            chunk_size: Target size of each chunk in characters.
            chunk_overlap: Overlap between consecutive chunks.

        Returns:
            List of (chunk_id, chunk_text, metadata) tuples.
        """
        chunks: list[tuple[str, str, dict]] = []

        if not text.strip():
            return chunks

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            if chunk_text.strip():  # Skip empty chunks
                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                chunk_metadata = {
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    **metadata,
                }
                chunks.append((chunk_id, chunk_text, chunk_metadata))
                chunk_index += 1

            # Move to next position with overlap
            next_start = end - chunk_overlap
            if next_start <= start:
                # Avoid infinite loop if overlap >= chunk_size
                next_start = end
            start = next_start

            # Stop if we've processed the entire text
            if end >= len(text):
                break

        return chunks

    def _chunk_char_split(
        self,
        doc_id: str,
        text: str,
        metadata: dict,
        split_sequence: str,
    ) -> list[tuple[str, str, dict]]:
        """Character/sequence-based splitting.

        Args:
            doc_id: Document identifier.
            text: Document text to chunk.
            metadata: Document metadata to include in chunks.
            split_sequence: Sequence to split on (e.g., ". ", "\\n").

        Returns:
            List of (chunk_id, chunk_text, metadata) tuples.
        """
        chunks: list[tuple[str, str, dict]] = []
        parts = text.split(split_sequence)

        for idx, part in enumerate(parts):
            chunk_text = part.strip()
            if chunk_text:
                chunk_id = f"{doc_id}_chunk_{idx}"
                chunk_metadata = {
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    **metadata,
                }
                chunks.append((chunk_id, chunk_text, chunk_metadata))

        return chunks

    def _embed_and_populate(
        self,
        collection: chromadb.Collection,
        chunks: list[tuple[str, str, dict]],
        batch_size: int,
        progress_callback: Callable[[EmbeddingProgress], None] | None,
        total_documents: int,
    ) -> None:
        """Embed chunks in batches and add to collection.

        Args:
            collection: ChromaDB collection to populate.
            chunks: List of (chunk_id, chunk_text, metadata) tuples.
            batch_size: Number of chunks per batch.
            progress_callback: Optional callback for progress updates.
            total_documents: Total number of source documents.
        """
        total_chunks = len(chunks)
        if total_chunks == 0:
            return

        total_batches = (total_chunks + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total_chunks)
            batch = chunks[start:end]

            ids = [c[0] for c in batch]
            documents = [c[1] for c in batch]
            metadatas = [c[2] for c in batch]

            # ChromaDB will use embedding_function to generate embeddings
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

            if progress_callback:
                progress_callback(
                    EmbeddingProgress(
                        total_documents=total_documents,
                        documents_processed=total_documents,  # Chunking is complete
                        total_chunks=total_chunks,
                        chunks_embedded=end,
                        current_batch=batch_idx + 1,
                        total_batches=total_batches,
                    )
                )
