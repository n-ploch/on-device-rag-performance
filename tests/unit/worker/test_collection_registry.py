"""Tests defining chunking-aware collection resolution and tree metadata."""

from __future__ import annotations

import json
from unittest.mock import patch

from shared_types.schemas import ChunkingConfig, RetrievalConfig


def _fixed(chunk_size: int, chunk_overlap: int = 64) -> RetrievalConfig:
    return RetrievalConfig(
        model="intfloat/multilingual-e5-small",
        quantization="fp16",
        dimensions=384,
        chunking=ChunkingConfig(
            strategy="fixed",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        k=3,
    )


def _char_split(split_sequence: str = ". ") -> RetrievalConfig:
    return RetrievalConfig(
        model="intfloat/multilingual-e5-small",
        quantization="fp16",
        dimensions=384,
        chunking=ChunkingConfig(
            strategy="char_split",
            split_sequence=split_sequence,
        ),
        k=3,
    )


class TestCollectionRegistry:
    def test_different_strategy_same_model_quant_dims_creates_new_collection(self, tmp_path):
        with patch.dict("os.environ", {"LOCAL_COLLECTIONS_DIR": str(tmp_path)}):
            from worker.services.collection_registry import CollectionRegistry

            registry = CollectionRegistry()
            fixed_path = registry.get_or_create_collection("scifact", _fixed(chunk_size=500))
            char_path = registry.get_or_create_collection("scifact", _char_split(split_sequence=". "))

            assert fixed_path != char_path
            assert fixed_path.name.endswith("__384_0")
            assert char_path.name.endswith("__384_1")

    def test_same_strategy_different_chunk_params_creates_new_collection(self, tmp_path):
        with patch.dict("os.environ", {"LOCAL_COLLECTIONS_DIR": str(tmp_path)}):
            from worker.services.collection_registry import CollectionRegistry

            registry = CollectionRegistry()
            first = registry.get_or_create_collection("scifact", _fixed(chunk_size=500, chunk_overlap=64))
            second = registry.get_or_create_collection("scifact", _fixed(chunk_size=800, chunk_overlap=32))

            assert first != second

    def test_tree_index_separates_fixed_vs_char_split_nodes(self, tmp_path):
        with patch.dict("os.environ", {"LOCAL_COLLECTIONS_DIR": str(tmp_path)}):
            from worker.services.collection_registry import CollectionRegistry

            registry = CollectionRegistry()
            registry.get_or_create_collection("scifact", _fixed(chunk_size=500))
            registry.get_or_create_collection("scifact", _char_split(split_sequence=". "))

            tree = json.loads((tmp_path / "metadata.json").read_text())
            chunking = (
                tree["datasets"]["scifact"]["models"]["multilingual-e5-small"]["quantizations"]["fp16"]["dimensions"]["384"][
                    "chunking"
                ]
            )
            assert "fixed" in chunking
            assert "char_split" in chunking

    def test_tree_index_separates_fixed_param_variants(self, tmp_path):
        with patch.dict("os.environ", {"LOCAL_COLLECTIONS_DIR": str(tmp_path)}):
            from worker.services.collection_registry import CollectionRegistry

            registry = CollectionRegistry()
            registry.get_or_create_collection("scifact", _fixed(chunk_size=500, chunk_overlap=64))
            registry.get_or_create_collection("scifact", _fixed(chunk_size=500, chunk_overlap=16))

            tree = json.loads((tmp_path / "metadata.json").read_text())
            fixed_nodes = (
                tree["datasets"]["scifact"]["models"]["multilingual-e5-small"]["quantizations"]["fp16"]["dimensions"]["384"][
                    "chunking"
                ]["fixed"]
            )
            assert len(fixed_nodes) == 2

    def test_exact_match_with_same_chunking_reuses_collection(self, tmp_path):
        with patch.dict("os.environ", {"LOCAL_COLLECTIONS_DIR": str(tmp_path)}):
            from worker.services.collection_registry import CollectionRegistry

            registry = CollectionRegistry()
            config = _fixed(chunk_size=500)
            path_a = registry.get_or_create_collection("scifact", config)
            path_b = registry.get_or_create_collection("scifact", config)

            assert path_a == path_b

            tree = json.loads((tmp_path / "metadata.json").read_text())
            fixed_nodes = (
                tree["datasets"]["scifact"]["models"]["multilingual-e5-small"]["quantizations"]["fp16"]["dimensions"]["384"][
                    "chunking"
                ]["fixed"]
            )
            signatures = list(fixed_nodes.values())
            by_hash = signatures[0]["by_config_hash"]
            assert len(by_hash) == 1

            leaf_metadata = json.loads((path_a / "metadata.json").read_text())
            assert leaf_metadata["retrieval_config"] == config.model_dump(mode="json", exclude_none=True)
