"""Collection registry and resolver for retrieval corpora."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from shared_types.naming import collection_base_key, normalize_model_name
from shared_types.schemas import ChunkingConfig, RetrievalConfig


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _hash_payload(payload: dict) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _chunking_strategy_and_signature(chunking: ChunkingConfig | None) -> tuple[str, str]:
    if chunking is None:
        return "none", "none"

    if chunking.strategy == "fixed":
        return "fixed", f"chunk_size={chunking.chunk_size};chunk_overlap={chunking.chunk_overlap}"

    return "char_split", f"split_sequence={chunking.split_sequence}"


class CollectionRegistry:
    """Maintains collection folders and a tree index metadata file."""

    INDEX_FILENAME = "metadata.json"

    def __init__(self, collections_dir: Path | None = None):
        if collections_dir is None:
            collections_dir = Path(os.environ["LOCAL_COLLECTIONS_DIR"])

        self.collections_dir = collections_dir
        self.collections_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.collections_dir / self.INDEX_FILENAME

    def get_or_create_collection(self, dataset_id: str, retrieval_config: RetrievalConfig) -> Path:
        """Return existing collection path for exact config, or create a new one."""
        index = self._load_index()
        retrieval_payload = retrieval_config.model_dump(mode="json", exclude_none=True)
        config_hash = _hash_payload(retrieval_payload)

        model_name = normalize_model_name(retrieval_config.model)
        strategy, signature = _chunking_strategy_and_signature(retrieval_config.chunking)
        signature_node = (
            index.setdefault("datasets", {})
            .setdefault(dataset_id, {"models": {}})
            .setdefault("models", {})
            .setdefault(model_name, {"quantizations": {}})
            .setdefault("quantizations", {})
            .setdefault(retrieval_config.quantization, {"dimensions": {}})
            .setdefault("dimensions", {})
            .setdefault(str(retrieval_config.dimensions), {"chunking": {}})
            .setdefault("chunking", {})
            .setdefault(strategy, {})
            .setdefault(signature, {"by_config_hash": {}, "entries": []})
        )

        existing = signature_node["by_config_hash"].get(config_hash)
        if existing is not None:
            return Path(existing["path"])

        collection_name = self._allocate_collection_name(retrieval_config)
        collection_path = self.collections_dir / collection_name
        collection_path.mkdir(parents=True, exist_ok=False)

        created_at = datetime.now(timezone.utc).isoformat()
        collection_metadata = {
            "dataset": dataset_id,
            "collection_name": collection_name,
            "created_at": created_at,
            "retrieval_config": retrieval_payload,
            "chunking": {"strategy": strategy, "signature": signature},
            "config_hash": config_hash,
        }
        (collection_path / self.INDEX_FILENAME).write_text(json.dumps(collection_metadata, indent=2, sort_keys=True))

        entry = {
            "collection_name": collection_name,
            "path": str(collection_path),
            "created_at": created_at,
        }
        signature_node["by_config_hash"][config_hash] = entry
        signature_node["entries"].append(
            {
                "config_hash": config_hash,
                "collection_name": collection_name,
                "path": str(collection_path),
                "retrieval_config": retrieval_payload,
            }
        )

        self._write_index(index)
        return collection_path

    def _load_index(self) -> dict:
        if not self.index_path.exists():
            return {"version": 1, "datasets": {}}
        return json.loads(self.index_path.read_text())

    def _write_index(self, index: dict) -> None:
        self.index_path.write_text(json.dumps(index, indent=2, sort_keys=True))

    def _allocate_collection_name(self, retrieval_config: RetrievalConfig) -> str:
        base = collection_base_key(
            retrieval_config.model,
            retrieval_config.quantization,
            retrieval_config.dimensions,
        )
        pattern = re.compile(rf"^{re.escape(base)}_(\d+)$")

        max_index = -1
        for path in self.collections_dir.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if match:
                max_index = max(max_index, int(match.group(1)))

        return f"{base}_{max_index + 1}"
