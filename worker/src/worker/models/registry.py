"""Model registry resolution for local GGUF artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from shared_types.naming import model_storage_key


def _load_registry(models_dir: Path) -> dict:
    registry_path = models_dir / "model_registry.json"
    return json.loads(registry_path.read_text())


def get_model_path(repo: str, quantization: str) -> Path:
    """Resolve the local path for a model/quantization pair."""
    models_root = Path(os.environ["LOCAL_MODELS_DIR"])
    registry = _load_registry(models_root)

    model_entry = registry["models"][repo]
    quant_entry = model_entry[quantization]
    filename = quant_entry["filename"]

    storage_dir = models_root / model_storage_key(repo, quantization)
    return storage_dir / filename
