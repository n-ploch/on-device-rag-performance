"""Orchestrator-side registry resolution mirroring worker logic."""

from __future__ import annotations

import json
import os
from pathlib import Path

from shared_types.naming import model_storage_key


def _load_registry(models_dir: Path) -> dict:
    return json.loads((models_dir / "model_registry.json").read_text())


def get_model_path(repo: str, quantization: str) -> Path:
    """Resolve model path using canonical `modelname__quantization` keying."""
    models_root = Path(os.environ["LOCAL_MODELS_DIR"])
    registry = _load_registry(models_root)
    filename = registry["models"][repo][quantization]["filename"]
    return models_root / model_storage_key(repo, quantization) / filename
