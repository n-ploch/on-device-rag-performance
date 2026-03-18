"""Worker model wrappers."""

from worker.models.llm import detect_backend
from worker.models.registry import get_model_path

__all__ = ["detect_backend", "get_model_path"]
