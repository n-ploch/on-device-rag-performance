"""Worker model wrappers."""

from worker.models.embedder import Embedder
from worker.models.generator import Generator, GenerationResult
from worker.models.llm import detect_backend
from worker.models.registry import get_model_path

__all__ = ["Embedder", "Generator", "GenerationResult", "detect_backend", "get_model_path"]
