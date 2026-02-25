"""Dataset loading and ground truth management for evaluation."""

from orchestrator.datasets.huggingface_ragbench import HuggingFaceRAGBench
from orchestrator.datasets.huggingface_scifact import HuggingFaceSciFact
from orchestrator.datasets.schemas import (
    DocumentEvidence,
    GroundTruthEntry,
    SentenceEvidence,
)

__all__ = [
    "DocumentEvidence",
    "GroundTruthEntry",
    "HuggingFaceRAGBench",
    "HuggingFaceSciFact",
    "SentenceEvidence",
]
