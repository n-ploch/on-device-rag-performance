"""Shared pytest fixtures for RAG evaluation system tests."""

import pytest
import sys
from pathlib import Path

# Add source directories to path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "shared" / "src"))
sys.path.insert(0, str(ROOT_DIR / "worker" / "src"))
sys.path.insert(0, str(ROOT_DIR / "orchestrator" / "src"))


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Create a temporary models directory with a mock registry."""
    import json

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    registry = {
        "models": {
            "TheBloke/Mistral-7B-Instruct-v0.1-GGUF": {
                "q4_k_m": {
                    "filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                    "size_bytes": 4368439296,
                    "sha256": "abc123",
                }
            },
            "intfloat/multilingual-e5-small": {
                "fp16": {
                    "filename": "multilingual-e5-small-fp16.gguf",
                    "size_bytes": 234567890,
                    "sha256": "def456",
                }
            },
        }
    }

    (models_dir / "model_registry.json").write_text(json.dumps(registry))
    return models_dir


@pytest.fixture
def sample_corpus():
    """Sample SciFact-like corpus for testing."""
    return [
        {"id": "doc1", "text": "Aspirin reduces inflammation and pain."},
        {"id": "doc2", "text": "Ibuprofen is a nonsteroidal anti-inflammatory drug."},
        {"id": "doc3", "text": "Paracetamol is effective for reducing fever."},
        {"id": "doc4", "text": "Statins lower cholesterol levels in blood."},
        {"id": "doc5", "text": "Metformin is used to treat type 2 diabetes."},
    ]


@pytest.fixture
def sample_queries():
    """Sample queries with relevance judgments."""
    return [
        {"id": "q1", "text": "What drugs reduce inflammation?", "relevant": ["doc1", "doc2"]},
        {"id": "q2", "text": "How to lower cholesterol?", "relevant": ["doc4"]},
        {"id": "q3", "text": "Treatment for diabetes?", "relevant": ["doc5"]},
    ]


@pytest.fixture
def sample_qrels(sample_queries):
    """Relevance judgments as dict."""
    return {q["id"]: set(q["relevant"]) for q in sample_queries}
