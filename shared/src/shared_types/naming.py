"""Shared naming helpers for model and collection storage keys."""

from __future__ import annotations


def normalize_model_name(model_id: str) -> str:
    """Return a local-friendly model name based on repository basename."""
    candidate = model_id.strip()
    if not candidate:
        raise ValueError("model_id cannot be empty")
    return candidate.rsplit("/", 1)[-1]


def model_storage_key(model_id: str, quantization: str) -> str:
    """Canonical storage key for local model directories."""
    return f"{normalize_model_name(model_id)}__{quantization}"


def collection_base_key(model_id: str, quantization: str, dimensions: int) -> str:
    """Base collection naming convention.

    Returns a key like `mistral-embed__q4_k_m__1024` which is used as the
    prefix for numbered collection folders (e.g., `mistral-embed__q4_k_m__1024_0`).
    """
    return f"{normalize_model_name(model_id)}__{quantization}__{dimensions}"
