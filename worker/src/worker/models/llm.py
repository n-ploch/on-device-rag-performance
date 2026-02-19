"""LLM backend detection helpers."""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover - exercised via mocked tests
    torch = None


def detect_backend() -> str:
    """Detect the preferred compute backend in priority order."""
    if torch is None:
        return "cpu"

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"
