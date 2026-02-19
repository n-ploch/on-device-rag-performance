"""Deterministic retrieval and abstention metrics."""

from __future__ import annotations


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(top_k)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    if not retrieved or not relevant:
        return 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def detect_abstention(output: str) -> bool:
    text = output.strip().lower()
    if not text:
        return True

    abstention_patterns = [
        "i don't know",
        "insufficient information",
        "cannot determine",
    ]
    return any(pattern in text for pattern in abstention_patterns)


def abstention_rate(outputs: list[str]) -> float:
    if not outputs:
        return 0.0
    abstentions = sum(1 for output in outputs if detect_abstention(output))
    return abstentions / len(outputs)
